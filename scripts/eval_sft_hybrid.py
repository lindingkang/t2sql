"""M4.5：SFT adapter + 自洽性投票杂交（SFT hybrid voting）。

动机
----
- SFT 单次生成 ER=0.9594，约 31 条样本连执行都失败（0.0306 * 1034 ≈ 32）。
  投票多次采样让"至少一条可执行"的概率大幅上升，是补齐 SFT 短板的精确工具。
- 但 SFT 后模型分布变尖（peaked），temperature=0.7 采 5 条可能高度雷同，
  投票退化为单采样 → 假性共识。因此：

  1. 先按 --vote-temperature 采 --n-samples 条；
  2. 执行全部可执行 SQL，统计有效结果集签名（松散签名）的簇数；
  3. 若签名簇数 <= --distinct-threshold（默认 2），把温度抬到
     --high-temperature（默认 0.97）逐条补采，直到签名簇数达到
     --target-signatures（默认 3）或补采量达到 --max-lift-samples（默认 10）；
  4. 两批合流后按执行结果签名聚簇，簇内 >= 2 条才形成共识，取最短 SQL；
     无共识就退回首条可执行 SQL。

  核心观测：first_sqldist（首批规范化 SQL 文本去重）、dist_first_exec /
  first_clusters（首批可执行数与签名簇数）、lift_added（补采量）、
  distinct_total（合流后签名簇数）、temp_up（是否触发抬温）。

用法
----
    # 冒烟：看 distinct 分布与抬温触发率，确认多样性问题是否出现
    python m4/eval_sft_hybrid.py --limit 50 \
        --sft-baseline m4/sft_eval_result_50.json

    # 全量
    python m4/eval_sft_hybrid.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from schema import load_schemas, get_ddl  # noqa: E402
from run_baseline import (  # noqa: E402
    SPIDER_DIR,
    execute_sql,
    extract_sql,
    results_equal_strict,
    results_equal_loose,
    classify_failure,
)

BASE_MODEL = "/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct"
ADAPTER_DIR = "/root/autodl-tmp/m4_lora/sft_cons_v1"
DEFAULT_OUT_JSON = ROOT / "m4" / "sft_hybrid_result.json"
DEFAULT_OUT_TXT = ROOT / "m4" / "sft_hybrid_summary.txt"

SYSTEM = (
    "You are a SQLite SQL expert. "
    "Given the database schema below, write a single SQLite SQL query "
    "to answer the question. Output ONLY the SQL query on ONE single line, "
    "with no markdown fences and no explanation."
)


def build_user(question: str, ddl: str) -> str:
    return f"Schema:\n{ddl}\n\nQuestion: {question}\n\nSQL:"


def load_model_and_adapter():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = True
    model = PeftModel.from_pretrained(model, ADAPTER_DIR, is_trainable=False)
    model.eval()
    return tokenizer, model


def load_records_or_empty(path: Path) -> list:
    if not path or not Path(path).exists():
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _sql_key(sql: str) -> str:
    """规范化 SQL 文本用于统计 distinct（只看文本种类，避免空白干扰）。"""
    return " ".join((sql or "").strip().split())


def _result_signature(rows):
    """复刻 agent 的结果集松散签名（忽略行/列顺序、数值类型差异）。"""

    def normalize(v):
        if v is None:
            return ("null", None)
        if isinstance(v, bool):
            return ("num", float(v))
        if isinstance(v, (int, float)):
            return ("num", float(v))
        if isinstance(v, str):
            s = v.strip().lower()
            try:
                return ("num", float(s))
            except ValueError:
                return ("str", s)
        return ("raw", str(v))

    return frozenset(
        tuple(sorted(normalize(v) for v in row)) for row in rows
    )


def _generate_one(model, tokenizer, inp, prompt_len, temperature, max_new_tokens):
    """一次采样生成；返回 (ok, raw, sql, new_tokens, error)。"""
    try:
        if temperature is not None and temperature > 0.0:
            kwargs = dict(do_sample=True, temperature=float(temperature), top_p=0.95)
        else:
            kwargs = dict(do_sample=False, temperature=None, top_p=None)

        with torch.inference_mode():
            gen = model.generate(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs,
            )
        new_ids = gen[0][prompt_len:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True)
        sql = extract_sql(raw)
        new_tokens = int(new_ids.shape[0])
        return True, raw, sql, new_tokens, None
    except Exception as exc:  # noqa: BLE001
        return False, "", "", 0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SFT adapter + 自洽性投票杂交（M4.5）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python m4/eval_sft_hybrid.py --limit 50 "
            "--sft-baseline m4/sft_eval_result_50.json\n"
            "  python m4/eval_sft_hybrid.py\n"
        ),
    )
    ap.add_argument("--limit", type=int, default=None, help="默认全量 1034")
    ap.add_argument("--n-samples", type=int, default=5, help="第一批采样数")
    ap.add_argument("--vote-temperature", type=float, default=0.7)
    ap.add_argument("--high-temperature", type=float, default=0.97)
    ap.add_argument("--distinct-threshold", type=int, default=2,
                    help="第一批有效结果签名数 <= 该值则抬温补采")
    ap.add_argument("--target-signatures", type=int, default=3,
                    help="抬温补采直到有效结果签名数达到该值")
    ap.add_argument("--max-lift-samples", type=int, default=10,
                    help="抬温补采最多新增样本数")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--output-json", default=str(DEFAULT_OUT_JSON))
    ap.add_argument("--output-txt", default=str(DEFAULT_OUT_TXT))
    ap.add_argument("--sft-baseline", default=str(ROOT / "m4" / "sft_eval_result.json"),
                    help="SFT 单次生成明细，用于 rescued/regressed 对比")
    ap.add_argument("--m3-baseline", default=str(ROOT / "m4" / "local_base_m3.json"),
                    help="local base M3 明细，用于额外对比")
    args = ap.parse_args()

    out_json = Path(args.output_json)
    out_txt = Path(args.output_txt)

    t_load0 = time.time()
    tokenizer, model = load_model_and_adapter()
    model_load_s = time.time() - t_load0

    dev = json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))
    schemas = load_schemas()
    sft_baseline = load_records_or_empty(Path(args.sft_baseline))
    m3_baseline = load_records_or_empty(Path(args.m3_baseline))

    total = len(dev) if args.limit is None else min(args.limit, len(dev))

    records: list[dict] = []
    n_correct = 0
    n_correct_loose = 0
    n_exec_ok = 0
    n_evaluable = 0
    n_gold_issue = 0
    total_calls = 0
    total_prompt_tokens = 0
    total_new_tokens = 0
    total_s = 0.0

    print(
        f"[sft_hybrid] start limit={total} model_load_s={model_load_s:.1f} "
        f"n_samples={args.n_samples} T={args.vote_temperature} "
        f"T_high={args.high_temperature} thr={args.distinct_threshold}",
        flush=True,
    )

    for idx, sample in enumerate(dev[:total], start=1):
        q_t0 = time.time()
        db_id = sample["db_id"]
        question = sample["question"]
        gold_sql = sample["query"].strip().rstrip(";")
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"

        ddl = get_ddl(schemas[db_id])
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(question, ddl)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inp = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = int(inp["input_ids"].shape[1])
        total_prompt_tokens += prompt_len

        pool: list[dict] = []
        temp_up = False
        lift_added = 0

        def run_pool(subset):
            """执行 subset 中尚未执行的 SQL 并记录结果签名；
            返回 (可执行条数, 结果签名集合)。"""
            ok_n = 0
            sigs = set()
            for d in subset:
                if not d["ok"] or not _sql_key(d["sql"]):
                    d["exec_ok"] = False
                    continue
                if not d.get("checked"):
                    rows, e = execute_sql(db_path, d["sql"])
                    d["checked"] = True
                    if e is None:
                        d["rows"] = rows
                        d["exec_ok"] = True
                        d["sig"] = _result_signature(rows)
                    else:
                        d["exec_err"] = e
                        d["exec_ok"] = False
                if d.get("exec_ok"):
                    ok_n += 1
                    sigs.add(d["sig"])
            return ok_n, sigs

        # 第一批：常规温度
        for _ in range(args.n_samples):
            ok, raw, sql, ntok, err = _generate_one(
                model, tokenizer, inp, prompt_len,
                args.vote_temperature, args.max_new_tokens,
            )
            total_calls += 1
            total_new_tokens += ntok
            pool.append({
                "ok": ok, "raw": raw, "sql": sql,
                "new_tokens": ntok, "err": err,
                "temperature": args.vote_temperature,
                "rows": None, "exec_ok": False, "sig": None,
                "checked": False, "exec_err": None,
            })

        # 签名会话：第一批 + 动态抬温补采
        dist_first_exec, first_sigs = run_pool(pool)
        first_clusters = len(first_sigs)
        first_sqldist = len({
            _sql_key(d["sql"]) for d in pool
            if d["ok"] and _sql_key(d["sql"])
        })

        if first_clusters <= args.distinct_threshold:
            temp_up = True
            while (len(first_sigs) < args.target_signatures
                   and lift_added < args.max_lift_samples):
                ok, raw, sql, ntok, err = _generate_one(
                    model, tokenizer, inp, prompt_len,
                    args.high_temperature, args.max_new_tokens,
                )
                total_calls += 1
                total_new_tokens += ntok
                d = {
                    "ok": ok, "raw": raw, "sql": sql,
                    "new_tokens": ntok, "err": err,
                    "temperature": args.high_temperature,
                    "rows": None, "exec_ok": False, "sig": None,
                    "checked": False, "exec_err": None,
                }
                pool.append(d)
                lift_added += 1
                # 每条补采即时执行，并重算当前 pool 的有效签名全集
                n_exec_first, first_sigs = run_pool(pool)

        # 确保补采截断后 pool 内所有样本都已执行并带签名
        pool_exec_ok, all_sigs = run_pool(pool)
        distinct_total = len(all_sigs)

        # 结果集签名聚簇：每条只用一次执行结果
        pred_sql = None
        pred_rows = None
        pred_err = "no_executable_sql"
        top_cnt = 0
        n_clusters = 0
        if pool_exec_ok:
            sig_counts: Counter = Counter()
            for d in pool:
                if d.get("exec_ok"):
                    sig_counts[d["sig"]] += 1
            n_clusters = len(sig_counts)
            top_sig, top_cnt = sig_counts.most_common(1)[0]
            if top_cnt >= 2:
                candidates = [
                    d["sql"] for d in pool
                    if d.get("exec_ok") and d["sig"] == top_sig
                ]
                pred_sql = min(candidates, key=len)
                pred_rows = next(
                    d["rows"] for d in pool
                    if d.get("exec_ok") and d["sig"] == top_sig
                    and d["sql"] == pred_sql
                )
                pred_err = None
            else:
                first_exec = next(d for d in pool if d.get("exec_ok"))
                pred_sql = first_exec["sql"]
                pred_rows = first_exec["rows"]
                pred_err = None
        else:
            # 全部执行失败 → 退回第一条非空 SQL；pred_err 保持非 None
            pred_sql = next(
                (d["sql"] for d in pool if d["ok"] and _sql_key(d["sql"])), None
            )

        exec_ok = pred_err is None
        if exec_ok:
            n_exec_ok += 1
        pred_res = pred_rows if exec_ok else []

        rec = {
            **sample,
            "gold": gold_sql,
            "pred": pred_sql,
            "pool_sqls": [d["sql"] for d in pool],
            "distinct_first": first_sqldist,
            "distinct_total": distinct_total,
            "distinct_first_exec": dist_first_exec,
            "first_clusters": first_clusters,
            "lift_added": lift_added,
            "temp_up": temp_up,
            "pool_size": len(pool),
            "pool_exec_ok": pool_exec_ok,
            "vote_top_count": top_cnt,
            "vote_clusters": n_clusters,
            "exec_success": exec_ok,
            "correct": False,
            "correct_loose": False,
            "failure_type": None,
        }

        gold_res, gold_err = execute_sql(db_path, gold_sql)
        if gold_err is not None:
            n_gold_issue += 1
            rec["failure_type"] = "gold_issue"
        elif pred_err is None:
            n_evaluable += 1
            correct = results_equal_strict(gold_res, pred_res)
            correct_loose = results_equal_loose(gold_res, pred_res)
            rec["correct"] = correct
            rec["correct_loose"] = correct_loose
            if correct:
                n_correct += 1
            if correct_loose:
                n_correct_loose += 1
            if not correct:
                rec["failure_type"] = classify_failure(
                    gold_sql, pred_sql, gold_res, pred_res)
        else:
            rec["failure_type"] = "exec_error"

        total_s += time.time() - q_t0
        mark = "✓" if rec["correct"] else ("!" if rec["exec_success"] else "✗")
        print(
            f"[{idx}/{total}] {mark} {db_id} "
            f"dS={first_sqldist} dX={dist_first_exec} c={first_clusters} cT={distinct_total} "
            f"up={int(temp_up)} lift={lift_added} ex={pool_exec_ok} v={top_cnt}",
            flush=True,
        )
        records.append(rec)

        if idx % 20 == 0:
            out_json.write_text(
                json.dumps(records, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    out_json.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 统计 distinct / 抬温分布
    d1_dist = Counter(r["distinct_first"] for r in records)
    dt_dist = Counter(r["distinct_total"] for r in records)
    pool_dist = Counter(r["pool_size"] for r in records)
    n_temp_up = sum(1 for r in records if r["temp_up"])
    n_all_exec_fail = sum(1 for r in records if r["pool_exec_ok"] == 0)
    n_no_consensus = sum(1 for r in records if r["vote_top_count"] < 2)
    avg_d1 = sum(r["distinct_first"] for r in records) / max(1, len(records))
    avg_dt = sum(r["distinct_total"] for r in records) / max(1, len(records))
    avg_pool = sum(r["pool_size"] for r in records) / max(1, len(records))

    failure_dist = Counter(
        r["failure_type"] for r in records
        if r["failure_type"] and r["failure_type"] != "gold_issue"
    )

    # rescued/regressed 相对 SFT 单次
    n_rescued_vs_sft = n_regressed_vs_sft = 0
    for i, r in enumerate(records):
        if i >= len(sft_baseline):
            break
        b = sft_baseline[i]
        if r["correct"] and not b.get("correct"):
            n_rescued_vs_sft += 1
        elif not r["correct"] and b.get("correct"):
            n_regressed_vs_sft += 1

    n_rescued_vs_m3 = n_regressed_vs_m3 = 0
    for i, r in enumerate(records):
        if i >= len(m3_baseline):
            break
        m = m3_baseline[i]
        if r["correct"] and not m.get("correct"):
            n_rescued_vs_m3 += 1
        elif not r["correct"] and m.get("correct"):
            n_regressed_vs_m3 += 1

    lines = [
        "=" * 78,
        f"SFT hybrid voting summary @ {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 78,
        f"adapter: {ADAPTER_DIR}",
        f"limit: {total}",
        f"n_samples_first: {args.n_samples}",
        f"vote_temperature: {args.vote_temperature}",
        f"high_temperature: {args.high_temperature}",
        f"distinct_threshold: {args.distinct_threshold}",
        f"model_load_s: {model_load_s:.1f}",
        f"total_time_s: {total_s:.1f}",
        f"avg_time_per_sample_s: {total_s / max(1, total):.2f}",
        "",
        f"samples: {total}",
        f"exec_ok: {n_exec_ok}",
        f"evaluable: {n_evaluable}",
        f"gold_issue: {n_gold_issue}",
        f"exec_rate: {n_exec_ok / max(1, total):.4f}",
        f"EA_strict: {n_correct / max(1, n_evaluable):.4f} ({n_correct}/{n_evaluable})",
        f"EA_loose: {n_correct_loose / max(1, n_evaluable):.4f} ({n_correct_loose}/{n_evaluable})",
        "",
        "-- diversity diagnostics (本次实验核心指标) --",
        f"distinct_first_counts: {dict(sorted(d1_dist.items()))}",
        f"distinct_total_counts: {dict(sorted(dt_dist.items()))}",
        f"pool_size_counts: {dict(sorted(pool_dist.items()))}",
        f"avg_distinct_first: {avg_d1:.2f}",
        f"avg_distinct_total: {avg_dt:.2f}",
        f"avg_pool_size: {avg_pool:.2f}",
        f"temp_up_samples: {n_temp_up}",
        f"no_consensus_samples: {n_no_consensus}",
        f"all_exec_fail_samples: {n_all_exec_fail}",
        "",
        "-- cost --",
        f"total_calls: {total_calls}",
        f"avg_calls_per_sample: {total_calls / max(1, total):.2f}",
        f"total_prompt_tokens: {total_prompt_tokens}",
        f"total_new_tokens: {total_new_tokens}",
        "",
        "-- vs SFT single --",
        f"rescued_vs_sft: {n_rescued_vs_sft}",
        f"regressed_vs_sft: {n_regressed_vs_sft}",
        "-- vs local base M3 --",
        f"rescued_vs_m3: {n_rescued_vs_m3}",
        f"regressed_vs_m3: {n_regressed_vs_m3}",
        "",
        "failure_dist:",
    ]
    for ft, cnt in failure_dist.most_common():
        lines.append(f"  {ft:17s}: {cnt}")

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print("DONE", flush=True)
    print(f"summary_written={out_txt}", flush=True)


if __name__ == "__main__":
    main()