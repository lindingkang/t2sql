"""本地未微调 Qwen2.5-Coder-7B-Instruct 的统一基准评测。

在 Spider dev 上评测：
  M0   -> 极简 zero-shot，单次生成
  M2   -> improved_v2 prompt，单次生成
  M1.5 -> 自洽性投票（不接反馈闭环/空结果确认）
  M3   -> 完整 Text2SQLAgent（安全层 + 差异化重试 + 投票 + 空结果确认）

用法：
    python m4/eval_local_base.py --mode m0 --limit 20
    python m4/eval_local_base.py --mode m15 --limit 50 --n-samples 5 --adaptive
    python m4/eval_local_base.py --mode m3 --limit 50
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

from schema import load_schemas  # noqa: E402
from run_baseline import (  # noqa: E402
    SPIDER_DIR,
    extract_sql,
    execute_sql,
    results_equal_strict,
    results_equal_loose,
    classify_failure,
)
from prompt_variants import build_prompt as build_variant_prompt  # noqa: E402
from local_llm import load_local_base_model, LocalOpenAIClient  # noqa: E402


def _result_signature(rows: list[tuple]) -> frozenset:
    """复刻 agent 的结果集宽松签名，用于投票聚类。"""
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=["m0", "m2", "m15", "m3"],
                    help="跑哪个基线")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-txt", default=None)
    ap.add_argument("--n-samples", type=int, default=5,
                    help="M1.5/M3 投票采样数")
    ap.add_argument("--vote-temperature", type=float, default=0.7)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--no-confirm-empty", dest="confirm_empty",
                    action="store_false", help="M3 关闭空结果确认")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    mode = args.mode
    out_json = Path(args.output_json) if args.output_json else (
        ROOT / "m4" / f"local_base_{mode}.json")
    out_txt = Path(args.output_txt) if args.output_txt else (
        ROOT / "m4" / f"local_base_{mode}_summary.txt")

    t_start = time.time()
    print(f"[local_base:{mode}] loading model ...", flush=True)
    tokenizer, model = load_local_base_model()
    client = LocalOpenAIClient(tokenizer, model, max_new_tokens=args.max_new_tokens)
    model_load_s = time.time() - t_start

    dev = json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))
    schemas = load_schemas()
    total = len(dev) if args.limit is None else min(args.limit, len(dev))

    m3_agent = None
    if mode == "m3":
        from agent import Text2SQLAgent
        # 关闭 L1 缓存，避免本地确定性缓存改变 M3 语义；日志也保持 None。
        m3_agent = Text2SQLAgent(
            client=client, model="local-qwen-7b-base", schemas=schemas,
            max_retries=args.max_retries, sleep=0.0,
            n_samples=args.n_samples,
            vote_temperature=args.vote_temperature,
            confirm_empty=args.confirm_empty,
            adaptive=args.adaptive,
            llm_cache=None, logger=None,
        )

    records: list[dict] = []
    n_correct = 0
    n_correct_loose = 0
    n_exec_ok = 0
    n_evaluable = 0
    total_gen_tokens = 0
    total_prompt_tokens = 0
    total_calls = 0
    total_s = 0.0

    print(f"[local_base:{mode}] start limit={total} load_s={model_load_s:.1f}",
          flush=True)

    for idx, sample in enumerate(dev[:total], start=1):
        q_t0 = time.time()
        db_id = sample["db_id"]
        question = sample["question"]
        gold_sql = sample["query"].strip().rstrip(";")
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"

        if mode == "m3":
            try:
                res = m3_agent.answer(db_id, question, question_id=str(idx))
            except Exception as exc:
                print(f"[{idx}/{total}] AGENT_CRASH {db_id}: {exc}", flush=True)
                res = {
                    "final_sql": None, "ok": False, "api_error": str(exc),
                    "attempts": None, "votes": None,
                    "empty_confirm": None, "meter": {},
                }
            meter = res.get("meter") or {}
            total_calls += meter.get("api_calls", 0)
            total_prompt_tokens += meter.get("prompt_tokens", 0)
            total_gen_tokens += meter.get("completion_tokens", 0)
            pred_sql = res.get("final_sql")
            exec_ok_base = bool(res.get("ok"))
            rec = {
                **sample, "gold": gold_sql, "pred": pred_sql,
                "api_error": res.get("api_error"),
                "exec_success": exec_ok_base,
                "attempts": res.get("attempts"),
                "votes": res.get("votes"),
                "empty_confirm": res.get("empty_confirm"),
                "meter": meter,
                "correct": False,
                "correct_loose": False,
                "failure_type": None,
            }
            # 评分沿用 eval_harness 口径
            if not res.get("api_error") and pred_sql:
                pred_res, pred_err = execute_sql(db_path, pred_sql)
                rec["exec_success"] = pred_err is None
            else:
                pred_res, pred_err = [], "api_error_or_empty"
            if pred_err is None:
                n_exec_ok += 1
            gold_res, gold_err = execute_sql(db_path, gold_sql)
            if gold_err is None and pred_err is None:
                n_evaluable += 1
                rec["correct"] = results_equal_strict(gold_res, pred_res)
                rec["correct_loose"] = results_equal_loose(gold_res, pred_res)
                if rec["correct"]:
                    n_correct += 1
                if rec["correct_loose"]:
                    n_correct_loose += 1
                if not rec["correct"]:
                    rec["failure_type"] = classify_failure(
                        gold_sql, pred_sql, gold_res, pred_res)
            elif gold_err is not None:
                rec["failure_type"] = "gold_issue"
        else:
            # M0/M2/M1.5：单次或投票
            variant = "baseline" if mode == "m0" else "improved"
            n_samples = args.n_samples if mode == "m15" else 1
            sample_temp = args.vote_temperature if mode == "m15" else 0.0
            prompts = []
            if mode in ("m0", "m2"):
                prompt = build_variant_prompt(
                    db_id, question, schemas[db_id], variant=variant)
                prompts = [prompt]
            else:
                base_prompt = build_variant_prompt(
                    db_id, question, schemas[db_id], variant="baseline")
                prompts = [base_prompt] * n_samples

            raws = []
            sqls = []
            ok = True
            api_error = None
            try:
                for p in prompts:
                    resp = client.chat.completions.create(
                        model="local-qwen-7b-base",
                        messages=[{"role": "user", "content": p}],
                        temperature=sample_temp,
                        max_tokens=args.max_new_tokens,
                    )
                    total_calls += 1
                    u = resp.usage
                    total_prompt_tokens += u.prompt_tokens
                    total_gen_tokens += u.completion_tokens
                    raw = resp.choices[0].message.content or ""
                    raws.append(raw)
                    sqls.append(extract_sql(raw))
            except Exception as exc:
                ok = False
                api_error = f"{type(exc).__name__}: {exc}"

            pred_sql = None
            exec_ok_base = False
            pred_res, pred_err = [], "gen_error"
            if ok:
                if mode in ("m0", "m2"):
                    pred_sql = sqls[0]
                    pred_res, pred_err = execute_sql(db_path, pred_sql)
                    exec_ok_base = pred_err is None
                else:
                    # M1.5：按结果集多数决，无共识则取第一条可执行 SQL
                    exec_sqls = []
                    for s in sqls:
                        rows, err = execute_sql(db_path, s)
                        if err is None:
                            exec_sqls.append((s, rows))
                    sig_counts: Counter[frozenset] = Counter()
                    for _, rows in exec_sqls:
                        sig = _result_signature(rows)
                        sig_counts[sig] += 1
                    if sig_counts:
                        top_sig, top_cnt = sig_counts.most_common(1)[0]
                        if top_cnt >= 2:
                            # 同簇最短 SQL 作为共识
                            candidates = [
                                s for s, rows in exec_sqls
                                if _result_signature(rows) == top_sig
                            ]
                            pred_sql = min(candidates, key=len)
                        elif exec_sqls:
                            pred_sql = exec_sqls[0][0]
                        pred_res, pred_err = execute_sql(db_path, pred_sql)
                        exec_ok_base = pred_err is None

            rec = {
                **sample, "gold": gold_sql, "pred": pred_sql,
                "raw": raws, "api_error": api_error,
                "exec_success": exec_ok_base,
                "correct": False,
                "correct_loose": False,
                "failure_type": None,
            }
            if pred_err is None:
                n_exec_ok += 1
            if ok:
                gold_res, gold_err = execute_sql(db_path, gold_sql)
                if gold_err is None and pred_err is None:
                    n_evaluable += 1
                    rec["correct"] = results_equal_strict(gold_res, pred_res)
                    rec["correct_loose"] = results_equal_loose(gold_res, pred_res)
                    if rec["correct"]:
                        n_correct += 1
                    if rec["correct_loose"]:
                        n_correct_loose += 1
                    if not rec["correct"]:
                        rec["failure_type"] = classify_failure(
                            gold_sql, pred_sql, gold_res, pred_res)
                elif gold_err is not None:
                    rec["failure_type"] = "gold_issue"
                else:
                    rec["failure_type"] = "exec_error"

        total_s += time.time() - q_t0
        mark = "✓" if rec["correct"] else ("!" if rec["exec_success"] else "✗")
        print(f"[{idx}/{total}] {mark} {db_id}", flush=True)
        records.append(rec)

        if idx % 20 == 0:
            out_json.write_text(
                json.dumps(records, indent=2, ensure_ascii=False),
                encoding="utf-8")

    out_json.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    denom = max(1, total)
    lines = [
        "=" * 70,
        f"local base [{mode}] @ {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        f"base: Qwen2.5-Coder-7B-Instruct (no adapter)",
        f"limit: {total}",
        f"model_load_s: {model_load_s:.1f}",
        f"total_time_s: {total_s:.1f}",
        f"avg_time_per_sample_s: {total_s / max(1, total):.2f}",
        "",
        f"samples: {total}",
        f"exec_ok: {n_exec_ok}",
        f"evaluable: {n_evaluable}",
        f"exec_rate: {n_exec_ok / denom:.4f}",
        f"EA_strict: {n_correct / max(1, n_evaluable):.4f} ({n_correct}/{n_evaluable})",
        f"EA_loose: {n_correct_loose / max(1, n_evaluable):.4f} ({n_correct_loose}/{n_evaluable})",
        "",
        f"total_calls: {total_calls}",
        f"avg_calls_per_sample: {total_calls / max(1, total):.2f}",
        f"total_prompt_tokens: {total_prompt_tokens}",
        f"total_gen_tokens: {total_gen_tokens}",
        "",
        "failure_dist:",
    ]
    fail_dist = Counter(r["failure_type"] for r in records if r["failure_type"])
    for ft, cnt in fail_dist.most_common():
        lines.append(f"  {ft:17s}: {cnt}")

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print("DONE", flush=True)
    print(f"summary_written={out_txt}", flush=True)


if __name__ == "__main__":
    main()