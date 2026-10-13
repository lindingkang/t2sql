"""M4-C：用 SFT LoRA adapter 在 Spider dev 集上做离线评测。

不调外部 API，直接本地 4bit 生成并执行 SQL。
复用 run_baseline 的执行器与评分口径，输出可与 M0/M3 对比的明细与汇总。

用法：
    python m4/eval_sft_adapter.py --limit 50       # 先小样本验证链路
    python m4/eval_sft_adapter.py                  # 全量 1034 条
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
DEFAULT_OUT_JSON = ROOT / "m4" / "sft_eval_result.json"
DEFAULT_OUT_TXT = ROOT / "m4" / "sft_eval_summary.txt"

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


def load_records_or_empty(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="只评估前 N 条；默认全量")
    ap.add_argument("--output-json", default=str(DEFAULT_OUT_JSON))
    ap.add_argument("--output-txt", default=str(DEFAULT_OUT_TXT))
    ap.add_argument("--baseline", default="baseline_full.json",
                    help="M0 baseline 明细，用于 rescued/regressed")
    ap.add_argument("--m3", default="agent_eval_m3.json",
                    help="M3 agent 明细，用于额外对比")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    t_model0 = time.time()
    tokenizer, model = load_model_and_adapter()
    model_load_s = time.time() - t_model0

    dev = json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))
    schemas = load_schemas()
    baseline = load_records_or_empty(ROOT / args.baseline)
    m3 = load_records_or_empty(ROOT / args.m3)

    total = len(dev) if args.limit is None else min(args.limit, len(dev))
    records: list[dict] = []

    n_correct = 0
    n_correct_loose = 0
    n_exec_ok = 0
    n_evaluable = 0
    n_gold_issue = 0
    total_prompt_tokens = 0
    total_new_tokens = 0
    total_s = 0.0

    print(f"SFT eval start: limit={total} model_load_s={model_load_s:.1f}", flush=True)

    for idx, sample in enumerate(dev[:total], start=1):
        sample_t0 = time.time()
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
        prompt_len = inp["input_ids"].shape[1]

        try:
            with torch.inference_mode():
                gen = model.generate(
                    **inp,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = gen[0][prompt_len:]
            raw = tokenizer.decode(new_ids, skip_special_tokens=True)
            pred_sql = extract_sql(raw)
            gen_ok = True
        except Exception as exc:
            raw = ""
            pred_sql = ""
            gen_ok = False
            gen_error = f"{type(exc).__name__}: {exc}"

        new_tokens = max(0, len(gen[0]) - prompt_len) if gen_ok else 0
        total_prompt_tokens += prompt_len
        total_new_tokens += new_tokens
        total_s += time.time() - sample_t0

        rec = {
            **sample,
            "gold": gold_sql,
            "pred_raw": raw,
            "pred": pred_sql if gen_ok else None,
            "gen_ok": gen_ok,
            "gen_error": (gen_error if not gen_ok else None)
                          if "gen_error" in locals() else None,
            "prompt_tokens": prompt_len,
            "new_tokens": new_tokens,
            "exec_success": False,
            "correct": False,
            "correct_loose": False,
            "failure_type": None,
        }
        # 更明确地处理 gen_error
        if not gen_ok:
            rec["gen_error"] = gen_error
        else:
            rec["gen_error"] = None

        if not gen_ok:
            records.append(rec)
            print(f"[{idx}/{total}] GEN_FAIL {db_id}", flush=True)
            continue

        # 执行 pred
        pred_res, pred_err = execute_sql(db_path, pred_sql)
        rec["exec_success"] = pred_err is None
        if pred_err is None:
            n_exec_ok += 1

        gold_res, gold_err = execute_sql(db_path, gold_sql)
        if gold_err is not None:
            n_gold_issue += 1
            rec["failure_type"] = "gold_issue"
        elif pred_err is None:
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
        else:
            rec["failure_type"] = "exec_error"

        mark = "✓" if rec["correct"] else ("!" if rec["exec_success"] else "✗")
        print(f"[{idx}/{total}] {mark} {db_id}", flush=True)
        records.append(rec)

        if idx % 20 == 0:
            Path(args.output_json).write_text(
                json.dumps(records, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    Path(args.output_json).write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    failure_dist = Counter(
        r["failure_type"] for r in records
        if r["failure_type"] and r["failure_type"] != "gold_issue"
    )
    exec_denom = max(1, total - sum(1 for r in records if not r["gen_ok"]))

    # rescued/regressed 相对 M0 baseline
    n_rescued_vs_baseline = 0
    n_regressed_vs_baseline = 0
    for i, r in enumerate(records):
        if i >= len(baseline):
            break
        b = baseline[i]
        if r["correct"] and not b.get("correct"):
            n_rescued_vs_baseline += 1
        elif not r["correct"] and b.get("correct"):
            n_regressed_vs_baseline += 1

    n_rescued_vs_m3 = 0
    n_regressed_vs_m3 = 0
    for i, r in enumerate(records):
        if i >= len(m3):
            break
        m = m3[i]
        if r["correct"] and not m.get("correct"):
            n_rescued_vs_m3 += 1
        elif not r["correct"] and m.get("correct"):
            n_regressed_vs_m3 += 1

    lines = [
        "=" * 70,
        f"SFT adapter evaluation summary @ {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        f"adapter: {ADAPTER_DIR}",
        f"limit: {total}",
        f"model_load_s: {model_load_s:.1f}",
        f"total_time_s: {total_s:.1f}",
        f"avg_time_per_sample_s: {total_s / max(1, total):.2f}",
        "",
        f"samples: {total}",
        f"gen_ok: {sum(1 for r in records if r['gen_ok'])}",
        f"exec_ok: {n_exec_ok}",
        f"evaluable: {n_evaluable}",
        f"gold_issue: {n_gold_issue}",
        f"exec_rate: {n_exec_ok / exec_denom:.4f}",
        f"EA_strict: {n_correct / max(1, n_evaluable):.4f} ({n_correct}/{n_evaluable})",
        f"EA_loose: {n_correct_loose / max(1, n_evaluable):.4f} ({n_correct_loose}/{n_evaluable})",
        "",
        f"avg_prompt_tokens: {total_prompt_tokens / max(1, total):.1f}",
        f"avg_new_tokens: {total_new_tokens / max(1, total):.1f}",
        f"total_prompt_tokens: {total_prompt_tokens}",
        f"total_new_tokens: {total_new_tokens}",
        "",
        f"rescued_vs_baseline: {n_rescued_vs_baseline}",
        f"regressed_vs_baseline: {n_regressed_vs_baseline}",
        f"rescued_vs_m3: {n_rescued_vs_m3}",
        f"regressed_vs_m3: {n_regressed_vs_m3}",
        "",
        "failure_dist:",
    ]
    for ft, cnt in failure_dist.most_common():
        lines.append(f"  {ft:17s}: {cnt}")

    Path(args.output_txt).write_text("\n".join(lines), encoding="utf-8")

    print("DONE", flush=True)
    print("summary_written", args.output_txt, flush=True)


if __name__ == "__main__":
    main()