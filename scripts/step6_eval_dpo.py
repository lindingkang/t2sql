"""step6_eval_dpo.py — M5 Step6: 评测 DPO adapter 在 dev 上的单次生成 + exec 修复。

口径与 M4 SFT / Step1 完全一致（分子/分母统一 1034），可直接对比。
链路：greedy 单次 -> extract -> 执行；失败则 exec_repair 修复再执行 -> strict/loose 评分。

用法：
    python m4/analysis/step6_eval_dpo.py --adapter /root/autodl-tmp/m4_lora/dpo_b01_v1 --limit 20
    python m4/analysis/step6_eval_dpo.py --adapter /root/autodl-tmp/m4_lora/dpo_b01_v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "m4"))
sys.path.insert(0, str(ROOT / "m4" / "analysis"))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

from eval_sft_adapter import SYSTEM, build_user  # noqa: E402
from exec_repair import repair_sql  # noqa: E402
from schema import load_schemas, get_ddl  # noqa: E402
from run_baseline import (  # noqa: E402
    SPIDER_DIR, execute_sql, extract_sql, results_equal_strict,
    results_equal_loose, classify_failure,
)

BASE_MODEL = "/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct"


def load_model(adapter: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.config.use_cache = True
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return tokenizer, model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out-json", default=str(ROOT / "m4" / "analysis" / "dpo_eval_result.json"))
    ap.add_argument("--out-txt", default=str(ROOT / "m4" / "analysis" / "dpo_eval_summary.txt"))
    args = ap.parse_args()

    tokenizer, model = load_model(args.adapter)
    schemas = load_schemas()
    dev = json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))
    total = len(dev) if args.limit is None else min(args.limit, len(dev))

    records = []
    n_correct = 0
    n_correct_loose = 0
    n_exec_ok = 0
    n_repair_fixed = 0
    ft_counter = Counter()
    t0 = time.time()

    for idx, sample in enumerate(dev[:total], start=1):
        db_id = sample["db_id"]
        question = sample["question"]
        gold_sql = sample["query"].strip().rstrip(";")
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"
        ddl = get_ddl(schemas[db_id])

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(question, ddl)},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inp["input_ids"].shape[1]

        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
        pred = extract_sql(raw)

        pred_rows, pred_err = execute_sql(db_path, pred)
        repaired = None
        if pred_err is not None:
            rep, _ = repair_sql(pred, schemas[db_id])
            if rep != pred:
                r_rows, r_err = execute_sql(db_path, rep)
                if r_err is None:
                    n_repair_fixed += 1
                    repaired = rep
                    pred_rows, pred_err = r_rows, None

        exec_ok = pred_err is None
        if exec_ok:
            n_exec_ok += 1
        gold_rows, gold_err = execute_sql(db_path, gold_sql)

        correct = correct_loose = False
        failure_type = None
        if gold_err is not None:
            failure_type = "gold_issue"
        elif exec_ok:
            correct = results_equal_strict(gold_rows, pred_rows)
            correct_loose = results_equal_loose(gold_rows, pred_rows)
            if correct:
                n_correct += 1
            if correct_loose:
                n_correct_loose += 1
            if not correct:
                failure_type = classify_failure(gold_sql, pred, gold_rows, pred_rows)
                ft_counter[failure_type] += 1
        else:
            failure_type = "exec_error"
            ft_counter["exec_error"] += 1

        records.append({
            "idx": idx, "db_id": db_id, "question": question, "gold": gold_sql,
            "pred_raw": raw, "pred": pred, "pred_repaired": repaired,
            "exec_ok": exec_ok, "correct": correct, "correct_loose": correct_loose,
            "failure_type": failure_type,
        })

        if idx % 100 == 0:
            print(f"[{idx}/{total}] corr={n_correct} exec_ok={n_exec_ok} elapsed={time.time()-t0:.0f}s", flush=True)

    Path(args.out_json).write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        "=" * 70,
        f"M5 DPO dev eval: {Path(args.adapter).name}",
        "=" * 70,
        f"samples: {total}",
        f"ER (exec_ok): {n_exec_ok}/{total} = {n_exec_ok/total:.4f}",
        f"EA_strict: {n_correct}/{total} = {n_correct/total:.4f}",
        f"EA_loose: {n_correct_loose}/{total} = {n_correct_loose/total:.4f}",
        f"repair_fixed_exec: {n_repair_fixed}",
        "",
        "对比基线: M4 SFT EA_strict=0.7959 | Step1 修复后 EA_strict=0.8104 | 目标 0.82",
        "",
        "failure_dist:",
    ]
    for ft, cnt in ft_counter.most_common():
        lines.append(f"  {ft:24s}: {cnt}")
    Path(args.out_txt).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("DONE")


if __name__ == "__main__":
    main()