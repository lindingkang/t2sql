"""step4_rejection.py — M5 Step4: 对 Step3 筛出的可用错误题做拒绝采样，构造 DPO 偏好对。

对每道错误题，用当前 SFT 模型 temp=1.0 采 N 条，逐条执行并 strict 匹配 gold：
  - 首次采到 (exec_ok 且 strict 匹配) 的样本作为 chosen（来自模型自身分布，严禁直接用 gold）
  - rejected = Step2 的 SFT 单次原始 pred（模型本来的错误输出）
  采不到 chosen 的题 -> 记入 no-chosen（衡量 DRO 可行性）

用法：
    python m4/analysis/step4_rejection.py --n-samples 8 --limit 20     # 冒烟
    python m4/analysis/step4_rejection.py --n-samples 8                 # 全量
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

import torch  # noqa: E402
from eval_sft_adapter import load_model_and_adapter, SYSTEM, build_user  # noqa: E402
from schema import load_schemas, get_ddl  # noqa: E402
from run_baseline import (  # noqa: E402
    SPIDER_DIR, execute_sql, extract_sql, results_equal_strict,
    results_equal_loose,
)

CLEAN_IN = ROOT / "m4" / "analysis" / "step3_clean_errors.jsonl"
OUT_PAIRS = ROOT / "m4" / "analysis" / "step4_pairs.jsonl"
OUT_NOCHOSEN = ROOT / "m4" / "analysis" / "step4_nochosen.jsonl"
OUT_REPORT = ROOT / "m4" / "analysis" / "step4_report.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-questions", type=int, default=800,
                    help="最多采样多少道题（通宵时间预算保护）")
    ap.add_argument("--limit", type=int, default=None, help="冒烟：只处理前 N 题")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    candidates = [
        json.loads(l) for l in CLEAN_IN.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    if args.limit is not None:
        candidates = candidates[: args.limit]
    candidates = candidates[: args.max_questions]
    total = len(candidates)
    if total == 0:
        print("no candidates; abort")
        return

    schemas = load_schemas()
    tokenizer, model = load_model_and_adapter()

    pairs: list[dict] = []
    nochosen: list[dict] = []
    t0 = time.time()
    chosen_counter = Counter()
    total_samples = 0

    f_pairs = open(OUT_PAIRS, "w", encoding="utf-8")
    f_no = open(OUT_NOCHOSEN, "w", encoding="utf-8")

    for i, e in enumerate(candidates):
        db_id = e["db_id"]
        question = e["question"]
        gold = e["gold"]
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"
        ddl = get_ddl(schemas[db_id])

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(question, ddl)},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inp["input_ids"].shape[1]

        gold_rows, gold_err = execute_sql(db_path, gold)
        if gold_err is not None:
            # 理论上 Step3 已排除，防御性跳过
            f_no.write(json.dumps({**e, "reason": "gold_err"}, ensure_ascii=False) + "\n")
            continue

        chosen = None
        chosen_raw = None
        n_exec_ok = 0
        n_correct = 0
        for _ in range(args.n_samples):
            with torch.inference_mode():
                gen = model.generate(
                    **inp,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
            pred = extract_sql(raw)
            total_samples += 1
            rows, err = execute_sql(db_path, pred)
            if err is not None:
                continue
            n_exec_ok += 1
            if results_equal_strict(gold_rows, rows):
                n_correct += 1
                if chosen is None:
                    chosen = pred
                    chosen_raw = raw

        if chosen is not None:
            pair = {
                "idx": e.get("idx"),
                "db_id": db_id,
                "question": question,
                "gold": gold,
                "prompt": messages,
                "chosen": chosen,
                "chosen_raw": chosen_raw,
                "rejected": e["pred"],
                "pred_repaired": e.get("pred_repaired"),
                "failure_type": e.get("failure_type"),
                "n_samples": args.n_samples,
                "n_exec_ok": n_exec_ok,
                "n_correct": n_correct,
            }
            pairs.append(pair)
            f_pairs.write(json.dumps(pair, ensure_ascii=False) + "\n")
            chosen_counter["chosen"] += 1
        else:
            nochosen.append({**e, "n_exec_ok": n_exec_ok, "n_correct": n_correct})
            f_no.write(json.dumps({**e, "n_exec_ok": n_exec_ok}, ensure_ascii=False) + "\n")
            chosen_counter["no_chosen"] += 1

        if (i + 1) % 20 == 0:
            f_pairs.flush(); f_no.flush()
            el = time.time() - t0
            print(f"[{i+1}/{total}] pairs={len(pairs)} no_chosen={len(nochosen)} "
                  f"recall={len(pairs)/(i+1):.3f} elapsed={el:.0f}s", flush=True)

    f_pairs.close(); f_no.close()

    recall = len(pairs) / total if total else 0.0
    lines = [
        "=" * 70,
        "M5 Step4 拒绝采样 / DPO 偏好对",
        "=" * 70,
        f"candidates: {total}",
        f"n_samples/题: {args.n_samples}  temp={args.temperature} top_p={args.top_p}",
        f"总采样次数: {total_samples}",
        f"chosen 命中(成对): {len(pairs)}",
        f"no_chosen: {len(nochosen)}",
        f"chosen 回收率: {recall:.4f}",
        "",
        "提示：回收率过低(<0.3)说明 SFT 分布过尖、temp=1.0 采样打不散，DPO 数据稀疏，需要改抬温/加多样化正则。",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"pairs -> {OUT_PAIRS.name}, no_chosen -> {OUT_NOCHOSEN.name}")


if __name__ == "__main__":
    main()