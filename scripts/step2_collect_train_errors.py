"""step2_collect_train_errors.py — M5 Step2: SFT 单次 + exec 修复在 train 全量上采集错误清单。

目标：得到 train（train_spider + train_others，8659 条）上"修复后仍错"的样本清单，
供 Step3 gold 噪声筛查 + Step4 拒绝采样构造 DPO 偏好对。

链路：greedy 单次生成 -> extract_sql -> 执行；若执行失败 -> exec_repair 修复 -> 再执行。
记录 raw pred 与 repaired pred、双端执行状态、failure_type。

用法：
    python m4/analysis/step2_collect_train_errors.py --limit 20           # 冒烟
    python m4/analysis/step2_collect_train_errors.py                       # 全量 8659（后台）
    python m4/analysis/step2_collect_train_errors.py --resume              # 断点续跑
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

from eval_sft_adapter import load_model_and_adapter, SYSTEM, build_user  # noqa: E402
from exec_repair import repair_sql  # noqa: E402
from schema import load_schemas, get_ddl  # noqa: E402
from run_baseline import (  # noqa: E402
    SPIDER_DIR, execute_sql, extract_sql,
    results_equal_strict, results_equal_loose, classify_failure,
)

OUT_RECORDS = ROOT / "m4" / "analysis" / "step2_train_records.jsonl"
OUT_ERRORS = ROOT / "m4" / "analysis" / "step2_train_errors.jsonl"
OUT_SUMMARY = ROOT / "m4" / "analysis" / "step2_train_summary.txt"


def load_train():
    ts = json.loads((SPIDER_DIR / "train_spider.json").read_text(encoding="utf-8"))
    to = json.loads((SPIDER_DIR / "train_others.json").read_text(encoding="utf-8"))
    return ts + to


def load_done(path: Path) -> set[int]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    train = load_train()
    schemas = load_schemas()
    total = len(train) if args.limit is None else min(args.limit, len(train))

    tokenizer, model = load_model_and_adapter()

    done = load_done(OUT_RECORDS) if args.resume else set()
    err_f = open(OUT_ERRORS, "a", encoding="utf-8") if args.resume else open(OUT_ERRORS, "w", encoding="utf-8")
    rec_f = open(OUT_RECORDS, "a", encoding="utf-8") if args.resume else open(OUT_RECORDS, "w", encoding="utf-8")

    n = Counter()
    n_repaired = 0
    n_repair_fixed = 0
    t0 = time.time()

    for idx, sample in enumerate(train[:total]):
        if idx in done:
            continue
        db_id = sample["db_id"]
        question = sample["question"]
        gold_sql = sample["query"].strip().rstrip(";")

        if db_id not in schemas:
            print(f"[{idx}/{total}] NOSCHEMA {db_id}", flush=True)
            continue

        ddl = get_ddl(schemas[db_id])
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(question, ddl)},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inp["input_ids"].shape[1]

        gen_ok = True
        try:
            with torch.inference_mode():
                gen = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, temperature=None, top_p=None,
                                     pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
            pred = extract_sql(raw)
        except Exception as exc:
            gen_ok = False
            raw, pred = "", ""

        if not gen_ok:
            n["gen_fail"] += 1
            err_f.write(json.dumps({"idx": idx, "db_id": db_id, "question": question,
                                    "gold": gold_sql, "gen_ok": False}, ensure_ascii=False) + "\n")
            print(f"[{idx}/{total}] GEN_FAIL {db_id}", flush=True)
            continue

        # 执行原始 pred
        pred_rows, pred_err = execute_sql(db_path, pred)
        repaired = pred
        repaired_by = False
        if pred_err is not None:
            repaired, notes = repair_sql(pred, schemas[db_id])
            if repaired != pred:
                repaired_by = True
                n_repaired += 1
                r_rows, r_err = execute_sql(db_path, repaired)
                if r_err is None:
                    n_repair_fixed += 1
                    pred_rows, pred_err = r_rows, None
                else:
                    repaired = pred  # 修复失败，回退原始
                    repaired_by = False

        exec_ok = pred_err is None
        gold_rows, gold_err = execute_sql(db_path, gold_sql)

        correct = correct_loose = False
        failure_type = None
        if gold_err is not None:
            failure_type = "gold_issue"
        elif exec_ok:
            correct = results_equal_strict(gold_rows, pred_rows)
            correct_loose = results_equal_loose(gold_rows, pred_rows)
            if not correct:
                ft = classify_failure(gold_sql, pred, gold_rows, pred_rows)
                failure_type = ft
        else:
            failure_type = "exec_error"

        n["total"] += 1
        if exec_ok:
            n["exec_ok"] += 1
        if correct:
            n["correct"] += 1
        if correct_loose:
            n["correct_loose"] += 1

        rec = {"idx": idx, "db_id": db_id, "question": question, "gold": gold_sql,
               "pred_raw": raw, "pred": pred, "pred_repaired": repaired if repaired_by else None,
               "exec_ok": exec_ok, "correct": correct, "correct_loose": correct_loose,
               "failure_type": failure_type}
        rec_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if not correct:
            err_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if n["total"] % 20 == 0:
            rec_f.flush(); err_f.flush()
        mark = "✓" if correct else ("!" if exec_ok else "✗")
        if n["total"] % 25 == 0:
            el = time.time() - t0
            print(f"[{idx}/{total}] done={n['total']} corr={n['correct']} err_total_so_far={n['total']-n['correct']} "
                  f"repair_fixed={n_repair_fixed} elapsed={el:.0f}s", flush=True)

    rec_f.close(); err_f.close()

    lines = [
        "=" * 70,
        f"SFT train error collection @ {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        f"samples: {total}",
        f"gen_fail: {n.get('gen_fail', 0)}",
        f"exec_ok: {n['exec_ok']}",
        f"correct_strict: {n['correct']} / {total} = {n['correct']/max(1,total):.4f}",
        f"correct_loose: {n['correct_loose']} / {total} = {n['correct_loose']/max(1,total):.4f}",
        f"errors_after_repair: {total - n['correct']}",
        f"repair_applied: {n_repaired}  repair_fixed_exec: {n_repair_fixed}",
        "",
        "failure_dist (errors):",
    ]
    ft_counter = Counter()
    if OUT_RECORDS.exists():
        for line in OUT_RECORDS.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if not r.get("correct") and r.get("failure_type"):
                    ft_counter[r["failure_type"]] += 1
            except Exception:
                pass
    for ft, cnt in ft_counter.most_common():
        lines.append(f"  {ft:20s}: {cnt}")
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("DONE")


if __name__ == "__main__":
    main()