"""eval_harness.py — M2：统一评测框架，支持 prompt 变体 A/B + 多次运行取统计。

为什么需要它：
  M1 发现千帆 temperature=0 也非确定性（单次跑噪声约 ±0.7%）。
  单次 A/B 的数字不可信——可能 improved 比 baseline 高 1% 只是运气。
  本框架对每个变体跑 N 次，输出 mean ± std，只有差距 > 2σ 才算显著提升。

用法：
    # 基线跑 2 次全量（已有 1 次 M0 结果，再补 1 次算方差）
    python eval_harness.py --variant baseline --runs 1 --tag baseline_run2

    # improved 变体跑 2 次全量
    python eval_harness.py --variant improved --runs 2 --tag improved

    # 汇总多次运行出对比报告
    python eval_harness.py --summarize --runs-dir eval_runs

每次运行输出：
    eval_runs/{variant}_{tag}_run{k}.json  单条明细
    eval_runs/{variant}_{tag}_run{k}.txt   该次统计摘要

汇总报告：eval_runs/summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from schema import load_schemas
from run_baseline import (
    SPIDER_DIR,
    execute_sql,
    results_equal_strict,
    results_equal_loose,
    classify_failure,
)
from prompt_variants import build_prompt

ROOT = Path(__file__).resolve().parent

# 从 run_baseline 复用提取逻辑（避免重复造轮子）
from run_baseline import extract_sql


def call_llm_once(client: OpenAI, model: str, prompt: str) -> str:
    """单次 LLM 调用，带重试。和 run_baseline 里的 call_llm 行为一致。"""
    from run_baseline import call_llm
    return call_llm(client, model, prompt)


def run_once(client: OpenAI, model: str, schemas: dict,
             variant: str, limit: int, sleep: float = 0.5) -> tuple[list[dict], dict]:
    """跑一次完整评测，返回 (明细列表, 统计字典)。"""
    dev = json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))
    samples = dev[:limit]
    records = []
    n_evaluable = 0
    n_correct = 0
    n_correct_loose = 0
    n_exec_ok = 0
    n_api_fail = 0

    for idx, sample in enumerate(samples, start=1):
        db_id = sample["db_id"]
        question = sample["question"]
        gold_sql = sample["query"].strip().rstrip(";")
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"

        prompt = build_prompt(db_id, question, schemas[db_id], variant)

        api_error = None
        pred_sql = None
        try:
            raw = call_llm_once(client, model, prompt)
            pred_sql = extract_sql(raw)
        except Exception as exc:
            api_error = str(exc)
        time.sleep(sleep)

        rec = {**sample, "gold": gold_sql, "pred": pred_sql,
               "api_error": api_error, "exec_success": False,
               "correct": False, "correct_loose": False,
               "failure_type": None}

        if api_error:
            n_api_fail += 1
            records.append(rec)
            continue

        pred_res, pred_err = execute_sql(db_path, pred_sql)
        rec["exec_success"] = pred_err is None
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

        mark = "✓" if rec["correct"] else ("!" if rec["exec_success"] else "✗")
        print(f"[{idx}/{len(samples)}] {mark} {db_id}", flush=True)
        records.append(rec)

    stats = {
        "total": len(samples),
        "api_fail": n_api_fail,
        "evaluable": n_evaluable,
        "exec_rate": n_exec_ok / max(1, len(samples) - n_api_fail),
        "ea_strict": n_correct / max(1, n_evaluable),
        "ea_loose": n_correct_loose / max(1, n_evaluable),
        "correct": n_correct,
        "correct_loose": n_correct_loose,
        "failure_dist": dict(Counter(r["failure_type"] for r in records
                                     if r["failure_type"])),
    }
    return records, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="baseline",
                    choices=["baseline", "improved"],
                    help="prompt 变体")
    ap.add_argument("--runs", type=int, default=1, help="运行次数")
    ap.add_argument("--limit", type=int, default=1034)
    ap.add_argument("--tag", default="run", help="本次运行的标签，用于文件名")
    ap.add_argument("--runs-dir", default="eval_runs", help="结果输出目录")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--summarize", action="store_true",
                    help="汇总 runs-dir 下所有结果出对比报告")
    args = ap.parse_args()

    runs_dir = ROOT / args.runs_dir
    runs_dir.mkdir(exist_ok=True)

    # ---------- 汇总模式 ----------
    if args.summarize:
        summarize_runs(runs_dir)
        return

    # ---------- 运行模式 ----------
    load_dotenv(ROOT / ".env")
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        base_url=os.getenv("LLM_BASE_URL", "https://qianfan.baidubce.com/v2").strip(),
    )
    model = os.getenv("LLM_MODEL", "").strip()
    schemas = load_schemas()

    all_stats = []
    for k in range(args.runs):
        print(f"\n{'='*56}\n{args.variant} | run {k+1}/{args.runs} "
              f"(limit={args.limit})\n{'='*56}", flush=True)
        records, stats = run_once(client, model, schemas,
                                  args.variant, args.limit, args.sleep)
        all_stats.append(stats)

        # 保存该次结果
        tag = f"{args.variant}_{args.tag}_run{k+1}"
        (runs_dir / f"{tag}.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        with open(runs_dir / f"{tag}.txt", "w") as f:
            f.write(format_stats(stats, tag))
        print(format_stats(stats, tag), flush=True)

    # 多次运行的统计汇总
    if args.runs > 1:
        eas = [s["ea_strict"] for s in all_stats]
        print(f"\n{args.variant} {args.runs} 次运行 strict EA: "
              f"{statistics.mean(eas):.3f} ± {statistics.stdev(eas):.3f}", flush=True)


def format_stats(s: dict, tag: str) -> str:
    lines = [
        f"=== {tag} ===",
        f"total:      {s['total']}",
        f"api_fail:   {s['api_fail']}",
        f"evaluable:  {s['evaluable']}",
        f"exec_rate:  {s['exec_rate']:.3f}",
        f"EA_strict:  {s['ea_strict']:.3f}  ({s['correct']}/{s['evaluable']})",
        f"EA_loose:   {s['ea_loose']:.3f}  ({s['correct_loose']}/{s['evaluable']})",
        f"failure_dist:",
    ]
    for ft, cnt in sorted(s["failure_dist"].items(), key=lambda x: -x[1]):
        lines.append(f"  {ft:17s}: {cnt}")
    return "\n".join(lines)


def summarize_runs(runs_dir: Path) -> None:
    """汇总目录下所有 .txt 统计，按变体分组输出 mean±std。"""
    from collections import defaultdict
    groups = defaultdict(list)
    for f in sorted(runs_dir.glob("*.txt")):
        variant = f.stem.split("_")[0]
        txt = f.read_text()
        ea = float(re.search(r"EA_strict:\s+([\d.]+)", txt).group(1))
        er = float(re.search(r"exec_rate:\s+([\d.]+)", txt).group(1))
        eal = float(re.search(r"EA_loose:\s+([\d.]+)", txt).group(1))
        groups[variant].append({"ea": ea, "er": er, "eal": eal, "file": f.name})

    lines = ["# M2 评测汇总报告\n",
             "| 变体 | 运行次数 | strict EA (mean±std) | loose EA | exec_rate |",
             "|------|----------|----------------------|----------|-----------|"]
    for variant, runs in sorted(groups.items()):
        eas = [r["ea"] for r in runs]
        eals = [r["eal"] for r in runs]
        ers = [r["er"] for r in runs]
        m_ea = statistics.mean(eas)
        s_ea = statistics.stdev(eas) if len(eas) > 1 else 0
        lines.append(
            f"| {variant} | {len(runs)} | {m_ea:.3f} ± {s_ea:.3f} | "
            f"{statistics.mean(eals):.3f} | {statistics.mean(ers):.3f} |")

    report = "\n".join(lines) + "\n"
    (runs_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
