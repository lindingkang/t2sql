"""eval_agent.py — M1 评测：在 dev 集上跑 Agent 闭环，与 M0 baseline 对比。

评测三个问题：
1. Agent 最终 EA 比 M0 的 79.9% 高多少？（主指标，严格口径）
2. 重试机制救回了多少条？（rescued：M0 错 → M1 对）
3. 有没有帮倒忙？（regressed：M0 对 → M1 错；理论上应接近 0，
   因为首轮 prompt/temperature 与 baseline 完全一致）

输出：
- agent_eval_full.json：每条样本的完整 trace（含每轮重试 SQL）
- 终端统计块：EA/ER/救回/退化/平均调用次数

用法：
    python eval_agent.py --limit 20      # 先小跑验证闭环
    python eval_agent.py                 # 全量 1034 条
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from schema import load_schemas
from run_baseline import SPIDER_DIR, execute_sql, results_equal_strict
from agent import Text2SQLAgent

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=1034)
    ap.add_argument("--output", default="agent_eval_full.json")
    ap.add_argument("--baseline", default="baseline_full.json",
                    help="M0 基线结果文件，用于对比救回/退化")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--n-samples", type=int, default=1,
                    help="自洽性投票采样数；1=纯重试，>=2=启用投票")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        base_url=os.getenv("LLM_BASE_URL", "https://qianfan.baidubce.com/v2").strip(),
    )
    model = os.getenv("LLM_MODEL", "").strip()

    dev = json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))
    schemas = load_schemas()
    agent = Text2SQLAgent(client, model, schemas,
                          max_retries=args.max_retries,
                          n_samples=args.n_samples)

    # M0 基线按下标对齐，用于计算 rescued / regressed
    baseline = json.loads((ROOT / args.baseline).read_text(encoding="utf-8"))

    total = min(args.limit, len(dev))
    records = []
    n_evaluable = 0
    n_correct = 0
    n_rescued = 0
    n_regressed = 0
    n_retry_success = 0   # 首轮执行失败、靠重试救回可执行性的条数
    n_api_fail = 0
    total_calls = 0
    n_vote_consensus = 0   # 投票有共识的条数
    n_vote_fallback = 0    # 投票无共识、fallback 到重试的条数

    print(f"开始 M1 Agent 评测：模型={model}，样本={total}，"
          f"最大重试={args.max_retries}\n")

    for idx, sample in enumerate(dev[:total], start=1):
        db_id = sample["db_id"]
        gold_sql = sample["query"].strip().rstrip(";")
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"

        result = agent.answer(db_id, sample["question"])
        total_calls += len(result["attempts"]) + (
            args.n_samples if result.get("votes") else 0)

        # 投票统计
        if result.get("votes"):
            if result["votes"]["consensus_sql"] is not None:
                n_vote_consensus += 1
            else:
                n_vote_fallback += 1

        rec = {**sample, "gold": gold_sql,
               "pred": result["final_sql"], "attempts": result["attempts"],
               "votes": result.get("votes"),
               "api_error": result["api_error"],
               "exec_success": result["ok"], "correct": False}

        if result["api_error"]:
            n_api_fail += 1
            records.append(rec)
            print(f"[{idx}/{total}] API 失败: {db_id}")
            continue

        gold_res, gold_err = execute_sql(db_path, gold_sql)
        if not gold_err and result["ok"]:
            n_evaluable += 1
            pred_res, _ = execute_sql(db_path, result["final_sql"])
            rec["correct"] = results_equal_strict(gold_res, pred_res)
            if rec["correct"]:
                n_correct += 1

            # 与 M0 同位置样本对比
            b = baseline[idx - 1]
            if not b["correct"] and rec["correct"]:
                n_rescued += 1
            elif b["correct"] and not rec["correct"]:
                n_regressed += 1

        # 首轮报错、后续轮次执行成功 = 重试机制的直接收益
        if (len(result["attempts"]) >= 2
                and not result["attempts"][0]["exec_ok"]
                and result["ok"]):
            n_retry_success += 1

        mark = "✓" if rec["correct"] else ("!" if result["ok"] else "✗")
        retry_note = f" (重试{len(result['attempts']) - 1}轮救回)" if (
            len(result["attempts"]) >= 2 and result["ok"]) else ""
        print(f"[{idx}/{total}] {mark} {db_id}{retry_note}", flush=True)
        records.append(rec)

        # 每 50 条保存一次，中途挂了不丢结果
        if idx % 50 == 0:
            (ROOT / args.output).write_text(
                json.dumps(records, indent=2, ensure_ascii=False),
                encoding="utf-8")

    print("\n" + "=" * 56)
    print("M1 Agent 评测结果")
    print("=" * 56)
    print(f"样本数:              {total}")
    print(f"API 失败:            {n_api_fail}")
    print(f"可评测:              {n_evaluable}")
    if n_evaluable:
        print(f"执行准确率 (EA):     {n_correct}/{n_evaluable} = {n_correct / n_evaluable:.1%}")
    print(f"较 M0 救回 (rescued): {n_rescued}")
    print(f"较 M0 退化:          {n_regressed}  （应≈0，否则说明闭环有副作用）")
    print(f"重试救回可执行性:    {n_retry_success}  （M0 的 execution_error 被修复）")
    if args.n_samples >= 2:
        print(f"投票有共识:          {n_vote_consensus}  （直接取最大簇结果）")
        print(f"投票无共识 fallback: {n_vote_fallback}  （走 M1 重试闭环）")
    print(f"平均 API 调用/题:    {total_calls / max(1, total - n_api_fail):.2f}")

    (ROOT / args.output).write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n完整 trace 已保存: {args.output}")


if __name__ == "__main__":
    main()
