"""analyze_failures.py — 给 baseline_full.json 的失败样本补齐分类标签，输出类型分布和详细案例。

为什么需要它：baseline_full.json 是在分类功能合入前生成的，只有 correct 字段，
没有 failure_type / correct_loose。补分类不需要重调 API——gold 和 pred 都存着，
在本地 SQLite 上重新执行一遍（免费、秒级、可重复）即可。
这正是"评测是纯计算而非纯消费"的体现：结果可离线复算，坏了不用花 API 钱重跑。

产出两样东西：
1. 终端报告：两个口径的 EA + 失败类型分布 + 每类真实案例（复盘素材）
2. failures_analysis.json：补全标签的完整记录，M2 评测体系的数据底座

用法：
    python analyze_failures.py                # 默认分析 baseline_full.json
    python analyze_failures.py --in baseline_100.json --out analysis_100.json
"""

import argparse
import json
from collections import Counter, defaultdict

from run_baseline import (
    SPIDER_DIR,
    execute_sql,
    results_equal_loose,
    classify_failure,
)

N_EXAMPLES = 3  # 每类展示的案例数


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_file", default="baseline_full.json")
    parser.add_argument("--out", dest="out_file", default="failures_analysis.json")
    args = parser.parse_args()

    with open(args.in_file, encoding="utf-8") as f:
        records = json.load(f)

    total = len(records)
    n_evaluable = sum(1 for r in records if r["gold_exec_success"] and r["exec_success"])
    n_correct = sum(1 for r in records if r["correct"])

    # ---------- 1. 非评测样本先定标签（这些不是模型的锅） ----------
    for r in records:
        if r["api_error"]:
            r["failure_type"] = "api_error"
        elif not r["gold_exec_success"]:
            r["failure_type"] = "gold_data_issue"   # gold 自己跑不了，Spider 瑕疵
        elif not r["exec_success"]:
            r["failure_type"] = "execution_error"   # pred 语法/引用错误

    # ---------- 2. 对"执行成功但结果不对"的样本：重执行 + 分类 ----------
    # 数学上"严格判对 ⇒ 宽松一定判对"（同一批行做确定性重排，集合相等保持），
    # 所以正确样本直接标 True，不必重执行。
    for r in records:
        if r["gold_exec_success"] and r["exec_success"] and r["correct"]:
            r["correct_loose"] = True

    wrong = [r for r in records
             if r["gold_exec_success"] and r["exec_success"] and not r["correct"]]
    for r in wrong:
        db_path = SPIDER_DIR / "database" / r["db_id"] / f"{r['db_id']}.sqlite"
        gold_res, _ = execute_sql(db_path, r["gold"])
        pred_res, _ = execute_sql(db_path, r["pred"])
        r["correct_loose"] = results_equal_loose(gold_res, pred_res)
        r["failure_type"] = classify_failure(r["gold"], r["pred"], gold_res, pred_res)

    # ---------- 3. 汇总报告 ----------
    n_loose = sum(1 for r in records if r.get("correct_loose"))
    print(f"总样本: {total} | 可评测: {n_evaluable}")
    print(f"严格 EA（官方口径）  : {n_correct}/{n_evaluable} = {n_correct / n_evaluable:.1%}")
    print(f"宽松 EA（忽略列顺序）: {n_loose}/{n_evaluable} = {n_loose / n_evaluable:.1%}")

    dist = Counter(r.get("failure_type") for r in records
                   if r.get("failure_type") and r["failure_type"] != "api_error")
    print("\n=== 失败类型分布 ===")
    for ft, cnt in dist.most_common():
        print(f"  {ft:17s}: {cnt:4d}  ({cnt / total:.1%} of all)")

    # ---------- 4. 每类打印详细案例 ----------
    buckets = defaultdict(list)
    for r in records:
        ft = r.get("failure_type")
        if ft and ft != "api_error":
            buckets[ft].append(r)

    for ft, cnt in dist.most_common():
        print(f"\n### {ft}（共 {cnt} 条，展示 {min(N_EXAMPLES, cnt)} 条）")
        for r in buckets[ft][:N_EXAMPLES]:
            print(f"\n  [{r['db_id']}] {r['question']}")
            print(f"    gold: {r['gold']}")
            print(f"    pred: {r['pred']}")
            if r["failure_type"] == "execution_error":
                print(f"    错误: {r['error']}")

    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n补全标签后的完整记录已保存: {args.out_file}")


if __name__ == "__main__":
    main()
