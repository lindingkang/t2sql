"""M4 第 0 步：训练数据 gold 可信度审计。

目标：SFT/DPO 的监督信号来自 gold SQL，如果 gold 本身有错，会反向教坏模型。
因此先执行全部训练样本的 gold SQL，并用启发式规则标记不可信样本，
随后人工抽样审计确认。

用法：
    # 冒烟：只审计训练集前 50 条
    python m4/data_audit.py --source train --limit 50

    # 全量训练数据（train_spider + train_others）
    python m4/data_audit.py --source train

    # 审计 dev（测试集 gold 完整性检查）
    python m4/data_audit.py --source dev
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_baseline import SPIDER_DIR, execute_sql  # noqa: E402


# ---------------------------------------------------------------------------
# 启发式规则：标记"可能不可信"的 gold，但最终由人工确认
# ---------------------------------------------------------------------------

SCALAR_PATTERNS = [
    r"^how many",
    r"^how much",
    r"^what is the number of",
    r"^what is the count of",
    r"^what is the total number of",
    r"^count how many",
    r"^what is the average",
    r"^what is the total",
    r"^what is the sum",
    r"^what is the minimum",
    r"^what is the maximum",
    r"^what is the min",
    r"^what is the max",
]

# 问题里出现这些词，说明它想要按维度拆开返回多行，不能当作单一标量题
GROUP_INDICATORS = [
    "each", "per ", "per-", "group by", "group them",
    "by country", "by year", "by name", "by type", "by gender",
    "by age", "by city", "by state", "by district", "by nation",
    "different", "list", "names of", "show all",
]


def is_scalar_question(question: str) -> bool:
    """判断问题是否期望一个单值答案（如 how many / what is the average）。"""
    q = question.lower().strip().rstrip("?").strip()
    if not any(re.match(p, q) for p in SCALAR_PATTERNS):
        return False
    if any(g in q for g in GROUP_INDICATORS):
        return False
    return True


def has_aggregate(sql: str) -> bool:
    """SQL 的 SELECT 中是否包含聚合函数。用于辅助判断 group-by 完整性。"""
    return bool(re.search(
        r"\b(count|sum|avg|min|max)\s*\(", sql, re.I))


def flag_gold(question: str, sql: str, rows: list[tuple], gold_err: str | None) -> list[str]:
    """对一条 gold 样本生成可疑标记列表。返回空列表表示暂无可疑。"""
    flags: list[str] = []

    if gold_err:
        # 执行本身失败：SQL 错误、超时或依赖缺失，绝对不能进训练
        flags.append(f"gold_exec_error: {gold_err[:120]}")
        return flags

    n_rows = len(rows)

    # H1：单值问题却返回多行 → 很可能漏了外层聚合
    if is_scalar_question(question) and n_rows > 1:
        flags.append("scalar_question_multirow")

    # H2：gold 返回空结果 → 对 Text2SQL 训练信号极弱，且可能是值大小写/拼写问题
    if n_rows == 0:
        flags.append("empty_result")

    # H3：问题问总数，SQL 有 GROUP BY 或结果多行但无聚合函数，可能是展开式写法
    q = question.lower()
    if ("how many" in q or "how much" in q) and not any(g in q for g in GROUP_INDICATORS):
        if re.search(r"\bgroup by\b", sql, re.I) and not has_aggregate(sql):
            flags.append("groupby_no_aggregate")

    return flags


def load_samples(source: str) -> list[dict]:
    """加载 train（train_spider + train_others）或 dev。"""
    if source == "train":
        samples = []
        for name in ("train_spider", "train_others"):
            data = json.loads((SPIDER_DIR / f"{name}.json").read_text(encoding="utf-8"))
            for i, s in enumerate(data):
                samples.append({**s, "_source": name, "_index": i})
        return samples
    if source == "dev":
        data = json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))
        return [{**s, "_source": "dev", "_index": i} for i, s in enumerate(data)]
    raise ValueError(f"unknown source: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["train", "dev"], default="train")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    samples = load_samples(args.source)
    total = min(args.limit, len(samples)) if args.limit else len(samples)

    records: list[dict] = []
    n_exec_fail = 0
    n_flagged = 0
    flag_counter: dict[str, int] = {}

    print(f"审计 {args.source}: {total} 条")

    for idx, s in enumerate(samples[:total], start=1):
        db_id = s["db_id"]
        question = s["question"]
        gold_sql = s["query"].strip().rstrip(";")
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"

        rows, gold_err = execute_sql(db_path, gold_sql)
        flags = flag_gold(question, gold_sql, rows, gold_err)

        if gold_err:
            n_exec_fail += 1
        if flags:
            n_flagged += 1
            for f in flags:
                base = f.split(":")[0]
                flag_counter[base] = flag_counter.get(base, 0) + 1

        records.append({
            "source": s["_source"],
            "index": s["_index"],
            "db_id": db_id,
            "question": question,
            "gold": gold_sql,
            "exec_ok": gold_err is None,
            "num_rows": len(rows),
            "num_cols": len(rows[0]) if rows else 0,
            "gold_error": gold_err,
            "flags": flags,
            "needs_audit": bool(flags),
        })

        if idx % 1000 == 0:
            print(f"  [{idx}/{total}] 执行失败 {n_exec_fail}, 标记 {n_flagged}", flush=True)

    out_file = ROOT / "m4" / f"{args.source}_gold_audit.json"
    out_file.write_text(json.dumps(records, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # 人工审计候选单独导出，按可疑程度排序
    audit_file = ROOT / "m4" / f"{args.source}_needs_audit.json"
    audit = [r for r in records if r["needs_audit"]]
    audit.sort(key=lambda r: (r["source"], r["index"]))
    audit_file.write_text(json.dumps(audit, indent=2, ensure_ascii=False),
                          encoding="utf-8")

    print("\n" + "=" * 50)
    print(f"{args.source} gold 审计结果")
    print("=" * 50)
    print(f"总样本:        {len(records)}")
    print(f"执行失败:      {n_exec_fail}")
    print(f"标记可疑:      {n_flagged}（{n_flagged / max(1, len(records)):.1%}）")
    print(f"可信样本:      {len(records) - n_flagged}")
    if flag_counter:
        print("\n可疑标记分布:")
        for k, v in sorted(flag_counter.items(), key=lambda x: -x[1]):
            print(f"  {k:28s}: {v}")
    print(f"\n全量结果: {out_file.name}")
    print(f"人工审计候选: {audit_file.name}")


if __name__ == "__main__":
    main()