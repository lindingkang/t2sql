"""M4 第 0 步：从审计结果构建 SFT 训练集。

策略：
1. 必然剔除：gold 执行失败（exec_ok=False）。
2. 保守版：保留空结果/标量多行候选，但每条都带上 audit flags，供后续排查。
3. 严格版：额外剔除 empty_result 与 scalar_question_multirow，用于消融对比。
4. 查重：
   a) 训练集内部按 (db_id, question, gold_sql) 去重；
   b) 训练集与 dev 做 question 规范化的精确问法泄漏检查（不直接删，输出报告）。

输出：
    m4/train_conservative.jsonl      # 保守版（只剔 exec_error）
    m4/train_strict.jsonl            # 严格版
    m4/leak_report.json              # dev 问法泄漏检查报告

消息格式与 M0 推理 prompt 保持一致：
    user: 角色 + DDL schema + question
    assistant: 仅 SQL
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from schema import load_schemas, get_ddl  # noqa: E402
from run_baseline import SPIDER_DIR  # noqa: E402

SYSTEM = (
    "You are a SQLite SQL expert. "
    "Given the database schema below, write a single SQLite SQL query "
    "to answer the question. Output ONLY the SQL query on ONE single line, "
    "with no markdown fences and no explanation."
)


def normalize_question(q: str) -> str:
    """题目级查重：压平空白、转小写、去尾部问号。"""
    return " ".join(q.lower().split()).rstrip("?").strip()


def normalize_sql(sql: str) -> str:
    """SQL 规范化：去尾分号、压平空白、转小写。"""
    return " ".join(sql.strip().rstrip(";").split()).lower()


def to_chat(question: str, ddl: str, gold_sql: str) -> dict:
    """转换为 TRL SFTTrainer 能直接消费的 messages 格式。"""
    user = f"Schema:\n{ddl}\n\nQuestion: {question}\n\nSQL:"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": gold_sql.strip().rstrip(";")},
        ]
    }


def main() -> None:
    audit_path = ROOT / "m4" / "train_gold_audit.json"
    if not audit_path.exists():
        raise SystemExit("找不到 train_gold_audit.json，请先运行 python m4/data_audit.py --source train")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    schemas = load_schemas()

    conservative: list[dict] = []
    strict: list[dict] = []
    seen_cons: set[tuple[str, str, str]] = set()
    dup_count = 0

    for r in audit:
        db_id = r["db_id"]
        question = r["question"]
        gold_sql = r["gold"]

        if not r["exec_ok"]:
            continue  # gold 执行失败，必然剔除

        key = (db_id, normalize_question(question), normalize_sql(gold_sql))
        if key in seen_cons:
            dup_count += 1
            continue
        seen_cons.add(key)

        record = {
            "db_id": db_id,
            "question": question,
            "gold": gold_sql,
            "audit_flags": r["flags"],
            "source": r["source"],
            "train_index": r["index"],
            "message": to_chat(question, get_ddl(schemas[db_id]), gold_sql),
        }
        conservative.append(record)

        # 严格版：剔除空结果与"标量问题返回多行"候选
        if "empty_result" in r["flags"] or "scalar_question_multirow" in r["flags"]:
            continue
        strict.append(record)

    out_cons = ROOT / "m4" / "train_conservative.jsonl"
    out_strict = ROOT / "m4" / "train_strict.jsonl"

    with out_cons.open("w", encoding="utf-8") as f:
        for rec in conservative:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with out_strict.open("w", encoding="utf-8") as f:
        for rec in strict:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- dev 问法泄漏检查 ----
    dev = json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))
    dev_norm = [(normalize_question(d["question"]), d["db_id"], d["question"])
                for d in dev]
    dev_set = {q for q, _, _ in dev_norm}

    leaks = []
    for rec in conservative:
        nq = normalize_question(rec["question"])
        if nq in dev_set:
            leaks.append({
                "train_db": rec["db_id"],
                "train_question": rec["question"],
                "dev_hits": [
                    {"db_id": db, "question": orig}
                    for q2, db, orig in dev_norm if q2 == nq
                ],
            })

    (ROOT / "m4" / "leak_report.json").write_text(
        json.dumps(leaks, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 50)
    print("训练集构建完成")
    print("=" * 50)
    print(f"审计样本:       {len(audit)}")
    print(f"剔除 exec_error: {sum(1 for r in audit if not r['exec_ok'])}")
    print(f"内部重复:       {dup_count}")
    print(f"保守版样本数:   {len(conservative)}")
    print(f"严格版样本数:   {len(strict)}")
    print(f"dev 问法泄漏:   {len(leaks)} 条训练问题与 dev 完全相同")
    print(f"\n输出文件:")
    print(f"  {out_cons}")
    print(f"  {out_strict}")
    print(f"  m4/leak_report.json")


if __name__ == "__main__":
    main()