#!/usr/bin/env python3
"""确认 SFT 训练数据相对 Spider dev 的泄漏情况。

重点区分四种重叠强度：
1. triple 级泄漏： (db_id, normalized question, normalized gold_sql) 完全相同
   —— 这才是真正把 dev 的“答案”拿去训练。
2. db+question 级重叠：同一数据库同一问题问法，但 SQL 不同。
3. question 级重叠：问题问法相同，但数据库不同（build_train_set 已报告过）。
4. SQL 级重叠：不同问题但 gold SQL 文本相同。
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_baseline import SPIDER_DIR  # noqa: E402


def normalize_question(q: str) -> str:
    return " ".join(q.lower().split()).rstrip("?").strip()


def normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").split()).lower()


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    dev = json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))
    cons = load_jsonl(ROOT / "m4" / "train_conservative.jsonl")
    strict = load_jsonl(ROOT / "m4" / "train_strict.jsonl")

    dev_triples = {
        (d["db_id"], normalize_question(d["question"]), normalize_sql(d["query"]))
        for d in dev
    }
    dev_db_q = {
        (d["db_id"], normalize_question(d["question"]))
        for d in dev
    }
    dev_q = {normalize_question(d["question"]) for d in dev}
    dev_sqls = {normalize_sql(d["query"]) for d in dev}

    def check(records: list[dict]) -> dict:
        triple = 0
        db_q = 0
        db_q_hits = []
        q_only = 0
        q_only_hits = []
        sql_only = 0
        sql_only_hits = []
        for r in records:
            db_id = r["db_id"]
            q = r["question"]
            sql = r["gold"]
            nq = normalize_question(q)
            nsql = normalize_sql(sql)

            key3 = (db_id, nq, nsql)
            key2 = (db_id, nq)
            in_dev_q = nq in dev_q
            in_dev_sql = nsql in dev_sqls

            if key3 in dev_triples:
                triple += 1
            if key2 in dev_db_q:
                db_q += 1
                db_q_hits.append({"db_id": db_id, "question": q, "gold": sql})
            if in_dev_q:
                q_only += 1
                if not (key3 in dev_triples) and not (key2 in dev_db_q):
                    q_only_hits.append({"db_id": db_id, "question": q, "gold": sql})
            if in_dev_sql:
                sql_only += 1
                if not (key3 in dev_triples):
                    sql_only_hits.append({"db_id": db_id, "question": q, "gold": sql})
        return {
            "n": len(records),
            "triple_exact_overlap": triple,
            "same_db_same_question": db_q,
            "same_db_same_question_hits": db_q_hits,
            "question_only_overlap": q_only,
            "question_only_hits": q_only_hits,
            "sql_only_overlap": sql_only,
            "sql_only_hits": sql_only_hits,
        }

    out = {
        "dev_n": len(dev),
        "conservative": check(cons),
        "strict": check(strict),
    }

    report = ROOT / "m4" / "leak_audit_final.json"
    report.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("=== Spider dev leakage audit ===")
    lines.append(f"dev examples: {out['dev_n']}")
    for name in ("conservative", "strict"):
        s = out[name]
        lines.append("")
        lines.append(f"[{name}] train examples: {s['n']}")
        lines.append(f"  triple_exact_overlap (db,q,sql all in dev): {s['triple_exact_overlap']}")
        lines.append(f"  same_db_same_question: {s['same_db_same_question']}")
        lines.append(f"  question_only_overlap (diff db): {s['question_only_overlap']}")
        lines.append(f"  sql_only_overlap: {s['sql_only_overlap']}")
    lines.append("")
    lines.append("question_only hits (train db != dev db, detail):")
    for r in out["conservative"]["question_only_hits"]:
        lines.append(f"  {r['db_id']} | {r['question']} | {r['gold']}")
    lines.append("")
    lines.append("same_db_same_question hits (detail):")
    for r in out["conservative"]["same_db_same_question_hits"]:
        lines.append(f"  {r['db_id']} | {r['question']} | {r['gold']}")

    (ROOT / "m4" / "leak_audit_final.txt").write_text(
        "\n".join(lines), encoding="utf-8")

    print(report)
    print("\n".join(lines))


if __name__ == "__main__":
    main()