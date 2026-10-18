"""build_repair_sft.py — M6 造数：exec 反馈自纠错 SFT（repair-aware SFT）。

把"agent 单步里的 act→observe→correct"从规则层蒸馏进模型权重。
数据三条来源（全部确定性、零 API 采样）：

  A  有机核心   train 里"模型执行失败 + exec_repair 修复回正确"的样本
               —— 挖 step2_train_errors.jsonl（pred_repaired 非空且 correct）
               —— 用 execute_sql 反推原始报错文本
  B  合成扩量   train_conservative 的 gold SQL，反向注入 exec_repair 那几类
               破坏（列名笔误 R3c / 双引号字符串 R2 / 限定符指错 R3b / 表名单复数 R1）
               —— 执行拿真实 sqlite 报错，目标 = 原 gold
  C  锚点       round-1 SFT（train_conservative 采样 50%）防遗忘（DPO v4b 教训）

输出：
  repair_sft_train.jsonl    训练集（每行 {"message":{"messages":[...]}} 与 train_sft.py 对齐）
  repair_sft_manifest.json  数据量/来源统计
  repair_bench.jsonl        dev 自纠错基准（仅评测，绝不入训练）：oracle 可修的 dev 执行失败样本

用法：
  python m4/analysis/build_repair_sft.py --syn-target 3000 --anchor-ratio 0.5 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "m4"))
sys.path.insert(0, str(ROOT / "m4" / "analysis"))

import sqlglot  # noqa: E402
from sqlglot import exp  # noqa: E402

from run_baseline import SPIDER_DIR, execute_sql  # noqa: E402
from schema import load_schemas, get_ddl  # noqa: E402

ANALYSIS = ROOT / "m4" / "analysis"
TRAIN_CONS = ROOT / "m4" / "train_conservative.jsonl"
STEP2_RECORDS = ANALYSIS / "step2_train_records.jsonl"  # 全量（含 repair-fixed 且 correct 的）
DEV_EVAL_JSON = ANALYSIS / "dpo_v1_fixed_eval_result.json"  # ≈ 基线(sft_cons_v1) 行为

OUT_TRAIN = ANALYSIS / "repair_sft_train.jsonl"
OUT_MANIFEST = ANALYSIS / "repair_sft_manifest.json"
OUT_BENCH = ANALYSIS / "repair_bench.jsonl"

SYSTEM = (
    "You are a SQLite SQL expert. "
    "Given the database schema below, write a single SQLite SQL query "
    "to answer the question. Output ONLY the SQL query on ONE single line, "
    "with no markdown fences and no explanation."
)


def repair_user_turn(question: str, ddl: str, wrong_sql: str, error: str) -> str:
    return (
        f"Schema:\n{ddl}\n\n"
        f"Question: {question}\n\n"
        f"Fix the SQL below so that it executes successfully.\n"
        f"Wrong SQL: {wrong_sql}\n\n"
        f"SQLite error: {error}\n\n"
        f"Output ONLY the corrected SQL on ONE single line, "
        f"with no markdown fences and no explanation."
    )


def make_message(question: str, ddl: str, wrong_sql: str, error: str, fixed_sql: str) -> dict:
    return {"message": {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": repair_user_turn(question, ddl, wrong_sql, error)},
        {"role": "assistant", "content": fixed_sql},
    ]}}


def db_path(db_id: str) -> Path:
    return SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"


def exec_error(sql: str, db_id: str):
    """返回 (无错?, 报错文本)。无错时 err 为 None。"""
    _, err = execute_sql(db_path(db_id), sql)
    return err is None, err


# --------------------------------------------------------------------------
# 合成破坏（exec_repair 各条规则的反向注入）。每条返回损坏后的 SQL 或 None。
# 所有正确性由后续 execute 校验兜底：无报错即丢弃，故不担心破坏碰巧合法。
# --------------------------------------------------------------------------
def _clone(sql: str):
    return sqlglot.parse_one(sql, read="sqlite")


def _sql(tree) -> str:
    return tree.sql(dialect="sqlite")


def corrupt_typo_col(sql: str) -> str | None:
    """R3c 反向：给第一个无限定列的列名追加字符（age -> agez）。"""
    try:
        tree = _clone(sql)
    except Exception:
        return None
    for c in tree.find_all(exp.Column):
        name = (c.name or "").strip()
        if not name or name == "*" or c.args.get("table"):
            continue
        c.set("this", exp.Identifier(this=name + "z", quoted=False))
        return _sql(tree)
    return None


def corrupt_dq_string(sql: str) -> str | None:
    """R2 反向：把字符串字面量改成双引号标识符。

    注意：SQLite 对双引号有"非标识符回退为字符串"的扩展，`"Yes"` 非列名时不报错，
    故该破坏几乎不产生 exec error，已从 CORRUPTORS 移除，仅留存记录。
    """
    try:
        tree = _clone(sql)
    except Exception:
        return None
    for lit in tree.find_all(exp.Literal):
        if lit.args.get("is_string"):
            val = lit.this
            lit.replace(exp.Identifier(this=val, quoted=True))
            return _sql(tree)
    return None


def corrupt_wrong_qualifier(sql: str) -> str | None:
    """R3b 反向：把第一个带限定符的列的限定符改成不存在的别名 ZZ。"""
    try:
        tree = _clone(sql)
    except Exception:
        return None
    for c in tree.find_all(exp.Column):
        name = (c.name or "").strip()
        if not name or name == "*" or not c.args.get("table"):
            continue
        c.set("table", exp.Identifier(this="ZZ", quoted=False))
        return _sql(tree)
    return None


def corrupt_table_plural(sql: str, tables_lower: set[str]) -> str | None:
    """R1 反向：把第一个表名改成加 s 的形式（head -> heads，若 heads 非真实表）。"""
    try:
        tree = _clone(sql)
    except Exception:
        return None
    for t in tree.find_all(exp.Table):
        raw = (t.name or "").strip()
        if not raw:
            continue
        wrong = raw + "s"
        if wrong.lower() in tables_lower:
            continue
        t.set("this", exp.Identifier(this=wrong, quoted=False))
        return _sql(tree)
    return None


CORRUPTORS = [
    ("col_typo", corrupt_typo_col),
    ("wrong_qualifier", corrupt_wrong_qualifier),
    ("table_plural", lambda sql, tl: corrupt_table_plural(sql, tl)),
]


def _collect_organic(schemas: dict) -> list[dict]:
    """来源 A：train 侧执行失败 + exec_repair 修复回严格正确的样本。"""
    out = []
    n_seen = 0
    for line in STEP2_RECORDS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        n_seen += 1
        if not rec.get("pred_repaired") or not rec.get("correct"):
            continue
        db_id = rec["db_id"]
        pred = rec["pred"]            # 原始错 SQL
        fixed = rec["pred_repaired"]  # oracle 修复后（已严格判对）
        if db_id not in schemas:
            continue
        ok, err = exec_error(pred, db_id)
        if ok or err is None:  # 理论上 pred_repaired 出现即代表 pred 曾报错，防御
            continue
        out.append({
            "db_id": db_id, "question": rec["question"], "gold": rec["gold"],
            "wrong": pred, "error": err, "fixed": fixed,
            "source": "organic",
        })
    print(f"[organic] 扫描 train 错误 {n_seen} 条，命中可修复且正确 {len(out)} 条")
    return out


def _collect_synthetic(schemas: dict, target: int, seed: int) -> list[dict]:
    """来源 B：gold SQL 反向注入破坏，执行拿真实报错。每种破坏各取 1/3（平衡列/表错误配比）。"""
    rng = random.Random(seed)
    golds = []
    for line in TRAIN_CONS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("audit_flags"):  # round-1 已筛掉可疑 gold，跳过
            continue
        golds.append(rec)
    rng.shuffle(golds)

    per_kind = max(1, target // len(CORRUPTORS))
    kind_counts = {name: 0 for name, _ in CORRUPTORS}
    out = []
    good = 0
    for rec in golds:
        if all(kind_counts[name] >= per_kind for name, _ in CORRUPTORS):
            break
        db_id = rec["db_id"]
        gold_sql = rec["gold"]
        if db_id not in schemas or not gold_sql or not gold_sql.strip():
            continue
        # 原 gold 必须本身可执行，才有"修回原样"的意义
        ok0, _ = exec_error(gold_sql, db_id)
        if not ok0:
            continue
        good += 1
        tables_lower = {t.lower() for t in schemas[db_id]["table_names_original"]}
        for name, fn in CORRUPTORS:
            if kind_counts[name] >= per_kind:
                continue
            try:
                corrupted = fn(gold_sql, tables_lower) if name == "table_plural" else fn(gold_sql)
            except Exception:
                corrupted = None
            if not corrupted or corrupted == gold_sql:
                continue
            ok, err = exec_error(corrupted, db_id)
            if ok or err is None:
                continue  # 破坏没造成报错 -> 丢弃
            out.append({
                "db_id": db_id, "question": rec["question"], "gold": gold_sql,
                "wrong": corrupted, "error": err, "fixed": gold_sql,
                "source": f"synthetic:{name}",
            })
            kind_counts[name] += 1
    print(f"[synthetic] 遍历 gold（可执行 {good} 条），产出 {len(out)} 条（各型配额 {per_kind}）")
    return out


def _collect_anchor(n: int, seed: int) -> list[dict]:
    """来源 C：round-1 SFT 锚点（防遗忘）。"""
    records = []
    for line in TRAIN_CONS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:n]


def _build_bench(schemas: dict) -> list[dict]:
    """dev 自纠错基准：基线模型【执行失败】的完整集合（仅评测用）。

    两类：
      - oracle-fixable：pred_repaired 非空（exec_repair 修到可执行）
      - oracle-unfixable：failure_type == exec_error（规则也修不了）
    """
    out = []
    n_fixable = n_unfixable = 0
    for rec in json.loads(DEV_EVAL_JSON.read_text(encoding="utf-8")):
        oracle_fix = None
        if rec.get("pred_repaired"):
            oracle_fix = rec["pred_repaired"]
            n_fixable += 1
        elif rec.get("failure_type") == "exec_error":
            oracle_fix = None
            n_unfixable += 1
        else:
            continue
        db_id = rec["db_id"]
        if db_id not in schemas:
            continue
        ok, err = exec_error(rec["pred"], db_id)
        if ok or err is None:
            continue
        out.append({
            "db_id": db_id, "idx": rec.get("idx"), "question": rec["question"],
            "gold": rec["gold"], "wrong": rec["pred"], "error": err,
            "oracle_fix": oracle_fix,
            "oracle_fixable": oracle_fix is not None,
        })
    print(f"[bench] dev 执行失败样本 {len(out)} 条（oracle 可修 {n_fixable} / 不可修 {n_unfixable}）")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--syn-target", type=int, default=3000)
    ap.add_argument("--anchor-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    schemas = load_schemas()

    organic = _collect_organic(schemas)
    synthetic = _collect_synthetic(schemas, args.syn_target, args.seed)
    repair_records = organic + synthetic
    n_repair = len(repair_records)
    n_anchor = int(round(n_repair * args.anchor_ratio / (1 - args.anchor_ratio)))
    anchor = _collect_anchor(n_anchor, args.seed)

    # 组装训练集
    lines = []
    for r in repair_records:
        lines.append(make_message(r["question"], get_ddl(schemas[r["db_id"]]),
                                  r["wrong"], r["error"], r["fixed"]))
    for a in anchor:
        lines.append({"message": a["message"]})
    rng = random.Random(args.seed)
    rng.shuffle(lines)

    OUT_TRAIN.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n",
                         encoding="utf-8")
    bench = _build_bench(schemas)
    OUT_BENCH.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in bench) + "\n",
                         encoding="utf-8")

    manifest = {
        "organic": len(organic),
        "synthetic": len(synthetic),
        "synthetic_by_kind": _count_kinds(synthetic),
        "repair_total": n_repair,
        "anchor": len(anchor),
        "train_total": len(lines),
        "bench_dev": len(bench),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    print("DONE")


def _count_kinds(synthetic: list[dict]) -> dict:
    from collections import Counter
    return dict(Counter(r["source"].split(":", 1)[1] for r in synthetic))


def _self_test() -> None:
    # 只验破坏函数在无 schema 依赖下的基本行为
    assert corrupt_typo_col("SELECT age FROM head") == "SELECT agez FROM head", corrupt_typo_col("SELECT age FROM head")
    assert corrupt_wrong_qualifier("SELECT T1.age FROM head AS T1") is not None
    assert corrupt_table_plural("SELECT * FROM head", {"head", "body"}) == "SELECT * FROM heads"
    print("build_repair_sft 自检通过 ✅")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    else:
        main()