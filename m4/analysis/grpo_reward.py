"""grpo_reward.py — M6 GRPO 奖励函数（稠密四档 + 严格目标）。

DPO/RFT 用稀疏 0/1 严格匹配，导致"71% 单签名"组的优势为零（组内零方差）。
GRPO 要拿到非零优势，奖励必须稠密。这里用四档可观测信号：

    0.00  执行失败
    0.25  可执行
    0.50  + 行数一致
    0.75  + 结果一致（忽略列序，loose）
    1.00  + 严格一致（strict）

复用 run_baseline.execute_sql（唯一合法执行路径）+ results_equal_*，零外部 API。

用法：
    python m4/analysis/grpo_reward.py --self-test
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "m4"))
sys.path.insert(0, str(ROOT / "m4" / "analysis"))

from run_baseline import (  # noqa: E402
    SPIDER_DIR, execute_sql, extract_sql,
    results_equal_strict, results_equal_loose,
)


def dense_exec_reward(completion: str, db_id: str, gold: str,
                      gold_rows: list | None = None) -> float:
    """单条 completion 的稠密执行奖励。"""
    db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"
    sql = extract_sql(completion)
    if not sql:
        return 0.0
    rows, err = execute_sql(db_path, sql)
    if err is not None:
        return 0.0
    if gold_rows is None:
        gold_rows, ge = execute_sql(db_path, gold)
        if ge is not None:
            return 0.0
    r = 0.25                                              # 可执行
    if len(rows) == len(gold_rows):
        r += 0.25                                          # 行数一致
        if results_equal_loose(gold_rows, rows):
            r += 0.25                                      # loose
        if results_equal_strict(gold_rows, rows):
            r += 0.25                                      # strict
    return r


def reward_fn(prompts: list[str], completions: list[str], gold: list[str],
              db_id: list[str], **kwargs) -> list[float]:
    """TRL GRPOTrainer 适用的批量奖励函数。

    TRL 会把 dataset 的额外列（gold/db_id）与 prompts/completions 对齐传入。
    """
    cache: dict[tuple, list] = {}
    out = []
    for _p, c, g, db in zip(prompts, completions, gold, db_id):
        key = (db, g)
        if key not in cache:
            rows, ge = execute_sql(SPIDER_DIR / "database" / db / f"{db}.sqlite", g)
            cache[key] = (rows if ge is None else None)
        gr = cache[key]
        if gr is None:
            out.append(0.0)
            continue
        out.append(dense_exec_reward(c, db, g, gold_rows=gr))
    return out


def _self_test() -> None:
    from schema import load_schemas
    schemas = load_schemas()
    # 找一个有黄金 SQL 的 dev 库做真实执行
    import json
    dev = json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))
    s = next(x for x in dev if x["db_id"] == "pets_1" and len(x["query"]) < 60)
    gold = s["query"].strip().rstrip(";")
    db = s["db_id"]
    db_path = SPIDER_DIR / "database" / db / f"{db}.sqlite"
    gold_rows, ge = execute_sql(db_path, gold)
    assert ge is None, f"gold 执行失败: {ge}"

    r_gold = dense_exec_reward(gold, db, gold)            # 严格 > 0
    assert r_gold >= 0.75, f"gold 应至少 loose 0.75, got {r_gold}"
    r_garbage = dense_exec_reward("SELECT nonexistent_col FROM nowhere", db, gold)
    assert r_garbage == 0.0, f"garbage 应 0.0, got {r_garbage}"
    r_wrong = dense_exec_reward("SELECT COUNT(*) AS c FROM pets", db, gold,
                                gold_rows=gold_rows)      # 可执行但结果可能不同
    assert 0.0 <= r_wrong <= 1.0

    # 批量 reward_fn 烟测（各列表同长，与 TRL 对齐传入一致）
    rs = reward_fn(["q1", "q2"], [gold, "SELECT bogus"], [gold, gold], [db, db])
    assert len(rs) == 2 and rs[0] >= 0.75 and rs[1] == 0.0, rs

    print(f"grpo_reward 自检通过 ✅ (gold={gold!r} -> reward={r_gold})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    else:
        print("usage: python m4/analysis/grpo_reward.py --self-test")