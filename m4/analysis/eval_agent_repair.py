"""eval_agent_repair.py — M6 端到端：repair_r1 接回 agent 重试路径，对比规则重试 vs 模型自纠错。

取 dev 上 agent 的【执行失败样本】（repair_bench.jsonl，46 条），对每个失败起点做三种纠正：
  ① single-pass   turn-0 原始输出（构造上全部执行失败 → 0）
  ② 规则重试        agent.py 现状：degrade.build_retry_hint + API LLM 重写（1 轮）   ← 本脚本调 API 算
  ③ 模型自纠错      repair_r1 本地 greedy 修正（已由 eval_self_correct 产出，读 selfcorrect_repair_r1.json）
  ④ 规则 oracle    exec_repair 确定性修复（上界）

成本：② 只在 46 条失败样本上各调 1 次 API（不必重跑 1034 条 turn-0）；③④ 复用已有结果。

用法：
  python m4/analysis/eval_agent_repair.py --limit 10      # 冒烟
  python m4/analysis/eval_agent_repair.py                 # 全 46 条
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "m4"))
sys.path.insert(0, str(ROOT / "m4" / "analysis"))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from schema import load_schemas, get_ddl  # noqa: E402
from run_baseline import (  # noqa: E402
    SPIDER_DIR, build_prompt, extract_sql, execute_sql,
    call_llm, results_equal_strict, results_equal_loose,
)
from degrade import classify_execution_error, build_retry_hint  # noqa: E402
from agent import Text2SQLAgent  # noqa: E402  (复用 _render_messages)

BENCH = ROOT / "m4" / "analysis" / "repair_bench.jsonl"
SELFCORRECT_JSON = ROOT / "m4" / "analysis" / "selfcorrect_repair_r1.json"


def db_path(db_id: str) -> Path:
    return SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"


def rule_retry_one(client, model: str, schemas: dict, b: dict) -> dict:
    """复刻 agent._retry_loop 的 turn-1：rule hint + API LLM 重写。"""
    db_id, question, gold, wrong, error = b["db_id"], b["question"], b["gold"], b["wrong"], b["error"]
    schema = schemas[db_id]
    gold_rows, _ge = execute_sql(db_path(db_id), gold)

    messages = [{"role": "user", "content": build_prompt(db_id, question, schema)}]
    messages.append({"role": "assistant", "content": wrong})
    error_type = classify_execution_error(error)
    hint = build_retry_hint(error_type, error, get_ddl(schema),
                            schema=schema, db_id=db_id, sql=wrong)
    messages.append({"role": "user", "content": hint})
    prompt = Text2SQLAgent._render_messages(messages)

    content = call_llm(client, model, prompt, temperature=0.0)
    new_sql = extract_sql(content)
    rows, err = execute_sql(db_path(db_id), new_sql)
    exec_ok = err is None
    strict = loose = False
    if exec_ok:
        strict = results_equal_strict(gold_rows, rows)
        loose = results_equal_loose(gold_rows, rows)
    return {"idx": b.get("idx"), "db_id": db_id, "question": question,
            "wrong": wrong, "retry_sql": new_sql, "exec_ok": exec_ok,
            "strict": strict, "loose": loose, "error_type": error_type}


def oracle_stats(bench: list[dict], schemas: dict) -> dict:
    """"""
    n = len(bench)
    exec_ok = strict = loose = 0
    for b in bench:
        gold_rows, _ = execute_sql(db_path(b["db_id"]), b["gold"])
        if not b["oracle_fix"]:
            continue
        rows, err = execute_sql(db_path(b["db_id"]), b["oracle_fix"])
        if err is not None:
            continue
        exec_ok += 1
        if results_equal_strict(gold_rows, rows):
            strict += 1
        if results_equal_loose(gold_rows, rows):
            loose += 1
    return {"n": n, "exec_ok": exec_ok, "strict": strict, "loose": loose}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        base_url=os.getenv("LLM_BASE_URL", "https://qianfan.baidubce.com/v2").strip(),
    )
    model = os.getenv("LLM_MODEL", "").strip()
    schemas = load_schemas()

    bench = [json.loads(x) for x in BENCH.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        bench = bench[:args.limit]

    # 模型自纠错已有结果（对齐 bench 顺序，按 db_id+idx+wrong 匹配）
    sc_rows = json.loads(SELFCORRECT_JSON.read_text(encoding="utf-8"))
    sc_map = {(r["db_id"], r["idx"], r["wrong"]): r for r in sc_rows}

    print(f"开始规则重试（API）x {len(bench)} …")
    rule_rows = []
    for b in bench:
        try:
            r = rule_retry_one(client, model, schemas, b)
        except Exception as exc:
            r = {"idx": b.get("idx"), "db_id": b["db_id"], "question": b["question"],
                 "wrong": b["wrong"], "retry_sql": None, "exec_ok": False,
                 "strict": False, "loose": False, "error_type": None,
                 "api_error": str(exc)}
        rule_rows.append(r)
        time.sleep(0.5)

    n = len(bench)
    def agg(rows, key):
        return sum(1 for r in rows if r[key])

    rule = {"exec_ok": agg(rule_rows, "exec_ok"), "strict": agg(rule_rows, "strict"),
            "loose": agg(rule_rows, "loose")}

    # 模型自纠错（对齐后的子集）
    sc_sub = []
    for b in bench:
        r = sc_map.get((b["db_id"], b["idx"], b["wrong"]))
        if r is None:
            r = {"exec_ok": False, "strict": False, "loose": False}
        sc_sub.append(r)
    model_sc = {"exec_ok": agg(sc_sub, "exec_ok"), "strict": agg(sc_sub, "strict"),
                "loose": agg(sc_sub, "loose")}

    ora = oracle_stats(bench, schemas)

    # 输出
    out = {
        "n": n,
        "single_pass": {"exec_ok": 0, "strict": 0, "loose": 0},
        "rule_retry_agent": rule,
        "model_selfcorrect_repair_r1": model_sc,
        "oracle_exec_repair": ora,
    }
    out_json = ROOT / "m4" / "analysis" / "agent_repair_compare.json"
    out_json.write_text(json.dumps({"summary": out, "rule_rows": rule_rows},
                                   ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        "=" * 72,
        "end-to-end: agent 重试路径 —— 规则重试 vs 模型自纠错（dev 执行失败样本）",
        "=" * 72,
        f"样本 n = {n}（agent turn-0 执行失败，全部作为纠正起点）",
        "",
        f"{'纠正器':<28}{'exec 恢复':>10}{'strict':>10}{'loose':>10}",
        f"{'single-pass (turn-0)':<28}{'0/'+str(n):>10}{'0/'+str(n):>10}{'0/'+str(n):>10}",
        f"{'规则重试 (hint+API 现状)':<28}{str(rule['exec_ok'])+'/'+str(n):>10}{str(rule['strict'])+'/'+str(n):>10}{str(rule['loose'])+'/'+str(n):>10}",
        f"{'模型自纠错 (repair_r1)':<28}{str(model_sc['exec_ok'])+'/'+str(n):>10}{str(model_sc['strict'])+'/'+str(n):>10}{str(model_sc['loose'])+'/'+str(n):>10}",
        f"{'规则 oracle (exec_repair)':<28}{str(ora['exec_ok'])+'/'+str(n):>10}{str(ora['strict'])+'/'+str(n):>10}{str(ora['loose'])+'/'+str(n):>10}",
        "",
    ]
    txt = "\n".join(lines)
    print(txt)
    (ROOT / "m4" / "analysis" / "agent_repair_compare.txt").write_text(txt, encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()