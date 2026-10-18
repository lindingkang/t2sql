"""step3_screen.py — M5 Step3: train 错误清单的 gold 噪声筛查（确定启发式 + LLM 二次确认）。

分两层：
  A. 确定性（无 API，零成本，绝不误伤）：
     R1 gold_exec_error   — gold 自己执行报错（漏列/漏表/语法错）
     R2 gold_empty        — gold 返回空结果（严格匹配退化为任意错 SQL）
     R3 scalar_multirow   — 问一个值却返回多行（值口径错，复用 train_gold_audit.json flags）
  B. LLM 二次确认（只对"模型 exec 成功但结果错"的暧昧题，判定 gold 是否真正确）：
     - 判定 NO  -> 排除（gold 语义错，避免把模型往错 gold 上带）
     - 判定 YES / 不可解析 / API 失败 -> 保留（fail-open，绝不阻塞流水线）
     LLM 调用按 --max-llm 封顶 + --llm-budget 秒墙钟预算，超了把剩余题 fail-open 保留。

用法：
    python m4/analysis/step3_screen.py                 # 默认开 LLM
    python m4/analysis/step3_screen.py --no-llm       # 只用确定性启发式（LLM 挂了用这个）
    python m4/analysis/step3_screen.py --max-llm 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "m4"))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402
from run_baseline import SPIDER_DIR, execute_sql, call_llm  # noqa: E402
from schema import load_schemas, get_ddl  # noqa: E402

OUT_CLEAN = ROOT / "m4" / "analysis" / "step3_clean_errors.jsonl"
OUT_EXCL = ROOT / "m4" / "analysis" / "step3_excluded.jsonl"
OUT_LLM = ROOT / "m4" / "analysis" / "step3_llm_decisions.jsonl"
OUT_REPORT = ROOT / "m4" / "analysis" / "step3_report.txt"

ERR_IN = ROOT / "m4" / "analysis" / "step2_train_errors.jsonl"
AUDIT_IN = ROOT / "m4" / "train_gold_audit.json"


def norm(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").split())


def _lines(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue  # 容忍并发写盘的半截行


# ---- 词频信号（只用于 LLM 优先级排序，不硬排除）----
COUNT_WORDS = ("how many", "number of", "how much", "how many of")
AVG_WORDS = ("average", "the mean", "avg")
SUM_WORDS = ("total", "sum", "combined", "overall")


def agg_suspect(question: str, gold: str) -> bool:
    """题干词频与 gold 顶层聚合不一致 -> 疑似 agg 类型错（soft 信号）。"""
    ql = question.lower()
    g = gold.lower()
    has_count = "count(" in g
    has_sum = "sum(" in g
    has_avg = "avg(" in g
    count_q = any(w in ql for w in COUNT_WORDS)
    avg_q = any(w in ql for w in AVG_WORDS)
    sum_q = any(w in ql for w in SUM_WORDS)
    if count_q and (has_sum or has_avg) and not has_count:
        return True
    if avg_q and has_count and not has_avg:
        return True
    if sum_q and (has_count or has_avg) and not has_sum:
        return True
    return False


def build_client():
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://qianfan.baidubce.com/v2").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not api_key or api_key == "在此填入你的key" or not model:
        return None, None
    return OpenAI(api_key=api_key, base_url=base_url), model


def ask_llm(client, model, question: str, ddl: str, gold: str):
    """返回 (verdict, content)。verdict ∈ {YES, NO, UNPARSEABLE, FAIL}。"""
    prompt = (
        "You are auditing a SQL gold reference for a text-to-SQL benchmark.\n"
        f"Schema:\n{ddl}\n\n"
        f"Question: {question}\n\n"
        f"Gold SQL: {gold}\n\n"
        "Is the Gold SQL a correct AND complete answer to the Question given the Schema? "
        "Consider column selection, aggregate type, value literals, joins, and row set. "
        "Answer with ONLY 'YES' or 'NO' on the first line, then optionally a one-line reason."
    )
    try:
        content = call_llm(client, model, prompt, max_retries=2, timeout=40, temperature=0.0)
    except Exception as exc:
        return "FAIL", str(exc)[:200]
    first = content.strip().splitlines()[0].strip().upper() if content and content.strip() else ""
    if first.startswith("YES"):
        return "YES", content.strip()[:400]
    if first.startswith("NO"):
        return "NO", content.strip()[:400]
    return "UNPARSEABLE", content.strip()[:400]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--max-llm", type=int, default=250)
    ap.add_argument("--llm-budget", type=int, default=1500,
                    help="LLM 阶段墙钟秒数预算，超出后剩余题 fail-open 保留")
    args = ap.parse_args()

    errors = [r for r in _lines(ERR_IN) if r.get("pred") and r.get("gen_ok") is not False]

    audit = {}
    if AUDIT_IN.exists():
        for r in json.loads(AUDIT_IN.read_text(encoding="utf-8")):
            if not isinstance(r, dict):
                continue
            audit[(r["db_id"], norm(r["gold"]))] = r
            audit.setdefault((r["db_id"], r["question"]), r)

    schemas = load_schemas()
    client, model = build_client()
    use_llm = (not args.no_llm) and (client is not None) and (model is not None)
    if not use_llm:
        print("LLM 关闭或不可用，仅用确定性启发式", flush=True)

    clean: list[dict] = []       # -> Step4
    excluded: list[dict] = []    # gold 噪声，剔除
    llm_pool: list[dict] = []    # exec_ok 但结果错 -> 送 LLM
    reason_counter = Counter()

    # ---------- Phase A 确定性 ----------
    for e in errors:
        db_id = e["db_id"]
        gold = e.get("gold", "")
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"
        g = audit.get((db_id, norm(gold))) or audit.get((db_id, e.get("question", "")))
        flags = list((g or {}).get("flags", []))

        gold_rows, gold_err = execute_sql(db_path, gold)
        if gold_err is not None:
            excluded.append({**e, "reason": "gold_exec_error", "flags": flags})
            reason_counter["gold_exec_error"] += 1
            continue
        nrows = len(gold_rows) if gold_rows is not None else 0

        if nrows == 0:
            excluded.append({**e, "reason": "gold_empty", "flags": flags})
            reason_counter["gold_empty"] += 1
            continue
        if "scalar_question_multirow" in flags:
            excluded.append({**e, "reason": "scalar_multirow", "flags": flags})
            reason_counter["scalar_multirow"] += 1
            continue

        # 模型自身故障（exec_fail 修复后仍错）-> 直接判 usable，无需 LLM
        if not e.get("exec_ok"):
            clean.append({**e, "gold_check": "model_fault", "flags": flags})
            continue

        # exec_ok 但结果错 -> 暧昧（可能是 gold 语义错）-> 送 LLM
        llm_pool.append({**e, "agg_suspect": agg_suspect(e["question"], gold), "flags": flags, "gold_points": [nrows, len(gold_rows[0]) if nrows else 0]})

    # ---------- Phase B LLM 二次确认 ----------
    # 优先级：agg 词频疑似 > 一般
    llm_pool.sort(key=lambda e: not e["agg_suspect"])
    n_agg_suspect = sum(1 for e in llm_pool if e["agg_suspect"])
    reason_counter["_llm_pool"] = len(llm_pool)

    f_llm = open(OUT_LLM, "w", encoding="utf-8")
    llm_t0 = time.time()
    n_llm_done = 0
    for e in llm_pool:
        if use_llm and n_llm_done < args.max_llm and (time.time() - llm_t0) < args.llm_budget:
            ddl = get_ddl(schemas[e["db_id"]]) if e["db_id"] in schemas else ""
            verdict, content = ask_llm(client, model, e["question"], ddl, e["gold"])
            n_llm_done += 1
            f_llm.write(json.dumps({
                "idx": e.get("idx"), "db_id": e["db_id"], "question": e["question"],
                "gold": e["gold"], "verdict": verdict,
                "agg_suspect": e["agg_suspect"], "detail": content,
            }, ensure_ascii=False) + "\n")
            if verdict == "NO":
                excluded.append({**e, "reason": "gold_noise_llm"})
                reason_counter["gold_noise_llm"] += 1
            elif verdict == "YES":
                clean.append({**e, "gold_check": "llm_yes"})
            else:  # UNPARSEABLE / FAIL -> fail-open 保留
                clean.append({**e, "gold_check": f"llm_{verdict.lower()}"})
                reason_counter[f"llm_{verdict.lower()}"] += 1
        else:
            # 预算耗尽 / 未开 LLM -> fail-open 保留
            clean.append({**e, "gold_check": "llm_capped" if use_llm else "no_llm"})
    f_llm.close()
    reason_counter["_llm_checked"] = n_llm_done

    OUT_CLEAN.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in clean) + ("\n" if clean else ""), encoding="utf-8")
    OUT_EXCL.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in excluded) + ("\n" if excluded else ""), encoding="utf-8")

    lines = [
        "=" * 70,
        "M5 Step3 gold 噪声筛查（启发式 + LLM 确认）",
        "=" * 70,
        f"Step2 错误总数(有pred): {len(errors)}",
        f"usable -> Step4: {len(clean)}",
        f"excluded: {len(excluded)}",
        f"LLM 池(exec_ok 但错): {len(llm_pool)}  (agg 词频疑似 {n_agg_suspect})",
        f"LLM 实际确认: {n_llm_done}",
        "",
        "排除原因分布:",
    ]
    for reason, cnt in reason_counter.most_common():
        if reason.startswith("_"):
            continue
        lines.append(f"  {reason:24s}: {cnt}")
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"clean -> {OUT_CLEAN.name} | excluded -> {OUT_EXCL.name} | llm -> {OUT_LLM.name}")


if __name__ == "__main__":
    main()