"""precise_diagnose.py — M2 精准归因：对每条 badcase 单独调用 LLM 做语义归因。

与 llm_diagnose.py 的区别：
1. 每条 badcase 单独调用（不是批量），上下文完整，归因更精准。
2. 输入包含：question、gold、pred、执行结果、错误信息、schema DDL、gold/pred 执行结果。
3. 要求 LLM 输出结构化 JSON：root_cause（一句话）、fix_prompt（prompt 中应加什么）、
   failure_type（分类）、is_gold_noise（gold 是否本身有错）。
4. 支持断点续跑：已归因的样本跳过。

用法：
    python precise_diagnose.py --input baseline_full.json --output badcase_precise.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from schema import load_schemas, get_ddl
from run_baseline import execute_sql, SPIDER_DIR

ROOT = Path(__file__).resolve().parent


def build_diagnose_prompt(sample: dict, schema: dict, gold_res: list, pred_res: list) -> str:
    """构造单条 badcase 的归因 prompt。"""
    ddl = get_ddl(schema)
    gold_preview = str(gold_res[:5]) if gold_res else "(empty)"
    pred_preview = str(pred_res[:5]) if pred_res else "(empty)"
    exec_error = sample.get("error", "")
    
    return f"""You are a Text2SQL error analyst. Given a question, the correct SQL (gold), the model's predicted SQL, and their execution results, diagnose WHY the prediction is wrong.

Database Schema:
{ddl}

Question: {sample['question']}

Gold SQL (correct): {sample['gold']}
Gold result preview: {gold_preview}

Predicted SQL (wrong): {sample['pred']}
Predicted result preview: {pred_preview}
Execution error: {exec_error}

Instructions:
1. Determine the ROOT CAUSE of the error in ONE sentence (be specific, not generic like "schema error").
2. If the gold SQL itself appears incorrect or inconsistent with the question, set is_gold_noise to true.
3. Suggest what should be added to the LLM prompt to prevent this specific error (fix_prompt).
4. Classify into ONE category: 
   - data_description_gap: model doesn't know data values (e.g., case sensitivity, exact strings)
   - sql_syntax_error: SQL is syntactically invalid
   - join_error: wrong JOIN logic (extra/missing/wrong path)
   - aggregation_error: wrong aggregation (GROUP BY, HAVING, etc.)
   - set_logic_error: INTERSECT/UNION/EXCEPT logic wrong
   - semantic_misunderstanding: model misunderstood the question's meaning
   - gold_noise: gold SQL is wrong/inconsistent
   - other: anything else

Output ONLY a JSON object with keys: root_cause, fix_prompt, failure_type, is_gold_noise."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="baseline_full.json")
    ap.add_argument("--output", default="badcase_precise.json")
    ap.add_argument("--limit", type=int, default=None, help="限制处理条数（调试用）")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        base_url=os.getenv("LLM_BASE_URL", "https://qianfan.baidubce.com/v2").strip(),
    )
    model = os.getenv("LLM_MODEL", "").strip()
    schemas = load_schemas()

    with open(ROOT / args.input, encoding="utf-8") as f:
        records = json.load(f)

    # 筛选需要归因的样本：执行成功但结果不对，或执行失败
    badcases = [r for r in records if r.get("exec_success") and not r.get("correct")]
    print(f"Total badcases to diagnose: {len(badcases)}")

    # 断点续跑
    output_path = ROOT / args.output
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            done = json.load(f)
        done_ids = {r["db_id"] + "|" + r["question"] for r in done}
        badcases = [r for r in badcases if r["db_id"] + "|" + r["question"] not in done_ids]
        print(f"Resuming: {len(done)} already done, {len(badcases)} remaining")
    else:
        done = []

    results = done
    for idx, sample in enumerate(badcases, start=1):
        if args.limit and idx > args.limit:
            break
            
        db_id = sample["db_id"]
        schema = schemas[db_id]
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"

        gold_res, _ = execute_sql(db_path, sample["gold"])
        pred_res, pred_err = execute_sql(db_path, sample["pred"])

        prompt = build_diagnose_prompt(sample, schema, gold_res, pred_res)
        
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            # 提取 JSON
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            diagnosis = json.loads(m.group(0)) if m else {"root_cause": "parse_error", "raw": raw}
        except Exception as e:
            diagnosis = {"root_cause": f"api_error: {e}", "failure_type": "other", "is_gold_noise": False}

        result = {
            "db_id": db_id,
            "question": sample["question"],
            "gold": sample["gold"],
            "pred": sample["pred"],
            "gold_res_preview": str(gold_res[:3]),
            "pred_res_preview": str(pred_res[:3]),
            "diagnosis": diagnosis,
        }
        results.append(result)
        
        # 每 10 条保存一次
        if idx % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"[{idx}/{len(badcases)}] saved checkpoint")
        
        time.sleep(args.sleep)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone. Total diagnosed: {len(results)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
