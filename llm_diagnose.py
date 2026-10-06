"""llm_diagnose.py — 调用大模型对失败样本批量归因，产出 bad case 报告。

为什么用 LLM 而不是纯规则：规则分类器（run_baseline.classify_failure）只能看
SQL 字符串表面特征（有没有 left join、有没有 union），无法理解语义，77 条掉进
other 兜底桶。LLM 能读懂问题意图、对比 gold/pred 的语义差异，给出更细的归因
和可解释的理由——这就是"规则发现异常，LLM 解释异常"的分工。

标签体系（对齐 M2 规划的五类 + 两个非模型因素）：
  schema_error        schema 理解错（选错表/列，含列名臆造）
  join_error          JOIN 错（类型 INNER/LEFT、路径、多连/漏连）
  aggregate_error     聚合错（聚合函数、GROUP BY、HAVING）
  value_error         值/条件错（WHERE 谓词、字符串值、比较运算）
  set_logic_error     集合/嵌套逻辑错（UNION/INTERSECT/EXCEPT/NOT IN/子查询）
  implicit_knowledge  隐性知识缺失（问题表述与库表约定之间的常识映射）
  column_order        仅列顺序不同（评测口径伪差，非模型错）
  gold_issue          gold SQL 本身有问题（Spider 数据瑕疵）
  other               无法归入以上

工程设计（批量 LLM 任务的保命四件套）：
1. 严格 JSON 输出 + 容错解析：模型偶尔带 markdown 或多字，解析失败不崩、记 raw
2. 断点续跑：结果逐条追加 JSONL，重跑自动跳过已完成的样本，不重复花钱
3. 先冒烟：--limit 10 验证返回结构，再跑全量
4. 串行 + 限速：无卡小机器 + 平台 RPM 限制，稳定第一

用法：
    python llm_diagnose.py --limit 10          # 冒烟
    python llm_diagnose.py                      # 全量 207 条（断点续跑）
    python llm_diagnose.py --report             # 只根据已有 JSONL 重新生成报告
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
LABELS = [
    "schema_error", "join_error", "aggregate_error", "value_error",
    "set_logic_error", "implicit_knowledge", "column_order",
    "gold_issue", "other",
]
JSONL_OUT = ROOT / "badcase_diagnosis.jsonl"
REPORT_OUT = ROOT / "badcase_report.md"

DIAG_PROMPT = """You are diagnosing a Text2SQL failure. Compare the question, the gold SQL (correct), and the predicted SQL (model output). Both run on SQLite.

Question: {question}
Gold SQL: {gold}
Pred SQL : {pred}{extra}

Decide the single primary reason the prediction differs from gold.
Return ONLY a JSON object, no markdown, in exactly this form:
{{"label": "<one of: {labels}>", "reason": "<one short English sentence stating the concrete semantic difference>", "model_at_fault": <true or false>}}

Rules:
- "column_order": rows are identical but only the SELECT column order differs. Then model_at_fault=false.
- "gold_issue": the gold SQL itself looks wrong/non-executable. Then model_at_fault=false.
- "join_error": wrong JOIN type (INNER vs LEFT), wrong join path, or extra/missing joined tables.
- "set_logic_error": UNION/INTERSECT/EXCEPT/NOT IN/EXISTS/subquery semantics wrong.
- "value_error": wrong filter value/predicate/comparison operator.
- "schema_error": chose wrong table or column, or invented a non-existent column.
- "implicit_knowledge": needs domain/common-sense mapping not explicit in the schema.
"""


def build_client() -> tuple[OpenAI, str]:
    load_dotenv(ROOT / ".env")
    key = os.getenv("LLM_API_KEY", "").strip()
    base = os.getenv("LLM_BASE_URL", "https://qianfan.baidubce.com/v2").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not key:
        raise SystemExit("错误：.env 里 LLM_API_KEY 为空")
    return OpenAI(api_key=key, base_url=base), model


def parse_json_lenient(text: str) -> dict | None:
    """从容错文本里提取 JSON 对象：先找花括号片段，再 json.loads。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj.get("label"), str):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def load_done() -> dict[int, dict]:
    """读已完成的 JSONL（断点续跑），键为样本在 records 里的下标。"""
    done = {}
    if JSONL_OUT.exists():
        for line in JSONL_OUT.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                done[obj["idx"]] = obj
    return done


def diagnose_one(client: OpenAI, model: str, rec: dict) -> dict:
    extra = ""
    if rec.get("error"):
        extra = f"\nSQLite error when running pred: {rec['error']}"
    prompt = DIAG_PROMPT.format(
        question=rec["question"], gold=rec["gold"], pred=rec["pred"],
        extra=extra, labels=", ".join(LABELS))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
        timeout=60,
    )
    raw = resp.choices[0].message.content or ""
    parsed = parse_json_lenient(raw)
    return parsed or {"label": "other", "reason": "LLM 返回无法解析",
                      "model_at_fault": True, "_unparsed": raw[:200]}


def run_diagnosis(limit: int | None) -> None:
    records = json.loads((ROOT / "baseline_full.json").read_text(encoding="utf-8"))
    # 只诊断失败样本：执行失败 + 执行成功但判错；gold 瑕疵也交给 LLM 复核
    targets = [(i, r) for i, r in enumerate(records)
               if not r["correct"] and not r["api_error"]]
    if limit:
        targets = targets[:limit]

    done = load_done()
    client, model = build_client()
    print(f"待诊断 {len(targets)} 条，已完成 {len(done)} 条，模型={model}")

    with open(JSONL_OUT, "a", encoding="utf-8") as fout:
        for n, (idx, rec) in enumerate(targets, start=1):
            if idx in done:
                continue
            for attempt in range(3):  # 每条最多重试 3 次
                try:
                    verdict = diagnose_one(client, model, rec)
                    break
                except Exception as exc:
                    if attempt == 2:
                        verdict = {"label": "other", "reason": f"API 失败: {exc}",
                                   "model_at_fault": True}
                    else:
                        time.sleep(2 ** (attempt + 1))
            out = {"idx": idx, "db_id": rec["db_id"], "question": rec["question"],
                   "gold": rec["gold"], "pred": rec["pred"], "error": rec.get("error"),
                   **verdict}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()  # 立刻落盘，崩了也不丢
            if n % 10 == 0 or n == len(targets):
                print(f"  进度 {n}/{len(targets)}")
            time.sleep(0.5)
    print(f"诊断完成，明细: {JSONL_OUT.name}")


def build_report() -> None:
    """根据 JSONL 生成 markdown bad case 报告。"""
    items = [json.loads(l) for l in
             JSONL_OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    total_fail = len(items)
    dist = Counter(it.get("label", "other") for it in items)
    model_fault = sum(1 for it in items if it.get("model_at_fault", True))

    lines = [
        "# Baseline Bad Case 归因报告（dev 全量 1034 条）",
        "",
        "- 严格 EA：822/1029 = **79.9%**；宽松 EA（忽略列顺序）：863/1029 = **83.9%**",
        f"- 失败样本 {total_fail} 条，经 LLM 逐条语义归因；其中判定为模型责任的 {model_fault} 条",
        "",
        "## 失败类型分布",
        "",
        "| 类型 | 数量 | 占失败比 |",
        "|------|------|----------|",
    ]
    for label, cnt in dist.most_common():
        lines.append(f"| {label} | {cnt} | {cnt / total_fail:.1%} |")

    buckets = defaultdict(list)
    for it in items:
        buckets[it.get("label", "other")].append(it)

    for label, cnt in dist.most_common():
        lines += ["", f"## {label}（{cnt} 条）", ""]
        for it in buckets[label][:5]:
            err = f"  \\n  ⚠️ {it['error']}" if it.get("error") else ""
            lines += [
                f"- **[{it['db_id']}]** {it['question']}",
                f"  - gold: `{it['gold']}`",
                f"  - pred: `{it['pred']}`{err}",
                f"  - 归因: {it.get('reason', '')}",
            ]

    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已生成: {REPORT_OUT.name}")
    print("\n=== 失败类型分布 ===")
    for label, cnt in dist.most_common():
        print(f"  {label:20s}: {cnt:3d} ({cnt / total_fail:.1%})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="只诊断前 N 条失败样本（冒烟用）")
    ap.add_argument("--report", action="store_true", help="不调 API，只用已有 JSONL 生成报告")
    args = ap.parse_args()
    if args.report:
        build_report()
    else:
        run_diagnosis(args.limit)
        build_report()


if __name__ == "__main__":
    main()
