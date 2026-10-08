"""prompt_variants.py — M2：基于失败归因的 prompt 工程变体库。

M0/M1 用的是极简 zero-shot prompt（只告诉模型"写 SQL"）。
M2 把 badcase_report.md 里暴露的高频错误模式，逐条翻译成 prompt 约束。

变体设计原则：
1. 每条约束都对应 badcase 报告里已验证的具体错误类型，不凭空加规则。
2. 约束要"短而硬"——大模型对长 prompt 的边际收益递减，还可能稀释 schema 注意力。
3. 所有变体共享 schema 注入和输出格式，唯一变量是"指令部分"，保证 A/B 对照干净。

目前有两个变体：
- baseline  : M0 原版，作为对照组
- improved  : 针对 value/join/aggregate/set_logic 四类真实错误加约束
未来可继续加：fewshot（带示例）、cot（让模型先推理再输出）等。
"""

from __future__ import annotations

from schema import get_ddl

# ---------------------------------------------------------------------------
# 各变体的"指令前缀"——schema 和问题部分所有变体一致，只改这一段
# ---------------------------------------------------------------------------

# baseline：M0 原版，极简 zero-shot
_BASELINE_INSTRUCTION = (
    "You are a SQLite SQL expert. "
    "Given the database schema below, write a single SQLite SQL query "
    "to answer the question. Output ONLY the SQL query on ONE single line, "
    "with no markdown fences and no explanation."
)

# improved：把 badcase 报告里的高频错误翻译成硬约束
# 每条约束都对应 badcase_report.md 里统计的具体错误类型：
#   - value_error(41)    -> "Match values exactly"
#   - join_error(29)     -> "Use foreign keys for JOINs"
#   - aggregate_error(25)-> "GROUP BY all non-aggregated columns"
#   - set_logic_error(15)-> "INTERSECT=both, UNION=either"
#   - schema_error(28)   -> "Use exact table/column names from DDL"
_IMPROVED_INSTRUCTION = (
    "You are a SQLite SQL expert. Write a single SQLite SQL query "
    "to answer the question, following these rules strictly:\n"
    "1. Use EXACT table and column names from the schema below.\n"
    "2. Only JOIN tables that have a foreign key relationship in the schema. "
    "Do not join unrelated tables.\n"
    "3. Match text values exactly as they appear in the question "
    "(use = for exact match, LIKE only for partial match).\n"
    "4. When using aggregate functions (COUNT/SUM/AVG/MAX/MIN), "
    "GROUP BY every non-aggregated column in the SELECT.\n"
    "5. If the question asks for items satisfying BOTH conditions "
    "(e.g. 'countries that have both X and Y'), use INTERSECT. "
    "Use UNION only for EITHER condition.\n"
    "Output ONLY the SQL query on ONE single line, "
    "with no markdown fences and no explanation."
)


def build_prompt(db_id: str, question: str, schema: dict,
                 variant: str = "baseline") -> str:
    """构造指定变体的 prompt。

    所有变体共享相同的 schema 注入和输出格式锚点，
    唯一变量是 instruction 段，保证 A/B 对照实验干净。
    """
    if variant == "baseline":
        instr = _BASELINE_INSTRUCTION
    elif variant == "improved":
        instr = _IMPROVED_INSTRUCTION
    else:
        raise ValueError(f"未知 prompt 变体: {variant}")

    ddl = get_ddl(schema)
    return (
        f"{instr}\n\n"
        f"Schema:\n{ddl}\n\n"
        f"Question: {question}\n\n"
        "SQL:"
    )
