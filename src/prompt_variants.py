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

# improved_v1：第一版改进（已验证无效，保留作对照）
_IMPROVED_V1_INSTRUCTION = (
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

# improved_v2：基于 207 条精准 LLM 归因重新设计
# 上一版 improved 没有显著提升（0.799→0.805，在 ±0.003 噪声内），根因是归因太粗。
# 这一版每条约束都对应精准归因里的高频 root_cause + fix_prompt：
#   - data_description_gap(45): 大小写/类型/格式不感知
#   - join_error(36): JOIN 多了/少了/类型错
#   - aggregation_error(25): 子查询嵌套、GROUP BY 规范
#   - semantic_misunderstanding(69): 过度推断、漏列、误读"average"等词
#   - set_logic_error(4): INTERSECT/UNION 语义
_IMPROVED_V2_INSTRUCTION = (
    "You are a SQLite SQL expert. Write a single SQLite SQL query "
    "to answer the question, following these rules strictly:\n"
    "\n"
    "DATA AWARENESS:\n"
    "1. String comparisons in SQLite are CASE-SENSITIVE. Check the schema for exact "
    "column types; if a column is TEXT, its values may be lowercase (e.g. 'france' "
    "not 'France'). When unsure, use case-insensitive matching: WHERE LOWER(col) = LOWER('value').\n"
    "2. Do NOT CAST TEXT columns to numeric types for sorting or comparison unless the "
    "question explicitly requires numeric ordering. TEXT columns sort lexicographically.\n"
    "3. When the question mentions a word like 'average', 'highest', or 'result', "
    "FIRST check if a column with that exact name exists in the schema. If yes, "
    "SELECT that column directly — do NOT apply AVG()/MAX() to a different column.\n"
    "\n"
    "JOIN DISCIPLINE:\n"
    "4. Only JOIN tables that have a foreign key relationship declared in the schema.\n"
    "5. When counting child entities per parent (e.g. 'concerts per stadium'), use "
    "INNER JOIN — do not include parents with zero children unless the question says "
    "'all' or 'including those without'.\n"
    "6. When the question asks to 'list IDs' of an entity, JOIN to that entity's main "
    "table and SELECT the ID from there, not from a foreign key column in a child table.\n"
    "\n"
    "AGGREGATION RULES:\n"
    "7. When the question asks 'how many X satisfy condition Y', wrap the filtering "
    "logic in a subquery and apply COUNT(*) on the outside: "
    "SELECT count(*) FROM (SELECT ... GROUP BY ... HAVING ...).\n"
    "8. GROUP BY should include only the key columns needed for grouping. Do not add "
    "functionally dependent columns unless the question asks for them.\n"
    "\n"
    "SELECT PRECISION:\n"
    "9. The SELECT clause must include exactly the columns the question asks for — "
    "no more, no less. If the question mentions 'name and age', return both.\n"
    "10. Place aggregate columns (COUNT, AVG, MAX, etc.) before non-aggregated grouping "
    "columns in the SELECT clause.\n"
    "\n"
    "SET LOGIC:\n"
    "11. 'both X and Y' / 'X and Y are from' = INTERSECT. "
    "'X or Y' / 'either X or Y' = UNION.\n"
    "\n"
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
        instr = _IMPROVED_V2_INSTRUCTION
    elif variant == "improved_v1":
        instr = _IMPROVED_V1_INSTRUCTION
    else:
        raise ValueError(f"未知 prompt 变体: {variant}")

    ddl = get_ddl(schema)
    return (
        f"{instr}\n\n"
        f"Schema:\n{ddl}\n\n"
        f"Question: {question}\n\n"
        "SQL:"
    )
