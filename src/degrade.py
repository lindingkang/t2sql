"""degrade.py — M3 模块2：失败分类与降级策略。

生产环境没有 gold SQL，"结果对不对"无法自动判断；降级策略只处理 agent
能观测到的失败信号。M3 把失败从原来的 3 类细化为 6 类，每类一条明确路径：

    错误类型              触发信号                降级路径                  重试
    ------------------    --------------------    ----------------------    -----
    safety_blocked        SAFETY_BLOCKED 前缀     告知是写操作，改 SELECT    1 次
    syntax_error          syntax error / parse    提示常见语法坑            2 次
    schema_error          no such column/table    推荐相似列+该表可用列清单  2 次
    timeout_error         EXEC_TIMEOUT 前缀       提示笛卡尔积/缺 JOIN      1 次
    row_limit_error       ROW_LIMIT_EXCEEDED      提示加 WHERE/LIMIT        1 次
    constraint/type/...   其他 sqlite 报错        通用修正提示              1 次

核心加分项——幻觉字段（schema_error）不做"把原始报错丢回去"，而是：
    1. 从 "no such column: X" 提取模型臆造/拼错的列名 X；
    2. 在 schema 全部真实列里用编辑距离找最相似的 top-k（治拼写错误）；
    3. 用 sqlglot 解析 SQL 实际引用的表，列出这些表的真实列清单（治臆造列）。
模型看到"你要的 X 不存在，这些表里的可用列是 ..."，修正率远高于裸报错。

接口：
    classify_execution_error(error_msg) -> error_type
    build_retry_hint(error_type, error_msg, schema_ddl="", schema=None,
                     db_id=None, sql=None) -> hint_text
"""

from __future__ import annotations

import difflib
import re

import sqlglot
from sqlglot import exp


# 失败类型常量
SYNTAX_ERROR = "syntax_error"
SCHEMA_ERROR = "schema_error"
CONSTRAINT_ERROR = "constraint_error"
TYPE_ERROR = "type_error"
TIMEOUT_ERROR = "timeout_error"
ROW_LIMIT_ERROR = "row_limit_error"
SAFETY_BLOCKED = "safety_blocked"
UNKNOWN_ERROR = "unknown_error"


def classify_execution_error(error_msg: str | None) -> str | None:
    """分类失败类型。

    M3 加固层的标记前缀（safety/timeout/row_limit）语义最明确，最先识别；
    随后是 sqlite 原生报错的语法/schema/约束/类型。
    """
    if not error_msg:
        return None

    if error_msg.startswith("SAFETY_BLOCKED"):
        return SAFETY_BLOCKED
    if error_msg.startswith("EXEC_TIMEOUT"):
        return TIMEOUT_ERROR
    if error_msg.startswith("ROW_LIMIT_EXCEEDED"):
        return ROW_LIMIT_ERROR

    msg = error_msg.lower()

    if "syntax error" in msg or "incomplete input" in msg or "unrecognized token" in msg:
        return SYNTAX_ERROR
    if "no such column" in msg or "no such table" in msg or "no such index" in msg:
        return SCHEMA_ERROR
    if "constraint" in msg:
        return CONSTRAINT_ERROR
    if "no such function" in msg or "datatype mismatch" in msg:
        return TYPE_ERROR
    return UNKNOWN_ERROR


def build_retry_hint(error_type: str, error_msg: str, schema_ddl: str = "",
                     schema: dict | None = None, db_id: str | None = None,
                     sql: str | None = None) -> str:
    """根据失败类型生成针对性重试提示。

    schema/sql 用于 schema_error 的幻觉列推荐；不传时退化为只带 DDL 的
    基础提示，保持向后兼容。
    """
    if error_type == SAFETY_BLOCKED:
        reason = error_msg.split(":", 1)[1].strip() if ":" in error_msg else error_msg
        return (
            "Your query was BLOCKED by the read-only safety layer before execution.\n"
            f"Reason: {reason}\n"
            "Text2SQL must output a single read-only SELECT query. "
            "Rewrite it using SELECT (INSERT/UPDATE/DELETE/DROP are never allowed).\n"
            "Output ONLY the corrected SQL on one line, no explanation."
        )

    if error_type == TIMEOUT_ERROR:
        return (
            "Your query timed out during execution. This almost always means a "
            "Cartesian product caused by a missing/incorrect JOIN condition, or "
            "joining very large tables unnecessarily.\n"
            "Fix it by:\n"
            "  - Adding the correct foreign-key JOIN condition for every joined table\n"
            "  - Removing tables that are not needed for the answer\n"
            "  - Aggregating before joining when possible\n"
            f"Original error: {error_msg}"
        )

    if error_type == ROW_LIMIT_ERROR:
        return (
            "Your query returned more rows than the safety limit, which usually "
            "means it is an unbounded query (missing filtering).\n"
            "Add a WHERE condition or an appropriate LIMIT. If the question asks "
            "for an aggregate (count/max/...), compute it directly instead of "
            "dumping raw rows.\n"
            f"Original error: {error_msg}"
        )

    if error_type == SYNTAX_ERROR:
        return (
            "Your SQL has a syntax error. Common causes:\n"
            "  - Missing/wrong comma between columns\n"
            "  - Unmatched parentheses\n"
            "  - Wrong ORDER BY / GROUP BY placement\n"
            "  - In a compound query (UNION/INTERSECT/EXCEPT), ORDER BY/LIMIT may "
            "appear only once at the very end and can only reference final output "
            "columns; wrap branches in a subquery if needed.\n"
            "Please rewrite the SQL carefully.\n"
            f"Original error: {error_msg}"
        )

    if error_type == SCHEMA_ERROR:
        missing = _extract_missing_identifier(error_msg)
        hint = (
            "Your SQL references a column or table that does not exist in the schema.\n"
            f"Missing identifier: {missing}\n"
        )
        suggestion = _suggest_identifier(missing, schema, sql)
        if suggestion:
            hint += suggestion + "\n"
        elif schema_ddl:
            hint += f"Check the schema below and use EXACT names:\n\n{schema_ddl}\n"
        hint += f"Original error: {error_msg}"
        return hint

    if error_type == CONSTRAINT_ERROR:
        return (
            "Your SQL violates a database constraint.\n"
            "This usually means you're trying to INSERT/UPDATE data.\n"
            "Text2SQL should only produce SELECT queries.\n"
            f"Original error: {error_msg}"
        )

    if error_type == TYPE_ERROR:
        return (
            "Your SQL uses a function or type that SQLite doesn't support.\n"
            "Common fixes:\n"
            "  - Use IFNULL() instead of ISNULL()\n"
            "  - Use CAST(column AS REAL) for float division\n"
            "  - Don't use MySQL-only functions like IF()\n"
            f"Original error: {error_msg}"
        )

    return f"SQL execution failed: {error_msg}\nPlease fix the query."


def _extract_missing_identifier(error_msg: str) -> str:
    """从 'no such column: foo' 提取缺失标识符。

    模型常带表别名前缀（t.averge），取最后一个 '.' 之后部分做相似度匹配。
    """
    m = re.search(r"no such (?:column|table|index):\s*(\S+)", error_msg, re.I)
    if not m:
        return "(unknown)"
    ident = m.group(1).strip("'\"`")
    return ident.split(".")[-1]


def _suggest_identifier(missing: str, schema: dict | None,
                        sql: str | None) -> str:
    """针对幻觉/拼错的标识符，生成"相似列 + 相关表可用列"推荐文本。

    返回空串表示无法给出结构化推荐（调用方回退整份 DDL）。
    """
    if not schema or missing in ("", "(unknown)"):
        return ""

    table_names = schema.get("table_names_original", [])
    col_entries = schema.get("column_names_original", [])

    cols_by_table: dict[str, list[str]] = {t: [] for t in table_names}
    all_cols: list[str] = []
    for tidx, cname in col_entries:
        if tidx == -1:
            continue
        tname = table_names[tidx]
        cols_by_table.setdefault(tname, []).append(cname)
        all_cols.append(cname)

    if not all_cols:
        return ""

    lines: list[str] = []

    # 1) 拼写相似列（编辑距离，n 取小避免噪声；cutoff 保证相似度够高）
    close = difflib.get_close_matches(missing, set(all_cols), n=3, cutoff=0.5)
    if close:
        lines.append("Did you mean one of these columns: "
                     + ", ".join(close) + " ?")

    # 2) SQL 实际引用表的列清单（治臆造列）
    referenced = _tables_in_sql(sql) if sql else set()
    real_by_lower = {t.lower(): t for t in table_names}
    relevant_tables = [real_by_lower[t] for t in referenced if t in real_by_lower]
    if relevant_tables:
        lines.append("Columns available in the tables you queried:")
        for t in relevant_tables:
            cols = cols_by_table.get(t, [])
            if cols:
                lines.append(f"  {t}: {', '.join(cols)}")
    elif not close:
        sample = all_cols[:40]
        lines.append("Available columns in this database: "
                     + ", ".join(sample)
                     + (" ..." if len(all_cols) > 40 else ""))

    return "\n".join(lines)


def _tables_in_sql(sql: str | None) -> set[str]:
    """用 sqlglot 提取 SQL 中 FROM/JOIN 的真实表名（小写集合）。

    解析失败返回空集合，调用方回退到全库推荐——分析失败不应中断重试。
    """
    if not sql:
        return set()
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return set()
    return {tbl.name.lower() for tbl in tree.find_all(exp.Table) if tbl.name}


# ---------------------------------------------------------------------------
# 降级策略
# ---------------------------------------------------------------------------
class DegradationPolicy:
    """按错误类型差异化分配重试预算。

    重试不是越多越好：性能类错误重试价值低，多试只烧 API 预算；
    语法/schema 类错误模型通常能依据提示自我修正，值得多给一次机会。
    """

    MAX_RETRIES = {
        SYNTAX_ERROR: 2,
        SCHEMA_ERROR: 2,
        SAFETY_BLOCKED: 1,
        TIMEOUT_ERROR: 1,
        ROW_LIMIT_ERROR: 1,
        CONSTRAINT_ERROR: 1,
        TYPE_ERROR: 1,
        UNKNOWN_ERROR: 1,
    }

    def __init__(self, max_retries: int = 2):
        self.global_max = max_retries

    def should_retry(self, error_type: str, current_attempt: int) -> bool:
        type_max = self.MAX_RETRIES.get(error_type, 1)
        return current_attempt < min(self.global_max, type_max)

    def get_hint(self, error_type: str, error_msg: str,
                 schema_ddl: str = "", schema: dict | None = None,
                 db_id: str | None = None, sql: str | None = None) -> str:
        return build_retry_hint(error_type, error_msg, schema_ddl,
                                schema=schema, db_id=db_id, sql=sql)


def _self_test() -> None:
    cases = [
        ("SAFETY_BLOCKED: 禁止写操作：DELETE", SAFETY_BLOCKED),
        ("EXEC_TIMEOUT: 查询执行超过 10s 被中断", TIMEOUT_ERROR),
        ("ROW_LIMIT_EXCEEDED: 结果集超过 50000 行上限", ROW_LIMIT_ERROR),
        ("near \"ORDER\": syntax error", SYNTAX_ERROR),
        ("no such column: average", SCHEMA_ERROR),
        ("no such table: users", SCHEMA_ERROR),
        ("UNIQUE constraint failed: users.id", CONSTRAINT_ERROR),
        ("no such function: IF", TYPE_ERROR),
        ("database is locked", UNKNOWN_ERROR),
    ]
    for error, expected in cases:
        actual = classify_execution_error(error)
        status = "PASS" if actual == expected else "FAIL"
        print(f"  [{status}] {error[:46]:48s} -> {actual}")

    fake_schema = {
        "table_names_original": ["stadium", "singer"],
        "column_names_original": [
            [-1, "*"],
            [0, "Stadium_ID"], [0, "Location"], [0, "Name"],
            [0, "Capacity"], [0, "Highest"], [0, "Average"],
            [1, "Singer_ID"], [1, "Name"], [1, "Country"],
            [1, "Song_Name"], [1, "Song_release_year"], [1, "Age"],
        ],
    }
    hint = build_retry_hint(SCHEMA_ERROR, "no such column: Averge",
                            schema=fake_schema,
                            sql="SELECT max(Capacity), avg(Averge) FROM stadium")
    print("\n幻觉列推荐（拼错 Averge）:\n" + hint)
    assert "Average" in hint

    hint2 = build_retry_hint(SCHEMA_ERROR, "no such column: made_up_col",
                             schema=fake_schema,
                             sql="SELECT made_up_col FROM singer")
    print("\n臆造列推荐（引用表列清单）:\n" + hint2)
    assert "Columns available in the tables you queried" in hint2
    assert "Singer_ID" in hint2

    policy = DegradationPolicy(max_retries=2)
    assert policy.should_retry(SYNTAX_ERROR, 0) is True
    assert policy.should_retry(SYNTAX_ERROR, 2) is False
    assert policy.should_retry(TIMEOUT_ERROR, 1) is False
    print("\nDegrade 自检通过 ✅")


if __name__ == "__main__":
    _self_test()
