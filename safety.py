"""safety.py — M3 模块1：基于 sqlglot AST 的 SQL 只读安全层。

为什么不用 sqlparse / 正则 / 关键字匹配：
    1. 关键字字符串扫描会误杀合法列名：SELECT drop_rate, updated_at FROM t
       含 "DROP"/"UPDATE" 子串却是完全只读的查询；WHERE name='DROP' 这类
       字符串字面量也会被误判。
    2. 正则解析不了嵌套子查询和 CTE（WITH x AS (...) SELECT ...），会漏判。
    3. AST 能精确区分"语句节点"和"标识符/字面量"：列名 drop_rate 在 AST 里
       是 Column，不是 Drop 节点；字符串里的 DROP 是 Literal，也不是 Drop 节点。

两道防线（defense in depth）：
    第一层（本文件）：执行前 AST 白名单，主动拦截，并给出精确 reason 供 agent 重试。
    第二层（run_baseline.execute_sql）：mode=ro 只读连接兜底，即使本层漏判，
    磁盘也写不动。

白名单策略（fail-closed）：
    - 根节点必须是 Select / Union / Intersect / Except（显式枚举，不用
      isinstance(exp.Query)——旧版 sqlglot 中 Insert 曾继承 Query，显式枚举
      不依赖版本行为）。
    - 全树遍历写操作节点（Insert/Update/Delete/Drop/Alter/Create/Attach/
      Pragma/Command...），命中即拒。白名单下理论上不会命中，这是双保险，
      同时给出精确的拒绝原因。
    - 危险函数只保留"有副作用或文件读写能力"的：load_extension（加载任意
      动态库=任意代码执行）、readfile/writefile（任意文件读写）。
      旧版把 json_each、sqlite_compileoption_get 也列为危险函数是误判——
      它们纯只读，且 Spider 里可能合法使用。
    - 多语句（SELECT 1; DROP TABLE x）严格拒绝，不静默截断。截断会让
      "模型输出被悄悄改写"，安全层应 fail-closed：拒绝 → agent 重试 →
      修不好则返回"无法回答"。

接口（保持与 M1 agent.py 调用兼容）：
    check_readonly(sql) -> (ok: bool, reason: str | None)
    sanitize_sql(sql)   -> (clean_sql: str, ok: bool, reason: str | None)
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

# 显式枚举的只读查询根节点。WITH ... SELECT 在 sqlglot 中根节点仍是 Select
# （CTE 挂在 root.args["with"] 上），无需单独加 CTE 类型。
ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)

# 写操作 / 会话 / 事务类节点。遍历整棵树，命中即拒。
# - Insert 覆盖 INSERT、INSERT OR REPLACE、REPLACE INTO（sqlglot 均归为 Insert）
# - Command 兜底 VACUUM / REINDEX / BEGIN / COMMIT 等非查询命令
WRITE_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Merge,
    exp.Attach,
    exp.Detach,
    exp.Pragma,
    exp.Command,
)

# 有真实副作用的危险函数（AST Anonymous 节点的小写名匹配）。
DANGEROUS_FUNCTIONS = {
    "load_extension",  # 加载 .so/.dll → 任意代码执行
    "readfile",        # 读任意服务器文件
    "writefile",       # 写任意服务器文件
}


def check_readonly(sql: str) -> tuple[bool, str | None]:
    """检查 SQL 是否为安全的单条只读查询。

    返回 (ok, reason)：
        ok=True  : 安全，可以执行
        ok=False : 不安全，reason 是给 agent 重试看的精确原因
    """
    if not sql or not sql.strip():
        return False, "空 SQL"

    # 用 SQLite 方言解析：ATTACH/PRAGMA 等 SQLite 特有语句才能落到专门节点。
    try:
        statements = [s for s in sqlglot.parse(sql.strip(), read="sqlite")
                      if s is not None]
    except ParseError as exc:
        # 解析失败 = 无法证明安全，fail-closed 拒绝，交给重试循环修正语法。
        first_line = str(exc).splitlines()[0][:120]
        return False, f"SQL 解析失败，无法确认安全性: {first_line}"

    if not statements:
        return False, "空 SQL"

    # 规则 1：只允许单条语句（防 SELECT 1; DROP TABLE x 注入）。
    if len(statements) > 1:
        kinds = ", ".join(type(s).__name__ for s in statements)
        return False, f"检测到 {len(statements)} 条语句（{kinds}），只允许单条查询"

    root = statements[0]

    # 规则 2：根节点必须是允许的只读查询类型。
    if not isinstance(root, ALLOWED_ROOTS):
        return False, f"非只读查询类型: {type(root).__name__}（只允许 SELECT/UNION/INTERSECT/EXCEPT）"

    # 规则 3：全树扫描写操作节点（双保险 + 精确原因）。
    for node in root.walk():
        if isinstance(node, WRITE_NODES):
            kind = type(node).__name__.upper()
            # Command 节点带上具体命令名（如 VACUUM），原因更可读。
            extra = f" {node.name.upper()}" if isinstance(node, exp.Command) and node.name else ""
            return False, f"禁止写操作/会话命令: {kind}{extra}"

    # 规则 4：危险函数扫描。只检查 Anonymous（未被识别为内建的函数调用），
    # 列名/字面量不会出现在这里，因此不会误杀。
    for fn in root.find_all(exp.Anonymous):
        fname = fn.name.lower()
        if fname in DANGEROUS_FUNCTIONS:
            return False, f"检测到危险函数: {fname}()（可能造成任意代码执行或文件读写）"

    return True, None


def sanitize_sql(sql: str) -> tuple[str, bool, str | None]:
    """清理并校验 SQL。

    清理只做"无语义副作用"的规范化：去首尾空白、去末尾分号。
    不做语句截断（多语句直接拒绝），不做内容改写——安全层不偷偷改模型输出，
    问题暴露给重试循环，trace 里也能忠实记录。

    返回 (clean_sql, ok, reason)
    """
    if not sql:
        return "", False, "空 SQL"

    cleaned = sql.strip()
    # 去掉末尾一个或多个分号（模型常带尾分号，SQLite 不介意，统一去掉更干净）
    cleaned = cleaned.rstrip(";").strip()

    ok, reason = check_readonly(cleaned)
    return cleaned, ok, reason


# ---------------------------------------------------------------------------
# 自检：独立运行 python safety.py。覆盖放行、拦截、防误杀三类用例。
# ---------------------------------------------------------------------------
def _self_test() -> None:
    # (sql, expected_ok)
    test_cases = [
        # ---- 必须放行 ----
        ("SELECT * FROM users", True),
        ("SELECT count(*) FROM users WHERE age > 18", True),
        ("WITH t AS (SELECT 1 AS x) SELECT * FROM t", True),
        ("SELECT a FROM t1 UNION SELECT b FROM t2", True),
        ("SELECT a FROM t1 INTERSECT SELECT b FROM t2", True),
        ("SELECT a FROM t1 EXCEPT SELECT b FROM t2", True),
        ("  SELECT * FROM users  ", True),
        ("SELECT * FROM users;", True),
        ("select * from users", True),
        # 多层嵌套子查询
        ("SELECT * FROM (SELECT id FROM (SELECT id FROM t) x) y", True),
        # AST 方案的关键优势：列名/别名/字面量含写关键字不能误杀
        ("SELECT drop_rate, updated_at, insert_id FROM events", True),
        ("SELECT * FROM t WHERE remark = 'please DROP this UPDATE'", True),
        ("SELECT d.id FROM delete_log d JOIN create_user c ON d.uid = c.id", True),

        # ---- 必须拦截：写操作 ----
        ("INSERT INTO users VALUES (1, 'a')", False),
        ("UPDATE users SET name = 'b' WHERE id = 1", False),
        ("DELETE FROM users WHERE id = 1", False),
        ("DROP TABLE users", False),
        ("ALTER TABLE users ADD COLUMN email TEXT", False),
        ("CREATE TABLE users (id INT)", False),
        ("REPLACE INTO users VALUES (1)", False),
        ("ATTACH DATABASE 'x.db' AS x", False),
        ("PRAGMA journal_mode=WAL", False),
        ("VACUUM", False),

        # ---- 必须拦截：多语句注入 ----
        ("SELECT * FROM users; DROP TABLE users", False),
        ("SELECT 1; INSERT INTO users VALUES (1)", False),

        # ---- 必须拦截：危险函数 ----
        ("SELECT load_extension('evil')", False),
        ("SELECT readfile('/etc/passwd')", False),
        ("SELECT writefile('/tmp/x', 'y')", False),
    ]

    passed = failed = 0
    for sql, expected_ok in test_cases:
        clean, ok, reason = sanitize_sql(sql)
        status = "PASS" if ok == expected_ok else "FAIL"
        if ok == expected_ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] expect={str(expected_ok):5s} got={str(ok):5s} "
              f"{sql[:52]:54s} {reason or ''}")

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit("安全层自检失败")
    print("安全层自检全部通过 ✅")


if __name__ == "__main__":
    _self_test()
