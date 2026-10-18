"""exec_repair.py — M5 Step1: 确定性 SQL 执行修复器（无训练）。

对 SFT 单次生成中"可确定修复"的执行失败做后处理，目标 ER 0.9594 -> ~0.98。

修复规则（只做"正确性有把握"的项，宁可不改也不改错）：
  R1 表名：大小写精确 -> 单复数模糊（SECTION -> Sections）。
  R2 双引号字符串字面量 -> 单引号（该 token 不是任何真实列/表名时）。
  R3 列名（按顺序，取第一个命中）：
     a. 限定表内大小写精确 -> 修正大小写（Makeid -> MakeId）。
     b. 跨表精确重指派：该列名只在【另一个已 join 的表】中存在 -> 改写限定符
        （T2.document_name -> T1.Document_Name、T2.Name -> T3.Name）。
     c. 笔误模糊：该列名【不是任何真实列】（pet_type / IndependantYear）且逐表唯一
        -> 修正为真实列名（PetType / IndepYear）。

不做：跨表模糊（maker->make、airportname->destairport 这类会改语义的错误映射）、
     以及对已是真实列名但属于"缺 JOIN"结构错误的硬改。
"""
from __future__ import annotations

import difflib
import sqlglot
from sqlglot import exp

FUZZY_CUTOFF = 0.6
FUZZY_MARGIN = 0.10


def _build_index(schema) -> dict:
    tables_orig = schema["table_names_original"]
    col_map = {}
    cols_by_table = {}
    for i, tn in enumerate(tables_orig):
        cols_by_table.setdefault(tn.lower(), [])
    for tidx, cname in schema["column_names_original"]:
        if tidx < 0:
            continue
        tn = tables_orig[tidx]
        cols_by_table[tn.lower()].append((cname, cname.lower()))
        col_map.setdefault(cname.lower(), []).append((cname, tn.lower()))
    return {
        "tables_lower": {t.lower(): t for t in tables_orig},
        "cols_by_table": cols_by_table,
        "col_map": col_map,
        "all_cols_lower": set(col_map.keys()),
        "all_tables_lower": set(t.lower() for t in tables_orig),
    }


def _fuzzy_unique(name: str, candidates: list[str]):
    matches = difflib.get_close_matches(name, candidates, n=3, cutoff=FUZZY_CUTOFF)
    if not matches:
        return None
    top = matches[0]
    r_top = difflib.SequenceMatcher(None, name, top).ratio()
    if len(matches) > 1:
        r_2 = difflib.SequenceMatcher(None, name, matches[1]).ratio()
        if r_top - r_2 < FUZZY_MARGIN:
            return None
    return top


def repair_sql(sql: str, schema: dict):
    """返回 (repaired_sql, notes)。解析失败/无修改时 repaired_sql == sql。"""
    if not sql:
        return sql, ["empty"]
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return sql, ["parse_error"]

    idx = _build_index(schema)
    notes = []

    # R1 表名 + 建立 alias 映射
    alias_to_table = {}   # alias_lower -> lower real table
    table_to_alias = {}   # lower real table -> 可见名（alias 优先）
    for t in tree.find_all(exp.Table):
        raw = t.name or ""
        ln = raw.lower()
        if ln in idx["tables_lower"]:
            if raw != idx["tables_lower"][ln]:
                t.set("this", exp.Identifier(this=idx["tables_lower"][ln], quoted=False))
                notes.append(f"table_case:{raw}->{idx['tables_lower'][ln]}")
            real_lower = ln
        else:
            close = difflib.get_close_matches(ln, list(idx["tables_lower"]), n=1, cutoff=0.85)
            if close:
                t.set("this", exp.Identifier(this=idx["tables_lower"][close[0]], quoted=False))
                notes.append(f"table_fuzzy:{raw}->{idx['tables_lower'][close[0]]}")
                real_lower = close[0]
            else:
                real_lower = None
        if real_lower:
            alias = t.alias_or_name
            alias_to_table.setdefault(alias.lower(), real_lower)
            table_to_alias.setdefault(real_lower, alias or t.name)

    # R3 列名 + R2 双引号
    for c in tree.find_all(exp.Column):
        ident = c.args.get("this")
        is_quoted = isinstance(ident, exp.Identifier) and bool(ident.args.get("quoted"))
        cname = c.name or ""
        lname = cname.lower()
        if not lname:
            continue

        # R2 双引号、非真实列/表名 -> 字符串字面量
        if is_quoted and lname not in idx["all_cols_lower"] and lname not in idx["all_tables_lower"]:
            c.replace(exp.Literal.string(cname))
            notes.append(f"dq_string:{cname}")
            continue

        qualifier = c.table
        target_lower = None
        if qualifier:
            ql = qualifier.lower()
            target_lower = alias_to_table.get(ql)
            if target_lower is None and ql in idx["tables_lower"]:
                target_lower = ql

        # R3a 限定表内大小写精确
        if target_lower and target_lower in idx["cols_by_table"]:
            lower2orig = {cl: oc for oc, cl in idx["cols_by_table"][target_lower]}
            if lname in lower2orig:
                if cname != lower2orig[lname]:
                    c.set("this", exp.Identifier(this=lower2orig[lname], quoted=False))
                    notes.append(f"col_case:{cname}->{lower2orig[lname]}")
                continue

        # R3b 跨表精确重指派（该列只属于另一个已 join 的表）
        candidates = idx["col_map"].get(lname)
        if candidates:
            tables_of_col = {tl for _, tl in candidates}
            if target_lower not in tables_of_col:
                joined = [tl for tl in tables_of_col if tl in table_to_alias]
                if len(joined) == 1:
                    new_alias = table_to_alias[joined[0]]
                    orig_name = next(oc for oc, tl in candidates if tl == joined[0])
                    new_col = exp.Column(
                        this=exp.Identifier(this=orig_name, quoted=False),
                        table=exp.Identifier(this=new_alias, quoted=False),
                    )
                    c.replace(new_col)
                    notes.append(f"col_reassign:{cname}->{new_alias}.{orig_name}")
                    continue

        # R3c 笔误模糊：列名不是任何真实列，且跨全部列唯一命中
        if lname not in idx["all_cols_lower"]:
            close = _fuzzy_unique(lname, list(idx["all_cols_lower"]))
            if close:
                cands = idx["col_map"][close]
                unique_tables = {tl for _, tl in cands}
                if len(unique_tables) == 1:
                    orig_name = cands[0][0]
                    if target_lower and target_lower in table_to_alias and target_lower in unique_tables:
                        c.set("this", exp.Identifier(this=orig_name, quoted=False))
                    elif not target_lower:
                        c.set("this", exp.Identifier(this=orig_name, quoted=False))
                    else:
                        continue
                    notes.append(f"col_typo:{cname}->{orig_name}")
                    continue

    return tree.sql(dialect="sqlite"), notes