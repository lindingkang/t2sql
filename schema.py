"""schema.py — 把 tables.json 转成给模型看的 DDL 文本。

设计取舍：
1. 用 CREATE TABLE（DDL）格式，不用 JSON / 纯文本列表：
   LLM 预训练语料中 SQL DDL 出现频率极高，模型对"DDL 里的表/列/主键"
   天然熟悉，这是 zero-shot Text2SQL 最常用也最有效的 schema 表示。
2. 数据源用 tables.json，不直接读 .sqlite 的真实 DDL：
   纯函数、无额外 IO、可离线测试，且与 dev.json 的 db_id 严格对齐。
   代价：类型是 Spider 的抽象类型（text/number/...），需要做一次映射。
3. 外键以 SQL 注释形式附在文末：模型读得到，又不破坏 DDL 语法合法性
   （--verify 模式会把这段文本喂给 sqlite 真实建表来验证这一点）。

用法：
    python schema.py                  # 打印 concert_singer 的 DDL
    python schema.py --db wta_1       # 指定库
    python schema.py --verify         # 额外用 sqlite 内存库验证 DDL 语法合法
"""

import argparse
import json
from pathlib import Path

SPIDER_DIR = Path(__file__).resolve().parent / "spider"

# Spider 抽象类型 → SQL 类型的映射。
# 注意：这只是"给模型看的提示类型"，不用于真实建表。
# number 在真实库中可能是 INTEGER 也可能是 REAL，统一写 NUMBER
# 反而避免模型去猜具体精度，误导生成。
TYPE_MAP = {
    "text": "TEXT",
    "number": "NUMBER",
    "time": "DATE",
    "others": "TEXT",
    "boolean": "BOOLEAN",
}


def load_schemas() -> dict[str, dict]:
    """读 tables.json，建成 db_id -> schema 的字典（O(1) 按库查询）。"""
    with open(SPIDER_DIR / "tables.json", encoding="utf-8") as f:
        tables = json.load(f)
    return {t["db_id"]: t for t in tables}


def get_ddl(schema: dict) -> str:
    """把单个库的 schema 字典转成 DDL 文本。"""
    cols = schema["column_names_original"]  # [[table_idx, 列名], ...]
    types = schema["column_types"]          # 与 cols 平行的类型数组
    pk = schema["primary_keys"]             # 元素是 cols 的下标
    fk = schema["foreign_keys"]             # [[下标a, 下标b], ...]
    table_names = schema["table_names_original"]

    # 把扁平列数组按 table_idx 归组：table_idx -> [(列下标, 列名, 类型)]
    # （table_idx = -1 的 "*" 占位行在这里被丢弃，它不是真实列）
    by_table: dict[int, list[tuple[int, str, str]]] = {
        i: [] for i in range(len(table_names))
    }
    for col_idx, ((t_idx, col_name), col_type) in enumerate(zip(cols, types)):
        if t_idx != -1:
            by_table[t_idx].append((col_idx, col_name, col_type))

    pk_set = set(pk)

    # sqlite_sequence 是 sqlite 为 AUTOINCREMENT 自动建的内部簿记表，
    # 被 Spider 的导出工具误收录（world_1 / soccer_1 / store_1 三个库）。
    # 已验证全部 8034 条 gold SQL 没有一条查询它——它不是真实 schema，
    # 放进 prompt 只会浪费 token、干扰模型，直接过滤。
    real_tables = [i for i, n in enumerate(table_names)
                   if n.lower() != "sqlite_sequence"]

    statements = []
    for t_idx in real_tables:
        t_name = table_names[t_idx]
        # 本表拥有的主键列（联合主键可能有多个，且理论上都属同一张表）
        pk_in_table = [i for i in pk if i < len(cols) and cols[i][0] == t_idx]

        col_defs = []
        for col_idx, col_name, col_type in by_table[t_idx]:
            sql_type = TYPE_MAP.get(col_type, "TEXT")
            # 标识符统一加双引号：Spider 列名含大写、下划线甚至保留字，
            # quote 之后 DDL 在任何方言里都合法，模型也不会学到坏习惯。
            col_def = f'  "{col_name}" {sql_type}'
            # 单列主键行内标注；联合主键留到表尾统一声明（SQL 标准写法）
            if col_idx in pk_set and len(pk_in_table) <= 1:
                col_def += " PRIMARY KEY"
            col_defs.append(col_def)
        if len(pk_in_table) > 1:
            pk_cols = ", ".join(f'"{cols[i][1]}"' for i in pk_in_table)
            col_defs.append(f"  PRIMARY KEY ({pk_cols})")

        body = ",\n".join(col_defs)
        statements.append(f'CREATE TABLE "{t_name}" (\n{body}\n);')

    ddl = "\n\n".join(statements)
    if fk:
        fk_lines = ["-- Foreign keys:"]
        for a, b in fk:
            ta, ca = cols[a]
            tb, cb = cols[b]
            fk_lines.append(
                f'--   "{table_names[ta]}"."{ca}" -> "{table_names[tb]}"."{cb}"'
            )
        ddl += "\n\n" + "\n".join(fk_lines)
    return ddl


def verify_ddl(ddl: str) -> None:
    """用 sqlite 内存库真实执行 DDL，语法不合法会直接抛异常。

    这是对"生成文本"的自动化断言：不依赖人眼检查，和整个项目
    '可验证奖励'的思路一致。注释行会被 sqlite 自然跳过。
    """
    import sqlite3

    con = sqlite3.connect(":memory:")
    con.executescript(ddl)
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="concert_singer", help="db_id")
    parser.add_argument("--verify", action="store_true",
                        help="用 sqlite 内存库验证生成的 DDL 语法合法")
    args = parser.parse_args()

    schemas = load_schemas()
    if args.db not in schemas:
        print(f"未知 db_id: {args.db}（tables.json 中共 {len(schemas)} 个库）")
        return

    ddl = get_ddl(schemas[args.db])
    print(ddl)
    if args.verify:
        verify_ddl(ddl)
        print("\n[verify] DDL 可在 sqlite 中建表成功 ✓")


if __name__ == "__main__":
    main()
