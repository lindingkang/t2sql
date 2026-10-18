"""explore.py — M0 第一步：确认 Spider 数据能正常读入，并看清一条样本的完整结构。

本脚本验证三件事：
1. dev.json / train_spider.json / tables.json 能被正常解析（条数、字段名）
2. dev 集引用的每个 db_id，在 database/ 下都有对应的 .sqlite 文件
   （我注意到 database/ 下只有 166 个库目录，需要确认 dev 用到的库是否都在）
3. 演示 Spider 的"下标表示法"如何解码成表结构——这是 schema.py 的基础
"""

import json
from pathlib import Path

# 用 __file__ 定位项目根目录，而不是依赖运行时的当前目录（cwd）。
# 这样无论你 `cd` 到哪里执行 `python /root/t2sql/explore.py` 都能找到数据。
SPIDER_DIR = Path(__file__).resolve().parent / "spider"


def load_json(name: str) -> list:
    with open(SPIDER_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    dev = load_json("dev.json")
    train = load_json("train_spider.json")
    tables = load_json("tables.json")

    # ---------- 1. 数据规模 ----------
    print("=== 1. 数据规模 ===")
    print(f"dev 样本数:          {len(dev)}")
    print(f"train 样本数:        {len(train)}")
    print(f"tables.json 中的库数: {len(tables)}")
    print(f"dev 样本字段:         {sorted(dev[0].keys())}")

    # ---------- 2. 数据库文件完整性 ----------
    # dev 里每条样本的 db_id 都应该对应 database/<db_id>/<db_id>.sqlite
    print("\n=== 2. 数据库文件完整性 ===")
    db_ids = sorted({s["db_id"] for s in dev})
    missing = [
        d for d in db_ids
        if not (SPIDER_DIR / "database" / d / f"{d}.sqlite").exists()
    ]
    print(f"dev 涉及 {len(db_ids)} 个库")
    if missing:
        print(f"!! 缺失 {len(missing)} 个库文件: {missing}")
    else:
        print("所有库的 .sqlite 文件都在")

    # ---------- 3. 一条完整样本 ----------
    print("\n=== 3. 一条完整样本（dev[0]）===")
    sample = dev[0]
    print(json.dumps(sample, indent=2, ensure_ascii=False))

    # ---------- 4. 该样本对应库的 schema，演示下标解码 ----------
    print(f"\n=== 4. 对应库的 schema（db_id = {sample['db_id']}）===")
    schema = next(t for t in tables if t["db_id"] == sample["db_id"])

    # Spider 的关键设计：列不直接挂在表下，而是全库一个扁平数组，
    # column_names[i] = [table_idx, 列名]，用 table_idx 指回表。
    # table_idx = -1 是特殊约定，表示 "*"（SELECT * 的星号），不是真实列。
    # primary_keys / foreign_keys 里存的都是这个扁平数组的下标 i，不是列名。
    cols = schema["column_names_original"]   # [[table_idx, 列名], ...]
    types = schema["column_types"]           # 与 cols 平行的类型数组
    pk = set(schema["primary_keys"])         # 元素是 cols 的下标
    fk = schema["foreign_keys"]              # [[下标a, 下标b], ...]

    print(f"表: {schema['table_names_original']}")
    for t_idx, t_name in enumerate(schema["table_names_original"]):
        print(f"  {t_name}:")
        for i, ((owner, col), typ) in enumerate(zip(cols, types)):
            if owner != t_idx:  # owner==-1 的 "*" 行在这里自然被跳过
                continue
            mark = "  [PK]" if i in pk else ""
            print(f"    - {col} ({typ}){mark}")

    if fk:
        print("  外键:")
        for a, b in fk:
            ta, ca = cols[a]
            tb, cb = cols[b]
            print(f"    - {schema['table_names_original'][ta]}.{ca}"
                  f" -> {schema['table_names_original'][tb]}.{cb}")


if __name__ == "__main__":
    main()
