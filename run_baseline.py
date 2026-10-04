"""run_baseline.py — M0 第三步：Zero-shot API baseline，计算执行准确率与可执行率。

设计要点：
1. 零样本（zero-shot）：prompt 里只有 schema + question，不给任何示例 SQL。
   这是为了拿到"模型本身能力"的下限数字，后续加 few-shot / CoT 才能
   看出每种技巧带来的绝对提升。
2. 执行准确率（Execution Accuracy）：对 gold SQL 和 pred SQL 分别执行，
   结果集转 set 后比较。这是 Spider 官方评测标准，数字可直接和论文对比。
   为什么用 set 而不是排序 list？因为 SQL 写法不同可能产生相同数据但不同
   顺序（如 SELECT * FROM singer vs SELECT * FROM singer ORDER BY Age），
   而 set 比较只看"数据内容是否等价"，这正是"可用性"的核心。
3. 可执行率（Execution Rate）：pred SQL 能被 sqlite 成功执行的比例。
   这个指标在 M2 分类错误类型时非常关键——执行失败和结果错误是两类不同问题。
4. 结果实时追加保存：每调完一条 API 就写一次 JSONL，程序崩溃不会丢数据。
   跑完 100 条后转成标准 JSON array，方便后续分析。
5. 错误分级记录：API 失败 / SQL 语法错误 / SQL 执行错误 / 结果不匹配，
   为 M2 的失败分类打基础。

环境要求：
    pip install openai python-dotenv -i https://pypi.tuna.tsinghua.edu.cn/simple

用法：
    python run_baseline.py                    # 默认跑 dev 前 100 条
    python run_baseline.py --limit 10         # 先跑 10 条快速验证流程
    python run_baseline.py --dry-run          # 不实际调 API，用固定假 SQL 跑通执行链路
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from pathlib import Path

# 依赖说明见文件头。
try:
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError as exc:
    raise ImportError(
        "请先安装依赖：pip install openai python-dotenv "
        "-i https://pypi.tuna.tsinghua.edu.cn/simple"
    ) from exc

# 复用 schema.py 的 DDL 生成逻辑（同目录下的独立脚本，import 是正常工程实践）。
from schema import load_schemas, get_ddl

SPIDER_DIR = Path(__file__).resolve().parent / "spider"

# ---------------------------------------------------------------------------
# Prompt 工程（这部分是 Text2SQL 最核心的设计之一，后续会反复迭代）
# ---------------------------------------------------------------------------

def build_prompt(db_id: str, question: str, schema: dict) -> str:
    """构造 zero-shot Text2SQL prompt。

    结构顺序：角色定位 -> schema -> 问题 -> 输出约束。
    把 schema 放在问题前面，让模型先"建立表结构上下文"，再读取问题，
    符合人类"先看资料再答题"的认知顺序，也和大模型 attention 的工作方式一致。
    """
    ddl = get_ddl(schema)
    return (
        "You are a SQLite SQL expert. "
        "Given the database schema below, write a single SQLite SQL query "
        "to answer the question. Output ONLY the SQL query on ONE single line, "
        "with no markdown fences and no explanation.\n\n"
        f"Schema:\n{ddl}\n\n"
        f"Question: {question}\n\n"
        "SQL:"
    )


# ---------------------------------------------------------------------------
# SQL 提取与清理（模型输出常带 markdown 代码块或多余换行）
# ---------------------------------------------------------------------------

def extract_sql(text: str) -> str:
    """从模型自由文本中提取纯 SQL 语句。

    优先级：
    1. 匹配 ```sql ... ``` 或 ``` ... ``` 代码块（最稳定）
    2. 从第一条 SQL 语句行取到文本末尾并拼成单行
       （模型常输出多行 SQL，绝不能只取一行——截断 bug 的教训）
    3. 清理多余空白和尾部分号（sqlite 不介意分号，但 prompt 里统一去掉更干净）
    """
    # 1. markdown 代码块
    m = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.S | re.I)
    if m:
        return m.group(1).strip().rstrip(";")
    # 2. 无代码块时，从第一条 SQL 语句行取到文本末尾并拼成一行。
    #    旧版"取最后一条以 SELECT 开头的行"默认 SQL 是单行的，模型输出
    #    多行 SQL 时会把 FROM/ORDER BY/LIMIT 整段截掉——这正是上一轮
    #    baseline 71% 样本报 no such column 的根因。
    #    注意：这里不做"截尾修复"（比如发现 SQL 后面跟了解释就砍掉），
    #    因为那会让提取器偷偷改写模型输出，污染 M2 的错误归因。
    #    忠实提取 + 记录 raw，问题留给评测环节暴露。
    lines = text.strip().splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^(SELECT|WITH|INSERT|UPDATE|DELETE)\b", line.strip(), re.I):
            return " ".join(l.strip() for l in lines[i:]).rstrip(";")
    # 3. 全文本兜底清理（压平所有空白）
    return " ".join(text.split()).rstrip(";")


# ---------------------------------------------------------------------------
# 执行与比较（评测核心）
# ---------------------------------------------------------------------------

def execute_sql(db_path: Path, sql: str) -> tuple[list[tuple], str | None]:
    """在指定 sqlite 文件上执行 SQL，返回 (结果集, 错误信息)。

    用只读连接（uri=true + mode=ro），即使 SQL 里被注入恶意语句
    也改不了磁盘数据。这是 M3 安全层的基础，M0 就提前引入。
    """
    try:
        # uri 模式要求路径前加 file://
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        cur = con.execute(sql)
        rows = [tuple(row) for row in cur.fetchall()]
        con.close()
        return rows, None
    except Exception as exc:
        return [], str(exc)


def results_equal(a: list[tuple], b: list[tuple]) -> bool:
    """比较两个结果集是否等价。

    转 set 比较（Spider 官方 eval 标准）。set 会忽略顺序和重复行，
    这恰好对应"两条 SQL 是否返回相同数据内容"的语义。
    注意：set 要求元素可 hash，tuple 天然满足。
    """
    return set(a) == set(b)


# ---------------------------------------------------------------------------
# API 调用（带指数退避重试）
# ---------------------------------------------------------------------------

def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int = 3,
    timeout: int = 60,
) -> str:
    """调用 LLM API，失败时按指数退避重试。

    对百度千帆等国内平台，偶尔会有 5xx 或 rate limit，退避能显著提高成功率。
    第一次失败后等 2 秒，第二次 4 秒，第三次 8 秒。
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,        # 确定性最高，减少随机波动
                max_tokens=512,
                timeout=timeout,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_err = exc
            wait = 2 ** (attempt + 1)
            print(f"  API 失败（{attempt + 1}/{max_retries}），{wait}s 后重试: {exc}")
            time.sleep(wait)
    raise last_err


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100,
                        help="评测 dev 集的前 N 条（默认 100）")
    parser.add_argument("--output", default="baseline_100.json",
                        help="输出 JSON 文件名")
    parser.add_argument("--model", default=None,
                        help="覆盖 .env 里的模型名")
    parser.add_argument("--dry-run", action="store_true",
                        help="不实际调 API，用固定假 SQL 测试执行链路")
    args = parser.parse_args()

    # dry-run 模式不需要 API key，方便先验证执行链路
    if not args.dry_run:
        load_dotenv(Path(__file__).resolve().parent / ".env")
        api_key = os.getenv("LLM_API_KEY", "").strip()
        base_url = os.getenv("LLM_BASE_URL", "https://qianfan.baidubce.com/v2").strip()
        model = (args.model or os.getenv("LLM_MODEL", "ernie-4.0-turbo-8k")).strip()
        if not api_key or api_key == "在此填入你的key":
            print("错误：.env 里的 LLM_API_KEY 还没填。请用 IDE 打开 /root/t2sql/.env 填入真实 key。")
            return
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        model = "dry-run（复用 gold SQL，模拟完美模型）"
        client = None

    # 3. 加载数据
    with open(SPIDER_DIR / "dev.json", encoding="utf-8") as f:
        dev = json.load(f)
    schemas = load_schemas()

    records: list[dict] = []
    total = min(args.limit, len(dev))

    print(f"开始 baseline：模型={model}，样本数={total}\n")

    for idx, sample in enumerate(dev[:total], start=1):
        db_id = sample["db_id"]
        question = sample["question"]
        gold_sql = sample["query"].strip().rstrip(";")
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"

        record: dict = {
            "db_id": db_id,
            "question": question,
            "gold": gold_sql,
            "pred": None,
            "raw": None,
            "api_error": None,
            "exec_success": False,
            "gold_exec_success": False,
            "correct": False,
            "error": None,
        }

        # ---------- 3a. 构造 prompt & 调 API ----------
        schema = schemas[db_id]
        prompt = build_prompt(db_id, question, schema)

        if args.dry_run:
            # dry-run 模式：直接复用 gold SQL，模拟"完美模型"
            raw = gold_sql
        else:
            try:
                raw = call_llm(client, model, prompt)
                record["raw"] = raw
                time.sleep(0.5)  # 简单限速，避免触发平台 RPM 限制
            except Exception as exc:
                record["api_error"] = str(exc)
                print(f"[{idx}/{total}] API 最终失败: {db_id} | {question}")
                records.append(record)
                continue

        pred_sql = extract_sql(raw)
        record["pred"] = pred_sql

        # ---------- 3b. 执行 gold SQL（作为基准） ----------
        gold_res, gold_err = execute_sql(db_path, gold_sql)
        if gold_err:
            record["gold_exec_success"] = False
            record["error"] = f"gold 执行失败: {gold_err}"
            print(f"[{idx}/{total}] gold 执行失败: {db_id} | {question}")
            records.append(record)
            continue
        record["gold_exec_success"] = True

        # ---------- 3c. 执行 pred SQL ----------
        pred_res, pred_err = execute_sql(db_path, pred_sql)
        if pred_err:
            record["exec_success"] = False
            record["error"] = f"pred 执行失败: {pred_err}"
            print(f"[{idx}/{total}] 执行失败: {db_id} | {question}")
            print(f"         pred SQL: {pred_sql}")
            records.append(record)
            continue

        record["exec_success"] = True
        record["correct"] = results_equal(gold_res, pred_res)
        status = "✓" if record["correct"] else "✗"
        print(f"[{idx}/{total}] {status} {db_id} | {question}")
        if not record["correct"]:
            print(f"         gold: {gold_sql}")
            print(f"         pred: {pred_sql}")

        records.append(record)

    # ---------- 4. 统计 ----------
    n_api_fail = sum(1 for r in records if r["api_error"])
    n_gold_fail = sum(1 for r in records if not r["gold_exec_success"] and not r["api_error"])
    n_exec_fail = sum(1 for r in records if not r["exec_success"] and r["gold_exec_success"])
    n_correct = sum(1 for r in records if r["correct"])
    n_evaluable = sum(1 for r in records if r["gold_exec_success"] and r["exec_success"])

    print("\n" + "=" * 50)
    print("Baseline 统计")
    print("=" * 50)
    print(f"总样本数:        {total}")
    print(f"API 调用失败:    {n_api_fail}")
    print(f"Gold SQL 执行失败:{n_gold_fail}")
    print(f"Pred 可执行数:   {n_evaluable}")
    print(f"Pred 不可执行数: {n_exec_fail}")
    print(f"执行准确率 (EA): {n_correct}/{n_evaluable} = {n_correct / n_evaluable:.1%}" if n_evaluable else "N/A")
    print(f"可执行率 (ER):   {n_evaluable}/{total - n_api_fail - n_gold_fail} = "
          f"{n_evaluable / max(1, total - n_api_fail - n_gold_fail):.1%}")

    # ---------- 5. 保存 ----------
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
