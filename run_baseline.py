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
from collections import Counter
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

# M3 模块2：执行加固的两个阈值。
# 数值不是拍脑袋，是基于全量 1034 条 gold SQL 的实测分布定的：
#   - 最慢 gold 查询 4.3s（wta_1 大表聚合），故 timeout=10s 留 2 倍余量，
#     正常查询永不触发，只拦笛卡尔积炸弹（如三表自连接实测 0.3s 即被中断）。
#   - 最大 gold 结果集 20662 行（wta_1 全球员名单），故 row_limit=50000
#     留 2.4 倍余量，gold 永不被截断，评测口径保持不变。
DEFAULT_EXEC_TIMEOUT = 10.0   # 秒
DEFAULT_ROW_LIMIT = 50_000    # 单次查询最多返回行数


def execute_sql(db_path: Path, sql: str,
                timeout: float = DEFAULT_EXEC_TIMEOUT,
                row_limit: int = DEFAULT_ROW_LIMIT
                ) -> tuple[list[tuple], str | None]:
    """在指定 sqlite 文件上执行 SQL，返回 (结果集, 错误信息)。

    三道保护：
    1. 只读连接（uri=true + mode=ro）：即使 SQL 被注入写操作也改不了磁盘，
       这是 M3 安全层的第二道防线（第一道是 safety.py 的 AST 白名单）。
    2. 执行超时（set_progress_handler）：每 1000 条字节码指令回调一次，
       超过 timeout 秒返回 1 请求 sqlite 中断（抛 OperationalError:
       interrupted），防止笛卡尔积/全表扫描把进程拖死。投票阶段同题执行
       N 条 SQL，炸弹风险放大 N 倍，这层必不可少。
    3. 行数上限（fetchmany 惰性计数）：只 fetch row_limit+1 行，多取到 1 行
       即判定超限，防止内存爆炸。刻意不采用 "SELECT * FROM (<sql>) LIMIT n"
       的 SQL 包裹法——包裹会改变 ORDER BY 的列作用域（flight_2 的复合查询
       badcase 已验证），且可能改变执行计划；在 Python 侧计数对 SQL 零改写。

    超限时 err 带标准前缀（EXEC_TIMEOUT / ROW_LIMIT_EXCEEDED），
    供 degrade.classify_execution_error 识别并选择降级路径。
    """
    con = None
    try:
        start = time.monotonic()

        def _progress_handler() -> int:
            # sqlite 每执行约 1000 条虚拟机指令回调一次。
            # 返回 0 继续执行，返回非 0 请求中断（OperationalError: interrupted）。
            return 1 if (time.monotonic() - start) > timeout else 0

        # uri 模式要求路径前加 file://
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        # 关键：Spider 数据集中部分库（如 wta_1）的文本字段存储为 Latin-1/cp1252
        # 而非 UTF-8（如球员名 Albarracín 的 í = 单字节 0xED）。
        # 默认 text_factory 遇到非法 UTF-8 会抛异常，导致所有含非 ASCII 字符的
        # 查询结果集为空/报错。用 replacement char 兜底，保证至少能读出数据。
        con.text_factory = lambda b: b.decode("utf-8", errors="replace")
        con.row_factory = sqlite3.Row
        con.set_progress_handler(_progress_handler, 1000)
        cur = con.execute(sql)

        # 惰性取 row_limit+1 行：若真有第 n+1 行，说明结果集超限。
        fetched = cur.fetchmany(row_limit + 1)
        rows = [tuple(row) for row in fetched]
        if len(rows) > row_limit:
            return rows[:row_limit], (
                f"ROW_LIMIT_EXCEEDED: 结果集超过 {row_limit} 行上限，"
                f"已截断（请增加 WHERE 过滤或 LIMIT，避免无界查询）")
        return rows, None
    except sqlite3.OperationalError as exc:
        # progress handler 主动中断时 sqlite 的报错文本就是 "interrupted"，
        # 翻译成可读的超时标记，便于 degrade 走 timeout 降级路径。
        if str(exc).strip().lower() == "interrupted":
            return [], (f"EXEC_TIMEOUT: 查询执行超过 {timeout:g}s 被中断"
                        f"（可能存在笛卡尔积或缺失 JOIN 条件）")
        return [], str(exc)
    except Exception as exc:
        return [], str(exc)
    finally:
        if con is not None:
            con.close()


def results_equal_strict(a: list[tuple], b: list[tuple]) -> bool:
    """严格版比较（Spider 官方口径）。

    结果集转 set：忽略行顺序和重复行，但**列顺序敏感**——gold 返回
    (avg, max, type) 而 pred 返回 (type, avg, max) 判错。理由：用户
    指定的列顺序是需求的一部分。这个口径的数字可直接与论文对比。
    """
    return set(a) == set(b)


def results_equal_loose(a: list[tuple], b: list[tuple]) -> bool:
    """宽松版比较（数据等价口径）。

    每行的元素先排序再转 set：忽略行顺序、重复行，且**忽略列顺序**——
    只要每行的数据内容相同即判对。用途：衡量模型"是否真正理解了问题"，
    排除列顺序这种非语义差异的干扰。
    """
    if not a and not b:
        return True
    if not a or not b:
        return False
    if len(a[0]) != len(b[0]):
        return False
    # 同一行可能混含不同类型（如 ('dog', 5, 8)），Python 3 不允许 int 与 str
    # 直接比较，故排序键用 (类型名, 值)：先按类型归组，只在同类值之间比较。
    # 这样能忽略列顺序，又不会把数字 5 和文本 '5' 错误地等同。
    def row_key(row: tuple) -> tuple:
        return tuple(sorted(
            (type(v).__name__, v if v is None or isinstance(v, (int, float, str))
             else str(v))
            for v in row
        ))
    return {row_key(r) for r in a} == {row_key(r) for r in b}


def classify_failure(gold_sql: str, pred_sql: str,
                     gold_res: list[tuple], pred_res: list[tuple]) -> str:
    """对"执行成功但结果不对"的样本做失败类型分类（M2 体系的雏形）。

    用规则启发式粗分，按固定优先级返回第一个命中的类别：
    - column_order : 仅列顺序不同（宽松版判对，属测量口径差异）
    - join_type    : JOIN 类型不同（LEFT/RIGHT/INNER/FULL，裸 join 算 INNER）
    - set_op       : 集合操作不同（UNION/INTERSECT/EXCEPT）
    - aggregate    : 聚合函数不同（COUNT/SUM/AVG/MIN/MAX）
    - extra_join   : pred 的表引用次数多于 gold（一对多连表可能致行膨胀）
    - other        : 兜底，M2 再人工细分
    """
    # 1. 仅列顺序差异（数据内容其实等价）
    if results_equal_loose(gold_res, pred_res):
        return "column_order"

    gold_lower = gold_sql.lower()
    pred_lower = pred_sql.lower()

    # 2. JOIN 类型差异
    join_types = ["left join", "right join", "inner join", "full join"]
    gold_joins = {j for j in join_types if j in gold_lower}
    pred_joins = {j for j in join_types if j in pred_lower}
    if " join " in gold_lower and not gold_joins:
        gold_joins.add("inner join")
    if " join " in pred_lower and not pred_joins:
        pred_joins.add("inner join")
    if gold_joins != pred_joins:
        return "join_type"

    # 3. 集合操作差异
    set_ops = ["intersect", "except", "union"]
    gold_sets = {s for s in set_ops if s in gold_lower}
    pred_sets = {s for s in set_ops if s in pred_lower}
    if gold_sets != pred_sets:
        return "set_op"

    # 4. 聚合函数差异
    aggs = ["count(", "sum(", "avg(", "min(", "max("]
    gold_agg = {a for a in aggs if a in gold_lower}
    pred_agg = {a for a in aggs if a in pred_lower}
    if gold_agg != pred_agg:
        return "aggregate"

    # 5. 表引用次数差异（FROM + JOIN 次数粗略反映连表规模）
    gold_tables = gold_lower.count(" from ") + gold_lower.count(" join ")
    pred_tables = pred_lower.count(" from ") + pred_lower.count(" join ")
    if pred_tables > gold_tables:
        return "extra_join"

    return "other"


# ---------------------------------------------------------------------------
# API 调用（带指数退避重试）
# ---------------------------------------------------------------------------

def call_llm_with_usage(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int = 3,
    timeout: int = 60,
    temperature: float = 0.0,
) -> tuple[str, dict]:
    """调用 LLM API，返回 (content, usage)。失败时按指数退避重试。

    对百度千帆等国内平台，偶尔会有 5xx 或 rate limit，退避能显著提高成功率。
    第一次失败后等 2 秒，第二次 4 秒，第三次 8 秒。

    temperature 参数：
        0.0  = 确定性最高，用于 baseline 和单轮生成
        > 0  = 促进多样性，用于自洽性投票的多采样

    硬超时兜底：httpx 的 timeout 参数在 TCP 连接阶段可能不被严格遵守
    （实测会永久挂起），用 signal.alarm 做第二道防线。

    M3 新增 usage 返回：{"prompt_tokens": int, "completion_tokens": int,
    "total_tokens": int}，用于成本看板。接口未返回 usage 时给 0 兜底，
    不影响主流程（成本计量是可观测能力，不是正确性依赖）。
    """
    import signal

    def _alarm_handler(signum, frame):
        raise TimeoutError(f"API 调用超过 {timeout+30}s 硬超时")

    last_err = None
    for attempt in range(max_retries):
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout + 30)  # 比 httpx timeout 多 30s 兜底
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=512,
                timeout=timeout,
            )
            content = resp.choices[0].message.content or ""
            u = getattr(resp, "usage", None)
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                "total_tokens": getattr(u, "total_tokens", 0) or 0,
            }
            return content, usage
        except Exception as exc:
            last_err = exc
            wait = 2 ** (attempt + 1)
            print(f"  API 失败（{attempt + 1}/{max_retries}），{wait}s 后重试: {exc}")
            time.sleep(wait)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    raise last_err


def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int = 3,
    timeout: int = 60,
    temperature: float = 0.0,
) -> str:
    """旧接口：只返回 content。内部转调 call_llm_with_usage（M3 起带计量）。"""
    content, _usage = call_llm_with_usage(
        client, model, prompt, max_retries=max_retries,
        timeout=timeout, temperature=temperature)
    return content


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
            "correct": False,        # 严格版（官方口径）
            "correct_loose": False,  # 宽松版（忽略列顺序）
            "failure_type": None,    # 失败分类标签（M2 体系雏形）
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
        record["correct"] = results_equal_strict(gold_res, pred_res)
        record["correct_loose"] = results_equal_loose(gold_res, pred_res)
        if not record["correct"]:
            record["failure_type"] = classify_failure(
                gold_sql, pred_sql, gold_res, pred_res)
        status = "✓" if record["correct"] else "✗"
        print(f"[{idx}/{total}] {status} {db_id} | {question}")
        if not record["correct"]:
            print(f"         gold: {gold_sql}")
            print(f"         pred: {pred_sql}")
            print(f"         失败类型: {record['failure_type']}")

        records.append(record)

    # ---------- 4. 统计 ----------
    n_api_fail = sum(1 for r in records if r["api_error"])
    n_gold_fail = sum(1 for r in records if not r["gold_exec_success"] and not r["api_error"])
    n_exec_fail = sum(1 for r in records if not r["exec_success"] and r["gold_exec_success"])
    n_correct = sum(1 for r in records if r["correct"])
    n_correct_loose = sum(1 for r in records if r.get("correct_loose"))
    n_evaluable = sum(1 for r in records if r["gold_exec_success"] and r["exec_success"])

    # 失败类型分布（仅统计可评测且严格判错的样本）
    failure_dist = Counter(r["failure_type"] for r in records
                           if not r["correct"] and r["exec_success"] and r["gold_exec_success"])

    print("\n" + "=" * 50)
    print("Baseline 统计")
    print("=" * 50)
    print(f"总样本数:        {total}")
    print(f"API 调用失败:    {n_api_fail}")
    print(f"Gold SQL 执行失败:{n_gold_fail}")
    print(f"Pred 可执行数:   {n_evaluable}")
    print(f"Pred 不可执行数: {n_exec_fail}")
    if n_evaluable:
        print(f"执行准确率 (EA·严格): {n_correct}/{n_evaluable} = {n_correct / n_evaluable:.1%}  ← 官方口径，可与论文对比")
        print(f"执行准确率 (EA·宽松): {n_correct_loose}/{n_evaluable} = {n_correct_loose / n_evaluable:.1%}  ← 忽略列顺序，衡量真实理解")
    else:
        print("执行准确率: N/A")
    denom = max(1, total - n_api_fail - n_gold_fail)
    print(f"可执行率 (ER):   {n_evaluable}/{denom} = {n_evaluable / denom:.1%}")
    if failure_dist:
        print("\n失败类型分布（严格判错的样本）：")
        for ft, cnt in failure_dist.most_common():
            print(f"  {ft:15s}: {cnt}")

    # ---------- 5. 保存 ----------
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
