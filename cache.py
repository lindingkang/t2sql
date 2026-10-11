"""cache.py — M3 模块3：成本控制（两层磁盘缓存 + token 预算守卫）。

Agent 每题最多 15 次 API 调用（投票 5 + 重试 3 轮 × 多轮），全量 1034 题
约 1.5 万次调用。反复跑消融实验时，没缓存会重复烧大量预算。本文件提供
两层缓存：

    L1  LLM 响应缓存（LLMCache）
        缓存"确定性"API 调用（temperature=0：baseline 首轮、重试、空结果
        复查）。键 = sha256(model + temperature + max_tokens + prompt)。
        关键：temperature>0 的投票采样【默认不缓存】——5 次采样共享一个
        prompt，若按 prompt 缓存，5 次会命中同一条响应，投票直接退化为
        单采样，且错误地"看起来共识率 100%"。投票的复用走 L2。

    L2  题级结果缓存（ResultStore）
        缓存整题 answer trace（含 votes/attempts/最终 SQL）。键含 agent
        全部配置（model/n_samples/temperature/max_retries/版本标记）。
        评测脚本断点续跑时，已完成的题直接读 trace，一次 API 都不调。
        这才是"反复跑消融不烧钱"的主力。

存储选型：sqlite 单文件。零依赖、原子写、K-V 查询 O(1)，比成千上万的
小文件干净，也比单个大 JSON 重写安全。

token 预算守卫（truncate_ddl_to_budget）：
    prompt 超预算时按问题关键词保留相关表。实测 Spider 166 个库最大 DDL
    仅 8041 字符（约 2.3k tokens），默认预算下永不触发；保留函数是为
    schema 远大于 Spider 的生产场景。
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DB = ROOT / ".cache" / "m3_cache.sqlite"


def _hash_key(**parts) -> str:
    """把任意可序列化部件拼成稳定哈希键。"""
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class LLMCache:
    """L1：确定性 LLM 响应的 sqlite K-V 缓存。"""

    def __init__(self, db_path: Path | str = DEFAULT_CACHE_DB,
                 cache_temperature: float = 0.0):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 只有 temperature <= cache_temperature 的调用才缓存
        self.cache_temperature = cache_temperature
        self.hits = 0
        self.misses = 0
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_db(self) -> None:
        with self._conn() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS llm_response ("
                "  cache_key TEXT PRIMARY KEY,"
                "  content TEXT NOT NULL,"
                "  usage_json TEXT NOT NULL,"
                "  created_ts REAL NOT NULL)")

    @staticmethod
    def make_key(model: str, prompt: str, temperature: float,
                 max_tokens: int) -> str:
        return _hash_key(kind="llm", model=model, prompt=prompt,
                         temperature=round(float(temperature), 4),
                         max_tokens=max_tokens)

    def get(self, model: str, prompt: str, temperature: float,
            max_tokens: int) -> tuple[str, dict] | None:
        """命中返回 (content, usage)，否则 None。"""
        if temperature > self.cache_temperature:
            return None  # 投票采样不读缓存
        key = self.make_key(model, prompt, temperature, max_tokens)
        with self._conn() as con:
            row = con.execute(
                "SELECT content, usage_json FROM llm_response WHERE cache_key=?",
                (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return row[0], json.loads(row[1])

    def put(self, model: str, prompt: str, temperature: float,
            max_tokens: int, content: str, usage: dict) -> None:
        if temperature > self.cache_temperature:
            return None  # 投票采样不写缓存
        key = self.make_key(model, prompt, temperature, max_tokens)
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO llm_response "
                "(cache_key, content, usage_json, created_ts) VALUES (?,?,?,?)",
                (key, content, json.dumps(usage, ensure_ascii=False), time.time()))

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / max(1, self.hits + self.misses)}


class ResultStore:
    """L2：题级 answer trace 缓存，供评测断点续跑。

    用法：
        store = ResultStore("runs/agent_vote5.sqlite")
        key = store.make_key(model, "car_1", question, n_samples=5, ...)
        trace = store.get(key)          # 已完成则直接取
        if trace is None: trace = agent.answer(...); store.put(key, trace)
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        with self._conn() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS answer_trace ("
                "  cache_key TEXT PRIMARY KEY,"
                "  db_id TEXT NOT NULL,"
                "  question TEXT NOT NULL,"
                "  trace_json TEXT NOT NULL,"
                "  created_ts REAL NOT NULL)")

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    @staticmethod
    def make_key(model: str, db_id: str, question: str, **config) -> str:
        return _hash_key(kind="answer", model=model, db_id=db_id,
                         question=question.strip(), config=config)

    def get(self, key: str) -> dict | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT trace_json FROM answer_trace WHERE cache_key=?",
                (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, key: str, db_id: str, question: str, trace: dict) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO answer_trace "
                "(cache_key, db_id, question, trace_json, created_ts) "
                "VALUES (?,?,?,?,?)",
                (key, db_id, question,
                 json.dumps(trace, ensure_ascii=False, default=str),
                 time.time()))

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses}


# ---------------------------------------------------------------------------
# token 预算守卫：schema 截断（Spider 默认预算下不触发，保留给大库场景）
# ---------------------------------------------------------------------------

def truncate_ddl_to_budget(ddl: str, question: str,
                           max_chars: int = 21000) -> str:
    """DDL 超预算时，按问题关键词保留相关表，其余丢弃。

    取舍：宁可多保留也不漏表——除关键词命中外，总是保留外键被引用的表，
    避免把 JOIN 链上的关联表截掉。Spider 最大 DDL 8041 字符，默认
    max_chars=21000 永不触发；该函数为"大库生产场景"预留。
    返回原 DDL 或截断后的 DDL（带截断说明头）。
    """
    if len(ddl) <= max_chars:
        return ddl

    # 拆出每张表的 CREATE TABLE 块
    blocks = re.findall(r"CREATE TABLE .*?;", ddl, flags=re.S)
    if not blocks:
        return ddl[:max_chars]

    def table_name(block: str) -> str:
        m = re.search(r'CREATE TABLE\s+"?([A-Za-z0-9_]+)"?', block, re.I)
        return m.group(1).lower() if m else ""

    q_words = {w for w in re.findall(r"[a-z_]{3,}", question.lower())}
    scored = []
    for b in blocks:
        tname = table_name(b)
        # 表名按 _ 拆开参与关键词匹配（car_makers 命中 car/makers）
        name_tokens = set(re.split(r"[_\s]+", tname))
        score = len(q_words & name_tokens)
        scored.append((score, tname, b))

    kept = [b for score, _, b in sorted(scored, reverse=True) if score > 0]
    # 至少保留分数最高的 3 张表，防止问题里全是虚词时一张不剩
    if len(kept) < 3:
        kept = [b for _, _, b in sorted(scored, reverse=True)[:3]]

    truncated = ("-- [BUDGET] schema 超 token 预算，已按问题相关性截断，"
                 "仅保留相关表：\n\n" + "\n\n".join(kept))
    return truncated[:max_chars]


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_test(tmp_dir: str = "/tmp/m3_cache_test") -> None:
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    cache = LLMCache(Path(tmp_dir) / "llm.sqlite")
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    # 确定性调用：未命中 -> 写入 -> 命中
    assert cache.get("m", "p", 0.0, 512) is None
    cache.put("m", "p", 0.0, 512, "SELECT 1", usage)
    hit = cache.get("m", "p", 0.0, 512)
    assert hit == ("SELECT 1", usage), hit
    print("[PASS] 确定性调用读写缓存")

    # 不同 prompt/温度/模型 键隔离
    assert cache.get("m", "p2", 0.0, 512) is None
    assert cache.get("m2", "p", 0.0, 512) is None
    print("[PASS] 缓存键隔离")

    # 投票（temp>0）：不读不写，防止采样退化
    assert cache.get("m", "p3", 0.7, 512) is None
    cache.put("m", "p3", 0.7, 512, "SELECT 2", usage)
    assert cache.get("m", "p3", 0.7, 512) is None
    print("[PASS] 高温调用不缓存（投票多样性保护）")

    # L2 题级缓存
    store = ResultStore(Path(tmp_dir) / "result.sqlite")
    key = store.make_key("m", "car_1", "How many models?", n_samples=5)
    key2 = store.make_key("m", "car_1", "How many models?", n_samples=3)
    assert key != key2
    assert store.get(key) is None
    store.put(key, "car_1", "How many models?", {"final_sql": "SELECT 1"})
    assert store.get(key)["final_sql"] == "SELECT 1"
    print("[PASS] 题级结果缓存 + 配置指纹隔离")

    # token 守卫：小 DDL 原样返回
    small_ddl = "CREATE TABLE \"a\" (\n  \"x\" NUMBER\n);"
    assert truncate_ddl_to_budget(small_ddl, "x?", max_chars=100000) == small_ddl
    # 超大 DDL 按相关性截断且保留 car_makers
    big_ddl = "\n\n".join(
        f'CREATE TABLE "car_makers_{i}" (\n  "Id" NUMBER\n);' for i in range(200))
    out = truncate_ddl_to_budget(big_ddl, "How many car makers are there?",
                                 max_chars=2000)
    assert "[BUDGET]" in out and "car_makers" in out
    print("[PASS] token 预算守卫（截断且保留相关表）")

    print(f"\ncache 统计: {cache.stats()}")
    print("Cache 自检通过 ✅")


if __name__ == "__main__":
    _self_test()
