"""agent.py — M3：Text2SQL Agent（反馈闭环 + 投票 + 工程化加固 + 成本控制）。

能力演进：
    M1   执行反馈闭环：生成 SQL → 执行 → 报错？→ 针对性提示重写
    M1.5 自洽性投票：采样 N 条 → 按结果集聚类 → 多数决
    M3   工程化加固：
         1. 安全层：所有执行入口（含投票采样）先过 safety.sanitize_sql；
         2. 六类降级：按错误类型差异化重试预算，schema 错误给相似列+可用列；
         3. 执行保护：execute_sql 超时中断 + 行数上限；
         4. 空结果二次确认：值匹配疑似失败时复查，采纳前过结构锁；
         5. 成本控制：L1 确定性响应缓存 + token 计量 + 自适应早停投票。

关键设计取舍：
1. n_samples=1 退化为纯重试闭环；>=2 启用投票。
2. 安全层必须覆盖投票路径——每条采样都要执行，是攻击面最大的入口。
3. 重试预算按错误类型给，而非固定轮数：超时多是 JOIN 理解错，重试价值低。
4. 空结果确认采纳有 AST 结构锁：只许改值匹配方式，不许动表/列/JOIN，
   防止在"正确的空答案"上瞎放宽匹配。
5. L1 缓存只缓存 temperature=0 的确定性调用；投票(temp=0.7)若也缓存，
   5 次采样会命中同一响应，投票退化为单采样且"假性共识率 100%"。
6. 自适应投票采用"多数决早停"：采到某簇严格过半(n=5 需 3 票)即停，
   简单题 3 次调用、难题采满 5 次，信号完全数据驱动。

用法：
    python agent.py --db concert_singer --question "How many singers ...?"
    python agent.py --db car_1 --question "..." --n-samples 5 --adaptive
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import sqlglot
from sqlglot import exp
from openai import OpenAI
from dotenv import load_dotenv

from schema import load_schemas, get_ddl
from run_baseline import (
    SPIDER_DIR,
    build_prompt,
    extract_sql,
    execute_sql,
    call_llm_with_usage,
    DEFAULT_EXEC_TIMEOUT,
    DEFAULT_ROW_LIMIT,
)
from safety import sanitize_sql
from degrade import (
    classify_execution_error,
    build_retry_hint,
    DegradationPolicy,
)

ROOT = Path(__file__).resolve().parent

# 自洽性投票采样温度：0.7 足够产生多样性，又不至于跑题。
VOTE_TEMPERATURE = 0.7
# API 单次输出上限（token），也是 L1 缓存键的一部分（换上限=换响应空间）。
MAX_OUTPUT_TOKENS = 512

# 空结果二次确认提示。核心假设：0 行/聚合 0 常常是"值匹配失败"而非真无数据。
EMPTY_RESULT_NOTE = (
    "The query executed successfully but returned ZERO rows (or an aggregate of 0).\n"
    "This is often caused by a literal value failing to match (case sensitivity "
    "or spelling), rather than the answer being truly empty.\n"
    "Question: {question}\n"
    "Your query: {sql}\n\n"
    "Re-check the literal values in WHERE against the question. If a case "
    "mismatch is plausible, use LOWER(col) = LOWER('value'). If the value is "
    "genuinely absent, keep the query UNCHANGED. Do not alter tables, joins, "
    "selected columns, or GROUP BY.\n"
    "Output ONLY the final SQL on ONE single line, no explanation."
)


def _result_set_signature(rows: list[tuple]) -> frozenset:
    """结果集 -> 可哈希的"宽松签名"，用于投票聚类（忽略行/列顺序）。

    数值归一化处理：int 30 与 float 30.0 视为同一结果；
    字符串 strip + lower-case，能数值化的字符串也按数值比较，
    让 France/france/ FRANCE 这类大小写不一致的结果能收敛为同簇。
    """
    def normalize(v):
        if v is None:
            return ("null", None)
        if isinstance(v, bool):
            return ("num", float(v))
        if isinstance(v, (int, float)):
            return ("num", float(v))
        if isinstance(v, str):
            s = v.strip()
            low = s.lower()
            try:
                return ("num", float(low))
            except ValueError:
                return ("str", low)
        return ("raw", str(v))

    def row_key(row: tuple) -> tuple:
        return tuple(sorted(normalize(v) for v in row))
    return frozenset(row_key(r) for r in rows)


def _has_literal_filter(sql: str) -> bool:
    """SQL 是否带"含字面量的 WHERE"——空结果复查触发条件之一。"""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return bool(re.search(r"\bwhere\b", sql, re.I)) and ("'" in sql or '"' in sql)
    where = tree.args.get("where")
    if where is None:
        return False
    return any(where.find_all(exp.Literal))


def _is_aggregate(sql: str) -> bool:
    """是否聚合查询（含聚合函数或 GROUP BY）。解析失败按正则兜底。"""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return bool(re.search(r"\b(count|sum|avg|min|max)\s*\(", sql, re.I))
    if tree.args.get("group") is not None:
        return True
    agg_funcs = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
    return any(isinstance(n, agg_funcs) for n in tree.walk())


def _is_empty_signal(rows: list[tuple], sql: str) -> bool:
    """结果是否为"疑似无匹配"空信号：非聚合 0 行，或聚合 COUNT/SUM=0、
    AVG/MIN/MAX=NULL。非聚合的普通标量 0 不算。"""
    if len(rows) == 0:
        return True
    if len(rows) == 1 and len(rows[0]) == 1 and _is_aggregate(sql):
        v = rows[0][0]
        return v == 0 or v is None
    return False


def _revision_is_structurally_safe(orig_sql: str, revised_sql: str) -> bool:
    """空结果复查采纳前的结构锁：复查只能改值匹配方式，不能改查询结构。

    三条 AST 约束：表集合相同；去 WHERE/HAVING/LIMIT 且字面量归一化后的
    骨架相同；WHERE/HAVING 引用列集合相同。任一解析失败即拒绝采纳。
    """
    try:
        a = sqlglot.parse_one(orig_sql, read="sqlite")
        b = sqlglot.parse_one(revised_sql, read="sqlite")
    except Exception:
        return False

    tables_a = {t.name.lower() for t in a.find_all(exp.Table)}
    tables_b = {t.name.lower() for t in b.find_all(exp.Table)}
    if tables_a != tables_b:
        return False

    def skeleton(tree) -> str:
        c = tree.copy()
        for node in list(c.walk()):
            if isinstance(node, (exp.Where, exp.Having, exp.Limit)):
                node.pop()
        for lit in c.find_all(exp.Literal):
            lit.replace(exp.Placeholder())
        return c.sql(dialect="sqlite").lower()

    if skeleton(a) != skeleton(b):
        return False

    def filter_columns(tree) -> frozenset:
        cols = set()
        for key in ("where", "having"):
            node = tree.args.get(key)
            if node is not None:
                cols.update(c.name.lower() for c in node.find_all(exp.Column))
        return frozenset(cols)

    return filter_columns(a) == filter_columns(b)


class Text2SQLAgent:
    """带安全层、六类降级、执行保护、空结果确认、成本控制的 Text2SQL Agent。"""

    def __init__(self, client: OpenAI, model: str, schemas: dict,
                 max_retries: int = 2, sleep: float = 0.5,
                 n_samples: int = 1, vote_temperature: float = VOTE_TEMPERATURE,
                 confirm_empty: bool = True, adaptive: bool = False,
                 exec_timeout: float = DEFAULT_EXEC_TIMEOUT,
                 row_limit: int = DEFAULT_ROW_LIMIT,
                 llm_cache=None, logger=None):
        self.client = client
        self.model = model
        self.schemas = schemas
        self.max_retries = max_retries
        self.sleep = sleep
        self.n_samples = n_samples
        self.vote_temperature = vote_temperature
        self.confirm_empty = confirm_empty
        self.adaptive = adaptive
        self.exec_timeout = exec_timeout
        self.row_limit = row_limit
        # L1 缓存（None=不启用）。只缓存 temperature=0 的确定性调用。
        self.llm_cache = llm_cache
        # M3 结构化日志（None=不打日志，零侵入零开销）
        self.logger = logger
        self.qid = ""
        self.policy = DegradationPolicy(max_retries=max_retries)
        # 每题重置的成本计量
        self._reset_meter()

    def _emit(self, handler_name: str, *args, **kwargs) -> None:
        """日志发射辅助：logger 为 None 时静默跳过，埋点代码不污染主逻辑。"""
        if self.logger is not None:
            getattr(self.logger, handler_name)(*args, **kwargs)

    def _reset_meter(self) -> None:
        self.meter = {"api_calls": 0, "cache_hits": 0,
                      "prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0}

    # ------------------------------------------------------------------
    # 统一 LLM 调用入口：L1 缓存 → API → 计量。
    # 所有 prompt→response 都走这里，保证缓存与 token 统计无遗漏。
    # ------------------------------------------------------------------
    def _call_llm(self, prompt: str, temperature: float) -> str:
        # 1) 查缓存（高温投票采样 get 直接返回 None）
        if self.llm_cache is not None:
            cached = self.llm_cache.get(self.model, prompt, temperature,
                                        MAX_OUTPUT_TOKENS)
            if cached is not None:
                content, usage = cached
                self.meter["cache_hits"] += 1
                # 命中缓存不花钱，但 token 数仍计入"处理量"用于观察 prompt 规模
                self.meter["prompt_tokens"] += usage.get("prompt_tokens", 0)
                return content

        # 2) 调 API
        content, usage = call_llm_with_usage(
            self.client, self.model, prompt,
            temperature=temperature)
        self.meter["api_calls"] += 1
        self.meter["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.meter["completion_tokens"] += usage.get("completion_tokens", 0)
        self.meter["total_tokens"] += usage.get("total_tokens", 0)

        # 3) 写缓存（内部对高温调用自动跳过）
        if self.llm_cache is not None:
            self.llm_cache.put(self.model, prompt, temperature,
                               MAX_OUTPUT_TOKENS, content, usage)
        return content

    # ------------------------------------------------------------------
    # 统一执行入口：安全层 → 执行加固
    # ------------------------------------------------------------------
    def _safe_execute(self, db_path: Path, sql: str) -> tuple[list[tuple], str | None]:
        clean, safe, reason = sanitize_sql(sql)
        if not safe:
            self._emit("log_safety", self.qid, sql, reason)
            return [], f"SAFETY_BLOCKED: {reason}"
        return execute_sql(db_path, clean, timeout=self.exec_timeout,
                           row_limit=self.row_limit)

    def answer(self, db_id: str, question: str,
               question_id: str = "") -> dict:
        """对单个问题跑完整 Agent 循环，返回完整 trace（含成本计量）。

        question_id 仅用于日志关联（如评测下标），不影响生成逻辑。
        """
        self._reset_meter()
        self.qid = question_id or f"{db_id}:{question[:20]}"
        t_start = time.monotonic()
        self._emit("log_input", self.qid, db_id, question)
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"
        schema = self.schemas[db_id]

        # ---- 阶段 1：自洽性投票 ----
        if self.n_samples >= 2:
            votes = self._vote(db_path, schema, db_id, question)
            self._emit("log_vote", self.qid,
                       n_samples=votes.get("actual_n", 0),
                       n_clusters=len(votes.get("clusters", [])),
                       consensus=votes["consensus_sql"] is not None)
            if votes["consensus_sql"] is not None:
                final_sql = votes["consensus_sql"]
                final_rows = votes["consensus_rows"]
                final_sql, final_rows, confirm = self._maybe_confirm_empty(
                    db_path, schema, db_id, question, final_sql, final_rows)
                elapsed_ms = (time.monotonic() - t_start) * 1000
                self._emit("log_final", self.qid, final_sql, True, None,
                           elapsed_ms=elapsed_ms, meter=self.meter)
                return {
                    "final_sql": final_sql, "ok": True,
                    "final_rows": final_rows,
                    "attempts": [], "votes": votes,
                    "empty_confirm": confirm, "api_error": None,
                    "meter": self.meter, "elapsed_ms": round(elapsed_ms, 1),
                }
            fallback_sql = votes.get("first_executable_sql")
        else:
            votes = None
            fallback_sql = None

        # ---- 阶段 2：执行反馈闭环 ----
        result = self._retry_loop(db_path, schema, db_id, question,
                                  fallback_sql=fallback_sql, votes=votes)

        # ---- 阶段 3：空结果二次确认 ----
        if result["ok"]:
            new_sql, new_rows, confirm = self._maybe_confirm_empty(
                db_path, schema, db_id, question,
                result["final_sql"], result.get("final_rows", []))
            result["final_sql"] = new_sql
            result["final_rows"] = new_rows
            result["empty_confirm"] = confirm
        else:
            result["empty_confirm"] = None
        result["meter"] = self.meter
        result["elapsed_ms"] = round((time.monotonic() - t_start) * 1000, 1)
        self._emit("log_final", self.qid, result["final_sql"] or "",
                   result["ok"], None,
                   elapsed_ms=result["elapsed_ms"], meter=self.meter)
        return result

    # ------------------------------------------------------------------
    # 阶段 1：投票（自适应早停）
    # ------------------------------------------------------------------
    def _vote(self, db_path, schema, db_id, question) -> dict:
        """采样 SQL → 安全执行 → 结果集聚类。

        自适应模式（--adaptive）采用多数决早停：每采一条就重新聚类，
        一旦某簇 size 严格过半（n=5 需 3 票）立即停止；否则采满。
        非自适应模式总是采满 n_samples（用于公平对比消融）。
        """
        prompt = build_prompt(db_id, question, schema)
        samples: list[dict] = []
        n_blocked = 0
        n_timeout = 0
        majority = self.n_samples // 2 + 1  # 严格过半阈值

        def cluster_now() -> dict[frozenset, list[dict]]:
            cl: dict[frozenset, list[dict]] = {}
            for s in samples:
                if s["exec_ok"] and s["signature"]:
                    cl.setdefault(s["signature"], []).append(s)
            return cl

        for i in range(self.n_samples):
            try:
                raw = self._call_llm(prompt, self.vote_temperature)
            except Exception:
                time.sleep(self.sleep)
                continue
            time.sleep(self.sleep)

            sql = extract_sql(raw)
            # 投票样本同样必须过安全层
            rows, err = self._safe_execute(db_path, sql)
            error_type = classify_execution_error(err) if err else None
            if error_type == "safety_blocked":
                n_blocked += 1
            if error_type == "timeout_error":
                n_timeout += 1
            samples.append({
                "sql": sql, "exec_ok": err is None, "error": err,
                "error_type": error_type,
                "signature": _result_set_signature(rows)
                             if (err is None and rows) else None,
                "rows": rows if err is None else [],
            })

            # 早停判定：某结果簇严格过半（至少 2 票，防单条自证）
            if self.adaptive:
                cl = cluster_now()
                top_size = max((len(v) for v in cl.values()), default=0)
                if top_size >= majority and top_size >= 2:
                    break

        actual_n = len(samples)
        executable = [s for s in samples if s["exec_ok"]]
        first_executable = executable[0]["sql"] if executable else None
        clusters = cluster_now()

        def cluster_view(cl: dict) -> list[dict]:
            return [{"size": len(v), "sqls": [s["sql"] for s in v[:3]]}
                    for v in cl.values()]

        if not clusters:
            return {"consensus_sql": None, "consensus_rows": None,
                    "first_executable_sql": None, "clusters": [],
                    "n_blocked": n_blocked, "n_timeout": n_timeout,
                    "planned_n": self.n_samples, "actual_n": actual_n,
                    "early_stopped": False}

        max_size = max(len(v) for v in clusters.values())
        top_clusters = [v for v in clusters.values() if len(v) == max_size]

        # 非自适应模式：采满后若最大簇只有 1 且存在多簇 = 无共识，交重试闭环。
        # 自适应模式能走到这里说明没触发过半早停，同样按"是否有多簇分歧"判定。
        if max_size == 1 and len(clusters) > 1:
            return {"consensus_sql": None, "consensus_rows": None,
                    "first_executable_sql": first_executable,
                    "clusters": cluster_view(clusters),
                    "n_blocked": n_blocked, "n_timeout": n_timeout,
                    "planned_n": self.n_samples, "actual_n": actual_n,
                    "early_stopped": False}

        all_top = [s for cluster in top_clusters for s in cluster]
        best_sample = min(all_top, key=lambda s: len(s["sql"]))
        return {"consensus_sql": best_sample["sql"],
                "consensus_rows": best_sample["rows"],
                "first_executable_sql": first_executable,
                "clusters": cluster_view(clusters),
                "n_blocked": n_blocked, "n_timeout": n_timeout,
                "planned_n": self.n_samples, "actual_n": actual_n,
                "early_stopped": actual_n < self.n_samples}

    # ------------------------------------------------------------------
    # 阶段 2：执行反馈闭环（差异化重试）
    # ------------------------------------------------------------------
    def _retry_loop(self, db_path, schema, db_id, question,
                    fallback_sql=None, votes=None) -> dict:
        messages = [{"role": "user",
                     "content": build_prompt(db_id, question, schema)}]
        attempts: list[dict] = []
        final_sql = fallback_sql
        final_rows: list[tuple] = []

        # 投票 fallback 的 SQL 再走一次统一执行入口（幂等，defense in depth）
        if fallback_sql is not None:
            rows, err = self._safe_execute(db_path, fallback_sql)
            attempts.append({"turn": -1, "sql": fallback_sql,
                             "exec_ok": err is None, "error": err,
                             "error_type": classify_execution_error(err) if err else None,
                             "source": "vote_fallback"})
            if err is None:
                return {"final_sql": fallback_sql, "ok": True,
                        "final_rows": rows, "attempts": attempts,
                        "votes": votes, "api_error": None}

        # 差异化重试：是否继续由 policy 按每轮错误类型动态决定
        turn = 0
        while True:
            try:
                raw = self._call_llm(self._render_messages(messages), 0.0)
            except Exception as exc:
                if turn == 0 and not attempts:
                    return {"final_sql": None, "ok": False, "final_rows": [],
                            "attempts": [], "votes": votes,
                            "api_error": str(exc)}
                break
            time.sleep(self.sleep)

            sql = extract_sql(raw)
            final_sql = sql

            rows, err = self._safe_execute(db_path, sql)
            error_type = classify_execution_error(err)
            attempts.append({"turn": turn, "sql": sql,
                             "exec_ok": err is None, "error": err,
                             "error_type": error_type})

            if err is None:
                final_rows = rows
                break

            if not self.policy.should_retry(error_type, turn):
                break

            messages.append({"role": "assistant", "content": raw})
            # schema 感知提示：幻觉列给相似列 + 该表可用列清单
            schema_ddl = get_ddl(schema) if db_id in self.schemas else ""
            hint = build_retry_hint(error_type, err, schema_ddl,
                                    schema=schema, db_id=db_id, sql=sql)
            messages.append({"role": "user", "content": hint})
            turn += 1

        return {"final_sql": final_sql,
                "ok": attempts[-1]["exec_ok"] if attempts else False,
                "final_rows": final_rows,
                "attempts": attempts, "votes": votes, "api_error": None}

    # ------------------------------------------------------------------
    # 阶段 3：空结果二次确认
    # ------------------------------------------------------------------
    def _maybe_confirm_empty(self, db_path, schema, db_id, question,
                             sql: str, rows: list[tuple]
                             ) -> tuple[str, list[tuple], dict | None]:
        """执行成功但疑似值匹配失败时让模型复查；采纳须过结构锁。"""
        if not self.confirm_empty or not sql:
            return sql, rows, None
        if not _is_empty_signal(rows, sql) or not _has_literal_filter(sql):
            return sql, rows, None

        note = EMPTY_RESULT_NOTE.format(question=question, sql=sql)
        try:
            raw = self._call_llm(note, 0.0)
        except Exception:
            return sql, rows, None
        time.sleep(self.sleep)

        new_sql = extract_sql(raw)
        new_rows, err = self._safe_execute(db_path, new_sql)
        structurally_safe = _revision_is_structurally_safe(sql, new_sql)
        adopted = (err is None and len(new_rows) > 0 and structurally_safe)
        trace = {"triggered": True, "original_sql": sql,
                 "revised_sql": new_sql, "revised_ok": err is None,
                 "revised_error": err, "revised_nonempty": len(new_rows) > 0,
                 "structurally_safe": structurally_safe,
                 "adopted": adopted}
        self._emit("log_empty_confirm", self.qid, adopted,
                   structurally_safe, new_sql)
        if adopted:
            return new_sql, new_rows, trace
        return sql, rows, trace

    @staticmethod
    def _render_messages(messages: list[dict]) -> str:
        """把多轮历史拼成单次 prompt。"""
        if len(messages) == 1:
            return messages[0]["content"]
        blocks = [messages[0]["content"]]
        for m in messages[1:]:
            tag = "Your previous SQL" if m["role"] == "assistant" else "Feedback"
            blocks.append(f"{tag}:\n{m['content']}")
        return "\n\n".join(blocks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="db_id，如 concert_singer")
    ap.add_argument("--question", required=True, help="自然语言问题")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--n-samples", type=int, default=1,
                    help="自洽性投票采样数；1=纯重试，>=2=启用投票")
    ap.add_argument("--vote-temperature", type=float, default=VOTE_TEMPERATURE)
    ap.add_argument("--adaptive", action="store_true",
                    help="自适应投票：多数决早停（简单题少采样）")
    ap.add_argument("--no-confirm-empty", dest="confirm_empty",
                    action="store_false", help="关闭空结果二次确认")
    ap.add_argument("--exec-timeout", type=float, default=DEFAULT_EXEC_TIMEOUT)
    ap.add_argument("--row-limit", type=int, default=DEFAULT_ROW_LIMIT)
    ap.add_argument("--cache", action="store_true",
                    help="启用 L1 确定性响应磁盘缓存")
    ap.add_argument("--log-dir", default=None,
                    help="结构化 JSONL 日志目录（不指定则不打日志）")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        base_url=os.getenv("LLM_BASE_URL", "https://qianfan.baidubce.com/v2").strip(),
    )
    model = os.getenv("LLM_MODEL", "").strip()
    schemas = load_schemas()

    if args.db not in schemas:
        raise SystemExit(f"未知 db_id: {args.db}")

    llm_cache = None
    if args.cache:
        from cache import LLMCache
        llm_cache = LLMCache()

    # 日志器作为上下文管理器，脚本退出时自动 flush/close
    trace_logger = None
    if args.log_dir:
        from logger import AgentLogger
        trace_logger = AgentLogger(log_dir=args.log_dir, verbose=True)

    try:
        agent = Text2SQLAgent(
            client, model, schemas,
            max_retries=args.max_retries,
            n_samples=args.n_samples,
            vote_temperature=args.vote_temperature,
            confirm_empty=args.confirm_empty,
            adaptive=args.adaptive,
            exec_timeout=args.exec_timeout,
            row_limit=args.row_limit,
            llm_cache=llm_cache, logger=trace_logger)
        result = agent.answer(args.db, args.question)
    finally:
        if trace_logger is not None:
            trace_logger.close()

    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
