"""logger.py — M3：结构化日志追踪。

为什么需要它：
    Agent 的每一步（生成、执行、重试、投票）如果只 print 到终端，
    事后无法回放分析。M4 微调需要构造训练数据（成功/失败样本对），
    必须有完整的 trace 记录。结构化日志（JSONL）是最可靠的方式。

设计：
    1. JSONL 格式，每行一个事件，方便 grep/awk/jq 处理
    2. 事件类型：input / generate / execute / retry / vote / final
    3. 每个事件带 timestamp、question_id、step、payload
    4. 文件按日期分文件，避免单文件过大
    5. 可选地同时输出到终端（带颜色）

接口：
    AgentLogger(log_dir="logs", verbose=True)
    logger.log(event_type, **payload)
    logger.log_input(question_id, db_id, question)
    logger.log_generate(question_id, turn, prompt, raw_output, sql)
    logger.log_execute(question_id, sql, ok, error, n_rows)
    logger.log_final(question_id, final_sql, ok, correct)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class AgentLogger:
    """结构化 JSONL 日志，同时支持终端彩色输出。"""

    # 事件类型常量
    INPUT = "input"
    GENERATE = "generate"
    EXECUTE = "execute"
    RETRY = "retry"
    VOTE = "vote"
    FINAL = "final"
    ERROR = "error"
    # M3 新增：安全拦截 / 空结果复查 / 成本快照
    SAFETY = "safety"
    EMPTY_CONFIRM = "empty_confirm"

    # 终端颜色
    _COLORS = {
        INPUT: "\033[36m",
        GENERATE: "\033[34m",
        EXECUTE: "\033[33m",
        RETRY: "\033[35m",
        VOTE: "\033[36m",
        FINAL: "\033[32m",
        ERROR: "\033[31m",
        SAFETY: "\033[31m",
        EMPTY_CONFIRM: "\033[35m",
    }
    _RESET = "\033[0m"

    def __init__(self, log_dir: str = "logs", verbose: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        today = datetime.now().strftime("%Y%m%d")
        self.log_file = self.log_dir / f"agent_{today}.jsonl"
        self._fh = open(self.log_file, "a", encoding="utf-8")

    def log(self, event_type: str, question_id: str = "", **payload: Any) -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "event": event_type,
            "qid": question_id,
            **payload,
        }
        line = json.dumps(record, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()

        if self.verbose:
            color = self._COLORS.get(event_type, "")
            ts = record["ts"][11:23]
            qid = f"[{question_id}]" if question_id else ""
            summary = ", ".join(f"{k}={v}" for k, v in list(payload.items())[:2])
            if len(summary) > 80:
                summary = summary[:77] + "..."
            print(f"{color}{ts} {event_type:8s} {qid} {summary}{self._RESET}",
                  file=sys.stderr, flush=True)

    def log_input(self, question_id: str, db_id: str, question: str) -> None:
        self.log(self.INPUT, question_id, db_id=db_id, question_len=len(question))

    def log_generate(self, question_id: str, turn: int,
                     prompt: str, raw: str, sql: str) -> None:
        self.log(self.GENERATE, question_id, turn=turn,
                 prompt_len=len(prompt), raw_len=len(raw), sql=sql[:120])

    def log_execute(self, question_id: str, sql: str, ok: bool,
                    error: str | None, n_rows: int) -> None:
        self.log(self.EXECUTE, question_id, ok=ok, n_rows=n_rows,
                 error=error, sql=sql[:80])

    def log_retry(self, question_id: str, turn: int, error: str) -> None:
        self.log(self.RETRY, question_id, turn=turn, error=error[:120])

    def log_vote(self, question_id: str, n_samples: int, n_clusters: int,
                 consensus: bool) -> None:
        self.log(self.VOTE, question_id, n_samples=n_samples,
                 n_clusters=n_clusters, consensus=consensus)

    def log_final(self, question_id: str, final_sql: str,
                  ok: bool, correct: bool | None,
                  elapsed_ms: float | None = None,
                  meter: dict | None = None) -> None:
        payload: dict[str, Any] = {"ok": ok, "correct": correct,
                                   "sql": final_sql[:120]}
        if elapsed_ms is not None:
            payload["elapsed_ms"] = round(elapsed_ms, 1)
        if meter:
            # meter 随 final 事件落一条，看板无需跨事件拼接即可算单题成本
            payload["meter"] = meter
        self.log(self.FINAL, question_id, **payload)

    def log_safety(self, question_id: str, sql: str, reason: str) -> None:
        """安全层拦截事件（M3 模块1 的可观测出口）。"""
        self.log(self.SAFETY, question_id, reason=reason, sql=sql[:120])

    def log_empty_confirm(self, question_id: str, adopted: bool,
                          structurally_safe: bool, revised_sql: str) -> None:
        """空结果二次确认事件：是否采纳、结构锁是否通过。"""
        self.log(self.EMPTY_CONFIRM, question_id, adopted=adopted,
                 structurally_safe=structurally_safe, sql=revised_sql[:120])

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_traces(log_file: str) -> list[dict]:
    traces = []
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def group_by_question(traces: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for t in traces:
        qid = t.get("qid", "")
        groups.setdefault(qid, []).append(t)
    return groups


if __name__ == "__main__":
    with AgentLogger(log_dir="/tmp/test_logs", verbose=True) as log:
        log.log_input("q1", "test_db", "How many users?")
        log.log_generate("q1", 0, "prompt", "raw", "SELECT COUNT(*) FROM users")
        log.log_execute("q1", "SELECT COUNT(*) FROM users", True, None, 1)
        log.log_final("q1", "SELECT COUNT(*) FROM users", True, True)

    traces = load_traces(log.log_file.as_posix())
    print(f"\n日志文件: {log.log_file}")
    print(f"事件数: {len(traces)}")
    groups = group_by_question(traces)
    print(f"问题数: {len(groups)}")
    print("Logger 自检通过")
