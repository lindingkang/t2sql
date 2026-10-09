"""agent.py — M1 + M1.5：Text2SQL Agent（执行反馈闭环 + 自洽性投票）。

M1（执行反馈闭环）：
    question ──▶ LLM ──▶ SQL ──▶ 执行校验
                                   ├─ 成功 ──▶ 返回
                                   └─ 报错 ──▶ 把错误喂回 LLM 让它改（最多 max_retries 轮）

M1.5（自洽性投票）：
    question ──▶ 采样 N 条 SQL（temperature>0，促进多样性）
              ──▶ 每条执行，拿结果集
              ──▶ 按结果集聚类（忽略列顺序、类型归一化）
              ──▶ 取最大簇的代表 SQL ──▶ 返回
              ──▶ 若最大簇只有 1 条（无共识），fallback 到 M1 的执行重试

为什么自洽性投票能突破 prompt 工程的天花板：
    M2 已验证：prompt 工程对 EA 无显著提升（0.799→0.805→0.798，在噪声内）。
    根因是 69/207 条失败是 semantic_misunderstanding——模型单次理解错了题。
    自洽性投票利用"正确的 SQL 往往结果一致，错误的 SQL 结果五花八门"的特性，
    通过多采样 + 执行结果投票来抑制单次理解偏差。这不需要 gold，也不依赖 LLM 判断。

关键设计取舍：
1. n_samples=1 时退化为 M1 行为（对照组），n_samples>=2 时启用投票。
2. 投票温度用 0.7：足够产生多样性，又不至于输出完全跑题。
3. 聚类用"宽松结果集签名"（行内元素按类型+值排序，再转 frozenset），
   忽略列顺序和行顺序——和 results_equal_loose 同源逻辑。
4. 平票时取 SQL 文本最短的那条（更可能是正确的简洁解）。
5. 无共识（所有簇大小都是 1）时 fallback 到 M1 的执行重试闭环，不浪费已采样的信息。

用法：
    python agent.py --db concert_singer --question "How many singers do we have?"
    python agent.py --db concert_singer --question "..." --n-samples 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from schema import load_schemas, get_ddl
from run_baseline import (
    SPIDER_DIR,
    build_prompt,
    extract_sql,
    execute_sql,
    call_llm,
)

ROOT = Path(__file__).resolve().parent

# 重试时喂给模型的反馈模板。{error} 是 sqlite 的原始报错，{prev_sql} 是上一版 SQL。
RETRY_SYSTEM_NOTE = (
    "The SQL query you wrote failed in SQLite with this error:\n{error}\n\n"
    "Fix the query against the given schema. Output ONLY the corrected SQL "
    "on ONE single line, with no markdown fences and no explanation."
)

# 自洽性投票的采样温度：0.7 是经验值——足够产生 SQL 多样性，又不至于跑题。
VOTE_TEMPERATURE = 0.7


def _result_set_signature(rows: list[tuple]) -> frozenset:
    """把结果集转成可哈希的"宽松签名"，用于聚类。

    与 results_equal_loose 同源：忽略行顺序、列顺序，只比较数据内容。
    每行元素按 (类型名, 值) 排序，这样 (5, 'dog') 和 ('dog', 5) 签名相同。
    用 frozenset 包裹整个结果集，使顺序不影响签名。
    """
    def row_key(row: tuple) -> tuple:
        return tuple(sorted(
            (type(v).__name__, v if v is None or isinstance(v, (int, float, str))
             else str(v))
            for v in row
        ))
    return frozenset(row_key(r) for r in rows)


class Text2SQLAgent:
    """带执行反馈闭环 + 自洽性投票的 Text2SQL Agent。

    n_samples=1  → 纯 M1 行为（执行报错重试）
    n_samples>=2 → M1.5 行为（多采样投票，无共识时 fallback 到重试）
    """

    def __init__(self, client: OpenAI, model: str, schemas: dict,
                 max_retries: int = 2, sleep: float = 0.5,
                 n_samples: int = 1, vote_temperature: float = VOTE_TEMPERATURE):
        self.client = client
        self.model = model
        self.schemas = schemas
        self.max_retries = max_retries
        self.sleep = sleep
        self.n_samples = n_samples
        self.vote_temperature = vote_temperature

    def answer(self, db_id: str, question: str) -> dict:
        """对单个问题跑完整 Agent 循环。

        返回 trace 字典：
          final_sql   : 最终 SQL
          ok          : 最终 SQL 是否能执行
          attempts    : 每轮尝试（M1 重试的 trace）
          votes       : 投票详情（仅 n_samples>=2 时有值）
          api_error   : 首轮 API 失败的错误
        """
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"
        schema = self.schemas[db_id]

        # ---- 阶段 1：自洽性投票（仅 n_samples >= 2 时启用）----
        if self.n_samples >= 2:
            votes = self._vote(db_path, schema, db_id, question)
            if votes["consensus_sql"] is not None:
                # 有共识：直接返回得票最多的 SQL
                final_sql = votes["consensus_sql"]
                _, err = execute_sql(db_path, final_sql)
                return {
                    "final_sql": final_sql,
                    "ok": err is None,
                    "attempts": [],
                    "votes": votes,
                    "api_error": None,
                }
            # 无共识：fallback 到 M1 执行重试闭环
            # 但可以复用投票中"能执行成功"的 SQL 作为起点，减少重试次数
            fallback_sql = votes.get("first_executable_sql")
        else:
            votes = None
            fallback_sql = None

        # ---- 阶段 2：M1 执行反馈闭环 ----
        return self._retry_loop(db_path, schema, db_id, question,
                                fallback_sql=fallback_sql, votes=votes)

    def _vote(self, db_path, schema, db_id, question) -> dict:
        """自洽性投票：采样 N 条 SQL，按执行结果集聚类，取最大簇。

        返回：
          consensus_sql        : 得票最多的 SQL（无共识时为 None）
          first_executable_sql : 第一条能执行成功的 SQL（fallback 用）
          clusters             : 聚类详情 [{signature, size, sqls}]
        """
        prompt = build_prompt(db_id, question, schema)
        samples: list[dict] = []  # [{sql, exec_ok, error, signature}]

        for _ in range(self.n_samples):
            try:
                raw = call_llm(self.client, self.model, prompt,
                               temperature=self.vote_temperature)
            except Exception:
                time.sleep(self.sleep)
                continue
            time.sleep(self.sleep)

            sql = extract_sql(raw)
            rows, err = execute_sql(db_path, sql)
            samples.append({
                "sql": sql,
                "exec_ok": err is None,
                "error": err,
                "signature": _result_set_signature(rows) if err is None else None,
            })

        # 只对执行成功的样本聚类（执行失败的 SQL 结果集无意义）
        executable = [s for s in samples if s["exec_ok"]]
        first_executable = executable[0]["sql"] if executable else None

        # 按结果集签名聚类
        clusters: dict[frozenset, list[str]] = {}
        for s in executable:
            clusters.setdefault(s["signature"], []).append(s["sql"])

        # 找最大簇
        if not clusters:
            return {"consensus_sql": None, "first_executable_sql": None,
                    "clusters": []}

        max_size = max(len(v) for v in clusters.values())
        # 取所有达到最大簇大小的簇（平票情况）
        top_clusters = [sqls for sqls in clusters.values() if len(sqls) == max_size]

        if max_size == 1 and len(clusters) > 1:
            # 无共识：每个簇都只有 1 条，说明 SQL 们各说各话
            return {"consensus_sql": None,
                    "first_executable_sql": first_executable,
                    "clusters": [{"size": len(v), "sqls": v}
                                 for v in clusters.values()]}

        # 有共识（最大簇 >= 2，或只有一个簇）
        # 平票时选 SQL 文本最短的（简洁解更可能正确）
        best_sql = min(top_clusters[0], key=len)

        return {"consensus_sql": best_sql,
                "first_executable_sql": first_executable,
                "clusters": [{"size": len(v), "sqls": v[:3]}
                             for v in clusters.values()]}

    def _retry_loop(self, db_path, schema, db_id, question,
                    fallback_sql=None, votes=None) -> dict:
        """M1 执行反馈闭环：生成 → 执行 → 报错重试。"""
        messages = [{"role": "user",
                     "content": build_prompt(db_id, question, schema)}]
        attempts: list[dict] = []
        final_sql = fallback_sql

        # 如果有 fallback_sql（来自投票），先试它
        if fallback_sql is not None:
            _, err = execute_sql(db_path, fallback_sql)
            attempts.append({"turn": -1, "sql": fallback_sql,
                             "exec_ok": err is None, "error": err,
                             "source": "vote_fallback"})
            if err is None:
                return {"final_sql": fallback_sql, "ok": True,
                        "attempts": attempts, "votes": votes, "api_error": None}

        # 总轮数 = 1 次首生成 + max_retries 次修正
        for turn in range(self.max_retries + 1):
            try:
                raw = call_llm(self.client, self.model,
                               self._render_messages(messages))
            except Exception as exc:
                if turn == 0 and not attempts:
                    return {"final_sql": None, "ok": False, "attempts": [],
                            "votes": votes, "api_error": str(exc)}
                break
            time.sleep(self.sleep)

            sql = extract_sql(raw)
            final_sql = sql
            _, err = execute_sql(db_path, sql)
            attempt = {"turn": turn, "sql": sql,
                       "exec_ok": err is None, "error": err}
            attempts.append(attempt)

            if err is None:
                break

            if turn < self.max_retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                 "content": RETRY_SYSTEM_NOTE.format(
                                     error=err, prev_sql=sql)})

        return {"final_sql": final_sql,
                "ok": attempts[-1]["exec_ok"] if attempts else False,
                "attempts": attempts,
                "votes": votes,
                "api_error": None}

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

    agent = Text2SQLAgent(client, model, schemas,
                          max_retries=args.max_retries,
                          n_samples=args.n_samples,
                          vote_temperature=args.vote_temperature)
    result = agent.answer(args.db, args.question)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
