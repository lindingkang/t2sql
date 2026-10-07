"""agent.py — M1：Text2SQL Agent 骨架（执行反馈闭环）。

与 M0 一次性问答的区别：
    M0：question ──▶ LLM ──▶ SQL ──▶ 执行（错了就结束）
    M1：question ──▶ LLM ──▶ SQL ──▶ 执行校验
                                   ├─ 成功 ──▶ 返回
                                   └─ 报错 ──▶ 把错误喂回 LLM 让它改（最多 max_retries 轮）

关键设计取舍：
1. 只对"执行失败"（语法错误、no such column 等数据库报错）重试。
   生产环境没有 gold SQL，Agent 无法判断"结果对不对"，能自动拿到的唯一信号
   就是数据库的报错——这正是"可验证奖励"在 Agent 形态下的含义。
2. 反馈用多轮对话承载（消息历史里保留上一版 SQL + 报错），不是重新拼 prompt。
   模型因此能像人调试一样"看到自己写了什么、错在哪"。
3. 第一次生成与 run_baseline.py 完全一致（同 prompt、temperature=0）。
   这样 M1 相对 M0 的提升可以干净地归因于重试机制本身（对照组思维）。
4. 每次尝试都记进 trace：M3 日志追踪的地基，也是事后分析"重试有没有用"的数据。

用法：
    python agent.py --db concert_singer --question "How many singers do we have?"
    python agent.py --db concert_singer --question "..." --max-retries 2
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


class Text2SQLAgent:
    """带执行反馈闭环的 Text2SQL Agent。

    生命周期：一次 answer() 调用 = 一次完整的"生成-校验-修正"循环。
    无状态跨问题：每个 question 独立消息历史，避免上一题的错误污染下一题。
    """

    def __init__(self, client: OpenAI, model: str,
                 schemas: dict, max_retries: int = 2, sleep: float = 0.5):
        self.client = client
        self.model = model
        self.schemas = schemas
        self.max_retries = max_retries  # 首次生成之外的额外修正轮数
        self.sleep = sleep              # 调用间隔，规避平台 RPM 限制

    def answer(self, db_id: str, question: str) -> dict:
        """对单个问题跑完整 Agent 循环。

        返回 trace 字典：
          final_sql   : 最终一版 SQL（可能执行成功，也可能重试耗尽仍失败）
          ok          : 最终 SQL 是否能在数据库成功执行
          attempts    : 每一轮尝试 [{turn, sql, exec_ok, error}]
          api_error   : 首轮 API 就失败时的错误（重试轮的 API 失败记在 attempts）
        """
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"
        schema = self.schemas[db_id]

        # 多轮消息历史：首轮是 M0 同款 zero-shot prompt，之后追加"报错+要求修正"
        messages = [{"role": "user",
                     "content": build_prompt(db_id, question, schema)}]
        attempts: list[dict] = []
        final_sql = None

        # 总轮数 = 1 次首生成 + max_retries 次修正
        for turn in range(self.max_retries + 1):
            try:
                raw = call_llm(self.client, self.model,
                               self._render_messages(messages))
            except Exception as exc:
                # 首轮 API 就挂：没有任何可用 SQL，直接返回
                if turn == 0:
                    return {"final_sql": None, "ok": False, "attempts": [],
                            "api_error": str(exc)}
                # 修正轮 API 挂：沿用上一版 SQL 作为最终结果
                attempts[-1]["api_error"] = str(exc)
                break
            time.sleep(self.sleep)

            sql = extract_sql(raw)
            final_sql = sql
            _, err = execute_sql(db_path, sql)
            attempt = {"turn": turn, "sql": sql,
                       "exec_ok": err is None, "error": err}
            attempts.append(attempt)

            if err is None:
                break  # 执行通过，闭环成功退出

            # 还没到最后一轮：把"模型上一版输出 + 报错反馈"追加进消息历史
            if turn < self.max_retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                 "content": RETRY_SYSTEM_NOTE.format(
                                     error=err, prev_sql=sql)})

        return {"final_sql": final_sql,
                "ok": attempts[-1]["exec_ok"] if attempts else False,
                "attempts": attempts,
                "api_error": None}

    @staticmethod
    def _render_messages(messages: list[dict]) -> str:
        """把多轮历史拼成单次 prompt。

        为什么不用原生多轮接口？当前千帆模型的 chat 接口虽支持多轮，但 M0 的
        基线是单轮 user 消息。这里手工拼接、显式标注轮次，有两个好处：
        1) 首轮文本与 baseline 逐字一致，保证对照实验干净；
        2) 不同模型/平台行为统一，不依赖各家对多轮模板的隐式处理。
        """
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

    agent = Text2SQLAgent(client, model, schemas, max_retries=args.max_retries)
    result = agent.answer(args.db, args.question)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
