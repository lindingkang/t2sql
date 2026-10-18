"""metrics.py — M3 模块4：指标看板。

从评测输出的 trace JSON 聚合"一张表讲清系统表现"。指标口径：

    EA 执行准确率        correct 数 / 可评测数（严格口径，与 eval_agent 一致）
    ER 可执行率          exec_success 数 / 总题数（SQL 跑得通的比例）
    平均重试轮数         每题非首轮 attempt 数均值（M1 反馈闭环的工作量）
    平均采样数           投票题实际采样数均值（自适应早停的节省证据）
    投票共识率           有共识题 / 投票题 —— M1.5 机制有效性的直接证据
    早停率               自适应模式下未采满的题占比
    空结果复查           触发 / 采纳 / 采纳后仍正确（模块2 的净收益）
    安全拦截/超时/超限   六类降级触发次数（从 attempts/votes 的 error_type 聚合）
    成本                 API 调用数、token 总量、缓存命中率，按单价折算人民币

成本折算：千帆等平台按实际定价计费，单价通过 --price-in/--price-out 传入
（单位：元/百万 token），不写死单价避免过时；未传则只报 token 不报钱。

用法：
    python metrics.py agent_eval_full.json
    python metrics.py runs/agent_vote5.json --price-in 0.8 --price-out 2
    python metrics.py runs/agent_vote5.json --json > metrics.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _meter(rec: dict) -> dict:
    return rec.get("meter") or {}


def _iter_error_types(rec: dict):
    """从一条 trace 的 attempts 和投票 samples 里产出所有 error_type。"""
    for a in rec.get("attempts") or []:
        if a.get("error_type"):
            yield a["error_type"]
    votes = rec.get("votes")
    if votes:
        # votes.clusters 只存了成功样本，拦截/超时计数走专用字段
        if votes.get("n_blocked"):
            for _ in range(votes["n_blocked"]):
                yield "safety_blocked"
        if votes.get("n_timeout"):
            for _ in range(votes["n_timeout"]):
                yield "timeout_error"


def compute_metrics(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}

    n_api_fail = sum(1 for r in records if r.get("api_error"))
    n_exec_ok = sum(1 for r in records if r.get("exec_success"))
    n_correct = sum(1 for r in records if r.get("correct"))
    # 可评测 = gold/pred 都能执行的题（correct 字段有意义的题）
    n_evaluable = sum(1 for r in records if r.get("evaluable", r.get("exec_success")))

    # ---- 重试工作量 ----
    # attempts 含 vote_fallback 的 turn=-1，不计入重试；首轮 turn=0 也不计
    retry_rounds = 0
    for r in records:
        for a in r.get("attempts") or []:
            if (a.get("turn") or 0) > 0:
                retry_rounds += 1

    # ---- 投票统计 ----
    vote_qs = [r for r in records if r.get("votes")]
    n_vote = len(vote_qs)
    n_consensus = sum(1 for r in vote_qs
                      if r["votes"].get("consensus_sql") is not None)
    sample_counts = [r["votes"].get("actual_n", 0) for r in vote_qs]
    n_early_stop = sum(1 for r in vote_qs if r["votes"].get("early_stopped"))
    # 无共识题里最终被重试闭环救对的比例（fallback 路径价值）
    no_consensus = [r for r in vote_qs
                    if r["votes"].get("consensus_sql") is None]
    fallback_rescued = sum(1 for r in no_consensus if r.get("correct"))

    # ---- 空结果二次确认 ----
    confirms = [r.get("empty_confirm") for r in records if r.get("empty_confirm")]
    n_empty_trigger = len(confirms)
    n_empty_adopt = sum(1 for c in confirms if c.get("adopted"))
    n_empty_adopt_correct = sum(1 for r in records
                                if (r.get("empty_confirm") or {}).get("adopted")
                                and r.get("correct"))

    # ---- 六类降级触发分布 ----
    error_dist = Counter(_iter_error_types(records))

    # ---- 成本 ----
    api_calls = sum(_meter(r).get("api_calls", 0) for r in records)
    cache_hits = sum(_meter(r).get("cache_hits", 0) for r in records)
    prompt_tok = sum(_meter(r).get("prompt_tokens", 0) for r in records)
    completion_tok = sum(_meter(r).get("completion_tokens", 0) for r in records)
    total_tok = sum(_meter(r).get("total_tokens", 0) for r in records)
    elapsed = [r.get("elapsed_ms") for r in records if r.get("elapsed_ms")]

    return {
        "n": n,
        "api_failures": n_api_fail,
        "exec_success": n_exec_ok,
        "correct": n_correct,
        "evaluable": n_evaluable,
        "EA": n_correct / n_evaluable if n_evaluable else None,
        "ER": n_exec_ok / n,
        "retry_rounds_total": retry_rounds,
        "retry_rounds_per_q": retry_rounds / n,
        "vote": {
            "questions": n_vote,
            "consensus": n_consensus,
            "consensus_rate": n_consensus / n_vote if n_vote else None,
            "early_stopped": n_early_stop,
            "early_stop_rate": n_early_stop / n_vote if n_vote else None,
            "avg_samples": (sum(sample_counts) / n_vote) if n_vote else None,
            "no_consensus": len(no_consensus),
            "fallback_rescued": fallback_rescued,
        },
        "empty_confirm": {
            "triggered": n_empty_trigger,
            "adopted": n_empty_adopt,
            "adopt_rate": n_empty_adopt / n_empty_trigger if n_empty_trigger else None,
            "adopted_and_correct": n_empty_adopt_correct,
        },
        "error_type_distribution": dict(error_dist),
        "cost": {
            "api_calls": api_calls,
            "cache_hits": cache_hits,
            "cache_hit_rate": (cache_hits / (api_calls + cache_hits)
                               if (api_calls + cache_hits) else None),
            "prompt_tokens": prompt_tok,
            "completion_tokens": completion_tok,
            "total_tokens": total_tok,
            "avg_tokens_per_q": total_tok / n,
        },
        "latency_ms": {
            "avg": sum(elapsed) / len(elapsed) if elapsed else None,
            "p95": sorted(elapsed)[int(len(elapsed) * 0.95) - 1] if elapsed else None,
        },
    }


def cost_in_rmb(cost: dict, price_in: float, price_out: float) -> float:
    """按元/百万 token 单价折算总成本。"""
    return (cost["prompt_tokens"] / 1e6 * price_in
            + cost["completion_tokens"] / 1e6 * price_out)


def format_board(m: dict, price_in: float | None = None,
                 price_out: float | None = None) -> str:
    if m.get("n", 0) == 0:
        return "空 trace，无指标。"
    L: list[str] = []
    bar = "=" * 60
    L.append(bar)
    L.append("M3 指标看板")
    L.append(bar)
    L.append(f"总题数:               {m['n']}")
    L.append(f"API 失败:             {m['api_failures']}")
    L.append(f"可执行率 ER:          {m['exec_success']}/{m['n']} "
             f"= {m['ER']:.1%}")
    ea = m["EA"]
    L.append(f"执行准确率 EA:        {m['correct']}/{m['evaluable']} "
             f"= {ea:.1%}" if ea is not None else "执行准确率 EA:        N/A")

    L.append("-" * 60)
    L.append(f"重试总轮数:           {m['retry_rounds_total']}"
             f"（平均 {m['retry_rounds_per_q']:.2f} 轮/题）")

    v = m["vote"]
    if v["questions"]:
        L.append("-" * 60)
        L.append(f"投票题数:             {v['questions']}")
        L.append(f"投票共识率:           {v['consensus']}/{v['questions']}"
                 f" = {v['consensus_rate']:.1%}   <- M1.5 有效性证据")
        L.append(f"无共识 fallback:      {v['no_consensus']} 题，"
                 f"其中重试闭环救对 {v['fallback_rescued']}")
        L.append(f"平均实际采样数:       {v['avg_samples']:.2f}"
                 f"（计划 5 时的节省）")
        L.append(f"自适应早停率:         {v['early_stopped']}/{v['questions']}"
                 f" = {v['early_stop_rate']:.1%}")

    e = m["empty_confirm"]
    if e["triggered"]:
        L.append("-" * 60)
        L.append(f"空结果复查触发:       {e['triggered']}")
        L.append(f"采纳修正:             {e['adopted']}"
                 f"（采纳率 {e['adopt_rate']:.0%}），"
                 f"采纳后判对 {e['adopted_and_correct']}")

    L.append("-" * 60)
    L.append("降级触发分布:")
    for k, v_ in sorted(m["error_type_distribution"].items(),
                        key=lambda kv: -kv[1]):
        L.append(f"  {k:20s} {v_}")

    c = m["cost"]
    L.append("-" * 60)
    L.append(f"API 调用数:           {c['api_calls']}（缓存命中 {c['cache_hits']}，"
             f"命中率 "
             f"{c['cache_hit_rate']:.1%}" if c["cache_hit_rate"] is not None
             else f"API 调用数:           {c['api_calls']}")
    L.append(f"Token 总量:           {c['total_tokens']:,}"
             f"（prompt {c['prompt_tokens']:,} / completion "
             f"{c['completion_tokens']:,}）")
    L.append(f"平均 token/题:        {c['avg_tokens_per_q']:,.0f}")
    if price_in is not None and price_out is not None:
        rmb = cost_in_rmb(c, price_in, price_out)
        L.append(f"折算成本:             ¥{rmb:.2f}"
                 f"（单价 入 {price_in}/出 {price_out} 元每百万 token）")
    else:
        L.append("折算成本:             用 --price-in/--price-out 传入单价计算")

    lat = m["latency_ms"]
    if lat["avg"] is not None:
        L.append("-" * 60)
        L.append(f"单题耗时:             平均 {lat['avg']/1000:.1f}s，"
                 f"p95 {lat['p95']/1000:.1f}s")

    L.append(bar)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", help="评测输出的 trace JSON 文件")
    ap.add_argument("--price-in", type=float, default=None,
                    help="输入 token 单价（元/百万 token）")
    ap.add_argument("--price-out", type=float, default=None,
                    help="输出 token 单价（元/百万 token）")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="输出机器可读 JSON")
    args = ap.parse_args()

    records = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    m = compute_metrics(records)
    if args.as_json:
        print(json.dumps(m, indent=2, ensure_ascii=False))
    else:
        print(format_board(m, args.price_in, args.price_out))


if __name__ == "__main__":
    main()
