#!/usr/bin/env python3
"""汇总 M3 -> M4-SFT 三类 transition 样本。

类别定义（按 strict correct 字段）：
  Regressed   : M3 correct=True  -> M4 correct=False
  Still-wrong : M3 correct=False -> M4 correct=False
  Rescued     : M3 correct=False -> M4 correct=True

输出：
  m4/case_transitions_m3_m4.txt        完整样本（含 m3/m4 pred 与状态）
  m4/case_transitions_m3_m4_summary.txt 统计摘要（各类数目、failure_type 分布）
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"not a list: {path}")
    return data


def status(r: dict) -> str:
    c = "T" if r.get("correct") else "F"
    l = "T" if r.get("correct_loose") else "F"
    e = "ok" if r.get("exec_success") else "exec_fail"
    ft = r.get("failure_type") or "-"
    return f"strict={c} loose={l} exec={e} fail={ft}"


def fmt_case(category: str, idx: int, before: dict, after: dict) -> str:
    qid = f"[{category}] dev#{idx:04d}  {after['db_id']} | {after['question']}"
    return "\n".join([
        qid,
        f"  gold : {str(after.get('gold') or '').strip()}",
        f"  m3   : {str(before.get('pred') or '').strip()}",
        f"         [{status(before)}]",
        f"  m4   : {str(after.get('pred') or '').strip()}",
        f"         [{status(after)}]",
    ])


def main() -> None:
    m3 = load(ROOT / "m4" / "local_base_m3.json")
    m4 = load(ROOT / "m4" / "sft_eval_result.json")
    n = min(len(m3), len(m4))

    buckets = {
        "Regressed": [],
        "Still-wrong": [],
        "Rescued": [],
    }
    unmatched = 0

    for i in range(n):
        b = m3[i]
        a = m4[i]
        # 顺序对齐校验，与 generate_case_transitions.py 一致
        if b.get("db_id") != a.get("db_id") or b.get("question") != a.get("question"):
            unmatched += 1
            continue

        b_correct = bool(b.get("correct"))
        a_correct = bool(a.get("correct"))
        if b_correct and not a_correct:
            category = "Regressed"
        elif not b_correct and a_correct:
            category = "Rescued"
        elif not b_correct and not a_correct:
            category = "Still-wrong"
        else:
            continue
        buckets[category].append((i, b, a))

    order = ["Regressed", "Still-wrong", "Rescued"]

    out_lines: list[str] = []
    summary_lines: list[str] = [f"aligned={n}  unmatched={unmatched}"]

    for category in order:
        items = buckets[category]
        out_lines.append("=" * 80)
        out_lines.append(f"{category}（共 {len(items)} 条）")
        out_lines.append("=" * 80)

        m3_ft = Counter(
            (b.get("failure_type") or "-") for _, b, _ in items
            if not b.get("correct"))
        m4_ft = Counter(
            (a.get("failure_type") or "-") for _, _, a in items
            if not a.get("correct"))

        summary_lines.append(f"\n{category}: {len(items)}")
        summary_lines.append(f"  m3 failure_type: {dict(m3_ft.most_common())}")
        summary_lines.append(f"  m4 failure_type: {dict(m4_ft.most_common())}")

        for idx, b, a in items:
            out_lines.append(fmt_case(category, idx, b, a))
        out_lines.append("")

    (ROOT / "m4" / "case_transitions_m3_m4.txt").write_text(
        "\n".join(out_lines), encoding="utf-8")
    (ROOT / "m4" / "case_transitions_m3_m4_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8")

    for category in order:
        print(f"{category}: {len(buckets[category])}")


if __name__ == "__main__":
    main()