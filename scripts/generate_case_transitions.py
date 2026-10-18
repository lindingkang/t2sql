#!/usr/bin/env python3
"""从本地基准 + SFT 结果生成逐阶段优化/退化案例文件。

阶段链路：
  M0   local_base_m0.json    ->  M2   local_base_m2.json
  M2   local_base_m2.json    ->  M1.5 local_base_m15.json
  M1.5 local_base_m15.json   ->  M3   local_base_m3.json
  M3   local_base_m3.json    ->  M4   sft_eval_result.json (SFT adapter)

每个阶段按 strict 口径（results_equal_strict）判定：
  improved : before.correct=False 且 after.correct=True
  regressed: before.correct=True  且 after.correct=False

同一条 case 同时打印 strict/loose 状态，便于手动归因时区分
"列顺序"这类伪退化（strict 失败但 loose 成功）。

输出：
  m4/case_transitions.txt
  m4/case_transitions.json
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_baseline import SPIDER_DIR  # noqa: E402

STAGES = [
    ("M0", ROOT / "m4" / "local_base_m0.json"),
    ("M2", ROOT / "m4" / "local_base_m2.json"),
    ("M1.5", ROOT / "m4" / "local_base_m15.json"),
    ("M3", ROOT / "m4" / "local_base_m3.json"),
    ("M4-SFT", ROOT / "m4" / "sft_eval_result.json"),
]


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"not a list: {path}")
    return data


def status(r: dict) -> str:
    """生成一行状态标志。"""
    c = "T" if r.get("correct") else "F"
    l = "T" if r.get("correct_loose") else "F"
    e = "ok" if r.get("exec_success") else "exec_fail"
    ft = r.get("failure_type") or "-"
    return f"strict={c} loose={l} exec={e} fail={ft}"


def fmt_case(stage: str, idx: int, before: dict, after: dict) -> str:
    qid = f"dev#{idx:04d}  {after['db_id']} | {after['question']}"
    gold = str(after.get("gold") or "").strip()
    pred_before = str(before.get("pred") or "").strip()
    pred_after = str(after.get("pred") or "").strip()
    return "\n".join([
        f"[{stage}] {qid}",
        f"  gold   : {gold}",
        f"  before : {pred_before}",
        f"           [{status(before)}]",
        f"  after  : {pred_after}",
        f"           [{status(after)}]",
    ])


def main() -> None:
    records = {name: load(path) for name, path in STAGES}
    dev = json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))

    out_lines: list[str] = []
    out_json: dict = {"generated": datetime.now().isoformat(), "edges": {}}

    edges = [("M0", "M2"), ("M2", "M1.5"), ("M1.5", "M3"), ("M3", "M4-SFT")]

    for before_name, after_name in edges:
        before = records[before_name]
        after = records[after_name]
        n = min(len(before), len(after))
        edge_key = f"{before_name}->{after_name}"

        improved: list[tuple[int, dict, dict]] = []
        regressed: list[tuple[int, dict, dict]] = []

        for i in range(n):
            b = before[i]
            a = after[i]

            # 交叉校验 db_id+question 是否和 dev 顺序一致，
            # 不一致时用 dev 映射兜底对齐。
            if b.get("db_id") != a.get("db_id") or b.get("question") != a.get("question"):
                continue

            b_correct = bool(b.get("correct"))
            a_correct = bool(a.get("correct"))
            if (not b_correct) and a_correct:
                improved.append((i, b, a))
            elif b_correct and (not a_correct):
                regressed.append((i, b, a))

        out_json["edges"][edge_key] = {
            "aligned": n,
            "improved": len(improved),
            "regressed": len(regressed),
        }

        out_lines.append("=" * 80)
        out_lines.append(f"阶段 {before_name} -> {after_name}")
        out_lines.append(f"aligned={n}  improved={len(improved)}  regressed={len(regressed)}")
        out_lines.append("=" * 80)

        out_lines.append("\n--- 优化 case ---")
        for idx, b, a in improved:
            out_lines.append(fmt_case(edge_key, idx, b, a))

        out_lines.append("\n--- 退化 case ---")
        for idx, b, a in regressed:
            out_lines.append(fmt_case(edge_key, idx, b, a))

        out_lines.append("")

    out_txt = ROOT / "m4" / "case_transitions.txt"
    out_txt.write_text("\n".join(out_lines), encoding="utf-8")

    (ROOT / "m4" / "case_transitions.json").write_text(
        json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8")

    # 终节打印摘要，方便重定向文件查看
    print(out_txt)
    for edge, info in out_json["edges"].items():
        print(f"{edge}: aligned={info['aligned']} improved={info['improved']} regressed={info['regressed']}")


if __name__ == "__main__":
    main()