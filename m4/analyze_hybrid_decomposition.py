#!/usr/bin/env python3
"""M4.5 hybrid 相对 SFT single / M3 local 的 rescued & regressed 分解。

输出：
  1) hybrid 相对 SFT single 的 rescued（41）按 SFT 单次状态分桶
  2) hybrid 相对 SFT single 的 regressed（15）按 hybrid 状态分桶
  3) hybrid 相对 M3 local 的 rescued（73）/ regressed（34）分桶
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def main():
    hybrid = load(ROOT / "m4" / "sft_hybrid_result.json")
    sft = load(ROOT / "m4" / "sft_eval_result.json")
    m3 = load(ROOT / "m4" / "local_base_m3.json")

    n = min(len(hybrid), len(sft), len(m3))

    rescued_sft: list[tuple] = []
    regressed_sft: list[tuple] = []
    rescued_m3: list[tuple] = []
    regressed_m3: list[tuple] = []

    for i in range(n):
        h = hybrid[i]
        s = sft[i]
        m = m3[i]
        if s.get("db_id") != h.get("db_id") or s.get("question") != h.get("question"):
            continue
        h_ok = bool(h.get("correct"))
        s_ok = bool(s.get("correct"))
        m_ok = bool(m.get("correct"))
        if h_ok and not s_ok:
            rescued_sft.append((i, s, h))
        elif (not h_ok) and s_ok:
            regressed_sft.append((i, s, h))
        if h_ok and not m_ok:
            rescued_m3.append((i, m, h))
        elif (not h_ok) and m_ok:
            regressed_m3.append((i, m, h))

    def sft_state(x):
        if not x.get("exec_success"):
            return "exec_fail"
        return "exec_ok_wrong"

    def hybrid_why_bad(x):
        if not x.get("exec_success"):
            return "exec_fail"
        return x.get("failure_type") or "exec_ok_wrong"

    lines = []
    lines.append(f"hybrid vs SFT single: rescued={len(rescued_sft)} regressed={len(regressed_sft)}")
    lines.append("  rescued 按 SFT 单次状态:")
    lines.append("    " + str(dict(Counter(sft_state(r[1]) for r in rescued_sft).most_common())))
    lines.append("  rescued 中执行拯救的贡献（SFT exec_fail -> hybrid correct）:")
    er = [r for r in rescued_sft if sft_state(r[1]) == "exec_fail"]
    lines.append(f"    {len(er)} 条")
    for i, s, h in er[:999]:
        lines.append(f"      dev#{i:04d} {s['db_id']} | {s['question'][:60]}")
    lines.append("  regressed 按 hybrid 状态:")
    lines.append("    " + str(dict(Counter(hybrid_why_bad(r[2]) for r in regressed_sft).most_common())))
    lines.append("")
    lines.append(f"hybrid vs M3 local base: rescued={len(rescued_m3)} regressed={len(regressed_m3)}")
    lines.append("  rescued 按 M3 状态:")
    lines.append("    " + str(dict(Counter(sft_state(r[1]) for r in rescued_m3).most_common())))
    lines.append("  regressed 按 hybrid 状态:")
    lines.append("    " + str(dict(Counter(hybrid_why_bad(r[2]) for r in regressed_m3).most_common())))

    out = ROOT / "m4" / "hybrid_decomposition.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()