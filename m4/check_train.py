"""M4: 检查 SFT 冒烟训练进程/GPU/报告状态，写出稳定报告。"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path(__file__).resolve().parent / "train_check_report.txt"

def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout.strip() + ("\n" + p.stderr.strip() if p.stderr.strip() else "")

def main():
    lines = []
    lines.append("--- ps train_sft ---")
    lines.append(run(["bash", "-lc", "ps -eo pid,etime,cmd | grep train_sft.py | grep -v grep || true"]))
    lines.append("--- gpu ---")
    lines.append(run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"]))
    lines.append("--- sft report tail ---")
    p = ROOT / "m4" / "sft_train_report.txt"
    lines.append(p.read_text(encoding="utf-8")[-3000:] if p.exists() else "(missing)")
    lines.append("--- smoke dir ---")
    smoke = Path("/root/autodl-tmp/m4_lora/smoke")
    if smoke.exists():
        for f in sorted(smoke.rglob("*")):
            if f.is_file():
                lines.append(f"{f.relative_to(smoke)} {f.stat().st_size}")
    else:
        lines.append("(no smoke dir)")
    REPORT.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()