"""等待本地 M3 全量跑完后，自动生成 M0/M2/M15/M3/SFT 最终对比表。

只做文件级等待与摘要解析，不重复跑模型。
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M4 = ROOT / "m4"
M3_PID_FILE = M4 / "local_base_m3_full.pid"
M3_SUMMARY = M4 / "local_base_m3_summary.txt"
SFT_SUMMARY = M4 / "sft_eval_summary.txt"
OUT = M4 / "local_base_vs_sft_comparison.txt"
LOG = M4 / "wait_m3_and_merge.log"


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def read_pid() -> int | None:
    if not M3_PID_FILE.exists():
        return None
    txt = M3_PID_FILE.read_text(encoding="utf-8").strip()
    try:
        return int(txt)
    except ValueError:
        return None


def parse_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    def get(pattern: str):
        m = re.search(pattern, text)
        return m.group(1) if m else None
    return {
        "EA_strict": get(r"EA_strict:\s+([0-9.]+)\s+\((\d+)/(\d+)\)"),
        "EA_numer": get(r"EA_strict:\s+([0-9.]+)\s+\((\d+)/(\d+)\)"),
        # 上面两次以最后定义的为准不优雅，这里用显式分组一次性拿全
    }


def parse_metrics2(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    m_ea = re.search(r"EA_strict:\s+([0-9.]+)\s+\((\d+)/(\d+)\)", text)
    m_loose = re.search(r"EA_loose:\s+([0-9.]+)\s+\((\d+)/(\d+)\)", text)
    m_exec = re.search(r"exec_rate:\s+([0-9.]+)", text)
    return {
        "EA_strict": m_ea.group(1) if m_ea else "N/A",
        "EA_strict_n_d": f"{m_ea.group(2)}/{m_ea.group(3)}" if m_ea else "N/A",
        "EA_loose": m_loose.group(1) if m_loose else "N/A",
        "EA_loose_n_d": f"{m_loose.group(2)}/{m_loose.group(3)}" if m_loose else "N/A",
        "exec_rate": m_exec.group(1) if m_exec else "N/A",
    }


def main() -> None:
    if LOG.exists():
        LOG.unlink()
    log(f"wait_m3_and_merge start @ {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 若 M3 进程还存在则等待退出；之后再确认 summary 文件生成（可能还需 1-2 分钟）
    pid = read_pid()
    if pid is not None:
        while pid_alive(pid):
            log(f"M3 still running (pid={pid}), sleep 60s")
            time.sleep(60)
    else:
        log("no M3 pid file found, assuming not running")

    # 兼容写 summary 与进程退出的竞争：多等几次
    for i in range(10):
        if M3_SUMMARY.exists():
            break
        log(f"waiting for {M3_SUMMARY.name}, attempt {i+1}/10")
        time.sleep(30)

    if not M3_SUMMARY.exists():
        log(f"ERROR: {M3_SUMMARY.name} still missing after wait")
        return

    rows = [
        ("local base M0", M4 / "local_base_m0_summary.txt"),
        ("local base M2", M4 / "local_base_m2_summary.txt"),
        ("local base M1.5", M4 / "local_base_m15_summary.txt"),
        ("local base M3", M4 / "local_base_m3_summary.txt"),
        ("SFT adapter M4", SFT_SUMMARY),
    ]

    lines = [
        "=" * 80,
        f"同模型公平对比：Qwen2.5-Coder-7B-Instruct 基准 vs SFT adapter",
        f"generated @ {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
    ]
    for name, p in rows:
        m = parse_metrics2(p) if p.exists() else None
        if m is None:
            lines.append(f"\n{name:15s}: MISSING ({p.name})")
            continue
        lines.append(
            f"\n{name:15s}  ER={m['exec_rate']}  "
            f"EA_strict={m['EA_strict']} ({m['EA_strict_n_d']})  "
            f"EA_loose={m['EA_loose']} ({m['EA_loose_n_d']})"
        )
    lines.append("\n" + "=" * 80)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
