#!/usr/bin/env python3
"""Inspect full SFT status and write a small summary report."""
import os
import re
import time
from pathlib import Path

ROOT = Path("/root/t2sql")
ADAPTER_DIR = Path("/root/autodl-tmp/m4_lora/sft_cons_v1")

LOG_FULL = ROOT / "m4" / "sft_full_run.txt"
LOG_REPORT = ROOT / "m4" / "sft_train_report.txt"
LOG_CHECK = ROOT / "m4" / "check_final_status.txt"
PID_FILE = ROOT / "m4" / "sft_full.pid"

OUT = ROOT / "m4" / "sft_result_summary.txt"


def tail_lines(path: Path, n: int = 150) -> list[str]:
    if not path.exists():
        return [f"<file not found: {path}>"]
    # Read only the last chunk by bytes, then slice text lines.
    size = path.stat().st_size
    chunk = 1_000_000
    if size <= chunk:
        data = path.read_text(errors="replace")
    else:
        with path.open("rb") as f:
            f.seek(-chunk, os.SEEK_END)
            data = f.read().decode("utf-8", errors="replace")
    data = data.replace("\r", "\n")
    lines = [line.rstrip() for line in data.split("\n") if line.strip() != ""]
    if not lines and path.exists():
        lines = path.read_text(errors="replace").splitlines()
    return lines[-n:]


def contains(path: Path, patterns: list[str]) -> dict[str, bool]:
    result = {}
    if not path.exists():
        for p in patterns:
            result[p] = False
        return result
    text = path.read_text(errors="replace")
    for p in patterns:
        result[p] = p in text
    return result


def adapter_snapshot(path: Path) -> list[str]:
    if not path.exists():
        return [f"<adapter dir not found: {path}>"]
    rows = []
    for root, dirs, files in os.walk(path):
        rel = os.path.relpath(root, path)
        for fname in sorted(files):
            fp = Path(root) / fname
            st = fp.stat()
            rel_name = fname if rel == "." else os.path.join(rel, fname)
            rows.append(f"{rel_name}\t{st.st_size}\t{time.ctime(st.st_mtime)}")
    return rows


def pid_status(pid_file: Path) -> list[str]:
    if not pid_file.exists():
        return ["<pid file not found>"]
    raw = pid_file.read_text(errors="replace").strip()
    lines = [f"pid_file={pid_file}", f"pid_raw={raw}"]
    try:
        pid = int(raw)
    except ValueError:
        lines.append("<pid not an int>")
        return lines
    lines.append(f"pid_alive={Path(f'/proc/{pid}').exists()}")
    proc_cmd = Path(f"/proc/{pid}/cmdline")
    if proc_cmd.exists():
        lines.append(f"cmdline={proc_cmd.read_text(errors='replace').replace(chr(0), ' ')}")
    return lines


def main() -> None:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"SFT INSPECTION SUMMARY @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    lines.append("")
    lines.append("[1] PID STATUS")
    lines.extend(pid_status(PID_FILE))

    lines.append("")
    lines.append("[2] FULL LOG KEYWORDS")
    kw = contains(LOG_FULL, [
        "TRAINING_DONE",
        "DONE",
        "SAVED_MODEL",
        "Traceback",
        "RuntimeError",
        "CUDA out of memory",
        "Error",
        "error",
    ])
    for k, v in kw.items():
        lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("[3] FULL LOG TAIL")
    lines.extend(tail_lines(LOG_FULL, 120))

    lines.append("")
    lines.append("[4] REPORT LOG KEYWORDS")
    kw2 = contains(LOG_REPORT, [
        "TRAINING_DONE",
        "DONE",
        "SAVED_MODEL",
        "SAVING",
        "Traceback",
        "RuntimeError",
        "CUDA out of memory",
    ])
    for k, v in kw2.items():
        lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("[5] REPORT LOG TAIL")
    lines.extend(tail_lines(LOG_REPORT, 80))

    lines.append("")
    lines.append("[6] CHECK STATUS TAIL")
    lines.extend(tail_lines(LOG_CHECK, 60))

    lines.append("")
    lines.append("[7] ADAPTER SNAPSHOT")
    lines.extend(adapter_snapshot(ADAPTER_DIR))

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"summary_written={OUT} lines={len(lines)}")


if __name__ == "__main__":
    main()