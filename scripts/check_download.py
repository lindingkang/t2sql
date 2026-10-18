"""M4: check base model download progress and write a stable report file."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

DEST = Path("/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct")
REPORT = Path(__file__).resolve().parent / "download_progress.txt"


def main() -> None:
    lines = []
    lines.append(f"dest_exists={DEST.exists()}")
    if DEST.exists():
        files = [p for p in DEST.rglob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        lines.append(f"file_count={len(files)}")
        lines.append(f"total_GB={total/1e9:.2f}")
        for p in sorted(files):
            lines.append(f"{p.relative_to(DEST)} {p.stat().st_size/1e6:.1f}MB")
    else:
        lines.append("no dest yet")

    proc = subprocess.run(
        ["bash", "-lc", "ps -eo pid,etime,cmd | grep -E 'hf download|snapshot_download|xet' | grep -v grep || true"],
        capture_output=True, text=True,
    )
    lines.append("--- download processes ---")
    lines.append(proc.stdout.strip() or "(none)")

    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()