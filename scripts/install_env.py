"""M4 环境依赖安装脚本：安装 SFT/DPO 训练所需包。

安装日志直接写 m4/pip_install_report.txt，避免 shell 重定向在远程环境下不可靠。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PACKAGES = [
    "transformers",
    "peft",
    "trl",
    "bitsandbytes",
    "accelerate",
    "datasets",
    "sentencepiece",
]
INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
REPORT = Path(__file__).resolve().parent / "pip_install_report.txt"


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        *PACKAGES,
        "-i",
        INDEX,
    ]
    REPORT.write_text(
        f"# M4 dependency install\n# started\nCMD: {subprocess.list2cmdline(cmd)}\n",
        encoding="utf-8",
    )
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        REPORT.write_text(
            f"# finished\nexit={proc.returncode}\n\n{proc.stdout}",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        REPORT.write_text(f"# exception\n{exc}\n", encoding="utf-8")


if __name__ == "__main__":
    main()