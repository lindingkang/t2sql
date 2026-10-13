"""M4：下载 Qwen2.5-Coder-7B-Instruct 基座到数据盘。

优先使用 hf-mirror.com 加速；下载日志写入 m4/model_download_report.txt。
"""

from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
LOCAL_DIR = Path("/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct")
REPORT = Path(__file__).resolve().parent / "model_download_report.txt"


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(line)


def ensure_hf():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        log("缺少 huggingface_hub，开始安装")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-U",
                        "huggingface_hub", "-i",
                        "https://pypi.tuna.tsinghua.edu.cn/simple"],
                       check=True)


def main() -> None:
    REPORT.write_text(f"# model download {REPO_ID}\n", encoding="utf-8")
    ensure_hf()

    # 优先镜像站；Windows 下 mirror 兼容性不同，但这是 Linux，设置无害。
    if os.getenv("HF_ENDPOINT") is None:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    log(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")

    from huggingface_hub import snapshot_download, hf_hub_download

    # 统一用 snapshot 拉权重/配置/tokenizer
    try:
        path = snapshot_download(
            REPO_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            max_workers=4,
            resume_download=True,
        )
        log(f"SNAPSHOT_OK path={path}")
    except Exception as exc:  # noqa: BLE001
        log(f"snapshot_download failed: {exc}")
        # 退化为逐个下载必需文件
        log("fallback to hf_hub_download key files...")
        must_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
        ]
        for fn in must_files:
            try:
                hf_hub_download(REPO_ID, fn, local_dir=LOCAL_DIR,
                                local_dir_use_symlinks=False)
                log(f"downloaded {fn}")
            except Exception as exc2:  # noqa: BLE001
                log(f"failed {fn}: {exc2}")
        # 权重文件：让 snapshot 只做权重
        try:
            snapshot_download(
                REPO_ID,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
                allow_patterns=["*.safetensors", "*.json", "*.txt",
                                "*.model", "tokenizer*", "vocab*", "merges*"],
                resume_download=True,
            )
            log("fallback snapshot weights done")
        except Exception as exc3:  # noqa: BLE001
            log(f"fallback snapshot weights failed: {exc3}")

    # 汇总目标目录大小
    total = sum(f.stat().st_size for f in LOCAL_DIR.rglob("*")
                if f.is_file())
    log(f"TOTAL_SIZE_GB={total/1e9:.2f}")
    log("DONE")


if __name__ == "__main__":
    main()