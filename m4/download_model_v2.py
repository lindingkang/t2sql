"""M4: 稳健下载 Qwen2.5-Coder-7B-Instruct，适配 huggingface_hub 2.2.0。

显式禁用 Xet；只用当前版本 snapshot_download 支持的参数。
日志写 m4/model_download_v2_report.txt。
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

REPO_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
LOCAL_DIR = Path("/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct")
REPORT = Path(__file__).resolve().parent / "model_download_v2_report.txt"

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_XET_DISABLE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    REPORT.write_text("# v2 download\n", encoding="utf-8")

    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(
            REPO_ID,
            local_dir=LOCAL_DIR,
            max_workers=4,
        )
        log(f"SNAPSHOT_OK path={path}")
    except Exception as exc:  # noqa: BLE001
        log(f"SNAPSHOT_FAILED {type(exc).__name__}: {exc}")
        # 逐文件兜底
        from huggingface_hub import hf_hub_download
        files = [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ]
        for fn in files:
            try:
                hf_hub_download(
                    REPO_ID,
                    fn,
                    local_dir=LOCAL_DIR,
                )
                log(f"FILE_OK {fn}")
            except Exception as exc2:  # noqa: BLE001
                log(f"FILE_FAIL {fn}: {exc2}")

    total = 0
    count = 0
    if LOCAL_DIR.exists():
        for p in LOCAL_DIR.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
                count += 1
    log(f"FILES={count} SIZE_GB={total/1e9:.2f}")
    log("DONE")


if __name__ == "__main__":
    main()