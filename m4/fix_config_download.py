"""M4: 补齐模型加载必需的 config.json / generation_config.json。

huggingface_hub 2.2.0 对某些小文件下载时触发兼容错误，
这里直接用标准库从 hf-mirror 下载。
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

DEST = Path("/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct")
REPORT = Path(__file__).resolve().parent / "model_files_check.txt"
BASE = "https://hf-mirror.com/Qwen/Qwen2.5-Coder-7B-Instruct/resolve/main/"
FILES = ["config.json", "generation_config.json"]


def main() -> None:
    lines = []
    for fn in FILES:
        target = DEST / fn
        if target.exists() and target.stat().st_size > 0:
            lines.append(f"EXISTS {fn} size={target.stat().st_size}")
            continue
        url = BASE + fn
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            urlretrieve(url, tmp)
            tmp.replace(target)
            lines.append(f"DOWNLOADED {fn} size={target.stat().st_size}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"FAILED {fn}: {exc}")

    # 列出最终关键文件
    for p in sorted(DEST.iterdir()):
        if p.is_file():
            lines.append(f"{p.name} {p.stat().st_size/1e6:.2f}MB")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()