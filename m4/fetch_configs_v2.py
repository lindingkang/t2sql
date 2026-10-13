"""M4: 用带 UA 的 urllib 请求补齐 config.json / generation_config.json。

hf-mirror 对默认 urllib 会返回 403，本脚本：
1. 设置浏览器 UA + Accept-Encoding: identity
2. 先试 hf-mirror，再试官方 huggingface.co
3. 校验文件为合法 JSON 后写入目标目录
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

DEST = Path("/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct")
REPORT = Path(__file__).resolve().parent / "config_fetch_v2_report.txt"
FILES = ["config.json", "generation_config.json"]
HOSTS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept-Encoding": "identity",
}


def fetch(url: str, target: Path) -> str | None:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        # 校验 JSON
        json.loads(data)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)
        return f"OK size={len(data)}"
    except Exception as exc:  # noqa: BLE001
        return f"ERR {type(exc).__name__}: {exc}"


def main() -> None:
    lines = []
    for fn in FILES:
        target = DEST / fn
        lines.append(f"--- {fn} ---")
        ok = False
        for host in HOSTS:
            url = f"{host}/Qwen/Qwen2.5-Coder-7B-Instruct/resolve/main/{fn}"
            msg = fetch(url, target)
            lines.append(f"{host}: {msg}")
            if msg.startswith("OK"):
                ok = True
                break
        lines.append(f"RESULT {fn}: {'OK' if ok else 'FAILED'}")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()