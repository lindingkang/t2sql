"""M4: 重试补齐 config.json，带退避和多源头。

目标：
1. hf-mirror 限流(429)时自动等待重试
2. 镜像拿不到时尝试 ModelScope
3. 成功后校验 JSON，并写 m4/config_fetch_v3_report.txt
"""
from __future__ import annotations

import json
import time
from pathlib import Path

DEST = Path("/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct")
REPORT = Path(__file__).resolve().parent / "config_fetch_v3_report.txt"
TARGET = DEST / "config.json"

SOURCES = [
    "https://hf-mirror.com/Qwen/Qwen2.5-Coder-7B-Instruct/resolve/main/config.json",
    "https://modelscope.cn/models/Qwen/Qwen2.5-Coder-7B-Instruct/resolve/master/config.json",
    "https://www.modelscope.cn/api/v1/models/Qwen/Qwen2.5-Coder-7B-Instruct/repo?Revision=master&FilePath=config.json",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept-Encoding": "identity",
}


def fetch_with_requests(url: str) -> str:
    import requests
    attempts = 5
    last_err = ""
    for i in range(attempts):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}"
            elif resp.status_code == 200:
                data = resp.content
                json.loads(data)  # validate
                tmp = TARGET.with_suffix(".json.tmp")
                tmp.write_bytes(data)
                tmp.replace(TARGET)
                return f"OK size={len(data)}"
            else:
                last_err = f"HTTP {resp.status_code} body={resp.text[:120]}"
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
        wait = 2 ** i  # 2,4,8,16s
        time.sleep(wait)
    return f"ERR {last_err}"


def main() -> None:
    lines = []
    for url in SOURCES:
        lines.append(f"--- {url}")
        msg = fetch_with_requests(url)
        lines.append(msg)
        if msg.startswith("OK"):
            lines.append("DONE")
            break
    lines.append(f"TARGET_EXISTS={TARGET.exists()} size={TARGET.stat().st_size if TARGET.exists() else 0}")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()