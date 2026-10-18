"""把 cli 下载日志尾部写入可读文件，用于远程排错。"""
from pathlib import Path

src = Path(__file__).resolve().parent / "model_download_cli3.txt"
out = Path(__file__).resolve().parent / "cli3_tail.txt"

text = src.read_text(errors="replace") if src.exists() else "(missing)"
out.write_text("LEN=" + str(len(text)) + "\n" + text[-6000:], encoding="utf-8")
