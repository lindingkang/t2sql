"""M4: 验证已下载的 Qwen2.5-Coder-7B-Instruct 可加载与可生成。

加载 4bit + NF4 模型，运行一个前向，并生成一小段文本。
所有输出写 m4/verify_model_report.txt。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MODEL_DIR = "/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct"
REPORT = Path(__file__).resolve().parent / "verify_model_report.txt"


def log(lines: list[str]) -> None:
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = []
    t0 = time.time()

    # 1. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True, use_fast=True
    )
    out.append(f"tokenizer OK pad={tokenizer.pad_token_id} eos={tokenizer.eos_token_id}")

    # 2. 4bit model
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    out.append(f"model OK dtype={model.dtype}")
    out.append(f"device={model.device}")
    out.append(f"hf_device_map={getattr(model.hf_device_map, '_asdict', lambda: None)() if hasattr(model, 'hf_device_map') else 'n/a'}")
    out.append(f"load_time_s={time.time()-t0:.1f}")

    # 3. 前向
    model.eval()
    inp = tokenizer(
        "Schema:\nCREATE TABLE A (x int);\n\nQuestion: what is 1+1?\n\nSQL:",
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        gen = model.generate(
            **inp,
            max_new_tokens=20,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    # 只显示生成尾段，避免 prompt 过长
    prompt_len = len(inp["input_ids"][0])
    generated = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
    out.append(f"generate OK prompt_tokens={prompt_len}")
    out.append(f"generated={generated!r}")
    out.append("RESULT=PASS")

    log(out)


if __name__ == "__main__":
    main()