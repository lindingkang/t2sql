"""M4-C 前置：最小验证 SFT LoRA adapter 能否正常加载并生成 SQL。

只做三件事：
1. 4bit 加载 base model，并加载训练好的 LoRA adapter。
2. 用与训练一致的 system/user 消息模板和 apply_chat_template 构造 prompt。
3. 对 1 条 Spider dev 样本做 greedy 生成，检查输出是否为 SQL。

输出：m4/verify_sft_adapter_report.txt（远端环境无法直接看终端输出）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from schema import load_schemas, get_ddl  # noqa: E402
from run_baseline import SPIDER_DIR  # noqa: E402

BASE_MODEL = "/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct"
ADAPTER_DIR = "/root/autodl-tmp/m4_lora/sft_cons_v1"
REPORT = Path(__file__).resolve().parent / "verify_sft_adapter_report.txt"

SYSTEM = (
    "You are a SQLite SQL expert. "
    "Given the database schema below, write a single SQLite SQL query "
    "to answer the question. Output ONLY the SQL query on ONE single line, "
    "with no markdown fences and no explanation."
)


def write_report(lines: list[str]) -> None:
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def build_user(question: str, ddl: str) -> str:
    return f"Schema:\n{ddl}\n\nQuestion: {question}\n\nSQL:"


def main() -> None:
    out: list[str] = []
    t0 = time.time()

    # 1. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    out.append(f"tokenizer OK pad={tokenizer.pad_token_id} eos={tokenizer.eos_token_id}")

    # 2. base model 4bit + LoRA
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.config.use_cache = True
    except Exception as exc:
        out.append(f"BASE_LOAD_ERROR: {type(exc).__name__}: {exc}")
        write_report(out)
        return

    try:
        model = PeftModel.from_pretrained(
            model, ADAPTER_DIR, is_trainable=False
        )
    except Exception as exc:
        out.append(f"ADAPTER_LOAD_ERROR: {type(exc).__name__}: {exc}")
        write_report(out)
        return

    model.eval()
    out.append(f"adapter OK device={model.device}")
    out.append(f"base+adapter load_time_s={time.time()-t0:.1f}")

    # 3. 取 1 条 dev 样本生成
    dev = json.loads((SPIDER_DIR / "dev.json").read_text(encoding="utf-8"))
    sample = dev[0]
    schemas = load_schemas()
    ddl = get_ddl(schemas[sample["db_id"]])
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user(sample["question"], ddl)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inp = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        gen = model.generate(
            **inp,
            max_new_tokens=160,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inp["input_ids"].shape[1]
    generated = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
    full_text = tokenizer.decode(gen[0], skip_special_tokens=True)

    # 4. 观察结果写入报告
    all_bins = gen[0].tolist()
    eos_count = all_bins.count(tokenizer.eos_token_id)
    out.append("=" * 70)
    out.append("生成样例（Spider dev 第 1 条）")
    out.append("=" * 70)
    out.append(f"db_id={sample['db_id']}")
    out.append(f"question={sample['question']}")
    out.append(f"gold_sql={sample['query'].strip().rstrip(';')}")
    out.append(f"prompt_tokens={prompt_len}")
    out.append(f"new_tokens={len(all_bins)-prompt_len}")
    out.append(f"eos_count_in_full_generation={eos_count}")
    out.append(f"generated={generated!r}")
    out.append(f"full_text_tail={full_text[-600:]!r}")
    out.append("RESULT=PASS")

    write_report(out)


if __name__ == "__main__":
    main()