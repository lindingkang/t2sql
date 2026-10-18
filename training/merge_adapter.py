"""merge_adapter.py — 把 LoRA adapter 合并进底座，产出 bf16 全量模型。

用途：GRPO 要"从 SFT 起步"，但 GRPOTrainer 的稳路径是 `model=<dir> + quantization + peft`，
无法直接传"底座+adapter"作为初始策略。合并成一个全量模型后即可当普通 base 用。

用法：
    python m4/analysis/merge_adapter.py \
        --adapter /root/autodl-tmp/m4_lora/sft_cons_v1 \
        --out /root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-SFT-cons
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "m4"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402

BASE_MODEL = "/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default=BASE_MODEL)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"MERGED -> {args.out}")


if __name__ == "__main__":
    main()