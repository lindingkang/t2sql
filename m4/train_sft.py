"""M4 第 1 步：Qwen2.5-Coder-7B-Instruct QLoRA SFT（先让模型学会执行）。

设计：
- 使用 bitsandbytes 4bit NF4 + LoRA，适配 RTX 4080 SUPER 32GB。
- 只用 assistant 回答部分计算交叉熵，system/user prompt 掩码为 -100。
- 数据集来自 m4/train_conservative.jsonl 或 m4/train_strict.jsonl。
- 训练日志由 TrainerCallback 写 m4/sft_train_report.txt，便于远程环境读取。

用法：
    # 冒烟：只取 100 条，2 步，验证加载/前向/反向/保存
    python m4/train_sft.py --limit 100 --max-steps 3 --output-dir /root/autodl-tmp/m4_lora/smoke

    # 正式训练
    python m4/train_sft.py --train-file m4/train_conservative.jsonl \
        --output-dir /root/autodl-tmp/m4_lora/sft_cons_v1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

DEFAULT_BASE = "/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct"
DEFAULT_TRAIN = str(ROOT / "m4" / "train_conservative.jsonl")
DEFAULT_OUT = "/root/autodl-tmp/m4_lora/sft_cons_v1"
_REPORT = ROOT / "m4" / "sft_train_report.txt"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}\n"
    with _REPORT.open("a", encoding="utf-8") as f:
        f.write(line)


class ReportCallback(TrainerCallback):
    """把关键训练指标增量写入报告文件，方便远程查看。"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            keys = ["loss", "learning_rate", "epoch", "grad_norm"]
            msg = " ".join(f"{k}={logs[k]:.4g}" for k in keys if k in logs)
            log(f"step={state.global_step} {msg}")

    def on_train_end(self, args, state, control, **kwargs):
        log(f"TRAINING_DONE global_step={state.global_step}")


def load_train_file(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec["message"])
            if limit and len(records) >= limit:
                break
    return records


def tokenize_example(example: dict[str, Any], tokenizer, max_len: int) -> dict[str, Any]:
    """只监督 assistant 回复。

    messages 预期格式：
      [system, user, assistant]
    将 system+user+assistant_prefix 作为 prompt，assistant 内容作为 label。
    """
    msgs = example["messages"]
    if len(msgs) < 3:
        raise ValueError(f"messages 长度不足: {msgs}")

    prompt_msgs = msgs[:-1]
    assistant_msg = msgs[-1]
    assistant_text = assistant_msg["content"]

    # prompt：开启 generation prompt，让模型知道后面该生成 assistant 内容
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # 老版本 fallback
        prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in prompt_msgs)
        prompt_text += "\nassistant: "

    response_text = assistant_text + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False,
                           truncation=True, max_length=max_len)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    # 序列过长时优先截断 prompt 的左侧，保留问题尾部与完整回答
    avail = max_len - len(response_ids)
    if avail <= 0:
        # 连回答都放不下：极端情况截断回答尾部
        response_ids = response_ids[:max_len]
        prompt_ids = []
    else:
        prompt_ids = prompt_ids[-avail:]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_collator(tokenizer):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        print("warning: tokenizer.pad_token_id 为 None，使用 0")
        pad_id = 0
    return DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        padding=True,
    )


def _compute_warmup_steps(args: argparse.Namespace, n_train: int) -> int:
    """transformers 5.x 已移除 warmup_ratio，这里折算为 warmup_steps。"""
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        per_device = max(1, args.batch_size)
        grad_accum = max(1, args.grad_accum)
        total_steps = max(1, math.ceil(n_train / (per_device * grad_accum) * args.epochs))
    return max(0, min(total_steps // 10, 200))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--train-file", default=DEFAULT_TRAIN)
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=None,
                   help="限制训练样本数（冒烟用）")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="-1 表示按 epoch 训练完整个数据集")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-fraction", type=float, default=0.0,
                   help="保留 0.02 作为训练内 eval；0 表示不切")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _REPORT.write_text("# M4 SFT training\n", encoding="utf-8")
    log(f"base={args.base_model}")
    log(f"train={args.train_file} limit={args.limit}")
    log(f"max_seq={args.max_seq_length} batch={args.batch_size} grad_accum={args.grad_accum} epochs={args.epochs}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    log("tokenizer loaded")

    # 2. 数据
    records = load_train_file(Path(args.train_file), args.limit)
    log(f"loaded {len(records)} examples")

    ds = Dataset.from_list([{"messages": r["messages"]} for r in records])
    ds = ds.map(
        lambda ex: tokenize_example(ex, tokenizer, args.max_seq_length),
        remove_columns=["messages"],
        num_proc=1,
    )
    log(f"tokenized dataset columns={ds.column_names}")

    if args.eval_fraction > 0:
        split = ds.train_test_split(test_size=args.eval_fraction, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]
        log(f"train={len(train_ds)} eval={len(eval_ds)}")
    else:
        train_ds = ds
        eval_ds = None

    # 3. 模型 + 4bit QLoRA
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    log("peft model ready")

    # 4. Trainer
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=_compute_warmup_steps(args, len(train_ds)),
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        lr_scheduler_type="cosine",
        seed=42,
    )

    try:
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=build_collator(tokenizer),
            callbacks=[ReportCallback()],
        )

        log("trainer start")
        trainer.train()
    except Exception:
        import traceback
        log(traceback.format_exc())
        raise

    # 5. 保存 adapter
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log(f"SAVED_MODEL {args.output_dir}")
    log("DONE")


if __name__ == "__main__":
    main()