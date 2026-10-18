"""step5_train_dpo.py — M5 Step5: 用 Step4 的偏好对做 QLoRA DPO。

设计：
  - 从 SFT LoRA 初始化 policy（可训练），另加载一份 SFT LoRA 做 frozen reference（KL 锚点）。
  - 数据格式用 trl 1.15 的 conversation 格式：{prompt:[system,user], chosen:[assistant], rejected:[assistant]}，
    由 DPOTrainer 自动套 chat template 并对 prompt 部分 mask。
  - β 默认 0.1（DPO 正则强度）；lr 取 1e-5（远低于 SFT 的 2e-4，防遗忘）。

用法：
    python m4/analysis/step5_train_dpo.py --limit 100 --max-steps 2 --output-dir .../dpo_smoke   # 冒烟
    python m4/analysis/step5_train_dpo.py --beta 0.1 --output-dir /root/autodl-tmp/m4_lora/dpo_b01_v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import PeftModel, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback,
)
from trl import DPOTrainer, DPOConfig  # noqa: E402

BASE_MODEL = "/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct"
SFT_ADAPTER = "/root/autodl-tmp/m4_lora/sft_cons_v1"
PAIRS_IN = ROOT / "m4" / "analysis" / "step4_pairs.jsonl"
DEFAULT_OUT = "/root/autodl-tmp/m4_lora/dpo_b01_v1"
REPORT = ROOT / "m4" / "analysis" / "step5_report.txt"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}\n"
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


class ReportCB(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            keys = ["loss", "learning_rate", "epoch", "grad_norm", "rewards/margins", "rewards/accuracies"]
            msg = " ".join(f"{k}={logs[k]:.4g}" for k in keys if k in logs)
            log(f"step={state.global_step} {msg}")

    def on_train_end(self, args, state, control, **kwargs):
        log(f"TRAINING_DONE global_step={state.global_step}")


def load_pairs(path, limit: int | None) -> Dataset:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({
            "prompt": r["prompt"],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
        })
        if limit and len(rows) >= limit:
            break
    return Dataset.from_list(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--sft-adapter", default=SFT_ADAPTER)
    p.add_argument("--pairs", default=str(PAIRS_IN))
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--limit", type=int, default=None, help="冒烟：只取前 N 对")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--grad-ckpt", action="store_true", default=False, help="训练时是否开启 gradient checkpointing（默认关，用于定位梯度为零的 bug）")
    p.add_argument("--use-reentrant", action="store_true", default=False, help="gradient_checkpointing_kwargs 的 use_reentrant")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    REPORT.write_text("# M5 DPO training\n", encoding="utf-8")
    log(f"base={args.base_model}")
    log(f"sft_adapter={args.sft_adapter}")
    log(f"pairs={args.pairs} limit={args.limit}")
    log(f"beta={args.beta} lr={args.lr} epochs={args.epochs} batch={args.batch_size} grad_accum={args.grad_accum}")

    # 1. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    log("tokenizer loaded")

    # 2. data
    ds = load_pairs(args.pairs, args.limit)
    n = len(ds)
    log(f"loaded {n} preference pairs")
    if n == 0:
        log("ABORT: no pairs")
        return

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. policy model（SFT 初始化，可训练 LoRA）
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.config.use_cache = False
    # 关键顺序：先 prepare_model_for_kbit_training（冻 base / cast fp32 / enable gc），
    # 再 PeftModel.from_pretrained 加载 LoRA（可训练）。顺序反了会把 LoRA 也一起冻死 -> 0 可训练参数 -> grad_norm=0。
    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    log("policy model ready")

    # 4. reference model（SFT 冻结）
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(ref_model, args.sft_adapter, is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    log("reference model ready")

    # 5. DPO config
    total_steps = math.ceil(n / (args.batch_size * args.grad_accum) * args.epochs)
    if args.max_steps > 0:
        total_steps = args.max_steps
    dpo_cfg = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        loss_type="sigmoid",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        warmup_steps=max(0, min(total_steps // 10, 200)),
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": args.use_reentrant} if args.grad_ckpt else None,
        optim="paged_adamw_8bit",
        report_to=[],
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        seed=42,
    )
    log(f"est_total_steps={total_steps} max_length={args.max_length}")

    # ---- trl 1.15 DPO 的 fused_lm_head 补丁对 ref_model(PeftModel) 有 bug：
    # policy 路径会 get_base_model() 解包，ref 路径却直接 patch PeftModel，
    # 导致 backbone_attr="base_model" 解析成 CausalLM（返回 CausalLMOutputWithPast），
    # 缺 last_hidden_state 抛 AttributeError。这里统一解包后再打补丁。
    from trl.trainer import dpo_trainer as _dpo_trainer
    _orig_add_fused_lm_head = _dpo_trainer.add_fused_lm_head

    def _patched_add_fused_lm_head(model, *a, **kw):
        get_base = getattr(model, "get_base_model", None)
        if callable(get_base):
            model = get_base()
        return _orig_add_fused_lm_head(model, *a, **kw)

    _dpo_trainer.add_fused_lm_head = _patched_add_fused_lm_head

    # 6. train
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        callbacks=[ReportCB()],
    )
    log("trainer start")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log(f"SAVED_MODEL {args.output_dir}")
    log("DONE")


if __name__ == "__main__":
    main()