"""run_grpo.py — M6 GRPO 在线 RL（验证心态：先冒烟验 rail，再 sanity 验假设）。

训练集用 train（dev 只做评测）。奖励 = 稠密 exec 奖（grpo_reward.py）。
prompt = 与 SFT 完全一致的 system + (question, DDL) 模板。

用法：
  # 冒烟：8 题 × 2 步 × G4，验整条链 rail（奖励被调用 / loss 在动 / grad 非零）
  python m4/analysis/run_grpo.py --smoke
  # sanity：200 题 × 100 步 × G8（验"reward 曲线 + advantage 方差"假设）
  python m4/analysis/run_grpo.py --n-prompts 200 --steps 100 --G 8 --out-dir /root/autodl-tmp/m4_lora/grpo_sanity
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "m4"))
sys.path.insert(0, str(ROOT / "m4" / "analysis"))

import torch  # noqa: E402
from transformers import AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402
from peft import LoraConfig  # noqa: E402

from grpo_reward import reward_fn  # noqa: E402
from schema import load_schemas, get_ddl  # noqa: E402
from run_baseline import SPIDER_DIR  # noqa: E402
from eval_sft_adapter import SYSTEM, build_user  # noqa: E402
from datasets import Dataset  # noqa: E402

BASE_MODEL = "/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct"
TRAIN_SPIDER = SPIDER_DIR / "train_spider.json"
TRAIN_OTHERS = SPIDER_DIR / "train_others.json"


def build_dataset(n: int, seed: int = 42, base_model: str = BASE_MODEL) -> list[dict]:
    """取 train 洗牌后 n 题（跳过无 schema 的），构造 {prompt(已模板化文本), gold, db_id}。"""
    schemas = load_schemas()
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    samples = json.loads(TRAIN_SPIDER.read_text(encoding="utf-8"))
    samples += json.loads(TRAIN_OTHERS.read_text(encoding="utf-8"))
    random.Random(seed).shuffle(samples)
    rows = []
    for s in samples:
        if len(rows) >= n:
            break
        db_id, question, gold = s["db_id"], s["question"], s["query"].strip().rstrip(";")
        if db_id not in schemas:
            continue
        ddl = get_ddl(schemas[db_id])
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user(question, ddl)}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": prompt, "gold": gold, "db_id": db_id})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--out-dir", default="/root/autodl-tmp/m4_lora/grpo_smoke")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n = 8 if args.smoke else args.n_prompts
    steps = 2 if args.smoke else args.steps
    G = 4 if args.smoke else args.G
    if args.smoke:
        args.out_dir = "/root/autodl-tmp/m4_lora/grpo_smoke"

    ds = Dataset.from_list(build_dataset(n, args.seed, args.base_model))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    peft = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    training_args = GRPOConfig(
        output_dir=args.out_dir,
        num_train_epochs=1,
        max_steps=steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        beta=args.beta,
        num_generations=G,
        generation_batch_size=G,
        max_completion_length=256,
        temperature=1.0,
        logging_steps=1,
        report_to=[],
        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        seed=42,
    )

    trainer = GRPOTrainer(
        model=args.base_model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=ds,
        quantization_config=bnb,
        peft_config=peft,
    )
    trainer.train()


if __name__ == "__main__":
    main()