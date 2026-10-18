"""诊断 DPO 梯度为零：最小前向+反向，检查 log_probs 是否 requires_grad、LoRA 是否有非零梯度。"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

BASE = "/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct"
SFT = "/root/autodl-tmp/m4_lora/sft_cons_v1"

def lora_grad_stats(model, tag):
    n_req = 0
    nonzero = 0
    total = 0
    for n, p in model.named_parameters():
        if p.requires_grad and "lora" in n:
            n_req += 1
            if p.grad is not None:
                total += 1
                if p.grad.abs().max().item() != 0.0:
                    nonzero += 1
    print(f"[{tag}] lora params requires_grad={n_req}, with grad tensor={total}, with NON-ZERO grad={nonzero}")

tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto",
                                             torch_dtype=torch.bfloat16, trust_remote_code=True)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)          # 先冻 base
model = PeftModel.from_pretrained(model, SFT, is_trainable=True)  # 后加 LoRA
print("=== trainable after CORRECT order ===")
model.print_trainable_parameters()

base = model.get_base_model()
print("base type:", type(base).__name__)
print("hasattr(base, 'base_model'):", hasattr(base, "base_model"))
print("base.base_model type:", type(getattr(base, "base_model", None)).__name__)

inputs = tokenizer("SELECT name FROM station WHERE city = 'San Jose'", return_tensors="pt").to(model.device)
labels = inputs["input_ids"].clone()

# ---- 路径 A：普通 forward（CE loss 由 base 产生） ----
model.zero_grad(set_to_none=True)
out_a = model(**inputs, labels=labels)
loss_a = out_a.loss
print(f"\n[NORMAL fwd] loss={loss_a.item():.4f} requires_grad={loss_a.requires_grad}")
loss_a.backward()
lora_grad_stats(model, "NORMAL fwd backward")

# ---- 路径 B：trl 融合头 forward ----
from trl.trainer import dpo_trainer as dt
model.zero_grad(set_to_none=True)
dt.add_fused_lm_head(base)
out_b = model(**inputs, labels=labels, fused_lm_head=True)
loss_b = out_b.loss
print(f"\n[FUSED fwd] loss={loss_b.item():.4f} requires_grad={loss_b.requires_grad}")
print(f"[FUSED fwd] log_probs requires_grad={out_b.log_probs.requires_grad}")
loss_b.backward()
lora_grad_stats(model, "FUSED fwd backward")

print("\nDONE")