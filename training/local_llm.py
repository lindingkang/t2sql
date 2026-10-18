"""本地 7B base 模型的 OpenAI 兼容包装器。

让 eval_harness.run_once 和 agent.Text2SQLAgent 无需改动即可：
    client.chat.completions.create(messages=[...], temperature=..., max_tokens=...)
内部转成 transformers 本地生成。

支持：
- temperature=0 -> greedy 生成
- temperature>0 -> sampling 生成（用于 M1.5 自洽性投票）
- 返回 usage token，用于 agent 的 meter 统计
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct"


def load_local_base_model():
    """4bit 加载 base model（不加载任何 LoRA）。"""
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = True
    model.eval()
    return tokenizer, model


class LocalOpenAIClient:
    """提供与 OpenAI client 相同的 chat.completions.create 接口。"""

    def __init__(self, tokenizer, model, max_new_tokens: int = 256,
                 max_input_tokens: int = 4096):
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        # 构造 client.chat.completions.create(...) 调用路径
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _extract_prompt(self, messages) -> str:
        if isinstance(messages, str):
            return messages
        if isinstance(messages, list):
            parts = []
            for m in messages:
                if isinstance(m, dict):
                    content = m.get("content")
                    if content:
                        parts.append(str(content))
                elif isinstance(m, str):
                    parts.append(m)
            return "\n".join(parts)
        return str(messages)

    def _create(self, model=None, messages=None, temperature: float = 0.0,
                max_tokens: int = 512, timeout: int = 60, **kwargs):
        prompt = self._extract_prompt(messages)
        inp = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)
        prompt_len = int(inp["input_ids"].shape[1])

        max_new = int(max_tokens or 256)
        max_new = min(max_new, self.max_new_tokens)

        with torch.inference_mode():
            if temperature is not None and temperature > 0.0:
                gen = self.model.generate(
                    **inp,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                gen = self.model.generate(
                    **inp,
                    max_new_tokens=max_new,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        new_ids = gen[0][prompt_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        completion_tokens = int(new_ids.shape[0])

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text)
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=prompt_len,
                completion_tokens=completion_tokens,
                total_tokens=prompt_len + completion_tokens,
            ),
        )