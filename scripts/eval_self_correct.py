"""eval_self_correct.py — M6 自纠错基准评测（exec-feedback 自纠错能力）。

给模型喂 [question + DDL + 它自己/基线生成的错 SQL + sqlite 报错]，看它能否自己改成
可执行（进一步：改到对）的 SQL。衡量"把 exec_repair 规则层的纠错能力蒸馏进权重的程度"。

三份对照（同口径）：
  ① single-pass    [question+DDL]       —— 这些题基线必 exec 失败，= 0
  ② oracle         exec_repair 规则修复   —— 上界（非全部可修）
  ③ model+feedback [question+DDL+错SQL+报错] -> adapter 自纠错（本脚本测这个）

画法：
  python m4/analysis/eval_self_correct.py --adapter /root/autodl-tmp/m4_lora/sft_cons_v1   # 基线对照
  python m4/analysis/eval_self_correct.py --adapter /root/autodl-tmp/m4_lora/repair_r1     # 训后
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "m4"))
sys.path.insert(0, str(ROOT / "m4" / "analysis"))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

from build_repair_sft import SYSTEM, repair_user_turn  # noqa: E402
from schema import load_schemas, get_ddl  # noqa: E402
from run_baseline import SPIDER_DIR, execute_sql, extract_sql, results_equal_strict, results_equal_loose  # noqa: E402

BASE_MODEL = "/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct"
BENCH = ROOT / "m4" / "analysis" / "repair_bench.jsonl"


def load_model(adapter: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.config.use_cache = True
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return tokenizer, model


def oracle_score(bench: list[dict], schemas: dict) -> dict:
    """规则 oracle（exec_repair 的 oracle_fix）在 bench 上的 executability/correctness。"""
    n = len(bench)
    exec_ok = strict = loose = 0
    for b in bench:
        gold = b["gold"]
        gold_rows, _ge = execute_sql(SPIDER_DIR / "database" / b["db_id"] / f"{b['db_id']}.sqlite", gold)
        if b["oracle_fix"] is None:
            continue
        rows, err = execute_sql(SPIDER_DIR / "database" / b["db_id"] / f"{b['db_id']}.sqlite", b["oracle_fix"])
        if err is not None:
            continue
        exec_ok += 1
        if results_equal_strict(gold_rows, rows):
            strict += 1
        if results_equal_loose(gold_rows, rows):
            loose += 1
    return {"n": n, "exec_ok": exec_ok, "strict": strict, "loose": loose}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--bench", default=str(BENCH))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out-txt", default=None)
    args = ap.parse_args()

    bench = [json.loads(x) for x in Path(args.bench).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        bench = bench[:args.limit]
    schemas = load_schemas()
    tokenizer, model = load_model(args.adapter)

    name = Path(args.adapter).name
    n_fixable = sum(1 for b in bench if b["oracle_fixable"])
    rows_out = []
    sub = Counter()   # oracle-fixable 子集上的分数
    sub_n = 0
    for b in bench:
        db_id, question = b["db_id"], b["question"]
        ddl = get_ddl(schemas[db_id])
        gold_rows, _ = execute_sql(SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite", b["gold"])
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": repair_user_turn(question, ddl, b["wrong"], b["error"])},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        plen = inp["input_ids"].shape[1]
        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(gen[0][plen:], skip_special_tokens=True)
        fixed = extract_sql(raw)
        rows, err = execute_sql(SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite", fixed)
        exec_ok = err is None
        strict = loose = False
        if exec_ok:
            strict = results_equal_strict(gold_rows, rows)
            loose = results_equal_loose(gold_rows, rows)
        rows_out.append({"idx": b["idx"], "db_id": db_id, "question": question,
                         "wrong": b["wrong"], "fixed": fixed, "exec_ok": exec_ok,
                         "strict": strict, "loose": loose, "oracle_fixable": b["oracle_fixable"]})
        if b["oracle_fixable"]:
            sub_n += 1
            if exec_ok:
                sub["exec_ok"] += 1
            if strict:
                sub["strict"] += 1
            if loose:
                sub["loose"] += 1

    n = len(bench)
    tot = Counter(r["exec_ok"] for r in rows_out)[True]
    tot_strict = sum(1 for r in rows_out if r["strict"])
    tot_loose = sum(1 for r in rows_out if r["loose"])
    ora = oracle_score(bench, schemas)

    out_json = (ROOT / "m4" / "analysis" / f"selfcorrect_{name}.json")
    out_json.write_text(json.dumps(rows_out, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        "=" * 70,
        f"self-correct eval: {name}",
        "=" * 70,
        f"bench(dev 执行失败): {n} 条（oracle 可修 {n_fixable} / 不可修 {n - n_fixable}）",
        "",
        "[oracle 上界 = exec_repair 规则]",
        f"  exec_ok: {ora['exec_ok']}/{ora['n']} = {ora['exec_ok']/ora['n']:.4f}",
        f"  strict : {ora['strict']}/{ora['n']} = {ora['strict']/ora['n']:.4f}",
        f"  loose  : {ora['loose']}/{ora['n']} = {ora['loose']/ora['n']:.4f}",
        "",
        f"[模型自纠错（{name}，喂错SQL+报错）]",
        f"  exec_ok: {tot}/{n} = {tot/n:.4f}",
        f"  strict : {tot_strict}/{n} = {tot_strict/n:.4f}",
        f"  loose  : {tot_loose}/{n} = {tot_loose/n:.4f}",
        "",
        f"[oracle 可修子集({sub_n} 条)上]",
    ]
    if sub_n:
        lines.append(f"  exec_ok: {sub['exec_ok']}/{sub_n} = {sub['exec_ok']/sub_n:.4f}   "
                     f"(内部化率 vs oracle {ora['exec_ok']}/{sub_n})")
        lines.append(f"  strict : {sub['strict']}/{sub_n} = {sub['strict']/sub_n:.4f}")
    lines.append("")
    lines.append("备注: single-pass 基线 = 0（这些题本就被选为执行失败）。"
                 "对比 ③ 基线 adapter 自纠错 vs ④ repair-SFT 自纠错 才是技能增益。")

    txt = "\n".join(lines)
    out_txt = args.out_txt or (ROOT / "m4" / "analysis" / f"selfcorrect_{name}_summary.txt")
    Path(out_txt).write_text(txt, encoding="utf-8")
    print(txt)
    print("DONE")


if __name__ == "__main__":
    main()