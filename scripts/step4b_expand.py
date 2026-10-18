"""step4b_expand.py — M5 Step4 扩展版：拒绝采样构造 DPO 偏好对（错误题全采 + 已做对题按硬度分层采）。

方案（用户拍板）：
  1. 错误题（greedy 单次做错，Step3 已 gold 筛查）全部采 N=8 次。
  2. 已做对题抽样：按 SQL 复杂度从难到易排序，分批采 N=6 次，挖出"偶尔才错"的样本；
     若最终对数不够 --target-pairs，就再抽下一批（自适应补量）。
  3. 对子生成带保护：
     - 每道题 chosen 去重（规范化签名）、封顶 3 个 distinct；
     - rejected：错误题用 greedy pred + 采样错样本（封顶 2）；已做对题用采样错样本（封顶 2）；
     - 每道题对数封顶 --max-pairs-per-q。
  4. chosen 只来自模型自身采样（strict 执行匹配 gold），严禁用 gold 直接当 chosen。

用法：
    python m4/analysis/step4b_expand.py --limit 24                    # 冒烟
    python m4/analysis/step4b_expand.py                                # 全量（后台）
"""
from __future__ import annotations

import argparse
import json
import sqlglot
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "m4"))

import torch  # noqa: E402
from eval_sft_adapter import load_model_and_adapter, SYSTEM, build_user  # noqa: E402
from schema import load_schemas, get_ddl  # noqa: E402
from run_baseline import SPIDER_DIR, execute_sql, extract_sql, results_equal_strict  # noqa: E402

ERR_IN = ROOT / "m4" / "analysis" / "step3_clean_errors.jsonl"      # 359 可用错误题（已筛）
REC_IN = ROOT / "m4" / "analysis" / "step2_train_records.jsonl"     # 8659 全量（含 correct 标记）
AUDIT_IN = ROOT / "m4" / "train_gold_audit.json"

OUT_PAIRS = ROOT / "m4" / "analysis" / "step4b_pairs.jsonl"
OUT_NOCHOSEN = ROOT / "m4" / "analysis" / "step4b_nochosen.jsonl"
OUT_REPORT = ROOT / "m4" / "analysis" / "step4b_report.txt"


def norm(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").split())


def sql_sig(sql: str) -> str:
    """规范化签名，用于 chosen/rejected 去重。"""
    try:
        return sqlglot.parse_one(sql, read="sqlite").sql(dialect="sqlite")
    except Exception:
        return norm(sql).lower()


def hardness(gold: str) -> int:
    """SQL 复杂度打分（越高越难）。仅作排序代理。"""
    try:
        t = sqlglot.parse_one(gold, read="sqlite")
    except Exception:
        return 0
    n_tables = len(list(t.find_all(sqlglot.exp.Table)))
    n_sub = len(list(t.find_all(sqlglot.exp.Subquery)))
    n_set = len(list(t.find_all(sqlglot.exp.Union, sqlglot.exp.Except, sqlglot.exp.Intersect)))
    n_cols = len(list(t.find_all(sqlglot.exp.Column)))
    has_gb = bool(next(t.find_all(sqlglot.exp.Group), None))
    has_nested = bool(next(t.find_all(sqlglot.exp.In), None))
    return n_tables * 2 + n_sub * 3 + n_set * 3 + n_cols + (2 if has_gb else 0) + (2 if has_nested else 0)


def _lines(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def gen_samples(model, tokenizer, prompt_text, inp, prompt_len, n, max_new_tokens, top_p):
    """采样 n 条，返回 (raw_list, pred_list)。"""
    raws, preds = [], []
    for _ in range(n):
        with torch.inference_mode():
            gen = model.generate(
                **inp, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=1.0, top_p=top_p, pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
        raws.append(raw)
        preds.append(extract_sql(raw))
    return raws, preds


def make_pairs(correct_sigs, wrong_sigs, greedy_pred, greedy_is_wrong, max_pairs):
    """correct_sigs/wrong_sigs 为已去重、含 (文本,签名) 的列表。返回 [(chosen, rejected), ...]
    rejected 候选 = greedy pred(若错) + 采样错样本，按签名去重，封顶 2。"""
    seen_sig = set()
    rcands = []
    if greedy_is_wrong and greedy_pred:
        s = sql_sig(greedy_pred)
        rcands.append((greedy_pred, s)); seen_sig.add(s)
    for w, sig in wrong_sigs:
        if sig not in seen_sig and len(rcands) < 2:
            rcands.append((w, sig)); seen_sig.add(sig)
    if not correct_sigs or not rcands:
        return []
    pairs = []
    for ch, _c in correct_sigs:
        if len(pairs) >= max_pairs:
            break
        pairs.append((ch, rcands[0][0]))
    if len(rcands) > 1 and len(pairs) < max_pairs:
        pairs.append((correct_sigs[0][0], rcands[1][0]))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-pairs", type=int, default=2000)
    ap.add_argument("--n-error", type=int, default=8)
    ap.add_argument("--n-correct", type=int, default=6)
    ap.add_argument("--max-correct", type=int, default=3000, help="已做对题最多采多少道")
    ap.add_argument("--batch-correct", type=int, default=500)
    ap.add_argument("--max-pairs-per-q", type=int, default=4)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="冒烟：只处理前 N 道错误题（不采正确题）")
    ap.add_argument("--resume", action="store_true", help="断点续跑：跳过已产出的 (source,db,question)")
    args = ap.parse_args()

    # ---- 断点续跑：加载已处理 key ----
    done = set()
    if args.resume:
        for p in (OUT_PAIRS, OUT_NOCHOSEN):
            for r in _lines(p):
                done.add((r.get("source", ""), r["db_id"], norm(r.get("question", ""))))
        print(f"resume: 已跳过 {len(done)} 道题", flush=True)

    schemas = load_schemas()
    audit = {}
    if AUDIT_IN.exists():
        for r in json.loads(AUDIT_IN.read_text(encoding="utf-8")):
            if isinstance(r, dict):
                audit[(r["db_id"], norm(r["gold"]))] = r
                audit.setdefault((r["db_id"], r["question"]), r)

    def is_gold_noise(db_id, question, gold):
        g = audit.get((db_id, norm(gold))) or audit.get((db_id, question))
        if not g:
            return False
        for fl in g.get("flags", []):
            if fl.startswith("empty_result") or fl.startswith("scalar_question_multirow") or fl.startswith("gold_exec_error"):
                return True
        return False

    # ---- 错误题 ----
    errors = [r for r in _lines(ERR_IN)]
    if args.limit is not None:
        errors = errors[: args.limit]
    # ---- 已做对题（correct=True 且 gold 非噪声），按难度降序 ----
    correct_pool = []
    for r in _lines(REC_IN):
        if r.get("correct") is not True:
            continue
        if is_gold_noise(r["db_id"], r["question"], r["gold"]):
            continue
        correct_pool.append(r)
    correct_pool.sort(key=lambda r: hardness(r["gold"]), reverse=True)

    tokenizer, model = load_model_and_adapter()
    t0 = time.time()

    mode = "a" if args.resume else "w"
    f_pairs = open(OUT_PAIRS, mode, encoding="utf-8")
    f_no = open(OUT_NOCHOSEN, mode, encoding="utf-8")
    stats = Counter()
    n_pairs = 0

    def process_one(e, greedy_is_wrong, n_samples):
        nonlocal n_pairs
        src = "error" if greedy_is_wrong else "correct"
        key = (src, e["db_id"], norm(e["question"]))
        if key in done:
            return
        done.add(key)
        db_id, question, gold = e["db_id"], e["question"], e["gold"]
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"
        ddl = get_ddl(schemas[db_id])
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": build_user(question, ddl)}]
        pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(pt, return_tensors="pt").to(model.device)
        plen = inp["input_ids"].shape[1]
        gold_rows, gold_err = execute_sql(db_path, gold)
        if gold_err is not None:
            return
        raws, preds = gen_samples(model, tokenizer, pt, inp, plen, n_samples, args.max_new_tokens, args.top_p)

        correct_sigs, wrong_sigs = [], []
        seen_c = set(); seen_w = set()
        for pred in preds:
            rows, err = execute_sql(db_path, pred)
            if err is not None:
                continue
            if results_equal_strict(gold_rows, rows):
                s = sql_sig(pred)
                if s not in seen_c and len(correct_sigs) < 3:
                    correct_sigs.append((pred, s)); seen_c.add(s)
            else:
                s = sql_sig(pred)
                if s not in seen_w and len(wrong_sigs) < 3:
                    wrong_sigs.append((pred, s)); seen_w.add(s)

        pairs = make_pairs(correct_sigs, wrong_sigs, e.get("pred"), greedy_is_wrong, args.max_pairs_per_q)
        if pairs:
            for chosen, rejected in pairs:
                f_pairs.write(json.dumps({
                    "db_id": db_id, "question": question, "gold": gold,
                    "prompt": messages, "chosen": chosen, "rejected": rejected,
                    "source": "error" if greedy_is_wrong else "correct",
                    "failure_type": e.get("failure_type"),
                }, ensure_ascii=False) + "\n")
            n_pairs += len(pairs)
            stats["paired_q"] += 1
        else:
            f_no.write(json.dumps({"db_id": db_id, "question": question, "gold": gold,
                                   "source": "error" if greedy_is_wrong else "correct"}, ensure_ascii=False) + "\n")
            stats["no_pair_q"] += 1
        stats["samples"] += n_samples

    # Phase 1: 错误题全采
    for i, e in enumerate(errors, 1):
        process_one(e, greedy_is_wrong=True, n_samples=args.n_error)
        if i % 20 == 0:
            f_pairs.flush(); f_no.flush()
            el = time.time() - t0
            print(f"[err {i}/{len(errors)}] pairs_so_far={n_pairs} elapsed={el:.0f}s", flush=True)
    stats["error_q"] = len(errors)

    # Phase 2: 已做对题按难度分批采，直到对数够或池空
    if args.limit is None:
        offset = 0
        batch_no = 0
        while offset < len(correct_pool) and offset < args.max_correct and n_pairs < args.target_pairs:
            batch = correct_pool[offset: offset + args.batch_correct]
            for i, e in enumerate(batch, 1):
                process_one(e, greedy_is_wrong=False, n_samples=args.n_correct)
            offset += len(batch)
            batch_no += 1
            f_pairs.flush(); f_no.flush()
            el = time.time() - t0
            print(f"[correct batch {batch_no} done] offset={offset} pairs={n_pairs} elapsed={el:.0f}s", flush=True)
        stats["correct_q"] = offset

    f_pairs.close(); f_no.close()
    recovered = stats.get("error_q", 0)
    lines = [
        "=" * 70, "M5 Step4b 扩展拒绝采样", "=" * 70,
        f"目标对数: {args.target_pairs}",
        f"错误题采样: {stats.get('error_q', 0)} 道 x N={args.n_error}",
        f"已做对题采样: {stats.get('correct_q', 0)} 道 x N={args.n_correct}",
        f"总采样次数: {stats.get('samples', 0)}",
        f"产出 DPO 对: {n_pairs}",
        f"有对的题数: {stats.get('paired_q', 0)}",
        f"无对题数: {stats.get('no_pair_q', 0)}",
        "",
        "注意：目标未达时要么调低 target，要么增大 --max-correct 继续采。",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"pairs -> {OUT_PAIRS.name}")


if __name__ == "__main__":
    main()