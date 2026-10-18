"""rft_build_train.py — M5 收尾后的 RFT round-2 训练集构造（分层加权）。

用 DPO 失败过程中已经采样的资产（step4b chosen / no-pair），构造新一轮 SFT：
  - base       : round-1 全量 gold-SFT（train_conservative.jsonl，8621 条）→ ×1
  - starchosen : 错误题里"rejection 采样能采到正确答案"的模型自身输出（chosen）→ ×2
  - hardgold   : 错误题里"采样 N 次也采不到正确"的 hard 题 gold → ×2
  - （correct 源 chosen/no-pair 一律跳过：greedy 已做对，纯锚点稀释，DPO v4b 已证明有害）

口径：
  - chosen 是模型自己的正确输出（STaR），不是 gold；hardgold 才是 gold（SFT 可用 gold）。
  - weight 通过物理重复实现（train_sft.py 的 load_train_file 只读 rec["message"]，
    直接重复行即可，无需改训练脚本）。
  - 每条输出记录带 layer/weight/db_id/question/gold 供溯源，训练时忽略这些字段。

用法：
    python m4/analysis/rft_build_train.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "m4"))

from build_train_set import to_chat  # noqa: E402   (与 round-1 完全同格式)
from schema import load_schemas, get_ddl  # noqa: E402

BASE_IN = ROOT / "m4" / "train_conservative.jsonl"
PAIRS_IN = ROOT / "m4" / "analysis" / "step4b_pairs.jsonl"
NOCHOSEN_IN = ROOT / "m4" / "analysis" / "step4b_nochosen.jsonl"

OUT_TRAIN = ROOT / "m4" / "analysis" / "rft_round2_train.jsonl"
OUT_MANIFEST = ROOT / "m4" / "analysis" / "rft_round2_manifest.json"

W_BASE = 1
W_CHOSEN = 2
W_HARDGOLD = 2


def read_lines(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    schemas = load_schemas()

    # ---- base（round-1 全量，直接复用其 message）----
    base = read_lines(BASE_IN)

    # ---- starchosen：error 源的 chosen ----
    pairs = read_lines(PAIRS_IN)
    chosen_recs = [p for p in pairs if p.get("source") == "error"]

    # ---- hardgold：error 源的 no-pair gold ----
    nochosen = read_lines(NOCHOSEN_IN)
    hard_recs = [p for p in nochosen if p.get("source") == "error"]

    total = 0
    counts = Counter()

    with OUT_TRAIN.open("w", encoding="utf-8") as f:
        def emit(layer: str, weight: int, db_id: str, question: str,
                 sql: str, message: dict, gold: str | None = None):
            nonlocal total
            rec = {
                "layer": layer,
                "weight": weight,
                "db_id": db_id,
                "question": question,
                "sql": sql,
                "gold": gold,
                "message": message,
            }
            for _ in range(weight):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
            counts[layer] += 1  # 每题只计一次（未乘权）

        for r in base:
            emit("base", W_BASE, r["db_id"], r["question"], r["gold"], r["message"], r["gold"])

        for r in chosen_recs:
            ddl = get_ddl(schemas[r["db_id"]])
            msg = to_chat(r["question"], ddl, r["chosen"])
            emit("starchosen", W_CHOSEN, r["db_id"], r["question"], r["chosen"], msg, r["gold"])

        for r in hard_recs:
            ddl = get_ddl(schemas[r["db_id"]])
            msg = to_chat(r["question"], ddl, r["gold"])
            emit("hardgold", W_HARDGOLD, r["db_id"], r["question"], r["gold"], msg, r["gold"])

    manifest = {
        "base_files": str(BASE_IN),
        "layers_unique": dict(counts),
        "weights": {"base": W_BASE, "starchosen": W_CHOSEN, "hardgold": W_HARDGOLD},
        "total_records_after_weighting": total,
        "unique_questions": len(base) + len(chosen_recs) + len(hard_recs),
        "chosen_from_error_questions": len({(r["db_id"], r["question"]) for r in chosen_recs}),
        "hardgold_from_error_questions": len(hard_recs),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print("RFT round-2 训练集构造完成")
    print("=" * 60)
    print(f"base         : {len(base)} 题 → ×{W_BASE} = {len(base) * W_BASE} 行")
    print(f"starchosen   : {len(chosen_recs)} 条 chosen → ×{W_CHOSEN} = {len(chosen_recs) * W_CHOSEN} 行"
          f"  (来自 {len({(r['db_id'], r['question']) for r in chosen_recs})} 道错误题)")
    print(f"hardgold     : {len(hard_recs)} 题 gold → ×{W_HARDGOLD} = {len(hard_recs) * W_HARDGOLD} 行")
    print(f"总计(加权重) : {total} 行")
    print(f"\n输出: {OUT_TRAIN}")
    print(f"清单: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()