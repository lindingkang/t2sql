"""统一运行本地未微调 7B 的 M0 / M1.5 / M2 / M3 基准。

用法：
    python m4/run_local_base_all.py --smoke              # 只跑 4 个模式的冒烟
    python m4/run_local_base_all.py                      # 只跑 4 个模式的全量
    python m4/run_local_base_all.py --smoke-then-full   # 冒烟成功后再跑全量

脚本按顺序执行：m0 -> m2 -> m15 -> m3。
每个阶段完成都会把摘要追加到 m4/local_base_all.log。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "m4" / "eval_local_base.py"
MODES = ["m0", "m2", "m15", "m3"]


def build_cmd(mode: str, limit: int | None, n_samples: int,
              smoke: bool) -> list[str]:
    tag = f"local_base_{mode}{'_smoke' if smoke else ''}"
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--mode", mode,
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if mode in ("m15", "m3"):
        cmd += ["--n-samples", str(n_samples)]
    cmd += [
        "--output-json", str(ROOT / "m4" / f"{tag}.json"),
        "--output-txt", str(ROOT / "m4" / f"{tag}_summary.txt"),
    ]
    return cmd


def run_batch(limit: int | None, n_samples: int, smoke: bool,
              log_path: Path) -> bool:
    all_ok = True
    with log_path.open("a", encoding="utf-8") as logf:
        header = (f"\n{'='*70}\n"
                  f"BATCH {'smoke' if smoke else 'full'} @ "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"limit={limit} n_samples={n_samples}\n"
                  f"{'='*70}\n")
        logf.write(header)
        logf.flush()

        for mode in MODES:
            tag = f"local_base_{mode}{'_smoke' if smoke else ''}"
            cmd = build_cmd(mode, limit, n_samples, smoke)
            logf.write(f"\n>>> START {mode}: cmd={' '.join(cmd)}\n")
            logf.flush()
            t0 = time.time()
            rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT).returncode
            dt = time.time() - t0
            logf.write(f"<<< END {mode}: rc={rc} time_s={dt:.1f}\n")
            logf.flush()
            if rc != 0:
                all_ok = False
                logf.write(f"!!! {mode} failed with rc={rc}, stopping batch.\n")
                logf.flush()
                break
    return all_ok


def write_combined_summary(smoke: bool, out_path: Path) -> None:
    lines = [
        "# local base all-modes combined summary",
        f"phase={'smoke' if smoke else 'full'}",
        "",
    ]
    for mode in MODES:
        tag = f"local_base_{mode}{'_smoke' if smoke else ''}"
        summary = ROOT / "m4" / f"{tag}_summary.txt"
        if summary.exists():
            lines.append(summary.read_text(encoding="utf-8").strip())
        else:
            lines.append(f"{mode}: MISSING summary")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="全量测试条数；默认 1034（None 表示全量）")
    ap.add_argument("--n-samples", type=int, default=5,
                    help="M1.5/M3 投票采样数")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-then-full", action="store_true")
    args = ap.parse_args()

    log_path = ROOT / "m4" / "local_base_all.log"
    log_path.write_text(
        f"# local base all-modes runner @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )

    if args.smoke or args.smoke_then_full:
        print("run smoke batch: all modes, limit=5, n_samples=3", flush=True)
        smoke_ok = run_batch(limit=5, n_samples=3, smoke=True, log_path=log_path)
        write_combined_summary(smoke=True,
                               out_path=ROOT / "m4" / "local_base_all_smoke_summary.txt")
        print(f"smoke_ok={smoke_ok}", flush=True)
        if not smoke_ok:
            print("SKIP full: smoke failed", flush=True)
            return
        if args.smoke:
            print("smoke done (not running full)", flush=True)
            return

    print(f"run full batch: all modes, limit={args.limit}, n_samples={args.n_samples}",
          flush=True)
    full_ok = run_batch(limit=args.limit, n_samples=args.n_samples,
                        smoke=False, log_path=log_path)
    write_combined_summary(smoke=False,
                           out_path=ROOT / "m4" / "local_base_all_summary.txt")
    print(f"full_ok={full_ok}", flush=True)


if __name__ == "__main__":
    main()