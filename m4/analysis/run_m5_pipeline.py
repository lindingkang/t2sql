"""run_m5_pipeline.py — M5 通宵编排器：串行执行 Step3→4→5→6，逐步写日志与状态文件。

- 等待 Step2（SFT 单次 train 全量错误采集）跑完（轮询 summary 文件 + 进程存活）。
- Step3 纯 CPU 秒级；Step4 GPU 拒绝采样；Step5 DPO 训练；Step6 dev 评测。
- 每步独立日志，状态写入 m4/analysis/M5_STATUS.md，明早直接看。
- 任一步失败即停并写清原因（不级联）。

用法：
    python m4/analysis/run_m5_pipeline.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "m4" / "analysis"
STATUS = ANALYSIS / "M5_STATUS.md"
LOG_DIR = ANALYSIS / "pipeline_logs"

DEPOT = "/root/autodl-tmp/m4_lora"
DPO_ADAPTER = DEPOT + "/dpo_b01_v1"

STEP2_SUMMARY = ANALYSIS / "step2_train_summary.txt"
STEP2_RECORDS = ANALYSIS / "step2_train_records.jsonl"
TRAIN_TOTAL = 8659

MIN_PAIRS_FOR_DPO = 50


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (ANALYSIS / "pipeline_run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(section: str) -> None:
    with STATUS.open("a", encoding="utf-8") as f:
        f.write(section + "\n")


def run_step(cmd: list[str], step_name: str, timeout: int | None = None) -> int:
    log(f"=== {step_name} 开始 ===")
    log(f"cmd: {' '.join(cmd)}")
    logpath = LOG_DIR / f"{step_name}.log"
    with logpath.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              cwd=str(ROOT), timeout=timeout)
    rc = proc.returncode
    log(f"=== {step_name} 结束 rc={rc} (日志 {logpath}) ===")
    return rc


def tail(path: Path, n: int = 25) -> str:
    if not path.exists():
        return "(no file)"
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-n:])


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open(encoding="utf-8"))


def wait_step2(max_wait_s: int = 6 * 3600) -> bool:
    log("等待 Step2 完成…")
    t0 = time.time()
    last_n = 0
    stall_since = time.time()
    while time.time() - t0 < max_wait_s:
        if STEP2_SUMMARY.exists():
            log("检测到 Step2 summary 文件，Step2 完成")
            return True
        n = count_lines(STEP2_RECORDS)
        if n != last_n:
            last_n = n
            stall_since = time.time()
            log(f"Step2 进行中: {n}/{TRAIN_TOTAL} 条")
        elif time.time() - stall_since > 1200:
            # 20 分钟无进展且无 summary -> 可能挂了
            log(f"Step2 疑似卡住（{int((time.time()-stall_since)/60)} 分钟无进展）")
            # 检查进程是否还活着
            alive = subprocess.run(["pgrep", "-f", "step2_collect_train_errors"],
                                   capture_output=True).returncode == 0
            if not alive and n < TRAIN_TOTAL:
                log("Step2 进程已退出但未跑到 8659 条，视为失败")
                return False
        time.sleep(60)
    log("Step2 超时")
    return False


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(
        "# M5 通宵流水线状态\n\n"
        f"启动: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        encoding="utf-8",
    )

    # ---------- Step 3（等 Step2 queue；Step3 依赖 Step2 输出）----------
    if not wait_step2():
        status("## Step2 未成功完成，流水线中止\n")
        log("ABORT: Step2 未完成")
        return

    status(f"## Step2 完成 @ {time.strftime('%H:%M:%S')}\n```\n{tail(STEP2_SUMMARY, 30)}\n```\n")

    # ---------- Step 3 ----------
    rc = run_step([sys.executable, "m4/analysis/step3_screen.py"], "step3_screen")
    if rc != 0:
        status("## Step3 失败，中止\n")
        return
    status(f"## Step3 完成\n```\n{tail(ANALYSIS / 'step3_report.txt')}\n```\n")

    # ---------- Step 4 ----------
    rc = run_step([sys.executable, "m4/analysis/step4_rejection.py",
                   "--n-samples", "8", "--max-questions", "800"],
                  "step4_rejection", timeout=10 * 3600)
    if rc != 0:
        status("## Step4 失败，中止\n")
        return
    status(f"## Step4 完成\n```\n{tail(ANALYSIS / 'step4_report.txt')}\n```\n")

    n_pairs = count_lines(ANALYSIS / "step4_pairs.jsonl")
    if n_pairs < MIN_PAIRS_FOR_DPO:
        status(f"## Step5 跳过：偏好对仅 {n_pairs} 条（< {MIN_PAIRS_FOR_DPO}），DPO 信噪比不足\n")
        log(f"ABORT: pairs={n_pairs} 太少，跳过 DPO")
        return

    # ---------- Step 5：先冒烟后正式 ----------
    smoke_rc = run_step([sys.executable, "m4/analysis/step5_train_dpo.py",
                         "--limit", "64", "--max-steps", "2",
                         "--output-dir", DEPOT + "/dpo_smoke"],
                        "step5_smoke", timeout=2 * 3600)
    if smoke_rc != 0:
        status("## Step5 冒烟失败，中止（见 pipeline_logs/step5_smoke.log）\n")
        return
    status(f"## Step5 冒烟通过 @ {time.strftime('%H:%M:%S')}\n")

    rc = run_step([sys.executable, "m4/analysis/step5_train_dpo.py",
                   "--beta", "0.1", "--output-dir", DPO_ADAPTER],
                  "step5_dpo", timeout=12 * 3600)
    if rc != 0:
        status("## Step5 DPO 训练失败，中止（见 pipeline_logs/step5_dpo.log）\n")
        return
    status(f"## Step5 DPO 训练完成\n```\n{tail(ANALYSIS / 'step5_report.txt', 8)}\n```\n")

    # ---------- Step 6 ----------
    rc = run_step([sys.executable, "m4/analysis/step6_eval_dpo.py",
                   "--adapter", DPO_ADAPTER],
                  "step6_eval", timeout=4 * 3600)
    if rc != 0:
        status("## Step6 评测失败（见 pipeline_logs/step6_eval.log）\n")
        return
    status(f"## Step6 完成\n```\n{tail(ANALYSIS / 'dpo_eval_summary.txt', 40)}\n```\n")

    status("## 流水线全部完成 ✓\n")
    log("ALL DONE")


if __name__ == "__main__":
    main()