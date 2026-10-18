# Text2SQL Agent（Spider）

一个从零实现的 Text2SQL / 执行反馈 Agent，完整跑通 **baseline → prompt 工程 → agent 反馈闭环 → SFT（QLoRA）→ 对齐（DPO / RFT / GRPO）→ 自纠错蒸馏** 一整条链路。以 Spider benchmark 的 dev 集（1034 条 / 166 个 sqlite 库）为评测对象，主指标是**执行准确率（Execution Accuracy，EA）**。

## 结果一览

| 方案 | EA_strict | 说明 |
|---|---|---|
| M0 基线（qwen3.5-122b，naive prompt） | 0.799 | 起点 |
| M3 Agent（反馈闭环 + 自适应投票） | 0.805 | 第一次「涨准确率的同时压住成本」（3.33 调用/题） |
| M4 SFT（本地 7B，单次前向） | 0.7959 | 把 agent 管线蒸馏进单次生成 |
| M5 确定性 exec 修复（零训练） | **0.8104** | 白捡 +15 strict，ER 0.9836 |
| M4.5 SFT + 自洽性投票杂交 | 0.8211 | 代价 14.3× 调用，收益 61% 是执行兜底 |
| M6 repair-aware SFT | 自纠错 71%→82% | 把「act→observe→correct」蒸馏进权重 |

> 更严谨的指标定义（ER / EA_strict / EA_loose 的口径与分母）、以及 DPO / RFT / GRPO 三种对齐范式的**负结果与根因**，见 `reports/` 与根目录 `learning_notes.md`（本项目的事实源）。

## 做什么

自然语言 → SQL。核心难点不在语法，而在**语义理解**（约 33% 的失败是 `semantic_misunderstanding`）、跨表 JOIN、以及字段大小写/格式导致的「数据描述缺口」。

项目主线不是刷榜，而是三件事：

1. **把 agent 做到「能安全地跑」**：SQL AST 白名单 fail-closed、只读连接 + 超时 + 行数上限、按错误类型差异化重试、空结果二次确认（带 AST 结构锁，防止把正确空答案改坏）。
2. **可复现的评测**：对每条失败样本逐条 LLM 归因；同一模型公平基准；本地 greedy 求值 vs API 非确定（temperature=0 也有 ±0.7pp 噪声）的取舍。
3. **诚实的归因**：三个对齐范式的失败根因全部挖到底（见下）。

## 关键发现（含负结果）

**DPO / RFT / GRPO 三范式撞上同一条墙**：0.8104 的 SFT 模型已经是其训练分布的局部最优，任何同分布微调都推不动最后那一两个点。

| 范式 | 现象 | 根因 |
|---|---|---|
| DPO（偏好式） | EA 与基线逐字节一致 | 加载顺序 bug 冻死 LoRA（0 可训练参数）；修好后仍只学 token 措辞、不迁移语义 |
| RFT / STaR（重放式） | 最佳 +0.10pp ≈ 无；干净对照 −0.19pp | 样本全是模型自己 train 错误里 pass 出来的，round-1 已训过，边际信息≈0 |
| GRPO（在线式） | loss→1e-9、entropy 0.10→0.015 | on-distribution 饱和 + 熵塌缩 → 零梯度 |

## 项目结构

```
t2sql/
├── src/                 核心库（schema、执行、agent、安全、降级、缓存、日志、成本）
├── training/            后训练（SFT / DPO / RFT / GRPO / repair-aware SFT + 权重合并）
├── scripts/             一次性分析、评测、诊断、环境脚本
├── reports/             报告与负结果总结
├── results/             评测数字（大 JSON 已 gitignore）
├── adapters/            LoRA 权重（gitignored，需自行训练/下载）
├── data/                训练数据构造产物（gitignored）
└── learning_notes.md    逐里程碑学习笔记（唯一事实源）
```

## 快速开始

**依赖**：
```bash
pip install -r requirements.txt          # 核心（推理 + 评测）
python scripts/install_env.py            # 后训练依赖（需 GPU）
```

**环境**：复制 `.env.example` 为 `.env`，填入 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`（`.env` 已 gitignore）。

**数据**：Spider 数据集自行下载到 `spider/`（gitignored），预期布局 `spider/dev.json`、`spider/tables.json`、`spider/database/<db_id>/<db_id>.sqlite`。

**运行**（从仓库根目录执行，核心模块在 `src/` 下，脚本通过 `PYTHONPATH=src` 引用）：
```bash
# 验证执行链（fixed/gold SQL，无需 API key）
PYTHONPATH=src python src/run_baseline.py --dry-run --limit 10

# 单题 agent
PYTHONPATH=src python src/agent.py --db concert_singer --question "How many singers ...?"

# 全量 agent 评测
PYTHONPATH=src python scripts/eval_agent.py --limit 20

# A/B 评测
PYTHONPATH=src python scripts/eval_harness.py --variant baseline|improved --runs 3 --tag X
```

## 里程碑

M0 基线 → M1 执行反馈闭环 → M1.5 自洽性投票 → M2 prompt 工程 + 精准归因 → M3 工程加固 → M4 SFT/QLoRA → M4.5 投票杂交 → M5 确定性 exec 修复 + DPO/RFT → M6 repair-aware SFT + GRPO。

每个里程碑的「学到什么 / 踩了什么坑 / 量化结果」都记录在 `learning_notes.md`；面向面试的精简总结见 `reports/FINAL_REPORT.md`。