# Text2SQL Agent 与后训练优化

> 用 **Qwen2.5-Coder-7B**，通过 agent 闭环 + SFT/蒸馏，在 Spider dev 上以**单次前向 + 确定性修复**达到 **EA 0.8104**，超过 122B 教师模型 + 完整 agent 管线（**0.805**，平均 3.33 次调用/题）。

<sub>口径：EA = 执行准确率（gold/pred 执行结果集逐条比对，不是 SQL 文本全等）。0.8104 含零模型调用的确定性 exec 修复；"122B" 指外部 `qwen3.5-122b-a10b` 的 M3 全开 agent。两者共用同一 dev 集（1034 条）与执行管线，但分属不同模型规模——这是跨模型对比，严格同模型对照见「结果总览」。</sub>

## 快速开始

```bash
pip install -r requirements.txt

# 验证执行链（fixed/gold SQL，无需 API key）
python src/run_baseline.py --dry-run --limit 10

# 单个自然语言问题 → SQL
python src/agent.py --db concert_singer --question "How many singers are there?"
```

> 先配 `.env`（`cp .env.example .env` 填 key）并下载 Spider 数据（见「复现步骤」）。

## 项目结构

```
t2sql/
├── src/         核心库：schema 生成、SQL 执行、agent 反馈闭环、SQL 白名单安全层、降级重试、缓存、成本核算
├── training/    后训练：SFT(QLoRA)、DPO、RFT、GRPO、repair-aware SFT、LoRA 权重合并
├── scripts/     一次性脚本：评测 driver、失败归因、诊断、数据/泄漏审计、环境安装
├── reports/     报告与负结果证据链（FINAL_REPORT 等）
├── results/     评测数字（保留小最终版 JSON，大件已 gitignore）
├── adapters/    LoRA 权重（gitignored）
├── data/        训练数据产物（gitignored）
└── learning_notes.md  逐里程碑学习笔记（唯一事实源）
```

## 结果总览

主指标 **EA_strict**（执行准确率，分母 1034）：

| 方案 | EA_strict | 成本 | 说明 |
|---|---|---|---|
| M0 外部 122B naive prompt | 0.799 | 1.0 调用/题 | 起点 |
| M1.5 外部 122B 投票 n=5 | 0.796 | 5.0 | 负收益：采样多样性不足 + 系统性误解投票纠不回 |
| M3 外部 122B agent 全开 | 0.805 | 3.33 | 反馈闭环 + 自适应早停 |
| M4 本地 7B SFT 单次前向 | 0.7959 | 1 次前向 | 把 agent 管线蒸馏进单次生成 |
| **M5 本地 7B + 确定性修复** | **0.8104** | 1 次前向 | 零训练白捡净 +15 条 strict（ER 0.9836） |

**同模型（本地 7B）帕累托前沿**：agent 管道（0.7834 @ 5.16 次调用）→ SFT 单次前向（0.7959 @ 1 次前向）→ 单次前向 + 修复（0.8104 @ 1 次前向）。**蒸馏把 5× 调用压成 1×，准确率不降反升，帕累托前沿同时向右、向下移动。**

## 复现步骤

1. 下载 Spider 数据集到 `spider/`（gitignored；官网 https://yale-lily.github.io/spider），预期布局：
   `spider/dev.json`、`spider/tables.json`、`spider/train_spider.json`、`spider/train_others.json`、`spider/database/<db_id>/<db_id>.sqlite`（166 个库）。
2. 构造 SFT 数据（外部 122B teacher 生成「问题 → 正确 SQL」，默认产出 `m4/train_conservative.jsonl` + `m4/train_strict.jsonl`）：
   ```bash
   python training/build_train_set.py
   ```
3. 训练（QLoRA；默认基座/输出路径是原训练机硬编码，记得显式覆盖）：
   ```bash
   python training/train_sft.py \
       --base-model <Qwen2.5-Coder-7B-Instruct 路径> \
       --train-file m4/train_conservative.jsonl \
       --output-dir adapters/sft_cons_v1
   ```
4. 评测：
   ```bash
   python scripts/eval_sft_adapter.py --limit 100     # 本地 7B SFT 单次前向
   python scripts/eval_agent.py --limit 100           # 外部 122B 全 agent 管线
   ```

## 关键结论

- **三种对齐范式（DPO / RFT / GRPO）在同一基线失效，根因统一为「训练分布饱和」**：EA 0.8104 已是该 SFT 模型训练分布的局部最优，任何同分布微调都推不动最后那一两个点（DPO 零梯度、RFT 边际信息≈0、GRPO 熵塌缩）。
- **agent 的纠错能力可蒸馏进权重**（repair-aware SFT：自纠错 71% → 82%）。

详见 `reports/FINAL_REPORT.md`（面试精简版）与根目录 `learning_notes.md`（完整事实源）。