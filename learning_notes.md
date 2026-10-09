# Text2SQL Agent 项目学习笔记

> 每个里程碑的关键决策、踩过的坑、验证过的结论。面试复盘的素材库。

---

## M0：环境与基线

### 学到的东西

1. **Spider 数据集结构**
   - `train_spider.json`（7000 条）、`dev.json`（1034 条）、`tables.json`（166 个库的 schema）、`database/`（166 个 sqlite 库）
   - 核心字段：`db_id`、`question`、`query`（gold SQL）
   - `query_toks` / `sql` 字段是官方为 exact-match 指标准备的，我们用执行准确率所以不用

2. **Schema 转 DDL**（[schema.py](file:///root/t2sql/schema.py)）
   - 把 `tables.json` 转成给模型看的 CREATE TABLE 文本，标注主键，外键用注释说明
   - 设计决策：DDL 里带外键注释，让模型一眼看出 JOIN 路径

3. **执行准确率（Execution Accuracy）vs Exact Match**
   - Exact Match：SQL 文本完全一致才算对。会因为"写法不同但结果一样"误判
   - Execution Accuracy：执行 gold 和 pred，比较结果集（去重排序后 set 比较）。更贴近"可用性"
   - 选 EA 作为主指标，因为这也是 M4 可验证奖励的基础

4. **比较函数用 set 不用排序 list**
   - SQL 写法不同会产生相同数据但不同行序（有没有 ORDER BY）
   - set 比较的是"数据内容是否等价"，和 Spider 官方评测一致

5. **只读连接** `mode=ro`
   - 模型生成的 SQL 理论上是 SELECT，但万一出现写操作也碰不到磁盘
   - 这是 M3 安全层的地基

6. **text_factory 编码兜底**
   - Spider 的 wta_1 库里有西班牙/拉美球员名（如 Albarracín），用 Latin-1/cp1252 存的（í = 0xED）
   - 必须设 `con.text_factory = lambda b: b.decode("utf-8", errors="replace")`，否则执行时抛 UnicodeDecodeError

7. **提取 SQL 要支持多行**
   - 早期 `extract_sql` 只取第一行 SQL，导致模型输出多行 SQL 时被截断
   - 修复：从第一条 SQL 行取到文本末尾拼成单行；prompt 同时要求单行输出

### 踩过的坑
- `-- limit 10`（`--` 后多空格）会被 argparse 当成位置参数报错 → 必须 `--limit 10`
- AutoDL 不同区的网关地址不同（westb vs westc），SSH 连接失败先去控制台拿最新登录指令

---

## M1：Agent 骨架（执行反馈闭环）

### 学到的东西

1. **当前 Agent = 执行报错重试器**
   - 流程：生成 SQL → 执行 → 报错？→ 把报错喂回模型重写（最多 2 轮）→ 返回
   - 覆盖面积：1034 条里只有 8 条（0.8%）首轮执行报错，所以闭环触发面很窄
   - 结果：救回 3 条，EA 持平 79.9%

2. **生产环境没有 gold SQL**
   - Agent 不能"对比正确答案"来判断对错——用 gold 是评测，不是 agent
   - Agent 能用的反馈信号只有三种：
     ① 执行报错（覆盖面 0.8%，已做）
     ② 执行结果的语义检查（LLM 看结果判断是否回答了问题，有不确定性）
     ③ 自洽性投票（多采样 N 条 SQL，执行后按结果集多数决，不需要 gold）

3. **temperature=0 在千帆上并不确定**
   - 相同 prompt 两次跑会随机翻转约 7 条（噪声约 ±0.7%）
   - 以后每次对比实验要么多跑几轮取均值，要么确认平台支持固定 seed
   - 这条直接决定 M4 微调收益的可信度

4. **闭环的能力边界**
   - 修得好"数据库报错"，修不好"语义理解错"
   - 执行报错只是语法/列名错，语义错（如 INTERSECT 写成 UNION）SQL 能跑但答案错，agent 察觉不到

---

## M2：评测体系 + Prompt 工程

### 学到的东西

1. **失败归因要精准，不能靠规则启发式**
   - 第一版归因（[llm_diagnose.py](file:///root/t2sql/llm_diagnose.py) 批量归因 + [analyze_failures.py](file:///root/t2sql/analyze_failures.py) 规则分类）质量很低，很多归因错误
   - 改进：[precise_diagnose.py](file:///root/t2sql/precise_diagnose.py) 对每条 badcase **单独调用 LLM**，输入包含完整 schema DDL、question、gold/pred SQL 及其执行结果
   - 分类的意义不是"贴标签"，而是"知道在 prompt 里写什么能防错"

2. **真实的失败分布（207 条精准归因）**

| 类型 | 条数 | 占比 | 说明 |
|------|------|------|------|
| semantic_misunderstanding | 69 | 33.3% | 模型误解了问题语义 |
| data_description_gap | 45 | 21.7% | 不知道数据存储细节（大小写、类型、格式） |
| join_error | 36 | 17.4% | JOIN 多了/少了/类型错 |
| aggregation_error | 25 | 12.1% | GROUP BY / HAVING / 子查询错 |
| other (column_order) | 24 | 11.6% | 列顺序伪错，宽松指标可消除 |
| set_logic_error | 4 | 1.9% | INTERSECT/UNION 语义错 |
| gold_noise | 4 | 1.9% | gold SQL 本身有错（毒样本，M4 要剔除） |

3. **Gold 噪声甄别**
   - 207 条里发现 9 条 gold 本身有问题（`is_gold_noise=True`）
   - 例如 battle_death 库：问题问 "name, date and result"，但 gold SQL 只 SELECT 了 name, date
   - 这些是毒样本，M4 训练时必须剔除，否则会把模型带偏

4. **Prompt 工程已触顶（核心结论）**

   | 变体 | EA_strict | EA_loose | run 次数 |
   |------|-----------|----------|----------|
   | baseline（M0 原版） | 0.799 | 0.839 | 1 |
   | improved_v1（5 条泛泛规则） | 0.805 ± 0.003 | 0.851 | 2 |
   | improved_v2（11 条精准规则） | 0.798 ± 0.002 | 0.842 | 2 |

   - v2 的 11 条规则每条都对应精准归因里的 fix_prompt，但 EA 不仅没提升，反而略降
   - 原因：
     ① 模型本来就知道这些规则，加更多规则只会让 prompt 变长、注意力被稀释
     ② 长 prompt 可能起反作用：v2 的 column_order 略好（48→45），但 join_type 变差（35→39）
     ③ 真正的瓶颈是语义理解（69 条 semantic_misunderstanding），规则救不了——模型不是"不知道怎么做"，而是"没读懂题"

5. **评测实验的严谨性**
   - 单次运行的数字不可信（±0.7% 噪声）
   - 必须跑多次算 mean ± std，差距 > 2σ 才算显著
   - [eval_harness.py](file:///root/t2sql/eval_harness.py) 的 `--summarize` 要区分全量和冒烟测试，否则 std 会被拉偏

### 面试可以讲的点
- "我对 207 条失败样本做了逐条 LLM 归因，识别出 7 类错误模式，据此设计了两版 prompt（v1 泛化规则、v2 精准规则），各跑 2 次全量对照实验。结果 EA 从 0.799 到 0.805 到 0.798，在 ±0.003 噪声内无显著变化。这证明对当前模型，prompt 工程已触顶——模型不是缺规则，而是缺语义理解能力。这把优化方向从 M2 prompt 工程推向了 M3 自洽性投票和 M4 微调。"

---

## M1.5：自洽性投票（Consensus Voting）

### 实验设计
- 对每个问题采样 N=5 条 SQL（temperature=0.7），各自执行
- 按结果集（宽松签名，忽略行/列顺序）聚类，取最大簇的代表 SQL
- 平票选最短 SQL，无共识（所有簇大小=1）fallback 到 M1 重试闭环

### 结果
| 指标 | M0 baseline | M1.5 投票 (n=5) | 变化 |
|------|-------------|-----------------|------|
| EA_strict | 0.799 | **0.796** | -0.3% |
| rescued | - | 10 | - |
| regressed | - | 9 | - |
| 平均调用/题 | 1.0 | 5.0 | 5x |

### 核心结论：自洽性投票没有提升

**为什么 5 倍 API 开销换不来提升？**

1. **多样性不足**：temperature=0.7 对当前模型（qwen3.5-122b）产生的 SQL 多样性不够。大多数问题 5 条采样的结果完全相同（最大簇大小=5），投票退化为"5 次重复生成同一个答案"。

2. **系统性错误无法被投票纠正**：对于 semantic_misunderstanding（33% 的失败），模型不是"随机猜错"，而是"系统性误解"——5 条采样会以同样的方式理解错，多数派仍然是错的。投票只能纠正"随机噪声"，纠正不了"系统性偏差"。

3. **温度的副作用**：9 条 regressed（M0 对、M1.5 错）说明 temperature=0.7 反而让某些简单题的输出不稳定，多数派偶尔选错。

4. **投票的前提不成立**：自洽性投票的理论假设是"正确 SQL 结果一致，错误 SQL 结果五花八门"。但实测发现：对当前模型，**错误 SQL 之间也高度一致**（因为是同一类系统性误解），所以投票没有区分度。

### 改进方向（留给 M3）
- 提高多样性：temperature 调高到 1.0+，或用不同 prompt 模板采样
- 加权投票：不只是数条数，而是对 SQL 做质量评分（如是否使用了 schema 中的列、是否简洁）
- 结果验证：不只投票，还可以让 LLM 看执行结果 + 问题判断"这个结果是否合理"（但这引入 LLM 判断不确定性）

### 工程改进
- `call_llm` 加了 `signal.alarm` 硬超时兜底（httpx 的 timeout 在 TCP 连接阶段可能不生效，导致进程永久挂起）
- `eval_agent.py` 每 50 条 checkpoint 保存，中途断了不丢数据
- `eval_agent.py` 记录 `votes` 字段，便于事后分析投票细节
