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

---

## M3：工程化加固 + 成本控制

### 目标

把 M1/M1.5 的 agent 从"评测脚本"升级为"不可打穿数据库、不会失控烧钱"的工程形态。

### 组件（全部有独立 self-test）

1. **安全层** [`safety.py`](file:///root/t2sql/safety.py)
   - 执行前 AST 白名单，fail-closed：发现解析不清一律判危险。
   - 只允许 SELECT/UNION/INTERSECT/EXCEPT 单条查询；阻止 Insert/Update/Delete/Drop/Attach/Pragma/多语句。
   - 显式拦截危险函数 `load_extension/readfile/writefile`。
   - self-test 覆盖 28 条正常/危险样本，28/28 PASS。

2. **六类降级** [`degrade.py`](file:///root/t2sql/degrade.py)
   - 错误分类：`syntax_error`、`schema_error`、`timeout_error`、`row_limit_error`、`safety_blocked`、`constraint_error`、`type_error`、`unknown_error`。
   - `should_retry()` 内聚了硬上限 `current_attempt < min(global_max, type_max)`，`while True` 不会无限循环。
   - schema 错误提示是 schema-aware 的：拼错列给相似列（difflib），臆造列给被引用表的可用列清单。

3. **执行保护** [`run_baseline.py`](file:///root/t2sql/run_baseline.py)
   - sqlite `mode=ro` + 超时中断 + 行数上限 + Latin-1 编码兜底，防止一条坏 SQL 拖死整个评测进程。

4. **空结果二次确认** [`agent.py`](file:///root/t2sql/agent.py)
   - 只对"非聚合 0 行 / 聚合 0 或 NULL"且 SQL 带字面量 WHERE 的情况触发。
   - 采纳前必须通过 AST **结构锁**：表集合不变；去 WHERE/HAVING/LIMIT 且字面量归一化后骨架不变；过滤列集合不变。
   - 核心判断：空答案常是"值大小写/拼写不匹配"，而不是真的没数据；结构锁防止把正确空答案改坏。

5. **成本控制** [`cache.py`](file:///root/t2sql/cache.py) + 自适应投票
   - L1 缓存只缓存 `temperature=0` 的确定性调用；`temperature=0.7` 的投票采样绝不缓存，避免 5 次采样命中同一响应。
   - L2 题级结果缓存带配置指纹，配置不同不会错误命中。
   - token 预算守卫会截断超长 schema，同时保留相关表。
   - 自适应投票：多数决早停，`n=5` 时某簇拿到 3 票立即停，省去无意义采样。

### M3 全量实验结果（2026-10-11）

| 指标 | M0 baseline | M1.5 投票 | M3 全开 |
|------|-------------|-----------|---------|
| EA_strict | 0.799 | 0.796 | **0.805** |
| rescued | - | 10 | **20** |
| regressed | - | 9 | 10 |
| 平均 API 调用/题 | 1.0 | 5.0 | **3.33** |
| 投票有共识 | - | 未单独记录 | 970 |
| 投票无共识 fallback | - | 未单独记录 | 64 |
| 空结果确认触发 | - | - | 77 |
| 空结果确认采纳 | - | - | 36 |

- 模型：`qwen3.5-122b-a10b`
- token 总消耗：1,440,236
- L1 缓存命中：0（单遍每题都是新问题，不会命中；缓存价值在重复评测/跨 run 复用）

### 关键结论

1. **M3 全开 + 自适应投票是正收益**
   - EA 从 M0 0.799 升到 **0.805**；M1.5 固定采 5 条反而 0.796。
   - 关键差异在自适应早停：大多数题 3 票即共识，平均调用 **3.33** 次/题，比固定 5 次省约 1/3。

2. **投票从"负收益"变成"正收益"的两个修复**
   - 之前空结果签名 `frozenset()` 会把"值大小写不匹配导致 0 行"聚成假共识；改为 `signature=None` 的空结果不参与聚类。
   - 之前平票只从 `top_clusters[0]` 选最短 SQL；改为展平所有最大簇再选。

3. **空结果确认是精确的救援工具**
   - 触发 77 次，采纳 36 次，说明约一半的空结果确实由值不匹配造成。
   - 结构锁保底：不会因为复查改坏表和 JOIN；本次没有大范围 regressed。

4. **成本控制的度量**
   - 平均 3.33 次调用/题，总 token 1.44M，对 1034 题属于可接受范围。
   - `--cache` 在单次全量评测中无命中，后续做多 run 对比/重复加载时才会体现价值。

### 本次自检修掉的 bug

1. `_result_set_signature` 用 `type(v).__name__` 导致 `30` 与 `30.0`、`France` 与 `france` 被错误拆簇。
2. `cluster_now` 把空结果 `frozenset()` 也聚成簇，造成空答案假性多数。
3. 平票只取 `top_clusters[0]`，改成展平所有最大簇。
4. `eval_agent.py` 支持 `--adaptive` / `--cache` / `--no-confirm-empty` / `--log-dir`，并记录 cost/meter/empty_confirm。

### 面试可以讲的点

> "M3 我把 agent 从能跑做到了能安全跑：安全层用 SQL AST 白名单 fail-closed，只允许只读单查询；执行层有超时、行上限和只读连接。反馈层按错误类型差异化重试；空结果二次确认带 AST 结构锁，不被 LLM 改坏结构。成本上自适应投票用多数决早停，把平均调用从 5 次压到 3.33 次。最终 EA 从 M0 0.799 升到 0.805，净救回 20 条，是第一次在提升准确率的同时压住了成本。"

---

## M4：SFT/QLoRA 与同模型公平对比

### 关键修正：同模型公平基准
- 之前的 M0 / M1.5 / M3 数字来自外部大模型（qwen3.5-122b），但 M4 SFT 后是本地 Qwen2.5-Coder-7B-Instruct + LoRA。两者不能直接对比。
- 因此用同一个本地 7B base，按 `m4/eval_local_base.py --limit 1034` 复跑 `M0 / M2 / M1.5 / M3`，和 `m4/sft_eval_result.json` 同口径对比。
- 全量等于 Spider dev 全量：1034 条。

### 口径（每个值都有分子/分母）
- `ER` = `exec_ok / 1034`：预测 SQL 执行成功比例。
- `EA_strict` = `正确条数 / 1034`，要求 gold 与 pred 的执行结果**按列顺序严格一致**（列顺序不同判错）。
- `EA_loose` = `正确条数 / 1034`，只看最终取值是否一致，**忽略列顺序**。

### 最终同模型对照表（分母统一 = 1034）

| 方案 | ER | EA_strict | EA_loose |
|------|----|-----------|----------|
| M0 local base | 956/1034 = 0.9246 | 759/1034 = 0.7340 | 787/1034 = 0.7611 |
| M2 local base improved_v2 prompt | 982/1034 = 0.9497 | 765/1034 = 0.7398 | 787/1034 = 0.7611 |
| M1.5 local base 投票 | 1023/1034 = 0.9894 | 797/1034 = 0.7708 | 830/1034 = 0.8027 |
| M3 local base agent | 1023/1034 = 0.9894 | 810/1034 = 0.7834 | 845/1034 = 0.8172 |
| M4 SFT（sft_cons_v1） | 992/1034 = 0.9594 | 823/1034 = 0.7959 | 851/1034 = 0.8230 |

### 成本叙事（这才是 M4 SFT 的价值）
- 同模型本地基准里，M3 不是早期的 3.33 次/题（那是另一个模型的旧结果）；本地复跑实际是：
  - `total_calls = 5331`，`5331 / 1034 = 5.16` 次/题。
- M4 SFT 每道题只做 **1 次前向**。
- 结论：本地 7B 单次前向 EA_strict `823/1034 = 0.7959`，略高于本地 agent 管道（5.16 次调用 + 重试 + 投票）的 `810/1034 = 0.7834`。
- 本质是「把 agent 管线的能力蒸馏进单次生成」。这是完整的部署故事，比「涨了 1.26 个点」更有说服力。

### M4 SFT 训练细节
- adapter：`/root/autodl-tmp/m4_lora/sft_cons_v1`
- 数据源：只来自 `train_spider.json + train_others.json`，`dev.json` 从未进入训练。
- QLoRA：4bit NF4 + double quant，LoRA `r=16 / alpha=32 / dropout=0.05`，target_modules 覆盖 q/k/v/o/gate/up/down projection。
- 监督方式：只对 assistant 输出计算交叉熵，system/user prompt 的 labels 用 `-100` 掩码。

### 保守版 SFT vs 严格版 SFT
两者共同点：都会剔除 `exec_ok=False` 的 gold 样本；都按 `(db_id, question, gold_sql)` 内部去重。

| 版本 | 剔除规则 | 文件 | 样本数 |
|------|----------|------|--------|
| 保守版 | 只剔 `exec_ok=False` | `m4/train_conservative.jsonl` | 8621 |
| 严格版 | 再剔 `empty_result` 和 `scalar_question_multirow` | `m4/train_strict.jsonl` | 7005 |

- 本次训练和评测用的是**保守版**：`train_conservative.jsonl`。
- `empty_result`：gold SQL 执行返回 0 行。
- `scalar_question_multirow`：问题在语义上是「单个标量」，gold 却返回了多行。

### Dev 泄漏审计
- 脚本：`m4/final_leak_audit.py`，输出 `m4/leak_audit_final.json`。
- 采样规范化后的 question / SQL 做四档检查（保守版 8621 条 vs dev 1034 条，严格版同结论）：

| 重叠类型 | 保守版 | 严格版 |
|----------|--------|--------|
| `(db_id, question, gold_sql)` 完全一致 | 0 | 0 |
| 同一数据库 + 同一问题问法 | 0 | 0 |
| 仅 question 字符串相同（不同数据库） | 7 | 7 |
| 仅 gold SQL 字符串相同（平凡 count(*) 等） | 10 | 10 |

- 结论：真泄漏（同库同题同 SQL）为 0。7 条 question-only 是 Spider train/dev 按数据库切分造成的同问法但不同库，10 条 sql-only 是 `SELECT count(*) FROM table` 这类平凡 SQL，均不构成 dev 泄漏。

### 逐阶段优化 / 退化案例
- 脚本：`m4/generate_case_transitions.py`
- 明细：`m4/case_transitions.txt`（含每条的 before/after SQL 和 `strict / loose / exec / failure_type` 状态）
- 摘要：`m4/case_transitions.json`

严格口径下的阶段变化：

| 阶段 | 优化 case | 退化 case |
|------|-----------|-----------|
| M0 → M2 | 60 | 54 |
| M2 → M1.5 | 82 | 50 |
| M1.5 → M3 | 51 | 38 |
| M3 → M4 SFT | 67 | 54 |

回看案例时务必同时看 `loose` 标记：有些「退化」可能只是列顺序变化，属于宽松口径下的伪退化。

---

## M4.5：SFT + 自洽性投票杂交（Hybrid Voting）

### 实验设计
- 把 M1.5/M3 的自洽性投票，从 base 模型搬到 M4 SFT adapter 上，目标是用多采样补齐 SFT 单次 `exec_fail` 的短板。
- 流程：首批 `temperature=0.7` 采 5 条 → 执行后按**松散结果签名**聚簇（忽略行/列顺序、数值类型差异）。
- 若首批签名簇数 `<= 2`，说明多样性不足，抬温到 `0.97` 逐条补采，直至签名簇数达到 3 或补采达到 10 条上限。
- 合流后簇内 `>= 2` 才算共识，取最短 SQL；无共识退回首条可执行 SQL。
- 脚本：[eval_sft_hybrid.py](file:///root/t2sql/m4/eval_sft_hybrid.py)，结果：`m4/sft_hybrid_result.json`。

### 结果（口径同 M4，分母统一 = 1034）

| 方案 | ER | EA_strict | EA_loose | 调用/题 |
|------|----|-----------|----------|--------|
| M4 SFT single | 0.9594 | 823/1034 = 0.7959 | 851/1034 = 0.8230 | 1.0 |
| **M4.5 hybrid** | **1.0000** | **849/1034 = 0.8211** | **883/1034 = 0.8540** | 14.31 |

- 相对 M4 SFT：**strict +2.52pp，loose +3.09pp**；相对 M3 agent：**strict +3.77pp**。
- `exec_rate: 1.0000`：投票让 1034 题**全部至少有一条可执行 SQL**，SFT 单次的 42 条执行失败被兜底覆盖。

### 收益分解（本次的核心结论）

**相对 SFT single：rescued 41 / regressed 15 / 净 +26**

| SFT 单次状态 | 条数 | 占比 | 含义 |
|---|---|---|---|
| `exec_fail`（执行都失败） | 25 | 61% | 纯执行兜底 |
| `exec_ok_wrong`（能执行但结果错） | 16 | 39% | 结果纠错 |

-> **超过六成收益来自「执行救援」而非「语义纠正」**。

15 条 regressed 的构成：`column_order=7 / other=6 / join_type=2`。

**相对 M3 local base：rescued 73 / regressed 34 / 净 +39**

| M3 状态 | 条数 |
|---|---|
| `exec_ok_wrong` | 65 |
| `exec_fail` | 8 |

-> 相对 M3 的提升主要由 **SFT 模型本身的语义能力**贡献，投票只补 8 条执行缺口。

34 条 regressed 的构成：`column_order=12 / other=12 / join_type=9 / aggregate=1`。

### 多样性诊断：M1.5 的老问题没有消失
- `temp_up_samples = 1007/1034 = 97.4%`：几乎每题首批 5 条的结果签名簇数都 `<= 2`，全部触发抬温补采。
- `avg_distinct_total = 1.41`；`distinct_total_counts = {1: 736, 2: 185, 3: 104, 4: 7, 5: 2}`。
- **71% 的题最终只有一种执行结果签名**：SFT 后模型分布高度尖峰，错误 SQL 之间也高度一致，投票的语义区分度依然很低。

### 成本与记账问题
- `avg_calls_per_sample = 14.31`，是 SFT 单次的 14.3 倍、M3 agent 的 2.77 倍；`total_new_tokens = 447266`，正好也是 14 倍。
- 抬温补采几乎跑满（`avg_pool_size = 14.31 ≈ 5 + 9.3`），说明模型已经尖峰时，抬温再采 10 条大部分是白采。
- **记账 bug**：`sft_hybrid_summary.txt` 里 `total_prompt_tokens=373619` 与 `total_calls=14795` 不一致。脚本每题只累加一次 `prompt_len`，没有按采样次数累乘。按 `361 token/call` 估算，真实 prompt token 约 `361 x 14795 = 5.34M`。

### 核心结论

> **M4.5 是一次执行层保险的正收益，不是语义层投票的正收益。**
> 在 M4 SFT 单次生成（0.7959，1 次前向）之上叠加自洽性投票，EA_strict 升到 0.8211，ER 从 0.9594 拉满到 1.0。但拆解 41 条 rescued 发现 61% 是多采样执行兜底（SFT 单次执行失败 -> 至少一条采样可执行），只有 39% 是执行成功但结果错的纠正。多样性诊断显示 97.4% 的题触发抬温、71% 的题最终只有一种执行签名——SFT 后分布依然尖峰，所以投票的语义纠错空间很小。净收益 +2.5pp 的代价是 14.3 倍调用成本，且 15 条 regressed 中约一半是 loose 签名聚簇引入的 column_order。

### 改进方向
- 若要继续提升，方向不是加大采样，而是：
  1. **打破 SFT 分布尖峰**：DPO / 多样性正则 / 训练时温度扰动。
  2. **直接修复 SFT 单次那 42 条执行失败**：它们贡献了 25/41 的 rescued，说明执行健壮性是 SFT 最明显的短板。
  3. **投票签名改成严格口径**：当前 loose 签名聚簇会系统性采纳列序错误的多数派，产生 12 条 column_order regressed。

