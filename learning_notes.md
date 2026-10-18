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

## M5：确定性 exec 修复 + DPO 规划（进行中）

### 口径修正（关键）：DPO 偏好对来自 train，不是 dev
- dev 是唯一评测集，只能测不能训。DPO 的 `rejected`/`chosen` 必须从训练集（`train_spider` 7000 + `train_others` 1659 = 8659 条）拒绝采样构造；dev 只在最后做配对检验。
- 目标不变：让 SFT 单次 `EA_strict ≥ 0.82`，摆脱 M4.5 的 14.31 倍采样。

### Step 1 — 确定性 exec 修复（无训练，无 LLM，已完成）
- 对 SFT 单次生成做后处理，只改"正确性有把握"的三类机械错误：
  1. **双引号字符串字面量 → 单引号**（SQLite 里双引号是标识符）：`"dog"` → `'dog'`。
  2. **表/列名大小写、单复数、笔误**：`SECTION`→`Sections`、`pet_type`→`PetType`、`IndependantYear`→`IndepYear`、`Makeid`→`MakeId`。
  3. **跨表列重指派**（列只在另一个已 join 表里有）：`T2.document_name`→`T1.Document_Name`、`T2.Name`→`T3.Name`。
- 实现见 `m4/analysis/exec_repair.py`（sqlglot AST 改写 + difflib 模糊，宁可不改也不改错）；逐条明细 `m4/analysis/step1_repair_result.json`。

### Step 1 结果（dev 1034，分母统一 1034）

| 指标 | 修复前 (M4 SFT) | 修复后 | Δ |
|---|---|---|---|
| ER | 992/1034 = 0.9594 | **1017/1034 = 0.9836** | +25 |
| EA_strict | 823/1034 = 0.7959 | **838/1034 = 0.8104** | +15 |
| EA_loose | 851/1034 = 0.8230 | **866/1034 = 0.8375** | +15 |

- 42 条 exec_fail：**25 条修复到可执行**，其中 15 条严格判对（确定性收益）、10 条仍错（可执行但另含 EXCEPT/聚合/值大小写等模型自身错误）；**17 条正确跳过**（缺 JOIN / 幻觉表 / 列归错表的结构错误，留给 DPO）。
- **结论**：确定性修复白捡 +25 ER / +15 EA_strict，零训练成本。**基线从 0.7959 抬到 0.8104，离 0.82 还差 ~10 条（约 1pp）**，由 DPO 补齐。

### 后续 5 步（规划）
- **Step 2** SFT 单次 + exec 修复跑 train 全量，采集错误清单（预计 1000~1500 条）。
- **Step 3** train gold 噪声筛查：确定性启发式（gold 执行错 / 空结果 / 标量多行）先剔，再对"exec 成功但结果错"的暧昧题用 LLM 二次确认 gold 是否真正确（判 NO 才剔，fail-open 不阻塞；软信号如 count-vs-sum 词频只用于排序不做硬杀）。
- **Step 4** 对错误题 temp=1.0 采 8~16 条拒绝采样，exec-ok 且 strict 匹配 gold 的作 `chosen`；**严禁用 gold 直接当 chosen**。
- **Step 5** DPO 训练混入保留 SFT 数据，β∈{0.1,0.3,0.5} 消融，**同时盯 EA 和 ER**（DPO 常见副作用是 ER 掉）。
- **Step 6** 只在 dev 配对检验（McNemar），千帆 temp=0 非确定、多跑取 mean±std、>2σ 才算显著。

### M5 执行结果（Step2–Step6 已完成，DPO 训练失败——梯度为零，待修）

#### 工程量改动（相对原计划）
- Step4 由"只采 359 道错误题"扩展为 **Step4b**：错误题全采（359×N8）+ 已做对题按 SQL 复杂度从难到易分层采（2000×N6），自适应补量直到对数达标。共 14848 次采样，产出 **2047 对**（808 题成对 / 1547 题无对；错误题产能 ~2.5 对/题，已做对题产能 ~0.7 对/题且随难度下滑）。

#### 结果：两版 DPO 全部零效果（关键负面结论）

| 版本 | 对数 | EA_strict | ER |
|---|---|---|---|
| SFT + Step1 修复（基线） | — | 0.8104 | 0.9836 |
| DPO v1（薄） | 243 | 0.8104 | 0.9836 |
| DPO v2（扩展） | 2047 | 0.8104 | 0.9836 |

- v1/v2 的 EA/ER/failure_dist 与基线**逐字节一致**（other 80 / join_type 29 / column_order 28 / aggregate 21 / exec_error 17 / set_op 12 / extra_join 9），不是"接近"，是"一模一样"。
- **直接证据**：`dpo_b01_v2/adapter_model.safetensors` 与 SFT adapter 逐张量 bit 完全相同（392/392 无差异、max 相对偏移 0.0），optimizer 一步都没更新。
- 训练日志：256 步 loss 钉在 **0.693 ≈ ln2 = -log σ(0)**（偏好 margin 恒 0），`grad_norm=0` 全程。

#### 根因（已定位并修复：加载顺序 bug）
- 243 对 vs 2047 对结果完全一致 → **"对数太少"假说证伪**，阻塞在训练环路本身。
- **真正病因**：`prepare_model_for_kbit_training` 与 `PeftModel.from_pretrained` 顺序颠倒。前者无条件 `param.requires_grad = False` 冻结所有参数（含 LoRA）。step5 是"先 PeftModel 加 LoRA，再 prepare"→ LoRA 被冻死 → 0 可训练参数 → grad_norm=0 → optimizer 一步没动。而 `train_sft.py` 是正确顺序（先 prepare 冻 base，再 get_peft_model 加 LoRA），所以 SFT 没踩中。
- 铁证：`dpo_b01_v2` 权重与 SFT adapter 逐位相同；diag 脚本证实修复前 trainable=0、修复后 trainable=40370176，且 392/392 LoRA 参数在普通与融合头 forward 的反向里都拿到非零梯度。
- 修复：调换顺序（先 prepare 再 PeftModel），冒烟 loss 0.693→0.6859、grad_norm=0→2.05、rewards/accuracies=0.36→0.725。
- 资产保留：`m4/analysis/step4b_pairs.jsonl`（2047 对）可直接复用。

### M5 DPO 最终结论（负面结果，已收尾）
dev 1034 全版本（含 loose）：

| 版本 | EA_strict | EA_loose | ER | 说明 |
|---|---|---|---|---|
| 基线 SFT+exec修复 | 0.8104 | 0.8375 | 0.9836 | 无 DPO |
| v1 薄243 / v2 厚2047(bug) | 0.8104 | 0.8375 | 0.9836 | 训练无梯度 → no-op |
| v1_fixed 薄243 | 0.8104 | 0.8385 | 0.9826 | 修复后 |
| v3 全量×原配方(β0.1/lr1e-5/1ep) | **0.8114** | **0.8395** | 0.9826 | 最佳，仍等于基线 |
| v4b 仅错误×原配方 | 0.8075 | 0.8356 | 0.9826 | 去锚点变差 |
| v4d 仅错误×强配方 | 0.7863 | 0.8269 | 0.9778 | 过冲 |
| v4c 全量×强配方(β0.05/lr3e-5/2ep) | 0.7340 | 0.7718 | 0.9362 | 崩，ER 掉到 .936 |

- 训练信号强（v3 loss 0.693→0.668、rewards/acc 0.87）但 greedy 评测纹丝不动 → 偏好对学的是 token 措辞偏好，不迁移为语义纠错。
- 强配方过冲：β 0.05 + lr 3e-5 + 2ep 破坏模型（v4c exec_error 17→66、v4d 列序→42）；"正确题弱对"是必要的反遗忘锚点（删掉即 v4b 变差）。
- **结论**：DPO 对"强 SFT + exec 修复"基线到头了，最佳仅 +0.001，任何加力/提纯都更差。后续应转向：加强 SFT 本身（数据+时长）、或逐类拆解语义失败（other ~80 / join_type ~28 / aggregate ~24）做定向修复。

### M5 收尾后：RFT / STaR round-2（DPO 的替代主线）

**弃 DPO 转 RFT 的根因**（详见 `m5/DPO_NEGATIVE_RESULT.md`）：
- DPO 三层失效：① 拒绝采样选择偏差（只有"差一点就对"的机械题才有 pair，join_type/aggregate/set_op 等硬骨头 no-pair）；② anchor 对占 71% 稀释信号；③ DPO 改不动 greedy 路径（loss 降到 0.668、rewards 0.87 但 greedy 纹丝不动）。
- RFT/SFT 三处全绕开：不用负样本 → no-pair 题可用 gold 补；全正样本无锚点稀释；SFT 直接改 token 分布。

**数据重构**（把拒绝采样资产变废为宝，关键事实）：
- 1547 no-pair = error 源 **123** + correct 源 **1424**（92% 是 greedy 已做对的题，纯锚点稀释）。
- 2047 chosen = error 源 **589** + correct 源 **1458**。
- **真信号只有 error 源**：589 chosen（STaR 正样本）+ 123 hardgold（模型采 N 次也够不到的 hard gold）。
- chosen 定义为"模型**原样执行就 strict 匹配 gold**"的原始输出，**未经过 exec 修复**（step4b 无 repair_sql；与 Step1 是两条独立路径）。

分层加权配方（`m4/analysis/rft_round2_train.jsonl`，物理重复实现加权）：

| 层 | 来源 | 权重 | 行数 |
|---|---|---|---|
| base | round-1 8621 | ×1 | 8621 |
| starchosen | 589 chosen | ×2 | 1178 |
| hardgold | 123 gold | ×2 | 246 |

总计 10045 行。

**实验设计**（拆"加数据"vs"多训一轮"两个变量，均 from base 重训 + rank16/lr2e-4）：

| run | 数据 | epochs | 目的 |
|---|---|---|---|
| 基线 | 8621 | 1 | = M4，已训出 0.8104 |
| control | 8621 | 2 | 单独测"epochs 1→2" |
| main | 10045 | 2 | 全套配方（先跑） |
| main_e1 | 10045 | 1 | 补跑：剥掉 epochs2 过拟合，单看纯数据 |

**结果（dev 1034，含 exec 修复）**：

| run | EA_strict | EA_loose | ER | repair_fixed |
|---|---|---|---|---|
| 基线 (8621, e1) | **0.8104** | 0.8375 | 0.9836 | ~25 |
| control (8621, e2) | 0.8027 | 0.8317 | 0.9826 | ~19 |
| main (10045, e2) | 0.8114 | 0.8404 | 0.9865 | ~18 |
| main_e1 (10045, e1) | 0.8085 | 0.8366 | 0.9797 | 35 |

2×2（EA_strict）：

| 数据 \ epochs | e1 | e2 |
|---|---|---|
| base 8621 | 0.8104 | 0.8027 |
| +数据 10045 | 0.8085 | 0.8114 |

四个格子全挤在 0.8027~0.8114，**没有一个到 0.82**。

**归因（重要修正——推翻上一版"数据有真信号"的判断）**：
- 之前的 `main(e2) − control(e2) = +0.87pp` 结论是**错的**：control(e2) 已被 epochs=2 过拟合打残（−0.77pp），那 +0.87pp 本质是"数据抵消过拟合"，不是"在健康基线上叠出新能力"。
- 干净对照看 e1 列：数据净贡献 = `main_e1 − 基线 = 0.8085 − 0.8104 = −0.19pp` → **略负面**。
- 数据唯一价值 = epochs=2 时当"防过拟合缓冲"（e2 列 +数据把 −0.77 拉回 +0.29），而非提升准确率。
- main_e1 的 ER 反而最差（0.9797）：exec_error 17→21、other 80→88、repair_fixed 25→35 → 加数据在 e1 反而轻微伤了执行稳健性。

**最终结论（与 DPO 同款负面结论）**：
- RFT round-2（589 STaR chosen + 123 hardgold）**没能突破 0.8104→0.82 的墙**，最佳仍是 main(e2)=0.8114（= 基线 +0.10pp ≈ 无）。
- 根子：589+123 全是模型自己 359 道 train 错误里 pass 出来的，而这 359 道在 round-1 的 gold 上已训过一遍（模型仍错），再喂自身采样，边际信息量≈0。
- 与 DPO 合流成同一结论：**对"强 SFT 单次 + exec 修复 = 0.8104"基线，靠模型自身错误重放（无论偏好式 DPO 还是生成式 RFT）都无法再抬 EA_strict**；0.82 缺口（~10 条）不是错误重放能补的。

**下一步候选（待定）**：
1. 定向挖 hard 题：按 dev 失败结构（join_type/aggregate/set_op 的 SQL 形态）去 train 全量挖同构难题，而非模型恰好错的 359 道；配合 rank 32。
2. 升容量 rank 16→32，看是否容量顶了。
3. 认 0.81 是单次天花板：回 M4.5 多采样兜底工程化，或换全参/更大模型。

---

## M6：把 agent 的「act→observe→correct」蒸馏进权重（repair-aware SFT）+ GRPO 验证

**目标转向**：不再挤 EA 零点几，而是补两块叙事空白——`training-time agentic`（exec 反馈自纠错 SFT）与 `online RL`（GRPO 第三条路）。两条都定位"能力 demo / 有根因的负结果"，不以涨 EA 为验收线。

### 定位与命名（面试用词要准）
- 我已有**规则层**的 lesson 闭环：`degrade.py`（错分类→提示→重试）+ `exec_repair.py`（确定性修复）+ `agent.py`（voting→retry→empty-result）。这些是 **inference-time** 的 agent。
- repair-aware SFT 是把这个"看到报错→自己改"的**单跳**从规则层蒸馏进**权重层**：训练模型在「自己生成的错 SQL + sqlite 报错」下直接吐出修正 SQL。这是"把 agent 能力内化进模型"叙事的第二块（第一块是"把 agent 管线蒸馏进单次生成"）。

### 数据结构（`m4/analysis/build_repair_sft.py`，零 API 采样、全确定性）
| 层 | 来源 | 数量 | 说明 |
|---|---|---|---|
| A 有机 | step2_train_records 里 `pred_repaired≠null 且 correct` | 27 | 真实模型 exec 错误 + exec_repair 修回严格正确（高保真核心） |
| B 合成 | train_conservative gold 反注入破坏 | 3000（列笔误/限定符/表名单复数 各 1000） | 执行拿真实报错，目标=原 gold；扩量 |
| C 锚点 | round-1 SFT | 3027 | 50% 反遗忘（DPO v4b 教训） |
合计 6054 行。

- 合成破坏的覆盖与 `exec_repair` 规则一一对应（R3c 列笔误、R3b 限定符、R1 表名）。**双引号字符串破坏不做**：SQLite 对双引号有"非标识符回退字符串"扩展，`"Yes"` 非列名时不报错，不构成 exec 错误源。
- 训练 prompt：system（同 round-1）+ user = `Schema + Question + Wrong SQL + SQLite error`，assistant = 修正 SQL。

### 训练
- `train_sft.py` 加 `--base-adapter`（从 `sft_cons_v1` 续训，非 from base；保底座）。rank16 / lr1e-4 / epochs1。
- adapter 输出：`/root/autodl-tmp/m4_lora/repair_r1`。
- 收敛：757 步，loss 0.024（比 RFT 低——合成数据是"近邻破坏"，学起来更简单）。

### 评测：dev 自纠错基准（`eval_self_correct.py`）
- bench = 基线模型在 dev 上**执行失败**的完整集合，共 **46 条**（oracle 可修 28 / 不可修 18）。
- **oracle 上界（exec_repair）**：exec 28/46=0.609、strict 16/46=0.348、loose 17/46=0.370。→ 规则层能修到"可执行"61%，但只修到"对"35%。
- 四配置消融：① single-pass（=0，构造上全是执行失败）② oracle 规则 ③ 基线 adapter 喂错SQL+报错自纠错 ④ repair_r1 喂错SQL+报错自纠错。**核心指标 = ③→④ 的技能增益（内部化率）**。

| 配置 | 总体 exec | 总体 strict | oracle-可修子集(28) exec |
|---|---|---|---|
| single-pass 基线 | 0/46 | 0/46 | 0/28 |
| 规则 oracle 上界 | 28/46=0.609 | 16/46=0.348 | 28/28 |
| ③ 基线 sft_cons_v1 自纠错 | 24/46=0.522 | 16/46=0.348 | 20/28=0.714 |
| ④ 训后 repair_r1 自纠错 | 27/46=0.587 | 17/46=0.370 | 23/28=0.821 |

**结论（第一块 agentic 训练已闭合，正信号）**：
1. **基座本就会自纠错**：sft_cons_v1 见报错即自修 71%（oracle-可修子集）。说明"看到报错→改"在 SFT 阶段已部分存在（important 基线，不能把功劳全记到 M6）。
2. **repair-aware SFT 抬到 82%**（+10.7pp，+3 条 exec 修正），严格正确 +1 条。
3. **正确性反超规则 oracle**：总体 strict 17/46 > oracle 16/46。模型学到的纠错**泛化到规则"不可修"的 18 条之外**（合成扩量教了更宽的修复），这正是"蒸馏进权重"比"规则层"值钱的那一点。
4. 量级诚实：46 条的绝对增量是 +3 exec/+1 strict，属"能力可量化、方向正确"的 demo，不是 EA 级别的突破。

### 端到端：repair_r1 接回 agent.py 重试路径（`eval_agent_repair.py`）
把 bench demo 升级成"真实 agent 循环里验证过的能力"。取 dev 的 46 条执行失败样本作纠正起点，对比三种纠正器：

| 纠正器 | exec 恢复 | strict | loose |
|---|---|---|---|
| single-pass (turn-0) | 0/46 | 0/46 | 0/46 |
| 规则 oracle (exec_repair) | 28/46 | 16/46 | 17/46 |
| 模型自纠错 (repair_r1) | 27/46 | 17/46 | 18/46 |
| 规则重试 (hint+API 现状) | **46/46** | **32/46** | 33/46 |

**结论**：
1. repair_r1 在真实重试循环里 ≈ 规则 oracle（27 vs 28 exec），strict 反超（17 vs 16）——"权重 ≈ 规则"在 agent 里成立。
2. 但 agent 现状（规则提示 + 外部大模型重写）能 **100% 恢复**（46/46 exec）。7B 蒸馏只捕获其 ~59%。差距来源：(a) **模型容量**（7B vs 外部大模型）；(b) **提示信息差**——repair_r1 训练时用裸报错，agent 现状喂的是 degrade.py 的"相似列+可用列"富提示。→ 直接指向 future work 的"模型容量"杠杆。
3. 诚实定位：repair_r1 是"本地免费的规则层等价物"，不是"替代大模型重试"；但零 API 成本/延迟，且已把"看到报错→改"内化成 7B 权重。

### GRPO（`run_grpo.py` + `grpo_reward.py`，验证心态）
- 定位：DPO(偏好式)/RFT(重放式) 都撞墙后，在线 RL 是第三条路。做它是完成"三种对齐范式全试过"的方法论闭环。环境已核实：**TRL 1.15 + transformers 5.19，GRPOTrainer 可用**（但 API 已大改：`max_prompt_length` 移除、新增 `generation_batch_size` 须被 `num_generations` 整除、`quantization_config/peft_config` 可内建）。
- reward = **稠密四档**（0 / 0.25 可执行 / 0.5 行数 / 0.75 loose / 1.0 strict），对抗"71% 单签名→组内零方差→优势崩"。
- 预期失败机制：组内零方差 → advantage 失效（用稠密 reward + G 采样缓解，仍大概率撞墙）。
- 进度：冒烟成功（8题×2步×G4）：`loss=-0.84、grad_norm=0.48（≠0，对比 DPO 的 0 梯度）、reward mean=0.625 std=0.433、frac_reward_zero_std=0`。**TRL 1.15 + QLoRA + 自定义稠密 reward 这套栈在本地可跑**（不是"论证 GRPO 会失败"而是"搭起来冒烟过了"）。坑：`train_dataset` 要 `datasets.Dataset`（list 会崩）。

### GRPO sanity 结果（200 题 × 100 步 × G8，从 sft_cons_v1 合并起步）
**结论：撞墙，且不是"会撞墙"的猜测，是精确复现了预言的失败机制。**

| 现象 | 数值 | 含义 |
|---|---|---|
| 零方差组占比 | **9/12 生成批次 std=0**（≈75%） | 你的预言"71% 单签名"精确复现 |
| 零方差批次的 reward | mean=1.0、std=0 | 模型对 train 已是满分 → 无学习信号 |
| 有方差的 3 批 | std 0.26 / 0.27 / 0.35 | 稠密 reward **确实**在难题上保住了方差（对策有效） |
| loss / grad_norm | 1.3 → **1e-6**，loss → 1e-9 | 但方差在，梯度还是死 |
| entropy | 0.10 → **0.015** | 策略塌缩成确定性 → advantage→0 |

**三层根因（比原假设更细）**：
1. **on-distribution 饱和**：sft_cons_v1 早在 train 上近乎满分（reward mean → 1.0），RL 在"已经学会的分布"上没有可学的残差。这跟 DPO/RFT 撞墙是**同一个根**——不是范式的问题，是**起点模型已经收敛**。
2. **零方差是果不是因的普通版**：易题 100% 零方差（满分），难题有方差（稠密 reward 起了作用）。
3. **有方差也没用**：熵塌缩（0.015）→ 采样退化成单点 → advantage 分母趋零 → 梯度死了。这印证了"G 采样组内零方差→优势崩"的那一半。

**附带基建坑**：96/100 步 CUDA OOM（completion 平均长度跑到 203 token、GRPO 显存随步增长）。查因：熵塌缩后模型开始吐长/不稳定输出 + 无 eval GC。修法：`eval_generation off` + `max_completion_length` 压到 ~64 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**三条路合流（面试可讲的一句话）**：DPO（偏好式）→ 零梯度；RFT（重放式）→ 边际信息≈0；GRPO（在线式）→ on-distribution 饱和 + 熵塌缩 → 零梯度。**三者同一根因：0.8104 的 SFT 模型已是其训练分布的局部最优，任何同分布微调（偏好/生成/RL）都推不动那一两个 pp。** 剩下的杠杆只剩：(a) 换更难/漂移的分布、(b) 容量（rank32/全参/更大模型）、(c) 认 0.81 是单次天花板。

### 状态
- 第一块（repair-aware SFT）：**闭合**，结果与结论已写入上文（内部化率 71%→82%，正确性反超 oracle）。
- 第二块（GRPO）：**闭合**。冒烟 rail 成功 + sanity 精确复现"零方差/熵塌缩"负结果，基建坑已定位。

