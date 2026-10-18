# Baseline Bad Case 精准归因报告（dev 全量 1034 条）

> 生成方式：对 207 条失败样本逐条单独调用 LLM 归因，输入包含完整 schema DDL、question、gold/pred SQL 及其执行结果。
> 区别于上一版：分类更细、归因更具体、每条都附 fix_prompt（prompt 中应加什么才能防错）。

## 总览

| 指标 | 数值 |
|------|------|
| 总样本 | 1034 |
| 可评测 | 1027（gold 能执行的） |
| 正确 | 820（严格 EA = 79.8%） |
| **失败待归因** | **207** |
| Gold 噪声样本 | 9（gold 本身有错，M4 训练时要剔除） |

## 失败类型分布

| 类型 | 条数 | 占比 | 说明 |
|------|------|------|------|
| semantic_misunderstanding | 69 | 33.3% | 模型误解了问题的语义 |
| data_description_gap | 45 | 21.7% | 模型不知道数据的具体存储方式（大小写、类型、格式） |
| join_error | 36 | 17.4% | JOIN 逻辑错（多余/缺失/类型错） |
| aggregation_error | 25 | 12.1% | 聚合逻辑错（GROUP BY / HAVING / 子查询） |
| other (column_order) | 24 | 11.6% | 列顺序不同（伪错，宽松指标可消除） |
| set_logic_error | 4 | 1.9% | INTERSECT/UNION 语义错 |
| gold_noise | 4 | 1.9% | gold SQL 本身有错 |

## 各类详细案例与 fix_prompt

### 1. semantic_misunderstanding（69 条）—— 最大错误源

模型对问题的理解出现了偏差，不是 schema 不知道，而是"读不懂题"。

**案例 1** [concert_singer] `What is the maximum capacity and the average of all stadiums?`
- gold: `SELECT max(capacity), average FROM stadium`
- pred: `SELECT max(capacity), avg(capacity) FROM stadium`
- root_cause: 问题问的是 "average" 这一列的值，模型理解成了要对 capacity 求 AVG()
- fix_prompt: 当问题提到"average"等词时，先检查 schema 中是否有同名列；有则直接 SELECT 该列，不要套 AVG()

**案例 2** [pets_1] `What is the first name of every student who has a dog but does not have a cat?`
- gold: `SELECT T1.Fname, T1.Age FROM Student ...`
- pred: `SELECT T1.Fname FROM Student ...`
- root_cause: 问题说 "first name"，gold 返回 first name 和 age 两列，模型只返回了一列
- fix_prompt: SELECT 的列要严格匹配问题中提到的所有属性

### 2. data_description_gap（45 条）—— 第二大错误源

模型不知道数据的存储细节：大小写、列类型（TEXT vs REAL）、特殊值格式。

**案例 1** [car_1] `How many car makers are there in france?`
- gold: `... WHERE T3.CountryName = 'france'`
- pred: `... WHERE T3.CountryName = 'France'`
- root_cause: 数据库里存的是小写 'france'，模型用了大写 'France'
- fix_prompt: 字符串比较是大小写敏感的；在 WHERE 里用值时，大小写要和数据库一致

**案例 2** [car_1] `What is the model of the car with the smallest amount of horsepower?`
- gold: `... ORDER BY Horsepower LIMIT 1`（TEXT 字典序）
- pred: `... ORDER BY CAST(Horsepower AS REAL) LIMIT 1`（数值序）
- root_cause: Horsepower 列是 TEXT 类型，gold 用字典序，模型转成了数值
- fix_prompt: 列类型是 TEXT 时，排序和比较按字典序；不要擅自 CAST

### 3. join_error（36 条）

JOIN 多了、少了、或类型错。

**案例 1** [concert_singer] `Show the stadium name and the number of concerts in each stadium.`
- gold: `INNER JOIN`（只列有演唱会的场馆）
- pred: `LEFT JOIN`（列了所有场馆，包括 0 场演唱会的）
- fix_prompt: 统计子实体的数量时用 INNER JOIN，除非问题明确要求"所有"包括零的

**案例 2** [pets_1] `For students who have pets, how many pets does each student have? List their ids instead of names.`
- gold: `JOIN Student`（取 Student.StuID）
- pred: 只用 Has_Pet 表（取 Has_Pet.StuID）
- root_cause: 问题说 "list their ids"，需要 JOIN Student 表取正式的 StuID
- fix_prompt: 当问题要求列出某实体的 ID 时，确保 JOIN 到该实体的主表

### 4. aggregation_error（25 条）

GROUP BY / HAVING / 子查询嵌套出错。

**案例 1** [car_1] `What is the number of countries with more than 2 car makers?`
- gold: `SELECT count(*) FROM (SELECT T2.CountryId ... GROUP BY T2.CountryId HAVING count(*) > 2)`
- pred: `SELECT T2.CountryId, count(*) ... GROUP BY T2.CountryId HAVING count(*) > 2`
- root_cause: 问题问"有多少个国家"，需要外层套 COUNT(*)；模型直接返回了国家列表
- fix_prompt: 问"有多少 X 满足条件"时，要把筛选逻辑包在子查询里，外层套 COUNT(*)

### 5. other / column_order（24 条）—— 伪错

列顺序不同，数据内容一样。宽松 EA 可消除。

### 6. set_logic_error（4 条）

**案例** [concert_singer] `Show countries where a singer above 40 and a singer below 30 are from.`
- gold: `INTERSECT`
- pred: `UNION`
- fix_prompt: "既...又..."用 INTERSECT；"...或..."用 UNION

### 7. gold_noise（4 条 + 5 条 is_gold_noise=True）—— 毒样本

gold SQL 本身有错或与问题不一致。这些样本在 M4 训练时要剔除。

**案例** [battle_death] `List the name, date and result of each battle.`
- gold: `SELECT name, date FROM battle`（漏了 result 列）
- 问题明确要 result，但 gold 没 SELECT 它——这是 Spider 数据标注错误

## 对 prompt 工程的指导

基于以上 207 条归因，新 prompt 应加入以下针对性约束：

1. **数据感知**：字符串值大小写要和 DB 一致；TEXT 列不要擅自 CAST；先查 schema 有无同名列
2. **JOIN 纪律**：统计子实体用 INNER JOIN；列 ID 时 JOIN 到主表；只 JOIN 有外键关系的表
3. **聚合规范**："有多少 X 满足条件"要嵌套子查询 + 外层 COUNT(*)；GROUP BY 只含主键
4. **SELECT 精确**：严格匹配问题提到的所有列；聚合列在前、分组列在后
5. **集合语义**：既...又...=INTERSECT；或...=UNION
6. **语义兜底**：问题含"average"等词时先查 schema 有无同名列；不要过度推断
