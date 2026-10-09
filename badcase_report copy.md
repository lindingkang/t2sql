# Baseline Bad Case 归因报告（dev 全量 1034 条）

## 数据的描述不足，背景知识不够，语义理解不对，sql语法错误

- **[car_1]** How many car models are produced in the usa?
  - gold: `SELECT count(*) FROM MODEL_LIST AS T1 JOIN CAR_MAKERS AS T2 ON T1.Maker  =  T2.Id JOIN COUNTRIES AS T3 ON T2.Country  =  T3.CountryId WHERE T3.CountryName  =  'usa'`
  - pred: `SELECT COUNT(*) FROM model_list ml JOIN car_makers cm ON ml.Maker = cm.Id JOIN countries c ON cm.Country = c.CountryId WHERE c.CountryName = 'USA'`
  - 归因: 是因为大模型不知道这个数据里面的countryName里面的usa应该是小写的。
  - 提示词：但是我们可以规定如果是这个数据库里面美国的countryname就是'usa'，除此之外没有这个提醒的匹配文本匹配筛选的时候绝大多数都按问题当中给出的大小写进行.
  - 错误类型：数据的描述不足

- **[car_1]** What is the count of the car models produced in the United States?
  - gold: `SELECT count(*) FROM MODEL_LIST AS T1 JOIN CAR_MAKERS AS T2 ON T1.Maker  =  T2.Id JOIN COUNTRIES AS T3 ON T2.Country  =  T3.CountryId WHERE T3.CountryName  =  'usa'`
  - pred: `SELECT COUNT(*) FROM model_list ml JOIN car_makers cm ON ml.Maker = cm.Id JOIN countries c ON cm.Country = c.CountryId WHERE c.CountryName = 'United States'`
  - 归因: 是因为大模型不知道这个数据里面的countryName里面的usa应该是小写的。
  - 提示词：但是我们可以规定如果是这个数据库里面美国的countryname就是'usa'，除此之外没有这个提醒的匹配文本匹配筛选的时候绝大多数都按问题当中给出的大小写进行.
  - 错误类型：数据的描述不足

- **[car_1]** Which countries in europe have at least 3 car manufacturers?
  - gold: `SELECT T1.CountryName FROM COUNTRIES AS T1 JOIN CONTINENTS AS T2 ON T1.Continent  =  T2.ContId JOIN CAR_MAKERS AS T3 ON T1.CountryId  =  T3.Country WHERE T2.Continent  =  'europe' GROUP BY T1.CountryName HAVING count(*)  >=  3`
  - pred: `SELECT c.CountryName FROM countries c JOIN car_makers cm ON c.CountryId = cm.Country JOIN continents cont ON c.Continent = cont.ContId WHERE cont.Continent = 'Europe' GROUP BY c.CountryId, c.CountryName HAVING COUNT(cm.Id) >= 3`
  - 归因: 是因为大模型不知道这个数据里面的countryName里面的usa应该是小写的。
  - 提示词：但是我们可以规定如果是这个数据库里面美国的countryname就是'usa'，除此之外没有这个提醒的匹配文本匹配筛选的时候绝大多数都按问题当中给出的大小写进行.
  - 错误类型：数据的描述不足
- **[car_1]** What are the names of all European countries with at least 3 manufacturers?
  - gold: `SELECT T1.CountryName FROM COUNTRIES AS T1 JOIN CONTINENTS AS T2 ON T1.Continent  =  T2.ContId JOIN CAR_MAKERS AS T3 ON T1.CountryId  =  T3.Country WHERE T2.Continent  =  'europe' GROUP BY T1.CountryName HAVING count(*)  >=  3`
  - pred: `SELECT c.CountryName FROM countries c JOIN car_makers cm ON c.CountryId = cm.Country JOIN continents cont ON c.Continent = cont.ContId WHERE cont.Continent = 'Europe' GROUP BY c.CountryId HAVING COUNT(cm.Id) >= 3`
  - 归因: 是因为大模型不知道这个数据continents里面的Continent欧洲europe应该是小写的。
  - 提示词：但是我们可以规定如果是这个数据库里面欧洲的Continent就是'europe'，除此之外没有这个提醒的匹配文本匹配筛选的时候绝大多数都按问题当中给出的大小写进行.
  - 错误类型：数据的描述不足

- **[flight_2]** What country is Jetblue Airways affiliated with?
  - gold: `SELECT Country FROM AIRLINES WHERE Airline  =  "JetBlue Airways"`
  - pred: `SELECT Country FROM airlines WHERE Airline = 'Jetblue Airways'`
  - 归因: 是因为大模型不知道这个数据Airline的JetBlue Airways大小写是这样子的。
  - 提示词：但是我们可以规定如果是这个数据库里面Airline的JetBlue Airways是这样子写，除此之外没有这个提醒的匹配文本匹配筛选的时候绝大多数都按问题当中给出的大小写进行.
  - 错误类型：数据的描述不足

- **[pets_1]** For students who have pets , how many pets does each student have ? list their ids instead of names .
  - gold: `select count(*) ,  t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid`
  - pred: `SELECT StuID, COUNT(*) AS pet_count FROM Has_Pet GROUP BY StuID`
  - 归因: The predicted SQL omits the join to the student table, failing to explicitly filter for students who have pets as required by the question's phrasing. 没有使用主表进行统计，而是仅使用了筛选表，筛选表里面的主id可能在主要统计表当中已经不在了。
  - 提示词：统计某个确定的对象时用这个对象的主表进行统计，不要用筛选条件的表的主键进行统计，因为筛选条件的表的维护可能跟主表不同步，要统计的那个对象的主表肯定是最准确的。
  - 错误类型：sql语法错误（join错误）

- **[car_1]** Which model of the car has the minimum horsepower?
  - gold: `SELECT T1.Model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id ORDER BY T2.horsepower ASC LIMIT 1`
  - pred: `SELECT T3.Model FROM cars_data AS T1 JOIN car_names AS T2 ON T1.Id = T2.MakeId JOIN model_list AS T3 ON T2.Model = T3.Model ORDER BY CAST(T1.Horsepower AS REAL) LIMIT 1`
  - 归因: 用的还是join，那么可能直接筛选掉部分不在model-list当中的数据，导致结果对不上。
  - 提示词：统计某个确定的对象时用这个对象的主表进行统计，其他表在连接时要谨慎进行join的筛选，不必要时不要增加多余的表进行连接。
  - 错误类型：sql语法错误（join错误）
  
- **[car_1]** What is the name of the different car makers who produced a car in 1970?
  - gold: `SELECT DISTINCT T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker JOIN CAR_NAMES AS T3 ON T2.model  =  T3.model JOIN CARS_DATA AS T4 ON T3.MakeId  =  T4.id WHERE T4.year  =  '1970'`
  - pred: `SELECT DISTINCT cm.Maker FROM car_makers cm JOIN car_names cn ON cm.Id = cn.MakeId JOIN cars_data cd ON cn.MakeId = cd.Id WHERE cd.Year = 1970`
  - 归因: The predicted SQL skips the MODEL_LIST table, creating an invalid direct join between car_makers and car_names that bypasses the required model linkage. 模型未使用外键链，靠字段名猜测连接（Maker/Make/MakeId/Model 高度相似）正确路径car_makers.Id → model_list.Maker → model_list.Model → car_names.Model → car_names.MakeId → cars_data.I
  - 提示词：join的时候要注意外键之间的联系，不能仅靠名字去猜，要认真读schema，使用外键图 BFS 求最短 JOIN 路径，注入 prompt
  - 错误类型：sql语法错误（join错误）

- **[car_1]** What is the maker of the carr produced in the earliest year and what year was it?
  - gold: `SELECT T2.Make ,  T1.Year FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T1.Year  =  (SELECT min(YEAR) FROM CARS_DATA)`
  - pred: `SELECT cm.Maker, cd.Year FROM cars_data cd JOIN car_names cn ON cd.Id = cn.MakeId JOIN model_list ml ON cn.Model = ml.Model JOIN car_makers cm ON ml.Maker = cm.Id ORDER BY cd.Year ASC LIMIT 1`
  - 归因: The predicted SQL incorrectly joins extra tables (model_list) and uses a different join path to retrieve the maker, whereas the gold SQL correctly joins only cars_data and car_names.用的还是join，那么可能直接筛选掉部分不在model-list当中的数据，导致结果对不上。
  - 提示词：统计某个确定的对象时用这个对象的主表进行统计，其他表在连接时要谨慎进行join的筛选，不必要时不要增加多余的表进行连接。
  - 错误类型：sql语法错误（join错误）

- **[car_1]** What is the accelerate of the car make amc hornet sportabout (sw)?
  - gold: `SELECT T1.Accelerate FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T2.Make  =  'amc hornet sportabout (sw)'`
  - pred: `SELECT T4.Accelerate FROM car_names AS T1 JOIN model_list AS T2 ON T1.Model = T2.Model JOIN car_makers AS T3 ON T2.Maker = T3.Id JOIN cars_data AS T4 ON T1.MakeId = T4.Id WHERE T3.Maker = 'amc' AND T1.Model = 'hornet sportabout (sw)'`
  - 归因: 用的还是join，那么可能直接筛选掉部分不在model-list当中的数据，导致结果对不上。
  - 提示词：统计某个确定的对象时用这个对象的主表进行统计，其他表在连接时要谨慎进行join的筛选，不必要时不要增加多余的表进行连接。要认真阅读外键的信息，根据外键信息选取join的最短路径。
  - 错误类型：sql语法错误（join错误）

- **[concert_singer]** What is the maximum capacity and the average of all stadiums ?
  - gold: `select max(capacity), average from stadium`
  - pred: `SELECT MAX(Capacity), AVG(Average) FROM stadium`
  - 归因: The model incorrectly applied the AVG aggregate function to a column named 'Average' instead of selecting the pre-computed 'average' column as specified in the gold SQL. 
  - 提示词：如果有预先算好的聚合，直接使用预先选好的聚合字段。
  - 错误类型：聚合错误

- **[car_1]** What is the model for the car with a weight smaller than the average?
  - gold: `SELECT T1.model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id WHERE T2.Weight  <  (SELECT avg(Weight) FROM CARS_DATA)`
  - pred: `SELECT T3.Model FROM cars_data AS T1 JOIN car_names AS T2 ON T1.Id = T2.MakeId JOIN model_list AS T3 ON T2.Model = T3.Model WHERE T1.Weight < (SELECT AVG(Weight) FROM cars_data)`
  - 归因: The predicted SQL incorrectly joins an extra table 'model_list' and selects from a non-existent column 'T3.Model', whereas the correct query retrieves the model directly from 'car_names'. 用的还是join，那么可能直接筛选掉部分不在model-list当中的数据，导致结果对不上。
  - 提示词：统计某个确定的对象时用这个对象的主表进行统计，其他表在连接时要谨慎进行join的筛选，不必要时不要增加多余的表进行连接。要认真阅读外键的信息，根据外键信息选取join的最短路径。
  - 错误类型：sql语法错误（join错误）

- **[car_1]** How many car makers are there in france?
  - gold: `SELECT count(*) FROM CAR_MAKERS AS T1 JOIN COUNTRIES AS T2 ON T1.Country  =  T2.CountryId WHERE T2.CountryName  =  'france'`
  - pred: `SELECT COUNT(*) FROM car_makers WHERE Country = (SELECT CountryId FROM countries WHERE CountryName = 'France')`
  - 归因: 是因为大模型不知道这个数据里面的countryName里面的france应该是小写的。
  - 提示词：但是我们可以规定如果是这个数据库里面法国的countryname就是'france'，除此之外没有这个提醒的匹配文本匹配筛选的时候绝大多数都按问题当中给出的大小写进行.
  - 错误类型：数据的描述不足
  
- **[car_1]** Find the make and production time of the cars that were produced in the earliest year?
  - gold: `SELECT T2.Make ,  T1.Year FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T1.Year  =  (SELECT min(YEAR) FROM CARS_DATA)`
  - pred: `SELECT cn.Make, cd.Year FROM car_names cn JOIN cars_data cd ON cn.MakeId = cd.Id ORDER BY cd.Year ASC LIMIT 1`
  - 归因: The model uses ORDER BY and LIMIT to get one row instead of using a subquery with MIN() to find the earliest year across all cars. 因为最早的那一年可能有很多的汽车厂商和产品，而不是最早的那一个商品。
  - 提示词：注意在最早年份等最什么的筛选条件时，要注意甄别时使用排序后再limit还是使用min,max进行筛选。
  - 错误类型：sql极值筛选错误

- **[car_1]** How many countries has more than 2 car makers ?
  - gold: `select count(*) from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country group by t1.countryid having count(*)  >  2`
  - pred: `SELECT COUNT(*) FROM (SELECT Country FROM car_makers GROUP BY Country HAVING COUNT(*) > 2)`
  - 归因:  没有使用主表进行统计，而是仅使用了筛选表，筛选表里面的主id可能在主要统计表当中已经不在了。
  - 提示词：统计某个确定的对象时用这个对象的主表进行统计，不要用筛选条件的表的主键进行统计，因为筛选条件的表的维护可能跟主表不同步，要统计的那个对象的主表肯定是最准确的。
  - 错误类型：sql语法错误（join错误）

- **[car_1]** What is the number of countries with more than 2 car makers ?
  - gold: `select count(*) from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country group by t1.countryid having count(*)  >  2`
  - pred: `SELECT COUNT(*) FROM ( SELECT c.CountryId FROM countries c JOIN car_makers cm ON c.CountryId = cm.Country GROUP BY c.CountryId HAVING COUNT(cm.Id) > 2 )`
  - 归因: 这个是给的gold是错误的，我的pred是正确的，给的标准答案输出的是车企的数量而不是国家的数量。
  - 错误类型：spider库噪声

- **[flight_2]** What is the code of airport that has fewest number of flights?
  - gold: `SELECT T1.AirportCode FROM AIRPORTS AS T1 JOIN FLIGHTS AS T2 ON T1.AirportCode  =  T2.DestAirport OR T1.AirportCode  =  T2.SourceAirport GROUP BY T1.AirportCode ORDER BY count(*) LIMIT 1`
  - pred: `SELECT SourceAirport AS AirportCode FROM flights GROUP BY SourceAirport UNION ALL SELECT DestAirport AS AirportCode FROM flights GROUP BY DestAirport ORDER BY COUNT(*) ASC LIMIT 1`  \n  ⚠️ pred 执行失败: 1st ORDER BY term does not match any column in the result set
  - 归因: 首先语法是错误的，order by 作用域不对。然后这个逻辑理解错误，这个逻辑算出来的话，同一个机场有两行，那么asc的时候排序只会采取落地或者起飞的某个排名第一的，但是正确的应当是总和第一。
  - 提示词： 补充知识：SQLite 限制：复合查询(UNION/EXCEPT/INTERSECT)的 ORDER BY，只能引用最终输出列，不能引用分支内的聚合表达式。需要排序时，把聚合列放进 SELECT，或整体包一层子查询。
  - 错误类型：模型理解错误/sql语法错误

- **[museum_visit]** What are the id, name and membership level of visitors who have spent the largest amount of money in total in all museum tickets?
  - gold: `SELECT t2.visitor_id ,  t1.name ,  t1.Level_of_membership FROM visitor AS t1 JOIN visit AS t2 ON t1.id  =  t2.visitor_id GROUP BY t2.visitor_id ORDER BY sum(t2.Total_spent) DESC LIMIT 1`
  - pred: `SELECT v.ID, v.Name, v.Level_of_membership FROM visitor v JOIN visit vis ON v.ID = vis.visitor_ID ORDER BY vis.Total_spent DESC LIMIT 1`
  - 归因: The model failed to aggregate the spending per visitor using SUM and GROUP BY, instead sorting individual visit records by their total spent.不要靠这个columns的名字去直接猜，这里的题目里的逻辑肯定是要聚合起来。
  - 提示词：注意题目中的in total等字眼，根据逻辑去判断要不要聚合。
  - 错误类型：聚合错误

- **[concert_singer]** Show countries where a singer above age 40 and a singer below 30 are from.
  - gold: `SELECT country FROM singer WHERE age  >  40 INTERSECT SELECT country FROM singer WHERE age  <  30`
  - pred: `SELECT Country FROM singer WHERE Age > 40 UNION SELECT Country FROM singer WHERE Age < 30`
  - 归因: 自然语言的 and 到底映射到 SQL 的 AND、OR、INTERSECT 还是自连接，取决于它修饰的是同一行的条件还是不同行的条件。
  - 提示词：When the question requires a country/entity to satisfy BOTH conditions (e.g. 'X and Y are from'), use INTERSECT, not UNION.
  - 错误类型：模型理解错误

- **[car_1]** Which are the car makers which produce at least 2 models and more than 3 car makers ? List the id and the maker .
  - gold: `select t1.id ,  t1.maker from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker group by t1.id having count(*)  >=  2 intersect select t1.id ,  t1.maker from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker join car_names as t3 on t2.model  =  t3.model group by t1.id having count(*)  >  3`
  - pred: `SELECT cm.Id, cm.Maker FROM car_makers cm JOIN model_list ml ON cm.Id = ml.Maker GROUP BY cm.Id, cm.Maker HAVING COUNT(ml.ModelId) >= 2 AND (SELECT COUNT(*) FROM car_makers) > 3`
  - 归因: 问题里的 and 连接的是两个独立的条件，不是同一行的 AND。more than 3 car makers 字面读起来就是"超过 3 个车企"。模型选择了最字面的解读，而 gold 的解读是"该车企关联超过 3 条 car_names 记录"。不过话说回来——哪怕按最宽松的标准，pred 也不对：它把本该是"该车企的某个属性"的条件，写成了全局常量。就算按字面的"超过 3 个车企"来理解，那这个条件也是对全表恒成立/恒不成立的荒谬条件，说明模型确实没理解题意。
  - 提示词：
  - 错误类型：spider库噪声/模型理解错误

- **[flight_2]** What is the code of airport that has the highest number of flights?
  - gold: `SELECT T1.AirportCode FROM AIRPORTS AS T1 JOIN FLIGHTS AS T2 ON T1.AirportCode  =  T2.DestAirport OR T1.AirportCode  =  T2.SourceAirport GROUP BY T1.AirportCode ORDER BY count(*) DESC LIMIT 1`
  - pred: `SELECT SourceAirport AS AirportCode FROM flights GROUP BY SourceAirport ORDER BY COUNT(*) DESC LIMIT 1 UNION ALL SELECT DestAirport AS AirportCode FROM flights GROUP BY DestAirport ORDER BY COUNT(*) DESC LIMIT 1 ORDER BY (SELECT COUNT(*) FROM flights WHERE SourceAirport = AirportCode) + (SELECT COUNT(*) FROM flights WHERE DestAirport = AirportCode) DESC LIMIT 1`  \n  ⚠️ pred 执行失败: near "ORDER": syntax error
  - 归因: The model incorrectly uses UNION ALL with nested ORDER BY clauses and a complex subquery in the final ORDER BY, which is syntactically invalid in SQLite and fails to correctly aggregate flight counts per airport before sorting.
  - 提示词：必须遵守的 SQLite 语法规则：1. 复合查询（UNION / UNION ALL / INTERSECT / EXCEPT）中，ORDER BY 和 LIMIT
   只能在最后出现一次，作用于整个结果集。各分支内不得单独使用，
   除非把每个分支包成子查询：SELECT ... FROM (SELECT ... UNION ALL SELECT ...) ORDER BY ...2. 复合查询的 ORDER BY 只能引用最终输出列，不能引用聚合表达式或分支内的列。3. 想用 UNION 之前，先判断条件能否用 OR 在单次 JOIN 中表达 ——
   通常更简洁，也永远不会触发上述语法问题。
  - 错误类型：sql语法错误

- **[flight_2]** Give the code of the airport with the least flights.
  - gold: `SELECT T1.AirportCode FROM AIRPORTS AS T1 JOIN FLIGHTS AS T2 ON T1.AirportCode  =  T2.DestAirport OR T1.AirportCode  =  T2.SourceAirport GROUP BY T1.AirportCode ORDER BY count(*) LIMIT 1`
  - pred: `SELECT AirportCode FROM airports WHERE AirportCode NOT IN (SELECT SourceAirport FROM flights) AND AirportCode NOT IN (SELECT DestAirport FROM flights) LIMIT 1`
  - 归因: The model incorrectly interprets 'least flights' as airports with zero flights (NOT IN subqueries) instead of counting and ordering by flight frequency.
  - 提示词：要正确理解问题当中的least，在sql问题当中不是0的意思。
  - 错误类型：模型理解错误

- **[flight_2]** Which airports do not have departing or arriving flights?
  - gold: `SELECT AirportName FROM Airports WHERE AirportCode NOT IN (SELECT SourceAirport FROM Flights UNION SELECT DestAirport FROM Flights)`
  - pred: `SELECT "AirportCode" FROM "airports" WHERE "AirportCode" NOT IN (SELECT "SourceAirport" FROM "flights") AND "AirportCode" NOT IN (SELECT "DestAirport" FROM "flights")`
  - 归因: 返回的值不对，要求返回的是airports，而不是airports的id.
  - 提示词：要注意题目的要求，严格按照题目要求的返回最后的列。如果没有说要返回id，一般返回的是name。
  - 错误类型：模型理解错误

- **[car_1]** What is the number of the cars with horsepower more than 150?
  - gold: `SELECT count(*) FROM CARS_DATA WHERE horsepower  >  150`
  - pred: `SELECT COUNT(*) FROM cars_data WHERE CAST(Horsepower AS INTEGER) > 150`
  - 归因: 这是因为cars_data里面horsepower里面给的类型是text，所以其实我们的pred是正确的
  - 错误类型：spider库噪声

- **[car_1]** What is the maximum miles per gallon of the car with 8 cylinders or produced before 1980 ?
  - gold: `select max(mpg) from cars_data where cylinders  =  8 or year  <  1980`
  - pred: `SELECT MAX(CAST(MPG AS REAL)) FROM cars_data WHERE Cylinders = 8 OR Year < 1980`
  - 归因: 这是因为cars_data里面mpg里面给的类型是text，所以其实我们的pred是正确的
  - 错误类型：spider库噪声



