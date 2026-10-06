# Baseline Bad Case 归因报告（dev 全量 1034 条）

- 严格 EA：822/1029 = **79.9%**；宽松 EA（忽略列顺序）：863/1029 = **83.9%**
- 失败样本 212 条，经 LLM 逐条语义归因；其中判定为模型责任的 137 条

## 失败类型分布

| 类型 | 数量 | 占失败比 |
|------|------|----------|
| column_order | 68 | 32.1% |
| value_error | 41 | 19.3% |
| join_error | 29 | 13.7% |
| schema_error | 28 | 13.2% |
| aggregate_error | 25 | 11.8% |
| set_logic_error | 15 | 7.1% |
| other | 5 | 2.4% |
| gold_issue | 1 | 0.5% |

## column_order（68 条）

- **[concert_singer]** Show the stadium name and the number of concerts in each stadium.
  - gold: `SELECT T2.name ,  count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.stadium_id`
  - pred: `SELECT s.Name, COUNT(c.concert_ID) AS number_of_concerts FROM stadium s LEFT JOIN concert c ON s.Stadium_ID = c.Stadium_ID GROUP BY s.Stadium_ID`
  - 归因: The predicted SQL returns the stadium name and concert count in a different order than the gold SQL, but the semantic result is equivalent.
- **[pets_1]** What is the first name of every student who has a dog but does not have a cat?
  - gold: `SELECT T1.fname ,  T1.age FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'dog' AND T1.stuid NOT IN (SELECT T1.stuid FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T3.petid  =  T2.petid WHERE T3.pettype  =  'cat')`
  - pred: `SELECT DISTINCT S.Fname FROM Student S JOIN Has_Pet HP ON S.StuID = HP.StuID JOIN Pets P ON HP.PetID = P.PetID WHERE P.PetType = 'dog' AND S.StuID NOT IN (SELECT HP2.StuID FROM Has_Pet HP2 JOIN Pets P2 ON HP2.PetID = P2.PetID WHERE P2.PetType = 'cat')`
  - 归因: The predicted SQL selects only the first name while the gold SQL incorrectly includes the age column, but both queries correctly identify the same set of students.
- **[pets_1]** Find the average and maximum age for each type of pet.
  - gold: `SELECT avg(pet_age) ,  max(pet_age) ,  pettype FROM pets GROUP BY pettype`
  - pred: `SELECT PetType, AVG(pet_age) AS avg_age, MAX(pet_age) AS max_age FROM Pets GROUP BY PetType`
  - 归因: The predicted SQL selects columns in a different order (PetType first) compared to the gold SQL, but the semantic content is identical.
- **[pets_1]** What is the average and maximum age for each pet type?
  - gold: `SELECT avg(pet_age) ,  max(pet_age) ,  pettype FROM pets GROUP BY pettype`
  - pred: `SELECT PetType, AVG(pet_age) AS avg_age, MAX(pet_age) AS max_age FROM Pets GROUP BY PetType`
  - 归因: The predicted SQL returns the same columns and aggregations as the gold SQL but in a different order.
- **[pets_1]** Find the average weight for each pet type.
  - gold: `SELECT avg(weight) ,  pettype FROM pets GROUP BY pettype`
  - pred: `SELECT PetType, AVG(weight) FROM Pets GROUP BY PetType`
  - 归因: The predicted SQL selects columns in a different order than the gold SQL but produces identical results.

## value_error（41 条）

- **[car_1]** How many car models are produced in the usa?
  - gold: `SELECT count(*) FROM MODEL_LIST AS T1 JOIN CAR_MAKERS AS T2 ON T1.Maker  =  T2.Id JOIN COUNTRIES AS T3 ON T2.Country  =  T3.CountryId WHERE T3.CountryName  =  'usa'`
  - pred: `SELECT COUNT(*) FROM model_list ml JOIN car_makers cm ON ml.Maker = cm.Id JOIN countries c ON cm.Country = c.CountryId WHERE c.CountryName = 'USA'`
  - 归因: The predicted SQL uses 'USA' in uppercase while the gold SQL uses 'usa' in lowercase, which may cause a mismatch if the database is case-sensitive.
- **[car_1]** What is the count of the car models produced in the United States?
  - gold: `SELECT count(*) FROM MODEL_LIST AS T1 JOIN CAR_MAKERS AS T2 ON T1.Maker  =  T2.Id JOIN COUNTRIES AS T3 ON T2.Country  =  T3.CountryId WHERE T3.CountryName  =  'usa'`
  - pred: `SELECT COUNT(*) FROM model_list ml JOIN car_makers cm ON ml.Maker = cm.Id JOIN countries c ON cm.Country = c.CountryId WHERE c.CountryName = 'United States'`
  - 归因: The model used 'United States' as the filter value while the gold SQL and likely database content use 'usa'.
- **[car_1]** Which countries in europe have at least 3 car manufacturers?
  - gold: `SELECT T1.CountryName FROM COUNTRIES AS T1 JOIN CONTINENTS AS T2 ON T1.Continent  =  T2.ContId JOIN CAR_MAKERS AS T3 ON T1.CountryId  =  T3.Country WHERE T2.Continent  =  'europe' GROUP BY T1.CountryName HAVING count(*)  >=  3`
  - pred: `SELECT c.CountryName FROM countries c JOIN car_makers cm ON c.CountryId = cm.Country JOIN continents cont ON c.Continent = cont.ContId WHERE cont.Continent = 'Europe' GROUP BY c.CountryId, c.CountryName HAVING COUNT(cm.Id) >= 3`
  - 归因: The model used 'Europe' with a capital E while the gold SQL and likely database content use lowercase 'europe'.
- **[car_1]** What are the names of all European countries with at least 3 manufacturers?
  - gold: `SELECT T1.CountryName FROM COUNTRIES AS T1 JOIN CONTINENTS AS T2 ON T1.Continent  =  T2.ContId JOIN CAR_MAKERS AS T3 ON T1.CountryId  =  T3.Country WHERE T2.Continent  =  'europe' GROUP BY T1.CountryName HAVING count(*)  >=  3`
  - pred: `SELECT c.CountryName FROM countries c JOIN car_makers cm ON c.CountryId = cm.Country JOIN continents cont ON c.Continent = cont.ContId WHERE cont.Continent = 'Europe' GROUP BY c.CountryId HAVING COUNT(cm.Id) >= 3`
  - 归因: The model used 'Europe' with a capital E while the gold SQL uses 'europe' in lowercase, which may cause a mismatch if the database is case-sensitive.
- **[flight_2]** What country is Jetblue Airways affiliated with?
  - gold: `SELECT Country FROM AIRLINES WHERE Airline  =  "JetBlue Airways"`
  - pred: `SELECT Country FROM airlines WHERE Airline = 'Jetblue Airways'`
  - 归因: The predicted SQL uses 'Jetblue Airways' with a lowercase 'b' instead of the correct 'JetBlue Airways' required for an exact string match.

## join_error（29 条）

- **[pets_1]** For students who have pets , how many pets does each student have ? list their ids instead of names .
  - gold: `select count(*) ,  t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid`
  - pred: `SELECT StuID, COUNT(*) AS pet_count FROM Has_Pet GROUP BY StuID`
  - 归因: The predicted SQL omits the join to the student table, failing to explicitly filter for students who have pets as required by the question's phrasing.
- **[car_1]** Which model of the car has the minimum horsepower?
  - gold: `SELECT T1.Model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id ORDER BY T2.horsepower ASC LIMIT 1`
  - pred: `SELECT T3.Model FROM cars_data AS T1 JOIN car_names AS T2 ON T1.Id = T2.MakeId JOIN model_list AS T3 ON T2.Model = T3.Model ORDER BY CAST(T1.Horsepower AS REAL) LIMIT 1`
  - 归因: The predicted SQL incorrectly joins an extra table 'model_list' and uses a different join path compared to the gold SQL.
- **[car_1]** What is the name of the different car makers who produced a car in 1970?
  - gold: `SELECT DISTINCT T1.Maker FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker JOIN CAR_NAMES AS T3 ON T2.model  =  T3.model JOIN CARS_DATA AS T4 ON T3.MakeId  =  T4.id WHERE T4.year  =  '1970'`
  - pred: `SELECT DISTINCT cm.Maker FROM car_makers cm JOIN car_names cn ON cm.Id = cn.MakeId JOIN cars_data cd ON cn.MakeId = cd.Id WHERE cd.Year = 1970`
  - 归因: The predicted SQL skips the MODEL_LIST table, creating an invalid direct join between car_makers and car_names that bypasses the required model linkage.
- **[car_1]** What is the maker of the carr produced in the earliest year and what year was it?
  - gold: `SELECT T2.Make ,  T1.Year FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T1.Year  =  (SELECT min(YEAR) FROM CARS_DATA)`
  - pred: `SELECT cm.Maker, cd.Year FROM cars_data cd JOIN car_names cn ON cd.Id = cn.MakeId JOIN model_list ml ON cn.Model = ml.Model JOIN car_makers cm ON ml.Maker = cm.Id ORDER BY cd.Year ASC LIMIT 1`
  - 归因: The predicted SQL incorrectly joins extra tables (model_list) and uses a different join path to retrieve the maker, whereas the gold SQL correctly joins only cars_data and car_names.
- **[car_1]** What is the accelerate of the car make amc hornet sportabout (sw)?
  - gold: `SELECT T1.Accelerate FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T2.Make  =  'amc hornet sportabout (sw)'`
  - pred: `SELECT T4.Accelerate FROM car_names AS T1 JOIN model_list AS T2 ON T1.Model = T2.Model JOIN car_makers AS T3 ON T2.Maker = T3.Id JOIN cars_data AS T4 ON T1.MakeId = T4.Id WHERE T3.Maker = 'amc' AND T1.Model = 'hornet sportabout (sw)'`
  - 归因: The predicted SQL uses an incorrect join path involving model_list and car_makers tables instead of directly joining CARS_DATA and CAR_NAMES as in the gold query.

## schema_error（28 条）

- **[concert_singer]** What is the maximum capacity and the average of all stadiums ?
  - gold: `select max(capacity), average from stadium`
  - pred: `SELECT MAX(Capacity), AVG(Average) FROM stadium`
  - 归因: The model incorrectly applied the AVG aggregate function to a column named 'Average' instead of selecting the pre-computed 'average' column as specified in the gold SQL.
- **[car_1]** What is the model of the car with the smallest amount of horsepower?
  - gold: `SELECT T1.Model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id ORDER BY T2.horsepower ASC LIMIT 1`
  - pred: `SELECT cn.Model FROM car_names cn JOIN cars_data cd ON cn.MakeId = cd.Id ORDER BY CAST(cd.Horsepower AS REAL) ASC LIMIT 1`
  - 归因: The predicted SQL uses incorrect column names 'Horsepower' and 'Model' instead of the correct schema names 'horsepower' and 'Model'.
- **[car_1]** What is the model for the car with a weight smaller than the average?
  - gold: `SELECT T1.model FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id WHERE T2.Weight  <  (SELECT avg(Weight) FROM CARS_DATA)`
  - pred: `SELECT T3.Model FROM cars_data AS T1 JOIN car_names AS T2 ON T1.Id = T2.MakeId JOIN model_list AS T3 ON T2.Model = T3.Model WHERE T1.Weight < (SELECT AVG(Weight) FROM cars_data)`
  - 归因: The predicted SQL incorrectly joins an extra table 'model_list' and selects from a non-existent column 'T3.Model', whereas the correct query retrieves the model directly from 'car_names'.
- **[car_1]** How many car makers are there in france?
  - gold: `SELECT count(*) FROM CAR_MAKERS AS T1 JOIN COUNTRIES AS T2 ON T1.Country  =  T2.CountryId WHERE T2.CountryName  =  'france'`
  - pred: `SELECT COUNT(*) FROM car_makers WHERE Country = (SELECT CountryId FROM countries WHERE CountryName = 'France')`
  - 归因: The predicted SQL references a non-existent 'Country' column in the 'car_makers' table, whereas the gold SQL correctly joins to the 'countries' table via 'CountryId'.
- **[car_1]** What is the number of makers of care in France?
  - gold: `SELECT count(*) FROM CAR_MAKERS AS T1 JOIN COUNTRIES AS T2 ON T1.Country  =  T2.CountryId WHERE T2.CountryName  =  'france'`
  - pred: `SELECT COUNT(*) FROM car_makers WHERE Country = (SELECT CountryId FROM countries WHERE CountryName = 'France')`
  - 归因: The predicted SQL incorrectly references a 'Country' column in the 'car_makers' table instead of joining with the 'countries' table.

## aggregate_error（25 条）

- **[car_1]** Find the make and production time of the cars that were produced in the earliest year?
  - gold: `SELECT T2.Make ,  T1.Year FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T1.Year  =  (SELECT min(YEAR) FROM CARS_DATA)`
  - pred: `SELECT cn.Make, cd.Year FROM car_names cn JOIN cars_data cd ON cn.MakeId = cd.Id ORDER BY cd.Year ASC LIMIT 1`
  - 归因: The model uses ORDER BY and LIMIT to get one row instead of using a subquery with MIN() to find the earliest year across all cars.
- **[car_1]** How many countries has more than 2 car makers ?
  - gold: `select count(*) from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country group by t1.countryid having count(*)  >  2`
  - pred: `SELECT COUNT(*) FROM (SELECT Country FROM car_makers GROUP BY Country HAVING COUNT(*) > 2)`
  - 归因: The model counts the number of countries meeting the condition instead of counting the total number of car makers across those countries.
- **[car_1]** What is the number of countries with more than 2 car makers ?
  - gold: `select count(*) from countries as t1 join car_makers as t2 on t1.countryid  =  t2.country group by t1.countryid having count(*)  >  2`
  - pred: `SELECT COUNT(*) FROM ( SELECT c.CountryId FROM countries c JOIN car_makers cm ON c.CountryId = cm.Country GROUP BY c.CountryId HAVING COUNT(cm.Id) > 2 )`
  - 归因: The predicted SQL counts the number of qualifying countries in a subquery but fails to wrap this count in an outer SELECT statement, resulting in a table of IDs rather than a single scalar count.
- **[flight_2]** What is the code of airport that has fewest number of flights?
  - gold: `SELECT T1.AirportCode FROM AIRPORTS AS T1 JOIN FLIGHTS AS T2 ON T1.AirportCode  =  T2.DestAirport OR T1.AirportCode  =  T2.SourceAirport GROUP BY T1.AirportCode ORDER BY count(*) LIMIT 1`
  - pred: `SELECT SourceAirport AS AirportCode FROM flights GROUP BY SourceAirport UNION ALL SELECT DestAirport AS AirportCode FROM flights GROUP BY DestAirport ORDER BY COUNT(*) ASC LIMIT 1`  \n  ⚠️ pred 执行失败: 1st ORDER BY term does not match any column in the result set
  - 归因: The model incorrectly applies COUNT(*) to each subquery in a UNION ALL before combining them, rather than counting the total occurrences of each airport code across both source and destination columns after unioning.
- **[museum_visit]** What are the id, name and membership level of visitors who have spent the largest amount of money in total in all museum tickets?
  - gold: `SELECT t2.visitor_id ,  t1.name ,  t1.Level_of_membership FROM visitor AS t1 JOIN visit AS t2 ON t1.id  =  t2.visitor_id GROUP BY t2.visitor_id ORDER BY sum(t2.Total_spent) DESC LIMIT 1`
  - pred: `SELECT v.ID, v.Name, v.Level_of_membership FROM visitor v JOIN visit vis ON v.ID = vis.visitor_ID ORDER BY vis.Total_spent DESC LIMIT 1`
  - 归因: The model failed to aggregate the spending per visitor using SUM and GROUP BY, instead sorting individual visit records by their total spent.

## set_logic_error（15 条）

- **[concert_singer]** Show countries where a singer above age 40 and a singer below 30 are from.
  - gold: `SELECT country FROM singer WHERE age  >  40 INTERSECT SELECT country FROM singer WHERE age  <  30`
  - pred: `SELECT Country FROM singer WHERE Age > 40 UNION SELECT Country FROM singer WHERE Age < 30`
  - 归因: The model used UNION to combine results instead of INTERSECT to find countries common to both age groups.
- **[car_1]** Which are the car makers which produce at least 2 models and more than 3 car makers ? List the id and the maker .
  - gold: `select t1.id ,  t1.maker from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker group by t1.id having count(*)  >=  2 intersect select t1.id ,  t1.maker from car_makers as t1 join model_list as t2 on t1.id  =  t2.maker join car_names as t3 on t2.model  =  t3.model group by t1.id having count(*)  >  3`
  - pred: `SELECT cm.Id, cm.Maker FROM car_makers cm JOIN model_list ml ON cm.Id = ml.Maker GROUP BY cm.Id, cm.Maker HAVING COUNT(ml.ModelId) >= 2 AND (SELECT COUNT(*) FROM car_makers) > 3`
  - 归因: The model incorrectly interprets 'more than 3 car makers' as a global count of all makers in the table rather than applying a second aggregate filter on the same grouped data, failing to use INTERSECT or a proper HAVING clause for the second condition.
- **[flight_2]** What is the code of airport that has the highest number of flights?
  - gold: `SELECT T1.AirportCode FROM AIRPORTS AS T1 JOIN FLIGHTS AS T2 ON T1.AirportCode  =  T2.DestAirport OR T1.AirportCode  =  T2.SourceAirport GROUP BY T1.AirportCode ORDER BY count(*) DESC LIMIT 1`
  - pred: `SELECT SourceAirport AS AirportCode FROM flights GROUP BY SourceAirport ORDER BY COUNT(*) DESC LIMIT 1 UNION ALL SELECT DestAirport AS AirportCode FROM flights GROUP BY DestAirport ORDER BY COUNT(*) DESC LIMIT 1 ORDER BY (SELECT COUNT(*) FROM flights WHERE SourceAirport = AirportCode) + (SELECT COUNT(*) FROM flights WHERE DestAirport = AirportCode) DESC LIMIT 1`  \n  ⚠️ pred 执行失败: near "ORDER": syntax error
  - 归因: The model incorrectly uses UNION ALL with nested ORDER BY clauses and a complex subquery in the final ORDER BY, which is syntactically invalid in SQLite and fails to correctly aggregate flight counts per airport before sorting.
- **[flight_2]** Give the code of the airport with the least flights.
  - gold: `SELECT T1.AirportCode FROM AIRPORTS AS T1 JOIN FLIGHTS AS T2 ON T1.AirportCode  =  T2.DestAirport OR T1.AirportCode  =  T2.SourceAirport GROUP BY T1.AirportCode ORDER BY count(*) LIMIT 1`
  - pred: `SELECT AirportCode FROM airports WHERE AirportCode NOT IN (SELECT SourceAirport FROM flights) AND AirportCode NOT IN (SELECT DestAirport FROM flights) LIMIT 1`
  - 归因: The model incorrectly interprets 'least flights' as airports with zero flights (NOT IN subqueries) instead of counting and ordering by flight frequency.
- **[flight_2]** Which airports do not have departing or arriving flights?
  - gold: `SELECT AirportName FROM Airports WHERE AirportCode NOT IN (SELECT SourceAirport FROM Flights UNION SELECT DestAirport FROM Flights)`
  - pred: `SELECT "AirportCode" FROM "airports" WHERE "AirportCode" NOT IN (SELECT "SourceAirport" FROM "flights") AND "AirportCode" NOT IN (SELECT "DestAirport" FROM "flights")`
  - 归因: The model uses two separate NOT IN conditions with AND instead of combining the flight columns with UNION before applying a single NOT IN check.

## other（5 条）

- **[car_1]** What is the number of the cars with horsepower more than 150?
  - gold: `SELECT count(*) FROM CARS_DATA WHERE horsepower  >  150`
  - pred: `SELECT COUNT(*) FROM cars_data WHERE CAST(Horsepower AS INTEGER) > 150`
  - 归因: The model added an unnecessary CAST operation to the horsepower column which is not present in the gold SQL.
- **[car_1]** What is the number of cars with a horsepower greater than 150?
  - gold: `SELECT count(*) FROM CARS_DATA WHERE horsepower  >  150`
  - pred: `SELECT COUNT(*) FROM cars_data WHERE CAST(Horsepower AS INTEGER) > 150`
  - 归因: The model added an unnecessary CAST operation to the horsepower column which is not present in the gold SQL.
- **[car_1]** What is the maximum miles per gallon of the car with 8 cylinders or produced before 1980 ?
  - gold: `select max(mpg) from cars_data where cylinders  =  8 or year  <  1980`
  - pred: `SELECT MAX(CAST(MPG AS REAL)) FROM cars_data WHERE Cylinders = 8 OR Year < 1980`
  - 归因: The predicted SQL adds an unnecessary CAST operation to the aggregate function which is not present in the gold SQL.
- **[car_1]** What is the maximum mpg of the cars that had 8 cylinders or that were produced before 1980 ?
  - gold: `select max(mpg) from cars_data where cylinders  =  8 or year  <  1980`
  - pred: `SELECT MAX(CAST(MPG AS REAL)) FROM cars_data WHERE Cylinders = 8 OR Year < 1980`
  - 归因: The predicted SQL adds an unnecessary CAST operation to the aggregate function which is not present in the gold SQL.
- **[wta_1]** What are the full names of all players, sorted by birth date?
  - gold: `SELECT first_name ,  last_name FROM players ORDER BY birth_date`
  - pred: `SELECT first_name || ' ' || last_name AS full_name FROM players ORDER BY birth_date`  \n  ⚠️ gold 执行失败: Could not decode to UTF-8 column 'last_name' with text 'Treyes Albarrac��N'
  - 归因: The model's use of string concatenation to create a 'full_name' column is semantically different from the gold SQL's separate columns, though the primary execution failure is caused by an unrelated data encoding issue in the dataset.

## gold_issue（1 条）

- **[wta_1]** List the first and last name of all players in the order of birth date.
  - gold: `SELECT first_name ,  last_name FROM players ORDER BY birth_date`
  - pred: `SELECT first_name, last_name FROM players ORDER BY birth_date`  \n  ⚠️ gold 执行失败: Could not decode to UTF-8 column 'last_name' with text 'Treyes Albarrac��N'
  - 归因: The gold SQL fails to execute due to invalid UTF-8 encoding in the 'last_name' column data.
