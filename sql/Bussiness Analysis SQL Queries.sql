SELECT
    COUNT(*) AS Total_Employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        AS Employees_Left,
    SUM(CASE WHEN Attrition = 'No' THEN 1 ELSE 0 END)
        AS Employees_Stayed,

    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*), 2
    ) AS Attrition_Rate,

    ROUND(AVG(Age), 2) AS Average_Age,
    ROUND(AVG(MonthlyIncome), 2) AS Average_Monthly_Income,
    ROUND(AVG(YearsAtCompany), 2) AS Average_Tenure

FROM Employees;

SELECT
    Department,
    COUNT(*) AS Total_Employees,

    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        AS Employees_Left,

    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*), 2
    ) AS Attrition_Rate

FROM Employees
GROUP BY Department
ORDER BY Attrition_Rate DESC;

SELECT
    JobRole,
    COUNT(*) AS Total_Employees,

    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        AS Employees_Left,

    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*), 2
    ) AS Attrition_Rate

FROM Employees
GROUP BY JobRole
ORDER BY Attrition_Rate DESC;


SELECT
    OverTime,
    COUNT(*) AS Total_Employees,

    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        AS Employees_Left,

    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*), 2
    ) AS Attrition_Rate

FROM Employees
GROUP BY OverTime;

SELECT
    CASE
        WHEN MonthlyIncome < 3000 THEN 'Low Income'
        WHEN MonthlyIncome BETWEEN 3000 AND 6000 THEN 'Medium Income'
        WHEN MonthlyIncome BETWEEN 6001 AND 10000 THEN 'High Income'
        ELSE 'Very High Income'
    END AS Income_Group,

    COUNT(*) AS Total_Employees,

    ROUND(
        SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*), 2
    ) AS Attrition_Rate

FROM Employees
GROUP BY Income_Group
ORDER BY Attrition_Rate DESC;