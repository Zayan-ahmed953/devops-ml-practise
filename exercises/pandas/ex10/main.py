import pandas as pd

df = pd.DataFrame({
    "department": ["IT", "HR", "IT", "Finance", "HR", "IT"],
    "salary": [70000, 50000, 75000, 65000, 52000, 72000]
})

# 1. Group by department
# 2. Calculate average salary per department
avg_salary = df.groupby("department")["salary"].mean().reset_index()

print(avg_salary)
