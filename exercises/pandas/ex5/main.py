import pandas as pd

df = pd.DataFrame({
    "age": [25, 32, 40, 23, 36],
    "salary": [50000, 80000, 120000, 35000, 95000],
    "has_credit": [1, 0, 1, 0, 1]
})

# TODO:
# Select rows where:
# - age > 30
# - salary between 60k and 100k
# - has_credit == 1



filtered = df[(df['age'] > 30) & 
              (df['salary'].between(60000,100000)) & 
              (df['has_credit'] == 1)]


print(filtered)
