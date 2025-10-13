import pandas as pd

df = pd.read_csv("data.csv")  # assume a dataset

# TODO:
# 1. Display top 5 rows
# 2. Show column names and data types
# 3. Check for missing values in each column


print(df.head())

print(df.dtypes)

print(df.isnull().sum())