import pandas as pd
data = {
    "age": [25, 30, None, 35, None],
    "salary": [50000, 54000, 58000, None, 62000],
}
df = pd.DataFrame(data)

# TODO:
# 1. Replace missing 'age' with mean
# 2. Replace missing 'salary' with median
# 3. Verify no NaN values remain


mean = df["age"].mean()

df["age"] = df["age"].fillna(mean)
print(df)

median = df["salary"].median()
print(median)

df["salary"] = df["salary"].fillna(median)
print(df)

sum = (df.isna().sum())
print(sum)