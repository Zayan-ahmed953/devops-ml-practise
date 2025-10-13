import pandas as pd

df = pd.DataFrame({
    "age": [25, -1, 32, 45, 200],
    "income": [50000, 60000, 55000, 70000, 65000]
})

# TODO:
# Remove rows where age < 0 or age > 120

# print(df)

fildtred = df[(df['age'] > 0) & (df['age'] < 120)]
print(fildtred)