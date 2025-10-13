import pandas as pd

df = pd.DataFrame({
    "age": [18, 25, 32, 45, 52, 61, 73]
})

# TODO:
# 1. Create a new column 'age_group' dividing ages into:
#    'Young' (0–30), 'Adult' (31–60), 'Senior' (61+)


print(df)

df["age_group"] = df['age'].apply(lambda x: 'Young' if 0 < x < 30 else ('Adult' if 31 <= x <= 60 else 'Seniour'))

print(df)


# df['BMI'].apply(lambda x: 'Overweight' if x > 25 else 'Healthy')