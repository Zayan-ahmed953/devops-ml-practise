import pandas as pd

df = pd.DataFrame({
    "height_cm": [180, 165, 170, 155],
    "weight_kg": [80, 55, 65, 50]
})

# TODO:
# 1. Create a new column 'BMI' = weight / (height_m^2)
# 2. Classify each row as 'Healthy' or 'Overweight' based on BMI > 25

height_m = df['height_cm'] / 100

df['BMI'] = df['weight_kg'] / height_m ** 2
print(df)

df['Status'] = df['BMI'].apply(lambda x: 'Overweight' if x > 25 else 'Healthy')
print(df)