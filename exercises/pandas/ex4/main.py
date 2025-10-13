import pandas as pd

df = pd.DataFrame({
    "name": ["Ali", "Sara", "Umer", "Hassan"],
    "age": [23, 27, 29, None],
    "city": ["Lahore", "Karachi", "Islamabad", "Peshawar"]
})

# TODO:
# 1. Drop the 'city' column
# 2. Drop any rows where 'age' is missing


dropped = df.drop(columns=["city"])
print(dropped)

fresh = dropped.dropna(subset=["age"])
print(fresh)