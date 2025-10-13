import pandas as pd

df = pd.DataFrame({
    "color": ["red", "blue", "green", "blue", "red"],
    "shape": ["circle", "square", "triangle", "circle", "square"]
})

# TODO:
# 1. Perform one-hot encoding on both columns
# 2. Merge encoded columns back into df

encoded = pd.get_dummies(df[["color", "shape"]])
print(encoded)