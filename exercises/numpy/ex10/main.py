import numpy as np

np.random.seed(42)
data = np.random.randint(-10, 100, size=(10, 3)).astype(float)
data[3, 1] = np.nan
data[7, 2] = np.nan

# TODO:
# Step 1: Replace NaNs with column means
# Step 2: Remove any rows where any value < 0 (invalid)
# Step 3: Normalize each feature to [0, 1]


print(data)

mean = np.nanmean(data, axis=0)
print(mean)

where_nan = np.where(np.isnan(data))
print(where_nan)

data[where_nan] = np.take(mean, where_nan[1])
print(data)