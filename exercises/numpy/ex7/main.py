import numpy as np

np.random.seed(0)
data = np.random.randn(100) * 10 + 50  # normally distributed
data[[5, 15, 50]] = np.nan  # simulate missing readings
data[[20, 80]] = [200, -100]  # simulate outliers



# 1. Fill NaNs with median 
# 2. Remove outliers beyond 2 std dev 
# 3. Reshape to (100, 1)

median1 = np.nanmedian(data)
print(median1)

filtered = np.nan_to_num(data, nan=median1)
print(filtered)

# Compute z-score
mean = np.mean(filtered)
std = np.std(filtered)
z_score = (filtered - mean) / std
print(z_score)

z_positive = filtered[np.abs(z_score) <= 2]
print(z_positive)

reshaped = np.reshape(z_positive, (98, 1))

print("\n\n\n\n",reshaped)