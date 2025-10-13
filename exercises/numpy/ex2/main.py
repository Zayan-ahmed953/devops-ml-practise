import numpy as np

data = np.array([2, 3, 2, 5, 2, 100, 3, 2, 4, 3])

# TODO:
# 1. Calculate z-scores for each value
# 2. Filter out any data points where |z| > 2


############

#SOLUTION

# 1️⃣ Calculate z-scores
mean = np.mean(data)
std = np.std(data)
z_scores = (data - mean) / std

print("Z-scores:", z_scores)

# 2️⃣ Filter out values where |z| > 2  (outliers)
positive = np.abs(z_scores)
print(positive)

filtered_data = data[np.abs(z_scores) <= 2]

print("Filtered data:", filtered_data)
