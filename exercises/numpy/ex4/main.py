import numpy as np

features = np.array([[10, 2000],
                     [20, 1800],
                     [15, 2400],
                     [25, 2100]])

# TODO:
# 1. Perform Min-Max scaling for each column
# 2. Perform Z-score normalization for each column

# 1️⃣ Min-Max normalization (values between 0 and 1)
min_vals = np.min(features, axis=0)
max_vals = np.max(features, axis=0)
min_max_scaled = (features - min_vals) / (max_vals - min_vals)

print("Min-Max Scaled:\n", min_max_scaled)

# 2️⃣ Z-Score
mean = np.mean(features, axis=0)
std = np.std(features, axis=0)
z_score_normalized = (features - mean) / std

print("\nZ-Score Normalized:\n", z_score_normalized)