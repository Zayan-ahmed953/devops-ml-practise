import numpy as np

# Suppose you have raw features:
raw_features = np.array([[1, 20, 3],
                         [2, 30, 2],
                         [3, 25, 4]])

# TODO:
# 1. Center each column around its mean
# 2. Normalize each column by its std deviation
# 3. Append a bias term (column of ones) to the left


# 1️⃣ Centering
mean = np.mean(raw_features, axis=0)
print(mean)

centered = raw_features - mean
print(centered)

std = np.std(centered, axis=0)
print(std)

normalized = centered / std
print(normalized)

bias = np.ones((raw_features.shape[0], 1))
print(bias)

final = np.hstack([bias, normalized])
print(final)