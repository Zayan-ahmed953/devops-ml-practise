import numpy as np

# Simulated dataset
data = np.array([10, np.nan, 25, np.nan, 40, 55, np.nan])

# TODO:
# 1. Count how many missing values exist
# 2. Replace np.nan with the mean of non-missing values
# 3. Normalize the data (values between 0 and 1)

###############

#Solution:

# 1️⃣ Count how many missing values exist
missing_count = np.isnan(data).sum()
print("Missing values:", missing_count)

# 2️⃣ Replace np.nan with the mean of non-missing values
mean_value = np.nanmean(data)        # mean of only non-missing numbers
final_data = np.nan_to_num(data, nan=mean_value)
print("After replacing NaN with mean:", final_data)

# 3️⃣ Normalize the data (values between 0 and 1)
normalized = (final_data - np.min(final_data)) / (np.max(final_data) - np.min(final_data))
print("Normalized data:", normalized)
