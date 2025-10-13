import numpy as np

data = np.array([[1, 45],
                 [2, -1],   # Invalid age
                 [3, 30],
                 [4, 200],  # Invalid age
                 [5, 25]])

# TODO:
# Remove rows where the age column (index 1) is not between 0 and 120


filtered_data = data[(data[:, 1] >= 0) & (data[:, 1] <= 120)]
print(filtered_data)