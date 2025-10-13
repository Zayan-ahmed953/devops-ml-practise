import numpy as np

dataset = np.array([
    [25, 50000, 1],
    [35, 80000, 0],
    [45, 120000, 1],
    [23, 35000, 0],
    [31, 90000, 1]
])

# Columns: [Age, Income, Has_Credit_Card]

# TODO:
# Select only rows where:
# - Age > 30
# - Income between 60k and 100k
# - Has_Credit_Card == 1


filtered = dataset[(dataset[:, 0] > 30) & (dataset[:,1] > 60000) & (dataset[:,1] < 100000) & (dataset[:, 2] == 1) ]
print(filtered)

filteree = dataset[(dataset[:, 0] > 30) & (dataset[:, 1] > 60000) & (dataset[:, 2] == 1)]
print(filteree)