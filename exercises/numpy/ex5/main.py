import numpy as np

age = np.array([20, 25, 30])
salary = np.array([50000, 60000, 65000])
experience = np.array([1, 3, 5])

# TODO:
# Combine these into a single 2D NumPy array (shape: (3,3))
# Then filter rows where salary > 55000 and experience >= 3


combined = np.column_stack((age, salary, experience))
print("Combined Array:\n", combined)

filtered = combined[(combined[:, 1] > 55000) & (combined[:, 2] >= 3)]
print("\nFiltered Rows (salary > 55000 and experience >= 3):\n", filtered)