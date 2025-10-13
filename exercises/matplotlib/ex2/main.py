import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np

# Generate random data
x = np.random.rand(50)     # 50 random values for x
y = np.random.rand(50)     # 50 random values for y

# TODO steps completed:
# 1. Plot a scatter chart
# 2. Change point color and size
# 3. Add labels

# plt.scatter(x, y, color='orange', s=100, alpha=0.7, edgecolors='black')
plt.scatter(x, y, color='blue', s = 50, alpha=0.6, edgecolors='black')

# Add title and axis labels
plt.title("Random Scatter Plot")
plt.xlabel("X-axis Values")
plt.ylabel("Y-axis Values")

# Show the chart
plt.show()

