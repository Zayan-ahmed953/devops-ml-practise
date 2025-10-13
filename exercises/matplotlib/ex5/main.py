import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np

x = np.random.rand(50)
y = x ** 2

# TODO:
# 1. Plot y = x^2
# 2. Add annotation text at a specific point (e.g., (2, 4))
# 3. Add a text box on the graph explaining the curve

plt.plot(x, y, color='red', marker='o', linestyle='-', linewidth=2, alpha=0.7)

plt.title('Styling with multiplication')
plt.xlabel('This is X axis')
plt.ylabel('This is Y axis')

plt.show()