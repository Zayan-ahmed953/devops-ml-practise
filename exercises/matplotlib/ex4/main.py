import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# TODO:
# 1. Plot y1 and y2 with different colors and line styles
# 2. Add legend, title, and labels
# 3. Use 'plt.style.use("seaborn")' or another style

plt.plot(x, y1, color='orange', marker='o', linestyle='-', linewidth=2, alpha=0.7, label='sin(x)')
plt.plot(x, y2, color='blue', marker='x', linestyle='--', linewidth=2, alpha=0.7, label='cos(x)')

# Add legend, title, and labels
plt.legend()
plt.title("Sine and Cosine Waves")
plt.xlabel('X Label')
plt.ylabel('Y Label')

plt.show()
