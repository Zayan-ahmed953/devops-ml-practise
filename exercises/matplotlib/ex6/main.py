import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np

x = np.arange(0, 10)
y1 = x
y2 = x**2
y3 = np.sqrt(x)

# TODO:
# 1. Create 3 subplots (1 row, 3 columns)
# 2. Plot each y on separate axes
# 3. Add individual titles and grid lines


plt.plot(x, y1, color='blue', marker='x', linestyle='--', linewidth=2, alpha=0.7, label='y=x')
plt.plot(x,y2, color='orange', marker='o', linestyle='-', linewidth=2, alpha=0.7, label="y2")
plt.plot(x,y3, color='purple', marker='v', linestyle=':', linewidth=2, alpha=0.7, label='y3')

plt.title('Multiplot')
plt.legend
plt.xlabel('This is x')
plt.ylabel('This is Y')

plt.show()