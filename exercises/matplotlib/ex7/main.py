import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np


# Apply style
plt.style.use('seaborn-v0_8')

# Generate data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# TODO:
# 1. Create subplots with shared x-axis
# 2. Plot sine and cosine on different subplots
# 3. Label x-axis only once (at the bottom)


# 1. Create subplots with shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# 2. Plot sine and cosine on different subplots
ax1.plot(x, y1, color='orange', label='sin(x)')
ax1.set_title('Sine Wave')
ax1.legend()

ax2.plot(x, y2, color='blue', label='cos(x)')
ax2.set_title('Cosine Wave')
ax2.legend()

# 3. Label x-axis only once (at the bottom)
ax2.set_xlabel('X Values')

# Add common Y-axis labels
ax1.set_ylabel('sin(x)')
ax2.set_ylabel('cos(x)')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
