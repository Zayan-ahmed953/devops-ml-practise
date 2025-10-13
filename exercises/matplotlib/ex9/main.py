import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# 1️⃣ Generate random data
data = np.random.randn(1000)  # 1000 random numbers from standard normal distribution

# 2️⃣ Plot histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)  # bins=30 controls number of bars

# 3️⃣ Add vertical line for mean
mean_value = np.mean(data)
plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_value:.2f}')

# Add title and labels
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show legend
plt.legend()

# Adjust layout
plt.tight_layout()

# Display plot
plt.show()
