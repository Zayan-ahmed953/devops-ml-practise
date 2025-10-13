import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np


data = {'Department': ['IT', 'HR', 'Finance', 'Marketing'],
        'Employees': [40, 10, 20, 15]}
df = pd.DataFrame(data)

# TODO:
# 1. Plot a bar chart using df.plot(kind='bar')
# 2. Add title, x/y labels, and rotate x ticks

df.plot(kind='bar', x='Department', y='Employees', color='skyblue', legend=False)

plt.title('Depratment and employees')
plt.xlabel('Depratment')
plt.ylabel('Employees')
plt.xticks(rotation = 45)

plt.tight_layout()
plt.show()