import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np

x = ['Math', 'Science', 'English']
y = [80, 90, 70]

# TODO:
# 1. Create a bar chart for subjects = ['Math', 'Science', 'English'] and marks = [80, 90, 70]
# 2. Add title, labels, and customize bar colors


plt.plot(x,y, color='red', marker='o', linestyle='-', linewidth = 2, alpha = 0.7)



plt.title('Subject and Marks')
plt.xlabel('Subjects')
plt.ylabel('Marks')

plt.show()