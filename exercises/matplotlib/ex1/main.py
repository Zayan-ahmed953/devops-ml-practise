import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


# Step 1: Prepare data
x = [1, 2, 3, 4, 5]
y = [10, 12, 8, 14, 10]

# Step 2: Plot the line graph
plt.plot(x, y)

# Step 3: Add a title and axis labels
plt.title("Simple Line Graph")       # Title of the graph
plt.xlabel("X-axis (Numbers)")       # Label for X-axis
plt.ylabel("Y-axis (Values)")        # Label for Y-axis

# Step 4: Show grid lines
plt.grid(True)

# Step 5: Display the graph
plt.show()
