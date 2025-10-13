import numpy as np

colors = np.array(["red", "blue", "green", "blue", "red"])

# TODO:
# Convert 'colors' into a one-hot encoded 2D array using only NumPy



# Solution:

unique_colors = np.unique(colors)
print(unique_colors)

# Create an empty 2D array (len(colors) x len(unique_colors))
one_hot = np.zeros((colors.size, unique_colors.size), dtype=int)
print(one_hot)

# Fill the one-hot matrix
for i, color in enumerate(colors):
    one_hot[i, unique_colors == color] = 1

print("Unique colors:", unique_colors)
print("One-hot encoded array:\n", one_hot)

