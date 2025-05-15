import re
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Input string (truncated in practice – here you use your full text)
with open(r"C:\Users\aggel\Desktop\your_log_file.txt", encoding="utf-8") as f:
    text = f.read()

# Extract positions from both formats
points_paren = re.findall(r'Position\s*=\s*\(([-\d.e+]+), ([-\d.e+]+), ([-\d.e+]+)\)', text)
points_brackets = re.findall(r'Position\s*=\s*\[\s*([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\s*\]', text)

# Combine and convert
all_points = points_paren + points_brackets

all_points_float = [(float(x), float(y), float(z)) for x, y, z in all_points]

point_paren_float = [(float(x), float(y), float(z)) for x, y, z in points_paren]
point_brackets_float = [(float(x), float(y), float(z)) for x, y, z in points_brackets]
# Create DataFrame
df_points = pd.DataFrame(all_points_float, columns=["X", "Y", "Z"])
df_points_paren = pd.DataFrame(point_paren_float, columns=["X", "Y", "Z"])

df_points_brackets = pd.DataFrame(point_brackets_float, columns=["X", "Y", "Z"])

# Plot only the points (no lines)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_points["X"], df_points["Y"], df_points["Z"], marker='o')  # ← scatter, not plot
ax.set_title("Extracted Trajectory Points")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load all data
positions = np.load(r"C:\Users\aggel\Desktop\planned_trajectory_positions.npy")
positions_cross = np.load(r"C:\Users\aggel\Desktop\planned_CROSS_positions.npy")
positions_convex = np.load(r"C:\Users\aggel\Desktop\planned_cONVEX_positions.npy")

# Convert to arrays (if not already)
positions = np.array(positions)
positions_cross = np.array(positions_cross)
positions_convex = np.array(positions_convex)

# Create a single 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each set with different styles
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'o-', label='Total Trajectory')
ax.plot(positions_cross[:, 0], positions_cross[:, 1], positions_cross[:, 2], 'x-', label='Cross Trajectory')
ax.plot(positions_convex[:, 0], positions_convex[:, 1], positions_convex[:, 2], '-c', label='Convex Trajectory')
ax.scatter(df_points_brackets["X"], df_points_brackets["Y"], df_points_brackets["Z"], marker='o')
# Labels & Legend
ax.set_title("Full Trajectory Visualization")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.tight_layout()
plt.show()
