import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# --- Argument parsing ---
if len(sys.argv) < 2:
    print("Usage: python visualize_pointcloud.py <pointcloud.csv>")
    sys.exit(1)

pointcloud_path = sys.argv[1]

# --- Load point cloud ---
df = pd.read_csv(pointcloud_path)

# If no headers present, uncomment this line:
# df = pd.read_csv(pointcloud_path, names=["x", "y", "z", "votes", "camMask"])

required_cols = {"x", "y", "z", "votes"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must include columns: {required_cols}")

# --- Plot setup ---
initial_min_votes = 1
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

def update_plot(min_votes):
    ax.clear()
    filtered = df[df['votes'] >= min_votes]
    sc = ax.scatter(filtered['x'], filtered['y'], filtered['z'],
                    c=filtered['votes'], cmap='viridis', s=5, alpha=0.7)
    ax.set_title(f"RayVote Point Cloud (min votes ≥ {int(min_votes)})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Votes")
    plt.draw()

# Initial render
update_plot(initial_min_votes)

# --- Slider for filtering ---
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Min Votes', 1, df['votes'].max(), 
                valinit=initial_min_votes, valstep=1)

slider.on_changed(update_plot)

plt.show()
