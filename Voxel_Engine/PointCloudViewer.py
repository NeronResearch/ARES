import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

<<<<<<< HEAD
=======
# --- Argument parsing ---
>>>>>>> 77a2458de1394621d4bf95842ad2549aa6ae38ce
if len(sys.argv) < 2:
    print("Usage: python visualize_pointcloud.py <pointcloud.csv>")
    sys.exit(1)

<<<<<<< HEAD
file_path = sys.argv[1]

# --- Custom CSV parsing ---
points = []
cameras = []

with open(file_path, "r") as f:
    lines = f.readlines()

in_cam_section = False
for line in lines[1:]:  # skip header
    line = line.strip()
    if not line:
        continue
    if line.startswith("#CAMERA_POSITIONS"):
        in_cam_section = True
        continue
    parts = line.split(",")
    if in_cam_section:
        cameras.append((float(parts[0]), float(parts[1]), float(parts[2])))
    else:
        points.append({
            "x": float(parts[0]),
            "y": float(parts[1]),
            "z": float(parts[2]),
            "votes": float(parts[3]),
            "camMask": int(parts[4])
        })

df = pd.DataFrame(points)
cam_df = pd.DataFrame(cameras, columns=["x", "y", "z"])

# --- Visualization ---
=======
pointcloud_path = sys.argv[1]

# --- Load point cloud ---
df = pd.read_csv(pointcloud_path)

# If no headers present, uncomment this line:
# df = pd.read_csv(pointcloud_path, names=["x", "y", "z", "votes", "camMask"])

required_cols = {"x", "y", "z", "votes"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must include columns: {required_cols}")

# --- Plot setup ---
>>>>>>> 77a2458de1394621d4bf95842ad2549aa6ae38ce
initial_min_votes = 1
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

<<<<<<< HEAD
filtered = df[df['votes'] >= initial_min_votes]
sc = ax.scatter(filtered['x'], filtered['y'], filtered['z'],
                c=filtered['votes'], cmap='viridis', s=5, alpha=0.7)
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, label="Votes")

# --- NEW: plot cameras ---
ax.scatter(cam_df['x'], cam_df['y'], cam_df['z'],
           c='red', s=60, marker='^', label='Cameras')

ax.legend()
ax.set_title(f"RayVote Point Cloud (min votes ≥ {initial_min_votes})")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.grid(True)

def update_plot(min_votes):
    min_votes = int(min_votes)
    filtered = df[df['votes'] >= min_votes]
    sc._offsets3d = (filtered['x'], filtered['y'], filtered['z'])
    sc.set_array(filtered['votes'])
    ax.set_title(f"RayVote Point Cloud (min votes ≥ {min_votes})")
    fig.canvas.draw_idle()

slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(slider_ax, 'Min Votes', 1, df['votes'].max(),
                valinit=initial_min_votes, valstep=1)
=======
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

>>>>>>> 77a2458de1394621d4bf95842ad2549aa6ae38ce
slider.on_changed(update_plot)

plt.show()
