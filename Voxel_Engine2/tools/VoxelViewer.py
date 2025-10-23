import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_voxels(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    xs = [v["x"] for v in data]
    ys = [v["y"] for v in data]
    zs = [v["z"] for v in data]
    intensities = [v["intensity"] for v in data]
    return np.array(xs), np.array(ys), np.array(zs), np.array(intensities)

def plot_voxels(xs, ys, zs, intensities):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize intensities for colormap
    norm = plt.Normalize(vmin=min(intensities), vmax=max(intensities))
    colors = plt.cm.viridis(norm(intensities))

    ax.scatter(xs, ys, zs, c=colors, s=5, marker='o', depthshade=True)
    ax.set_title("Sparse Voxel Projection", fontsize=14, pad=12)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    # Equal aspect ratio
    max_range = np.ptp([xs, ys, zs]).max() / 2.0
    mid_x = (np.max(xs) + np.min(xs)) * 0.5
    mid_y = (np.max(ys) + np.min(ys)) * 0.5
    mid_z = (np.max(zs) + np.min(zs)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.6, label='Intensity')

    plt.tight_layout()
    plt.show()

def main():
    json_path = "out/voxels.json"
    xs, ys, zs, intensities = load_voxels(json_path)
    print(f"Loaded {len(xs)} voxels from {json_path}")
    plot_voxels(xs, ys, zs, intensities)

if __name__ == "__main__":
    main()
