import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from matplotlib import colormaps

# ------------------------------------------------------------
ORIGIN_LAT = 34.7441
ORIGIN_LON = -86.6705
ORIGIN_ALT = 190.0
M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * np.cos(np.radians(ORIGIN_LAT))

# ------------------------------------------------------------
def load_voxels(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    xs = np.array([v["x"] for v in data], dtype=float)
    ys = np.array([v["y"] for v in data], dtype=float)
    zs = np.array([v["z"] for v in data], dtype=float)
    intensities = np.array([v["intensity"] for v in data], dtype=float)
    return xs, ys, zs, intensities

def geodetic_to_local(lat, lon, alt):
    dx = (lon - ORIGIN_LON) * M_PER_DEG_LON
    dy = (lat - ORIGIN_LAT) * M_PER_DEG_LAT
    dz = alt - ORIGIN_ALT
    return dx, dy, dz

def apply_transform(xs, ys, zs, pre_R, R, t):
    P = np.vstack((xs, ys, zs))
    P_std = pre_R @ P
    Pw = (R @ P_std) + np.reshape(t, (3,1))

    # Fix: Flip Y and Z to align world up correctly
    Pw[1, :] *= -1  # Invert Y
    Pw[2, :] *= -1  # Invert Z

    return Pw[0], Pw[1], Pw[2]


def plot_combined_voxels(voxel_sets, camera_positions_local):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatters = {}
    cmap_list = ['viridis', 'plasma', 'inferno']

    for i, (xs, ys, zs, intensities, label) in enumerate(voxel_sets):
        cmap = colormaps[cmap_list[i % len(cmap_list)]]
        norm = plt.Normalize(vmin=float(np.min(intensities)), vmax=float(np.max(intensities)))
        colors = cmap(norm(intensities))
        scatters[label] = ax.scatter(xs, ys, zs, c=colors, s=5, marker='o', depthshade=True, label=label)

    # Plot camera positions as small stars
    for i, (cam, (x, y, z)) in enumerate(camera_positions_local.items()):
        cmap = colormaps[cmap_list[i % len(cmap_list)]]
        star_color = cmap(0.5)  # middle of the colormap
        ax.scatter(x, y, z, marker='*', s=120, c=[star_color], edgecolor='black', label=f"{cam} (pos)")
        ax.text(x, y, z + 0.5, cam, color='black', fontsize=9, ha='center', weight='bold')

    ax.set_title("Multi-Camera Sparse Voxel Projection", fontsize=14, pad=12)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")

    # Equal aspect ratio
    all_x = np.concatenate([v[0] for v in voxel_sets] + [[v[0]] for v in camera_positions_local.values()])
    all_y = np.concatenate([v[1] for v in voxel_sets] + [[v[1]] for v in camera_positions_local.values()])
    all_z = np.concatenate([v[2] for v in voxel_sets] + [[v[2]] for v in camera_positions_local.values()])
    max_range = np.ptp([all_x, all_y, all_z]).max() / 2.0
    mid_x, mid_y, mid_z = np.mean(all_x), np.mean(all_y), np.mean(all_z)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Toggle buttons
    rax = plt.axes([0.02, 0.4, 0.16, 0.18])
    labels = [v[4] for v in voxel_sets]
    visibility = [True] * len(labels)
    check = CheckButtons(rax, labels, visibility)

    def toggle_visibility(label):
        s = scatters[label]
        s.set_visible(not s.get_visible())
        plt.draw()

    check.on_clicked(toggle_visibility)
    plt.tight_layout()
    plt.show()

def main():
    voxel_files = {
        "Camera1": "out/voxels_1.json",
        "Camera2": "out/voxels_2.json",
        "Camera3": "out/voxels_3.json",
    }

    camera_positions_geo = {
        "Camera1": (34.744090, -86.670359, 190.00),
        "Camera2": (34.744206, -86.670500, 190.00),
        "Camera3": (34.744090, -86.670641, 190.00),
    }

    rotations = {
        "Camera1": np.array([
            [ 1.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0, -0.0, -1.0],
        ], dtype=float),
        "Camera2": np.array([
            [-0.0,  1.0, -0.0],
            [ 1.0,  0.0, -0.0],
            [ 0.0, -0.0, -1.0],
        ], dtype=float),
        "Camera3": np.array([
            [-1.0, -0.0,  0.0],
            [-0.0,  1.0, -0.0],
            [ 0.0, -0.0, -1.0],
        ], dtype=float),
    }

    pre_R = {
        "Camera1": np.diag([1.0, -1.0, 1.0]),
        "Camera2": np.eye(3),
        "Camera3": np.eye(3),
    }

    voxel_sets = []
    camera_positions_local = {}
    for cam, path in voxel_files.items():
        xs, ys, zs, intensities = load_voxels(path)
        dx, dy, dz = geodetic_to_local(*camera_positions_geo[cam])
        rotation = rotations[cam]
        xs_w, ys_w, zs_w = apply_transform(xs, ys, zs, pre_R[cam], rotation, (dx, dy, dz))
        voxel_sets.append((xs_w, ys_w, zs_w, intensities, cam))
        camera_positions_local[cam] = (dx, dy, dz)

    print(f"Loaded voxel data from {len(voxel_sets)} cameras.")
    plot_combined_voxels(voxel_sets, camera_positions_local)

if __name__ == "__main__":
    main()
