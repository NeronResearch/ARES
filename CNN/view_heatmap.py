#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter


def visualize_heatmap_with_frame(csv_path: str, frame_path: str, sigma: float = 2.0):
    """
    Visualize 3D voxel data and prediction heatmap with interactivity.

    Args:
        csv_path (str): Path to a CSV file containing x,y,z,t,prob.
        frame_path (str): Path to a JSON file containing frame voxels, targets, cameras, and grid info.
        sigma (float): Gaussian smoothing for probability field.
    """

    # --- Load frame JSON ---
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"{frame_path} not found")
    with open(frame_path, "r") as f:
        frame_data = json.load(f)

    voxels = frame_data.get("voxels", [])
    if not voxels:
        raise RuntimeError(f"No voxels found in {frame_path}")

    coords_vox = np.array([v["coordinates"] for v in voxels], dtype=int)
    intensities = np.array([v.get("intensity", 0.0) for v in voxels], dtype=float)

    # --- Targets ---
    targets = frame_data.get("targets", [])
    target_pos = None
    if targets:
        pos_m = np.array(targets[0].get("position_m", [0, 0, 0]), dtype=float)
        grid_info = frame_data.get("grid_info", {})
        origin_m = np.array(grid_info.get("origin_m", [0, 0, 0]), dtype=float)
        voxel_size_m = float(grid_info.get("voxel_size_m", 1.0))
        target_pos = np.floor((pos_m - origin_m) / voxel_size_m).astype(int)

    # --- Cameras ---
    cameras = frame_data.get("cameras", [])
    camera_positions = []
    grid_info = frame_data.get("grid_info", {})
    origin_m = np.array(grid_info.get("origin_m", [0, 0, 0]), dtype=float)
    voxel_size_m = float(grid_info.get("voxel_size_m", 1.0))
    for cam in cameras:
        pos_m = np.array(cam.get("position_m", [0, 0, 0]), dtype=float)
        cam_pos = np.floor((pos_m - origin_m) / voxel_size_m).astype(int)
        camera_positions.append(cam_pos)
    camera_positions = np.array(camera_positions) if camera_positions else None

    # --- Load CSV ---
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    df = pd.read_csv(csv_path)
    required_cols = {"x", "y", "z", "t", "prob"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain columns: x,y,z,t,prob")

    t_sel = df["t"].iloc[0]
    df = df[df["t"] == t_sel]
    coords_pred = df[["x", "y", "z"]].values.astype(int)
    probs_pred = df["prob"].values.astype(float)

    # --- Normalize and align coordinates ---
    all_coords = np.vstack([coords_vox, coords_pred])
    offset = all_coords.min(axis=0)
    coords_vox -= offset
    coords_pred -= offset
    if target_pos is not None:
        target_pos = (target_pos - offset).astype(int)
    if camera_positions is not None:
        camera_positions = camera_positions - offset
    grid_shape = all_coords.max(axis=0) - all_coords.min(axis=0) + 1

    # --- Build and smooth probability grid ---
    grid = np.zeros(grid_shape, dtype=np.float32)
    for (x, y, z), p in zip(coords_pred, probs_pred):
        if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[2]:
            grid[x, y, z] = max(grid[x, y, z], p)

    heatmap = gaussian_filter(grid, sigma=sigma)
    heatmap /= heatmap.max() if heatmap.max() > 0 else 1.0

    # --- Plot setup ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.2, left=0.18)

    # Base voxel cloud
    sc_vox = ax.scatter(coords_vox[:, 0], coords_vox[:, 1], coords_vox[:, 2],
                        c=intensities, cmap="viridis", s=5, alpha=0.6, label="Frame Voxels")

    # Heatmap overlay (start at 25th percentile)
    thr_init = 0.25
    xs, ys, zs = np.where(heatmap > thr_init)
    vals = heatmap[heatmap > thr_init]
    sc_heat = ax.scatter(xs, ys, zs, c=vals, cmap="hot_r", s=8, alpha=0.35, label="Prediction Heatmap")

    # Target and cameras
    sc_target = None
    sc_cams = None
    if target_pos is not None:
        sc_target = ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]],
                               c='red', marker='*', s=120, edgecolor='k', label='Target')
    if camera_positions is not None and len(camera_positions) > 0:
        sc_cams = ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                             c='blue', marker='^', s=60, edgecolor='k', label='Cameras')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Frame with Prediction Heatmap (σ={sigma}, t={t_sel})")

    cb = plt.colorbar(sc_vox, ax=ax, pad=0.1, shrink=0.8)
    cb.set_label("Intensity")

    # --- Sliders ---
    ax_slider_int = plt.axes([0.25, 0.05, 0.6, 0.03])
    slider_int = Slider(ax_slider_int, "Min Intensity", 0.0, intensities.max(), valinit=0.0, valstep=0.1)

    ax_slider_heat = plt.axes([0.25, 0.01, 0.6, 0.03])
    slider_heat = Slider(ax_slider_heat, "Heatmap Threshold", 0.0, 1.0, valinit=thr_init, valstep=0.01)

    # --- Checkboxes ---
    ax_check = plt.axes([0.01, 0.35, 0.15, 0.22])
    labels = ['Voxel Grid', 'Heatmap', 'Target', 'Cameras']
    visibility = [True, True, True, True]
    check = CheckButtons(ax_check, labels, visibility)

    # --- Update logic ---
    def update(val=None):
        thr_int = slider_int.val
        thr_heat = slider_heat.val

        # Update voxel scatter
        mask = intensities >= thr_int
        sc_vox._offsets3d = (coords_vox[mask, 0], coords_vox[mask, 1], coords_vox[mask, 2])
        sc_vox.set_array(intensities[mask])

        # Update heatmap
        xs, ys, zs = np.where(heatmap > thr_heat)
        vals = heatmap[heatmap > thr_heat]
        sc_heat._offsets3d = (xs, ys, zs)
        sc_heat.set_array(vals)

        ax.set_title(f"Frame with Prediction Heatmap (σ={sigma}, t={t_sel}) - Thr={thr_heat:.2f}")
        fig.canvas.draw_idle()

    def toggle(label):
        idx = labels.index(label)
        if idx == 0:
            sc_vox.set_visible(check.get_status()[0])
        elif idx == 1:
            sc_heat.set_visible(check.get_status()[1])
        elif idx == 2 and sc_target is not None:
            sc_target.set_visible(check.get_status()[2])
        elif idx == 3 and sc_cams is not None:
            sc_cams.set_visible(check.get_status()[3])
        fig.canvas.draw_idle()

    slider_int.on_changed(update)
    slider_heat.on_changed(update)
    check.on_clicked(toggle)

    ax.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    visualize_heatmap_with_frame(
        csv_path=r"E:\Code\Neron\ARES\CNN\output\116-125_heatmap.csv",
        frame_path=r"E:\Code\Neron\ARES\Frames\FernBellPark\0125.json",
        sigma=2.0
    )
