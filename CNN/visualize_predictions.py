import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter

def visualize_frame_with_heatmap(npz_path: str, frames_dir: str, frame_name: str = "0009.json", sigma: float = 2.0):
    """
    Visualize a frame's voxels with a Gaussian heatmap overlay.
    Args:
        npz_path (str): Path to predictions_clip.npz.
        frames_dir (str): Directory containing JSON voxel frames.
        frame_name (str): Which frame to display (e.g., '0009.json').
        sigma (float): Gaussian smoothing sigma.
    """
    # --- Load frame JSON ---
    frame_path = os.path.join(frames_dir, frame_name)
    with open(frame_path, "r") as f:
        frame_data = json.load(f)
    voxels = frame_data.get("voxels", [])
    if not voxels:
        raise RuntimeError(f"No voxels in {frame_path}")

    coords = np.array([v["coordinates"] for v in voxels], dtype=np.int32)
    intensities = np.array([v.get("intensity", 0.0) for v in voxels], dtype=np.float32)

    print(f"Loaded {len(coords)} voxels from {frame_name}")

    # --- Parse target position (assume first target if multiple) ---
    targets = frame_data.get("targets", [])
    target_pos = None
    if targets:
        pos_m = np.array(targets[0].get("position_m", [0, 0, 0]), dtype=np.float32)
        # Convert to grid coordinates
        grid_info = frame_data.get("grid_info", {})
        origin_m = np.array(grid_info.get("origin_m", [0, 0, 0]), dtype=np.float32)
        voxel_size_m = float(grid_info.get("voxel_size_m", 1.0))
        target_pos = np.floor((pos_m - origin_m) / voxel_size_m).astype(int)

    # --- Parse camera positions (if available) ---
    cameras = frame_data.get("cameras", [])
    camera_positions = []
    for cam in cameras:
        pos_m = np.array(cam.get("position_m", [0, 0, 0]), dtype=np.float32)
        grid_info = frame_data.get("grid_info", {})
        origin_m = np.array(grid_info.get("origin_m", [0, 0, 0]), dtype=np.float32)
        voxel_size_m = float(grid_info.get("voxel_size_m", 1.0))
        cam_pos = np.floor((pos_m - origin_m) / voxel_size_m).astype(int)
        camera_positions.append(cam_pos)

    # --- Load prediction data ---
    pred_data = np.load(npz_path)
    coords_all = pred_data["coords"]  # shape [N,4]
    probs_all = pred_data["probs"]    # shape [N]
    # Only keep predictions for the last frame (t=9)
    last_frame_mask = coords_all[:, 3] == 9
    pred_coords = coords_all[last_frame_mask, :3].astype(int)
    pred_probs = probs_all[last_frame_mask].astype(np.float32)

    # Normalize coordinate ranges to align both datasets
    all_coords = np.vstack([coords, pred_coords])
    offset = all_coords.min(axis=0)
    coords -= offset
    pred_coords -= offset
    grid_shape = all_coords.max(axis=0) - all_coords.min(axis=0) + 1

    # --- Build Gaussian heatmap ---
    grid = np.zeros(grid_shape, dtype=np.float32)
    for (x, y, z), p in zip(pred_coords, pred_probs):
        if x < grid_shape[0] and y < grid_shape[1] and z < grid_shape[2]:
            grid[x, y, z] = max(grid[x, y, z], p)
    if grid.max() > 0:
        heatmap = gaussian_filter(grid, sigma=sigma)
        heatmap /= heatmap.max()
    else:
        heatmap = grid

    # --- Prepare plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.2)


    # Initial plot (show all voxels)
    sc_vox = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=intensities, cmap="viridis", s=5, alpha=0.6, label="Frame Voxels")

    # Heatmap overlay (semi-transparent red/orange, INVERTED colormap)
    xs, ys, zs = np.where(heatmap > 0.25)
    vals = heatmap[heatmap > 0.25]
    sc_heat = ax.scatter(xs, ys, zs, c=vals, cmap="hot_r", s=6, alpha=0.35, label="Prediction Heatmap")

    # Target position (red star)
    if target_pos is not None:
        sc_target = ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]],
                               c='red', marker='*', s=120, label='Target', edgecolor='k')
    else:
        sc_target = None

    # Camera positions (blue triangles)
    if camera_positions:
        camera_positions = np.array(camera_positions)
        sc_cams = ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                             c='blue', marker='^', s=60, label='Cameras', edgecolor='k')
    else:
        sc_cams = None

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Frame {frame_name} with Gaussian Prediction Heatmap (σ={sigma})")

    cb = plt.colorbar(sc_vox, ax=ax, pad=0.1)
    cb.set_label("Intensity")

    # --- Sliders for intensity and heatmap threshold ---
    ax_slider_int = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider_int = Slider(ax_slider_int, "Min Intensity", 0.0, intensities.max(), valinit=0.0, valstep=0.1)

    ax_slider_heat = plt.axes([0.25, 0.01, 0.5, 0.03])
    slider_heat = Slider(ax_slider_heat, "Heatmap Threshold", 0.0, 1.0, valinit=0.25, valstep=0.01)

    # --- Checkboxes for toggling elements ---
    from matplotlib.widgets import CheckButtons
    ax_check = plt.axes([0.01, 0.4, 0.15, 0.18])
    labels = ['Voxel Grid', 'Heatmap', 'Target', 'Cameras']
    visibility = [True, True, True, True]
    check = CheckButtons(ax_check, labels, visibility)

    def update(val=None):
        thr_int = slider_int.val
        thr_heat = slider_heat.val
        # Update voxel scatter
        mask = intensities >= thr_int
        sc_vox._offsets3d = (coords[mask, 0], coords[mask, 1], coords[mask, 2])
        sc_vox.set_array(intensities[mask])
        # Update heatmap scatter
        xs, ys, zs = np.where(heatmap > thr_heat)
        vals = heatmap[heatmap > thr_heat]
        sc_heat._offsets3d = (xs, ys, zs)
        sc_heat.set_array(vals)
        fig.canvas.draw_idle()

    def func_check(label):
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
    check.on_clicked(func_check)

    ax.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    visualize_frame_with_heatmap(
        npz_path="out/predictions_clip.npz",
        frames_dir=r"E:\Code\Neron\ARES\Frames\Scenario3",
        frame_name="0009.json",
        sigma=2.0
    )
