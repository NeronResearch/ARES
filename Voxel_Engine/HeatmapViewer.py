import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from scipy.ndimage import gaussian_filter

def visualize_heatmap(raw_voxel_path: str, inference_path: str, sigma: float = 2.0):
    # --- Load raw voxel JSON ---
    with open(raw_voxel_path, "r") as f:
        raw_data = json.load(f)

    voxels = raw_data.get("voxels", [])
    if not voxels:
        raise RuntimeError(f"No voxels found in {raw_voxel_path}")

    coords_raw = np.array([v["coordinates"] for v in voxels], dtype=int)
    intensities = np.array([v.get("intensity", 0.0) for v in voxels], dtype=np.float32)

    # --- Apply grid_info transform if available ---
    grid_info = raw_data.get("grid_info", {})
    origin = np.array(grid_info.get("origin_m", [0, 0, 0]), dtype=float)
    voxel_size = grid_info.get("voxel_size_m", 1.0)
    if isinstance(voxel_size, list):
        voxel_size = np.array(voxel_size, dtype=float)
    else:
        voxel_size = np.array([voxel_size] * 3, dtype=float)

    coords_raw_world = coords_raw * voxel_size + origin

    print(f"[INFO] Loaded {len(coords_raw)} raw voxels with origin={origin}, voxel_size={voxel_size}")

    # --- Parse targets ---
    targets = raw_data.get("targets", [])
    target_positions = []
    for t in targets:
        pos = t.get("position_m") or t.get("coordinates") or [0, 0, 0]
        target_positions.append(np.array(pos, dtype=float))

    # --- Parse cameras ---
    cameras = raw_data.get("cameras", [])
    camera_positions = []
    for c in cameras:
        pos = c.get("position_m") or c.get("coordinates") or [0, 0, 0]
        camera_positions.append(np.array(pos, dtype=float))

    # --- Load inference heatmap JSON ---
    with open(inference_path, "r") as f:
        infer_data = json.load(f)

    coords_inf = np.array(infer_data["coords"])[:, :3].astype(int)
    probs_inf = np.array(infer_data["probs"], dtype=np.float32)
    coords_inf_world = coords_inf * voxel_size + origin

    print(f"[INFO] Loaded {len(coords_inf)} inference voxels")

    # --- Build Gaussian heatmap grid ---
    grid_shape = coords_inf.max(axis=0) + 1
    grid = np.zeros(grid_shape, dtype=np.float32)
    for (x, y, z), p in zip(coords_inf, probs_inf):
        grid[x, y, z] = max(grid[x, y, z], p)
    heatmap = gaussian_filter(grid, sigma=sigma)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # --- Setup figure ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.25)

    # --- Initial render parameters ---
    voxel_thresh = 0.25
    intensity_thresh = 0.0

    # --- Raw voxel points ---
    sc_vox = ax.scatter(coords_raw_world[:, 0], coords_raw_world[:, 1], coords_raw_world[:, 2],
                        c=intensities, cmap="viridis", s=5, alpha=0.6, label="Raw Voxels")

    # --- Inference heatmap overlay ---
    xs, ys, zs = np.where(heatmap > voxel_thresh)
    vals = heatmap[heatmap > voxel_thresh]
    coords_heat_world = np.stack([xs, ys, zs], axis=1) * voxel_size + origin
    sc_heat = ax.scatter(coords_heat_world[:, 0], coords_heat_world[:, 1], coords_heat_world[:, 2],
                         c=vals, cmap="hot_r", s=6, alpha=0.35, label="Heatmap")

    # --- Targets ---
    sc_target = None
    if len(target_positions) > 0:
        tpos = np.array(target_positions)
        sc_target = ax.scatter(tpos[:, 0], tpos[:, 1], tpos[:, 2],
                               c='red', marker='*', s=120, edgecolor='k', label='Targets')

    # --- Cameras ---
    sc_cams = None
    if len(camera_positions) > 0:
        cpos = np.array(camera_positions)
        sc_cams = ax.scatter(cpos[:, 0], cpos[:, 1], cpos[:, 2],
                             c='blue', marker='^', s=60, edgecolor='k', label='Cameras')

    # --- Axes and titles ---
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Voxel Scene with Inference Heatmap (σ={sigma})")
    cb = plt.colorbar(sc_vox, ax=ax, pad=0.1)
    cb.set_label("Raw Intensity")

    # --- Sliders ---
    ax_slider_int = plt.axes([0.25, 0.08, 0.5, 0.03])
    slider_int = Slider(ax_slider_int, "Min Intensity", 0.0, intensities.max(), valinit=intensity_thresh, valstep=0.01)

    ax_slider_heat = plt.axes([0.25, 0.03, 0.5, 0.03])
    slider_heat = Slider(ax_slider_heat, "Heatmap Threshold", 0.0, 1.0, valinit=voxel_thresh, valstep=0.01)

    # --- Checkboxes for visibility toggles ---
    ax_check = plt.axes([0.01, 0.35, 0.15, 0.25])
    labels = ['Raw Voxels', 'Heatmap', 'Targets', 'Cameras']
    visibility = [True, True, True, True]
    check = CheckButtons(ax_check, labels, visibility)

    def update(_=None):
        thr_int = slider_int.val
        thr_heat = slider_heat.val

        # update raw voxels
        mask = intensities >= thr_int
        sc_vox._offsets3d = (coords_raw_world[mask, 0], coords_raw_world[mask, 1], coords_raw_world[mask, 2])
        sc_vox.set_array(intensities[mask])

        # update heatmap
        xs, ys, zs = np.where(heatmap > thr_heat)
        vals = heatmap[heatmap > thr_heat]
        coords_heat_world = np.stack([xs, ys, zs], axis=1) * voxel_size + origin
        sc_heat._offsets3d = (coords_heat_world[:, 0], coords_heat_world[:, 1], coords_heat_world[:, 2])
        sc_heat.set_array(vals)

        fig.canvas.draw_idle()

    def toggle(label):
        states = check.get_status()
        sc_vox.set_visible(states[0])
        sc_heat.set_visible(states[1])
        if sc_target:
            sc_target.set_visible(states[2])
        if sc_cams:
            sc_cams.set_visible(states[3])
        fig.canvas.draw_idle()

    slider_int.on_changed(update)
    slider_heat.on_changed(update)
    check.on_clicked(toggle)

    ax.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python HeatmapViewer.py <raw_voxel.json> <inference_heatmap.json>")
        sys.exit(1)
    raw_voxel_path = sys.argv[1]
    inference_path = sys.argv[2]
    visualize_heatmap(raw_voxel_path, inference_path, sigma=2.0)
