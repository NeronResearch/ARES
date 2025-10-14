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

    # --- Align target and camera positions with the same offset ---
    # target_pos and camera_positions were computed in original grid coordinates
    # so they must be shifted by the same offset used for coords and pred_coords.
    if target_pos is not None:
        try:
            target_pos = (target_pos - offset).astype(int)
        except Exception:
            # Fallback: ensure array conversion
            target_pos = np.array(target_pos, dtype=int) - offset

    if len(camera_positions) > 0:
        # camera_positions is a list of positions in grid coords; shift each
        try:
            camera_positions = [np.array(cp, dtype=int) - offset for cp in camera_positions]
            camera_positions = np.array(camera_positions)
        except Exception:
            # If already numpy array, just subtract
            camera_positions = np.array(camera_positions) - offset

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


    # Heatmap overlay: display top 1% most intense voxels by default
    nonzero_vals = heatmap[heatmap > 0.0]
    if nonzero_vals.size > 0:
        top1_thr = float(np.percentile(nonzero_vals, 99.0))
    else:
        top1_thr = 0.0

    mask_init = heatmap >= top1_thr
    xs, ys, zs = np.where(mask_init)
    vals = heatmap[mask_init]
    sc_heat = ax.scatter(xs, ys, zs, c=vals, cmap="hot_r", s=8, alpha=0.8, label="Prediction Heatmap")

    # Target position (red star)
    if target_pos is not None:
        sc_target = ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]],
                               c='red', marker='*', s=140, label='Target', edgecolor='k')
    else:
        sc_target = None

    # Camera positions (blue triangles)
    if len(camera_positions) > 0:
        camera_positions = np.array(camera_positions)
        sc_cams = ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                             c='blue', marker='^', s=80, label='Cameras', edgecolor='k')
    else:
        sc_cams = None

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Frame {frame_name} Prediction Heatmap (σ={sigma})")

    # Colorbar for heatmap
    cb = plt.colorbar(sc_heat, ax=ax, pad=0.1)
    cb.set_label("Prediction Score")

    # --- Fix the 3D view so it's always centered on the cameras (or fallback) ---
    def _compute_view_limits(camera_positions, target_pos, coords, grid_shape):
        # Prefer cameras; fallback to target; fallback to centroid of available coords
        if camera_positions is not None and len(camera_positions) > 0:
            pts = np.array(camera_positions)
        elif target_pos is not None:
            pts = np.array([target_pos])
        else:
            # Use centroid of data coords if nothing else
            pts = np.array(coords)

        center = pts.mean(axis=0).astype(float)

        # Compute an extent that covers all camera points with a margin
        dists = np.linalg.norm(pts - center, axis=1)
        max_rad = float(dists.max()) if dists.size > 0 else 0.0
        # Use a minimum radius so very close cameras still show a reasonable area
        min_rad = max(10.0, max(grid_shape) * 0.05)
        margin = max(5.0, min_rad * 0.2)
        extent = max(max_rad, min_rad) + margin

        # Ensure extent is at least 1 in case of degenerate data
        extent = max(extent, 1.0)

        xlim = (center[0] - extent, center[0] + extent)
        ylim = (center[1] - extent, center[1] + extent)
        zlim = (center[2] - extent, center[2] + extent)
        return xlim, ylim, zlim

    xlim, ylim, zlim = _compute_view_limits(camera_positions if len(camera_positions) > 0 else None,
                                            target_pos, coords, grid_shape)

    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)

    # Lock the view so autoscaling doesn't change limits when updating scatter
    ax.get_proj = lambda: Axes3D.get_proj(ax)

    # --- Slider for heatmap threshold only ---
    ax_slider_heat = plt.axes([0.25, 0.05, 0.5, 0.03])
    # Initialize slider to the computed top-1% threshold so the plot shows only top 1% by default
    slider_heat = Slider(ax_slider_heat, "Heatmap Threshold", 0.0, 1.0, valinit=top1_thr, valstep=0.01)

    # --- Checkboxes for toggling elements ---
    from matplotlib.widgets import CheckButtons
    ax_check = plt.axes([0.01, 0.4, 0.15, 0.12])
    labels = ['Heatmap', 'Target', 'Cameras']
    visibility = [True, True, True]
    check = CheckButtons(ax_check, labels, visibility)

    def update(val=None):
        thr_heat = slider_heat.val
        # Update heatmap scatter
        xs, ys, zs = np.where(heatmap > thr_heat)
        vals = heatmap[heatmap > thr_heat]
        sc_heat._offsets3d = (xs, ys, zs)
        sc_heat.set_array(vals)
        fig.canvas.draw_idle()

    def func_check(label):
        idx = labels.index(label)
        if idx == 0:
            sc_heat.set_visible(check.get_status()[0])
        elif idx == 1 and sc_target is not None:
            sc_target.set_visible(check.get_status()[1])
        elif idx == 2 and sc_cams is not None:
            sc_cams.set_visible(check.get_status()[2])
        fig.canvas.draw_idle()

    slider_heat.on_changed(update)
    check.on_clicked(func_check)

    ax.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    # Batch process: iterate npz files in infer_frames/ and generate PNGs
    def batch_process(infer_dir: str, frames_dir: str, out_dir: str, sigma: float = 2.0):
        os.makedirs(out_dir, exist_ok=True)
        # Look for .npz files in infer_dir
        npz_files = sorted([f for f in os.listdir(infer_dir) if f.lower().endswith('.npz')])
        if not npz_files:
            print(f"No .npz files found in {infer_dir}")
            return

        for npz_fname in npz_files:
            npz_path = os.path.join(infer_dir, npz_fname)
            # Expect file name like 0010.npz -> frame_name 0010.json
            base = os.path.splitext(npz_fname)[0]
            frame_name = f"{base}.json"
            frame_path = os.path.join(frames_dir, frame_name)
            if not os.path.exists(frame_path):
                print(f"Warning: matching frame JSON not found for {npz_fname}: expected {frame_path}")
                continue

            # Render to a figure but save instead of show
            try:
                # Create a figure within the visualize function by temporarily capturing plt
                # We'll replicate a simplified version of the plotting code but direct it to save
                # Reuse visualize_frame_with_heatmap but suppress interactive show by creating a wrapper
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")
                # Call the core visualization to populate the current figure by passing a small flag
                # Since visualize_frame_with_heatmap currently calls plt.show() at the end, we'll
                # instead load data and draw here by reusing its internals. To avoid duplication
                # complexity we will call visualize_frame_with_heatmap but monkey-patch plt.show
                real_show = plt.show
                plt.show = lambda *a, **k: None
                try:
                    visualize_frame_with_heatmap(npz_path=npz_path, frames_dir=frames_dir, frame_name=frame_name, sigma=sigma)
                finally:
                    plt.show = real_show

                # The above call creates its own figure and shows it suppressed; grab the current fig
                cur_fig = plt.gcf()
                out_path = os.path.join(out_dir, f"{base}.png")
                cur_fig.savefig(out_path, dpi=150)
                plt.close(cur_fig)
                print(f"Saved {out_path}")
            except Exception as e:
                print(f"Failed to render {npz_fname}: {e}")

    infer_dir = os.path.join(os.path.dirname(__file__), 'infer_frames')
    frames_dir = r"E:\Code\Neron\ARES\Frames\FernBellPark"
    out_dir = os.path.join(os.path.dirname(__file__), 'graphs')
    batch_process(infer_dir=infer_dir, frames_dir=frames_dir, out_dir=out_dir, sigma=2.0)