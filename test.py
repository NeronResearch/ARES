import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# Path to your JSON directory
json_dir = r"E:\Code\Neron\ARES\Frames\FernBellPark"

# Output video path
output_video = os.path.join(json_dir, "FernBellPark_adjusted_pan_translation.mp4")

# Collect and sort all JSON filenames
json_files = sorted(
    [f for f in os.listdir(json_dir) if f.endswith(".json")],
    key=lambda x: int(x.split(".")[0])
)

# Read one representative JSON to get camera positions
with open(os.path.join(json_dir, json_files[0]), "r") as f:
    base_data = json.load(f)

cameras = base_data.get("cameras", [])
camera_positions = np.array([cam["position_m"] for cam in cameras])

# Compute scene center and radius
center = camera_positions.mean(axis=0)
base_radius = np.linalg.norm(camera_positions - center, axis=1).max()

# Wider zoom
zoom_factor = 8.0
scene_radius = base_radius * zoom_factor

# --- View translation setup ---
# We translate the z-limits upward (so objects appear lower in frame)
translation_up = scene_radius * 0.3  # adjust fraction for desired offset

# Static scene bounds (centered on cameras but shifted upward)
xlim = (center[0] - scene_radius, center[0] + scene_radius)
ylim = (center[1] - scene_radius, center[1] + scene_radius)
zlim = (
    center[2] - scene_radius / 1.2 + translation_up,
    center[2] + scene_radius / 1.2 + translation_up
)

# Prepare directories and video writer
frame_dir = os.path.join(json_dir, "frames_adjusted_pan")
os.makedirs(frame_dir, exist_ok=True)

video_fps = 30
frames_per_json = 15  # 2 fps input → 30 fps output
frame_size = (1280, 720)
video_writer = cv2.VideoWriter(
    output_video, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, frame_size
)

# Iterate through frames
for idx, file_name in enumerate(json_files):
    path = os.path.join(json_dir, file_name)
    with open(path, "r") as f:
        data = json.load(f)

    voxels = data.get("voxels", [])
    targets = data.get("targets", [])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Frame {idx:04d} — Adjusted Pan and Translation")

    # Plot cameras (fixed)
    for cam in cameras:
        x, y, z = cam["position_m"]
        ax.scatter(x, y, z, c="cyan", s=60, marker="^")
        R = np.array(cam["rotation_matrix"])
        direction = R[:, 0] * 5.0
        ax.quiver(x, y, z, direction[0], direction[1], direction[2],
                  color="cyan", length=5)

    # Plot top 5% voxels with inverted intensity
    if voxels:
        intensities = np.array([v["intensity"] for v in voxels])
        threshold = np.percentile(intensities, 95)
        bright_voxels = [v for v in voxels if v["intensity"] >= threshold]

        if bright_voxels:
            xs = [v["position_m"][0] for v in bright_voxels]
            ys = [v["position_m"][1] for v in bright_voxels]
            zs = [v["position_m"][2] for v in bright_voxels]
            top_intensities = np.array([v["intensity"] for v in bright_voxels])

            # Normalize and invert intensity so brighter = darker
            norm_intensity = (top_intensities - top_intensities.min()) / (np.ptp(top_intensities) + 1e-6)
            inverted_intensity = 1 - norm_intensity

            p = ax.scatter(xs, ys, zs, c=inverted_intensity, cmap="gray", s=12, alpha=0.9)

    # Plot targets (e.g., Drone)
    for target in targets:
        x, y, z = target["position_m"]
        ax.scatter(x, y, z, c="lime", s=100, marker="*", label=target["name"])
        ax.text(x, y, z, target["name"], color="black", fontsize=9, weight="bold")  # black label

    # Set static scene bounds
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Pan down to prior view angle
    ax.view_init(elev=25, azim=45)  # lower elevation = looking more horizontally

    plt.tight_layout()

    frame_path = os.path.join(frame_dir, f"{idx:04d}.png")
    plt.savefig(frame_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    # Write each frame 15 times for 30fps playback
    frame = cv2.imread(frame_path)
    frame = cv2.resize(frame, frame_size)
    for _ in range(frames_per_json):
        video_writer.write(frame)

video_writer.release()
print(f"Video saved to: {output_video}")
