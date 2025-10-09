import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------
# Model Definition (must match train.py)
# --------------------------------------------------
class Temporal3DUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1):
        super(Temporal3DUNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, 3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, 3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_ch, 16)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)
        self.pool = nn.MaxPool3d(2)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.up2 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.dec2 = conv_block(32, 16)
        self.outc = nn.Conv3d(16, out_ch, 1)

    def forward(self, x):
        B, C, T, D, H, W = x.shape
        outs = []
        for t in range(T):
            xt = x[:, :, t]
            e1 = self.enc1(xt)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            d2 = self.up1(e3)
            d2 = torch.cat([d2, e2], dim=1)
            d2 = self.dec1(d2)
            d1 = self.up2(d2)
            d1 = torch.cat([d1, e1], dim=1)
            d1 = self.dec2(d1)
            out = torch.sigmoid(self.outc(d1))
            outs.append(out)
        return torch.stack(outs, dim=2)


# --------------------------------------------------
# Data Loader
# --------------------------------------------------
def load_voxel_frames(data_dir):
    json_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])
    all_voxels = []

    for jf in json_files:
        path = os.path.join(data_dir, jf)
        with open(path, "r") as f:
            data = json.load(f)

        voxels = data.get("voxels", [])
        grid_info = data.get("grid_info", {})
        dims = grid_info.get("dimensions", [64, 64, 64])

        arr = np.zeros((2, *dims), dtype=np.float32)
        out_of_bounds = 0

        for v in voxels:
            x, y, z = v["coordinates"]
            if (0 <= x < dims[0]) and (0 <= y < dims[1]) and (0 <= z < dims[2]):
                arr[0, x, y, z] = v.get("intensity", 0.0)
                arr[1, x, y, z] = v.get("motion_type", 0.0)
            else:
                out_of_bounds += 1

        if out_of_bounds > 0:
            print(f"[{jf}] Skipped {out_of_bounds} voxels (out of bounds)")

        all_voxels.append(arr)

    volume = np.stack(all_voxels, axis=0)
    volume = volume / (np.max(volume) + 1e-8)
    return volume


# --------------------------------------------------
# Visualization
# --------------------------------------------------
def visualize_voxels(voxel_sequence, interval=0.4):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt.ion()

    for t in range(voxel_sequence.shape[0]):
        ax.clear()
        ax.set_title(f"Predicted Drone Location - Frame {t}")
        v = voxel_sequence[t]
        coords = np.argwhere(v > 0.5)
        if len(coords) > 0:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                       c=v[v > 0.5], cmap="inferno", s=6)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.pause(interval)

    plt.ioff()
    plt.show()


# --------------------------------------------------
# Inference
# --------------------------------------------------
def infer(model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path}")

    model = Temporal3DUNet(in_ch=2, out_ch=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    voxels = load_voxel_frames(data_dir)
    print(f"Running inference on {voxels.shape[0]} frames...")

    X = np.transpose(voxels, (1, 0, 2, 3, 4))
    X = torch.from_numpy(X[None, ...]).float().to(device)

    with torch.no_grad():
        pred = model(X).cpu().numpy()[0, 0]  # (T, D, H, W)

    print("Inference complete. Displaying results...")
    visualize_voxels(pred)
    np.save("predicted_heatmaps.npy", pred)
    print("Saved predictions to predicted_heatmaps.npy")


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on voxel frames.")
    parser.add_argument("--model", required=True, help="Path to trained model (.pt)")
    parser.add_argument("--data_dir", required=True, help="Directory of JSON voxel frames")
    args = parser.parse_args()

    infer(args.model, args.data_dir)
