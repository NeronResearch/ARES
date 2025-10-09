import os
import json
import math
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME


# -------------------------------
# Utility functions
# -------------------------------

def load_scene(frames_dir, r_m=0.75):
    """Load 10 JSON frames as one sparse 4D sample (x,y,z,t)."""
    frames_dir = Path(frames_dir)
    json_files = sorted(frames_dir.glob("????.json"))
    assert len(json_files) == 10, f"Expected 10 JSON files, found {len(json_files)}"

    all_coords, all_feats, all_labels = [], [], []
    grid_info_ref = None

    # motion type mapping to embedding indices
    motion_map = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4}

    for t_idx, jf in enumerate(json_files):
        with open(jf, "r") as f:
            data = json.load(f)

        grid_info = data["grid_info"]
        if grid_info_ref is None:
            grid_info_ref = grid_info
        # else:
           #  assert grid_info == grid_info_ref, "Inconsistent grid_info across frames"

        voxels = data.get("voxels", [])
        if not voxels:
            continue

        voxel_size_m = grid_info["voxel_size_m"]
        origin_m = np.array(grid_info["origin_m"])

        coords_xyz = np.array([v["coordinates"] for v in voxels], dtype=np.int32)
        intensities = np.array([v["intensity"] for v in voxels], dtype=np.float32)
        motion_types = np.array([motion_map.get(v["motion_type"], 0) for v in voxels], dtype=np.float32)

        # normalize intensity
        if len(intensities) > 1:
            mean, std = intensities.mean(), intensities.std()
            intensities = np.clip((intensities - mean) / (std + 1e-6), -5, 5)
        else:
            intensities[:] = 0.0

        # target label generation (single primary target)
        targets = [t for t in data.get("targets", []) if t.get("frame", t_idx) == t_idx]
        if targets:
            tgt = targets[0]
            tgt_pos = np.array(tgt["position_m"])
            tgt_idx = np.round((tgt_pos - origin_m) / voxel_size_m).astype(int)
        else:
            tgt_idx = np.array([0, 0, 0])

        r_vox = int(math.ceil(r_m / voxel_size_m))

        coords_4d = np.hstack([coords_xyz, np.full((len(coords_xyz), 1), t_idx, dtype=np.int32)])

        # label = 1 if within radius r_vox
        diffs = coords_xyz - tgt_idx
        dist = np.linalg.norm(diffs, axis=1)
        labels = (dist <= r_vox).astype(np.float32)

        all_coords.append(coords_4d)
        all_feats.append(np.stack([intensities, motion_types], axis=1))
        all_labels.append(labels)

    coords = np.concatenate(all_coords, axis=0)
    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return coords, feats, labels, grid_info_ref


# -------------------------------
# Model definition
# -------------------------------

class Sparse4DUNet(nn.Module):
    def __init__(self, emb_dim=2):
        super().__init__()
        self.emb = nn.Embedding(5, emb_dim)

        self.e0 = ME.MinkowskiConvolution(2, 32, kernel_size=3, dimension=4)
        self.bn0 = ME.MinkowskiBatchNorm(32)
        self.temporal0 = ME.MinkowskiConvolution(32, 32, kernel_size=3, dimension=4)

        self.down1 = ME.MinkowskiConvolution(32, 64, kernel_size=3, stride=2, dimension=4)
        self.bn1 = ME.MinkowskiBatchNorm(64)
        self.temporal1 = ME.MinkowskiConvolution(64, 64, kernel_size=3, dimension=4)

        self.down2 = ME.MinkowskiConvolution(64, 96, kernel_size=3, stride=2, dimension=4)
        self.bn2 = ME.MinkowskiBatchNorm(96)

        self.bottleneck = nn.Sequential(
            ME.MinkowskiConvolution(96, 128, kernel_size=3, dimension=4),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiLeakyReLU(0.1),
            ME.MinkowskiDropout(0.1),
            ME.MinkowskiConvolution(128, 128, kernel_size=3, dimension=4),
            ME.MinkowskiBatchNorm(128),
        )

        self.up2 = ME.MinkowskiConvolutionTranspose(128, 96, kernel_size=2, stride=2, dimension=4)
        self.up1 = ME.MinkowskiConvolutionTranspose(96, 64, kernel_size=2, stride=2, dimension=4)
        self.final = ME.MinkowskiConvolution(64, 1, kernel_size=1, dimension=4)

    def forward(self, x):
        e0 = F.leaky_relu(self.bn0(self.e0(x)), 0.1)
        e0 = F.leaky_relu(self.temporal0(e0), 0.1)
        e1 = F.leaky_relu(self.bn1(self.down1(e0)), 0.1)
        e1 = F.leaky_relu(self.temporal1(e1), 0.1)
        e2 = F.leaky_relu(self.bn2(self.down2(e1)), 0.1)

        b = self.bottleneck(e2)
        u2 = self.up2(b)
        u1 = self.up1(u2 + e1)
        out = self.final(u1 + e0)
        return out



# -------------------------------
# Training routine
# -------------------------------

def train(scene_coords, scene_feats, scene_labels, out_file, epochs=10, lr=1e-3):
    device = torch.device("cpu")

    # Prepare data
    coords = torch.from_numpy(scene_coords).int()
    feats = torch.from_numpy(scene_feats).float()
    labels = torch.from_numpy(scene_labels).float()

    # Create sparse tensor directly
    sparse_tensor = ME.SparseTensor(
        features=feats,
        coordinates=coords,
        device=device
    )
    
    labels = labels.to(device)

    model = Sparse4DUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(sparse_tensor)
        logits = out.features.squeeze()

        # Align labels with output coordinates
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), out_file)
    print(f"Model saved to {out_file}")


# -------------------------------
# CLI
# -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 4D sparse CNN on voxel frames.")
    parser.add_argument("--frames_dir", required=True, help="Path to directory with 0000.json-0009.json")
    parser.add_argument("--out_file", required=True, help="Output model file (e.g., model.pth)")
    parser.add_argument("--radius_m", type=float, default=0.75, help="Target radius in meters")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    coords, feats, labels, grid_info = load_scene(args.frames_dir, r_m=args.radius_m)
    print(f"Loaded scene with {len(coords)} voxels")

    train(coords, feats, labels, args.out_file, epochs=args.epochs)
