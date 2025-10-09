import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchsparse
import torchsparse.nn as spnn
import torchsparse.nn.functional as spf
from torchsparse import SparseTensor
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# Sparse Voxel UNet-like CNN (TorchSparse)
# ============================================================
class SparseVoxelNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(SparseVoxelNet, self).__init__()

        self.enc1 = nn.Sequential(
            spnn.Conv3d(in_channels, 32, kernel_size=3, stride=1),
            spnn.BatchNorm(32),
            spnn.ReLU(True)
        )
        self.enc2 = nn.Sequential(
            spnn.Conv3d(32, 64, kernel_size=3, stride=2),
            spnn.BatchNorm(64),
            spnn.ReLU(True)
        )
        self.enc3 = nn.Sequential(
            spnn.Conv3d(64, 128, kernel_size=3, stride=2),
            spnn.BatchNorm(128),
            spnn.ReLU(True)
        )

        self.dec2 = nn.Sequential(
            spnn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            spnn.BatchNorm(64),
            spnn.ReLU(True)
        )
        self.dec1 = nn.Sequential(
            spnn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            spnn.BatchNorm(32),
            spnn.ReLU(True)
        )

        self.out_block = spnn.Conv3d(32, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(e3)
        d1 = self.dec1(d2)
        out = self.out_block(d1)
        return out


# ============================================================
# Data Loader (JSON → SparseTensor)
# ============================================================
def load_sparse_voxel_frames(data_dir):
    json_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])
    coords_list, feats_list = [], []

    for jf in json_files:
        path = os.path.join(data_dir, jf)
        with open(path, "r") as f:
            data = json.load(f)
        voxels = data.get("voxels", [])
        if not voxels:
            continue

        coords = np.array([v["coordinates"] for v in voxels], dtype=np.int32)
        feats = np.array([[v.get("intensity", 0.0), v.get("motion_type", 0.0)] for v in voxels], dtype=np.float32)

        # TorchSparse expects [N, 4] coords: batch index + xyz
        batch_col = np.zeros((coords.shape[0], 1), dtype=np.int32)
        coords = np.concatenate([batch_col, coords], axis=1)

        coords_list.append(coords)
        feats_list.append(feats)

    return coords_list, feats_list


def make_sparse_tensor(coords, feats, device):
    coords = torch.from_numpy(coords).to(device)
    feats = torch.from_numpy(feats).to(device)
    return SparseTensor(feats, coords)


# ============================================================
# Training Loop
# ============================================================
def train_sparse(data_dir, output_model, epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    coords_list, feats_list = load_sparse_voxel_frames(data_dir)
    print(f"Loaded {len(coords_list)} frames from {data_dir}")

    model = SparseVoxelNet(in_channels=2, out_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_hist = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for coords, feats in tqdm(zip(coords_list, feats_list), total=len(coords_list), desc=f"Epoch {epoch+1}/{epochs}"):
            if coords.shape[0] == 0:
                continue

            stensor = make_sparse_tensor(coords, feats, device)
            optimizer.zero_grad()
            output = model(stensor)

            # align features via coordinate intersection
            cm = output.c
            tgt_coords = stensor.c
            tgt_feats = stensor.f

            # Find matching coordinates
            _, idx_pred, idx_tgt = torchsparse.utils.sphash.hash_query(cm, tgt_coords)
            if len(idx_pred) == 0:
                continue

            pred_feats = output.f[idx_pred]
            target_feats = tgt_feats[idx_tgt]

            loss = criterion(pred_feats, target_feats)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(coords_list)
        loss_hist.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), output_model)
    print(f"Model saved to {output_model}")

    plt.figure()
    plt.plot(loss_hist, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("TorchSparse VoxelNet Training Loss")
    plt.legend()
    plt.show()


# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sparse voxel CNN with TorchSparse")
    parser.add_argument("--data_dir", required=True, help="Directory containing JSON voxel frames")
    parser.add_argument("--output_model", required=True, help="Output path for trained model")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    args = parser.parse_args()

    train_sparse(args.data_dir, args.output_model, args.epochs)
