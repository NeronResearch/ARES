import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '8')
import json
import glob
import math
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import MinkowskiEngine as ME


# -----------------------------
# Utilities
# -----------------------------

def load_clip_jsons(frame_dir: str) -> List[str]:
    import re

    # All JSON files in the directory (sorted by name)
    all_jsons = sorted(glob.glob(os.path.join(frame_dir, "*.json")))
    if len(all_jsons) == 0:
        raise RuntimeError(f"No JSON files found in {frame_dir}")

    # Map numeric frame index -> path. We look for the first group of digits
    # in the filename (without extension).
    idx_to_path: Dict[int, str] = {}
    for p in all_jsons:
        name = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"(\d+)", name)
        if not m:
            continue
        idx = int(m.group(1))
        # If multiple files map to same index, prefer the lexicographically
        # earlier one (sorted order of all_jsons already enforces this)
        if idx not in idx_to_path:
            idx_to_path[idx] = p

    # All files are candidates; collect the numeric indices that are present
    matched_indices: List[int] = sorted(idx_to_path.keys())

    if len(matched_indices) == 0:
        raise RuntimeError(f"No numeric-indexed JSON files found in {frame_dir}")

    # Try each matched index (sorted ascending) as a candidate start and
    # return the first contiguous run of 10 frames that exists. This allows
    # patterns that match a later frame (e.g. 0100) to work when 0100..0109
    # exist, but also tolerates patterns that match multiple possible starts.
    candidate_starts = sorted(set(int(x) for x in matched_indices))

    tried_info: Dict[int, List[int]] = {}
    for start_idx in candidate_starts:
        needed = [start_idx + i for i in range(10)]
        missing = [idx for idx in needed if idx not in idx_to_path]
        if not missing:
            # Found a full run
            return [idx_to_path[idx] for idx in needed]
        tried_info[start_idx] = missing

    # No candidate produced a full contiguous run. Build helpful message.
    msg_lines = ["No contiguous 10-frame clip found starting from any available start index in the directory"]
    msg_lines.append("Candidates tried and their missing indices:")
    for s, miss in tried_info.items():
        msg_lines.append(f"  start {s}: missing {miss}")
    raise RuntimeError("\n".join(msg_lines))


def find_10frame_segments_in_dir(frame_dir: str) -> List[List[str]]:
    """
    Return all contiguous 10-frame segments (sliding window) found within a single directory.
    Each segment is a list of 10 file paths.
    """
    import re
    all_jsons = sorted(glob.glob(os.path.join(frame_dir, "*.json")))
    if len(all_jsons) == 0:
        return []

    idx_to_path: Dict[int, str] = {}
    for p in all_jsons:
        name = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"(\d+)", name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx not in idx_to_path:
            idx_to_path[idx] = p

    matched_indices = sorted(idx_to_path.keys())
    if len(matched_indices) == 0:
        return []

    segments: List[List[str]] = []
    # For each possible start index, check contiguous run of 10 frames
    for start in matched_indices:
        needed = [start + i for i in range(10)]
        if all((n in idx_to_path) for n in needed):
            segments.append([idx_to_path[n] for n in needed])

    return segments


def find_all_10frame_segments(root_dir: str) -> List[List[str]]:
    """
    Recursively search root_dir and its subdirectories for all contiguous 10-frame segments.
    Returns a list of segments (each segment is a list of 10 json paths).
    """
    all_segments: List[List[str]] = []
    # Walk directories including root
    for dpath, dnames, fnames in os.walk(root_dir):
        segs = find_10frame_segments_in_dir(dpath)
        if segs:
            all_segments.extend(segs)
    return all_segments


def parse_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def one_hot_motion(motion_type: int) -> np.ndarray:
    """
    Map motion types {-1,0,1,2,3} to {0..4} one-hot.
    """
    mapping = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4}
    idx = mapping.get(int(motion_type), 0)
    vec = np.zeros(5, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def meters_to_grid(position_m: np.ndarray, origin_m: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """
    Convert continuous meters to integer grid indices using floor.
    """
    return np.floor((position_m - origin_m) / voxel_size_m).astype(np.int32)


def compute_labels_for_points(
    coords_xyz_it: np.ndarray,
    targets_per_t: Dict[int, np.ndarray],
    radius_vox: int
) -> np.ndarray:
    """
    Assign a binary label to each sparse point based on proximity (in grid indices) to target center
    at the same time index t. coords_xyz_it is [N,4] with (ix, iy, iz, it).
    """
    labels = np.zeros((coords_xyz_it.shape[0],), dtype=np.float32)
    if not targets_per_t:
        return labels

    # Pre-split indices per t to avoid scanning everything
    # Build dict t -> indices
    time_to_indices: Dict[int, List[int]] = {}
    for i, it in enumerate(coords_xyz_it[:, 3]):
        time_to_indices.setdefault(int(it), []).append(i)

    r2 = radius_vox * radius_vox
    for t, center_idx in targets_per_t.items():
        if t not in time_to_indices:
            continue
        c_ix, c_iy, c_iz = center_idx
        pts_idx = time_to_indices[t]
        pts = coords_xyz_it[pts_idx, :3]  # [M,3]
        d2 = np.sum((pts - np.array([c_ix, c_iy, c_iz], dtype=np.int32)) ** 2, axis=1)
        pos_mask = d2 <= r2
        labels_arr = labels[pts_idx]
        labels_arr[pos_mask] = 1.0
        labels[pts_idx] = labels_arr

    return labels


# -----------------------------
# Dataset
# -----------------------------

class Sparse4DClipDataset(torch.utils.data.Dataset):
    """
    Each sample is a 10-frame clip merged into one 4D sparse point cloud.
    Features per point: [intensity_norm, one_hot(motion,5), t_norm] => 7 dims.
    Labels: binary for objectness at active points, radius-based around target center per frame.
    """

    def __init__(self,
                 dir_path: str,
                 radius_vox: int = 2):
        super().__init__()
        self.dir_path = dir_path
        self.radius_vox = radius_vox
        # Find all contiguous 10-frame segments under dir_path (including subdirectories)
        self.clip_segments = find_all_10frame_segments(dir_path)
        if len(self.clip_segments) == 0:
            raise RuntimeError(f"No contiguous 10-frame segments found under {dir_path}")

        # Pre-parse and build arrays for each segment to keep __getitem__ simple
        self._samples = []  # list of tuples (coords, feats, labels, grid_info)
        for seg_idx, seg_files in enumerate(self.clip_segments):
            frames = [parse_json(p) for p in seg_files]
            # Validate grid consistency inside the segment
            grid_info = frames[0].get("grid_info", {})
            for i, fr in enumerate(frames[1:], 1):
                gi = fr.get("grid_info", {})
                for k in ["voxel_size_m", "origin_m", "dimensions"]:
                    if str(grid_info.get(k)) != str(gi.get(k)):
                        raise RuntimeError(f"grid_info mismatch in segment {seg_idx} between frame 0 and frame {i} for key '{k}'.")

            # store voxel/grid metadata for the sample
            voxel_size_m = float(grid_info.get("voxel_size_m", 1.0))
            origin_m = np.array(grid_info.get("origin_m", [0.0, 0.0, 0.0]), dtype=np.float32)

            coords, feats, labels = self._build_clip_arrays_from_frames(frames)
            self._samples.append((coords, feats, labels, grid_info))

        # Expose metadata of first sample for backward compatibility
        self.grid_info = self._samples[0][3]
        self.voxel_size_m = float(self.grid_info.get("voxel_size_m", 1.0))
        self.origin_m = np.array(self.grid_info.get("origin_m", [0.0, 0.0, 0.0]), dtype=np.float32)

    def _build_clip_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          coords_xyz_it: [N,4] int32 (ix, iy, iz, it)
          feats:         [N,7] float32 (intensity_norm, 5x onehot, t_norm)
          labels:        [N]   float32 (0/1) positive around target center
        """
        coords_list = []
        intens_list = []
        motion_oh_list = []
        tnorm_list = []
        targets_per_t: Dict[int, np.ndarray] = {}

        # Build per-frame targets in grid indices (assume single target per frame; if multiple, take the first)
        for it, fr in enumerate(self.frames):
            tlist = fr.get("targets", [])
            if len(tlist) > 0:
                # Choose the one whose 'frame' matches it if available, else the first
                sel = None
                for tgt in tlist:
                    if int(tgt.get("frame", it)) == it:
                        sel = tgt
                        break
                if sel is None:
                    sel = tlist[0]
                pos_m = np.array(sel.get("position_m", [0, 0, 0]), dtype=np.float32)
                center_idx = meters_to_grid(pos_m, self.origin_m, self.voxel_size_m)
                targets_per_t[it] = center_idx

        # Gather sparse voxels across frames
        for it, fr in enumerate(self.frames):
            vox = fr.get("voxels", [])
            if len(vox) == 0:
                continue
            # Coordinates in grid index space
            c = np.array([v["coordinates"] for v in vox], dtype=np.int32)  # [M,3]
            tcol = np.full((c.shape[0], 1), it, dtype=np.int32)
            coords_list.append(np.hstack([c, tcol]))

            intens = np.array([v.get("intensity", 0.0) for v in vox], dtype=np.float32)
            intens_list.append(intens)

            motion_oh = np.stack([one_hot_motion(v.get("motion_type", 0)) for v in vox], axis=0)  # [M,5]
            motion_oh_list.append(motion_oh)

            t_norm = np.full((c.shape[0], 1), float(it) / 9.0, dtype=np.float32)
            tnorm_list.append(t_norm)

        if len(coords_list) == 0:
            raise RuntimeError("No voxels found across the 10 frames.")

        coords_xyz_it = np.vstack(coords_list)                    # [N,4]
        intens = np.concatenate(intens_list, axis=0)              # [N]
        motion_oh = np.vstack(motion_oh_list)                     # [N,5]
        t_norm = np.vstack(tnorm_list)                            # [N,1]

        # Normalize intensity across the clip
        if len(intens) > 1:
            mu, sigma = float(np.mean(intens)), float(np.std(intens) + 1e-6)
            intens_norm = (intens - mu) / sigma
        else:
            intens_norm = intens

        feats = np.concatenate([
            intens_norm.reshape(-1, 1).astype(np.float32),
            motion_oh.astype(np.float32),
            t_norm.astype(np.float32)
        ], axis=1)  # [N,7]

        # Build labels at active points
        labels = compute_labels_for_points(coords_xyz_it, targets_per_t, radius_vox=self.radius_vox).astype(np.float32)  # [N]

        return coords_xyz_it.astype(np.int32), feats.astype(np.float32), labels


    def _build_clip_arrays_from_frames(self, frames: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build arrays for an arbitrary list of 10 frame dicts (same logic as before).
        """
        # Temporarily assign self.frames and grid metadata to reuse same code; avoid modifying state permanently
        old_frames = getattr(self, "frames", None)
        old_origin = getattr(self, "origin_m", None)
        old_voxel = getattr(self, "voxel_size_m", None)

        grid_info = frames[0].get("grid_info", {})
        try:
            self.frames = frames
            # set origin and voxel size expected by _build_clip_arrays
            self.origin_m = np.array(grid_info.get("origin_m", [0.0, 0.0, 0.0]), dtype=np.float32)
            self.voxel_size_m = float(grid_info.get("voxel_size_m", 1.0))
            return self._build_clip_arrays()
        finally:
            # restore previous attributes
            if old_frames is not None:
                self.frames = old_frames
            else:
                if hasattr(self, "frames"):
                    delattr(self, "frames")
            if old_origin is not None:
                self.origin_m = old_origin
            else:
                if hasattr(self, "origin_m"):
                    delattr(self, "origin_m")
            if old_voxel is not None:
                self.voxel_size_m = old_voxel
            else:
                if hasattr(self, "voxel_size_m"):
                    delattr(self, "voxel_size_m")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        coords, feats, labels, grid_info = self._samples[idx]
        return coords.copy(), feats.copy(), labels.copy(), grid_info


def sparse_collate_fn(batch):
    """
    Batch is a list of (coords_xyz_it, feats, labels, grid_info) with batch size B (here B=1 by default).
    Convert to ME-friendly batched sparse tensors by prepending batch index and merging.
    """
    batched_coords = []
    batched_feats = []
    batched_labels = []
    infos = []

    for bidx, (coords_xyz_it, feats, labels, info) in enumerate(batch):
        # print("coords_xyz_it shape:", coords_xyz_it.shape)
        # Expect coords_xyz_it to be [N, 4] (x, y, z, t)
        if coords_xyz_it.shape[1] != 4:
            raise ValueError(f"Expected coords_xyz_it shape [N, 4], got {coords_xyz_it.shape}")
        coords_me = coords_xyz_it.astype(np.int32)  # [N, 4]
        batched_coords.append(coords_me)
        batched_feats.append(feats)
        batched_labels.append(labels.reshape(-1, 1))  # keep as column
        infos.append(info)

    coords_me = np.vstack(batched_coords)
    feats_me = np.vstack(batched_feats)
    labels_me = np.vstack(batched_labels)

    # Minkowski sparse collate expects two lists: coordinates and features
    coords, feats = ME.utils.sparse_collate([coords_me], [feats_me])
    # labels must align with the order after collate; with single sample, order is preserved
    labels_t = torch.from_numpy(labels_me.astype(np.float32))  # [N,1]

    return coords, feats, labels_t, infos


# -----------------------------
# Model
# -----------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, dim=4, ks=3, stride=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=c_in, out_channels=c_out, kernel_size=ks,
                                    stride=stride, dilation=dilation, dimension=dim),
            ME.MinkowskiBatchNorm(c_out),
            ME.MinkowskiReLU(inplace=True),
        )

    def forward(self, x: ME.SparseTensor):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, c, dim=4):
        super().__init__()
        self.conv1 = ConvBNReLU(c, c, dim=dim)
        self.conv2 = ME.MinkowskiConvolution(c, c, kernel_size=3, stride=1, dimension=dim)
        self.bn2 = ME.MinkowskiBatchNorm(c)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor):
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out = out + x
        return self.relu(out)


class Simple4DUNet(nn.Module):
    def __init__(self, c_in=7, c_mid=32, dim=4):
        super().__init__()
        self.dim = dim
        # Encoder
        self.stem = ConvBNReLU(c_in, c_mid, dim=dim)
        self.enc1 = ResidualBlock(c_mid, dim=dim)
        self.down1 = ME.MinkowskiConvolution(c_mid, c_mid*2, kernel_size=2, stride=(2,2,2,2), dimension=dim)
        self.enc2 = ResidualBlock(c_mid*2, dim=dim)
        # Less aggressive temporal downsample in the second stage
        self.down2 = ME.MinkowskiConvolution(c_mid*2, c_mid*4, kernel_size=2, stride=(2,2,2,1), dimension=dim)
        self.bottleneck = ResidualBlock(c_mid*4, dim=dim)

        # Decoder
        self.up1 = ME.MinkowskiConvolutionTranspose(c_mid*4, c_mid*2, kernel_size=2, stride=(2,2,2,1), dimension=dim)
        self.dec1 = ResidualBlock(c_mid*2, dim=dim)
        self.up2 = ME.MinkowskiConvolutionTranspose(c_mid*2, c_mid, kernel_size=2, stride=(2,2,2,2), dimension=dim)
        self.dec2 = ResidualBlock(c_mid, dim=dim)

        # Head: 1-channel heatmap logits
        self.head = ME.MinkowskiConvolution(c_mid, 1, kernel_size=1, stride=1, dimension=dim)

    def forward(self, x: ME.SparseTensor):
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(ME.SparseTensor(
            features=self.down1(x1).F, coordinate_map_key=self.down1(x1).coordinate_map_key,
            coordinate_manager=self.down1(x1).coordinate_manager
        ))
        # Note: keep references to tensors after ops
        x2 = self.enc2(self.down1(x1))

        x3in = self.down2(x2)
        x3 = self.bottleneck(x3in)

        u1 = self.up1(x3)
        # ME supports concat via cat on SparseTensors if coords match; here we just apply a residual block
        d1 = self.dec1(u1)

        u2 = self.up2(d1)
        d2 = self.dec2(u2)

        logits = self.head(d2)  # [N,1]
        return logits


# -----------------------------
# Training
# -----------------------------

def train_one_epoch(model, loader, optimizer, device, pos_weight: float, show_progress: bool = True):
    """Train for one epoch.

    If show_progress is True, display a tqdm progress bar over batches for this epoch.
    Returns average loss (float).
    """
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc="Batches", unit="batch", leave=False)

    for coords, feats, labels, _ in iterator:
        coords = coords.to(device)
        feats = feats.to(device)
        labels = labels.to(device)  # [N,1]

        stensor = ME.SparseTensor(features=feats, coordinates=coords, device=device)
        logits = model(stensor)  # SparseTensor
        # Align features order with labels via .F (features)
        pred = logits.F  # [N,1]

        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        if show_progress:
            iterator.set_postfix(loss=total_loss / (iterator.n if hasattr(iterator, 'n') and iterator.n > 0 else 1))

    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device):
    model.eval()
    total_pos, correct_pos = 0, 0
    with torch.no_grad():
        for coords, feats, labels, _ in loader:
            coords = coords.to(device)
            feats = feats.to(device)
            labels = labels.to(device)

            stensor = ME.SparseTensor(features=feats, coordinates=coords, device=device)
            logits = model(stensor).F
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            total_pos += int(labels.sum().item())
            correct_pos += int((preds * labels).sum().item())
    precision_on_pos = (correct_pos / total_pos) if total_pos > 0 else 0.0
    return precision_on_pos


def main():
    ap = argparse.ArgumentParser("Sparse 4D CNN training (MinkowskiEngine)")
    ap.add_argument("--data_dir", type=str, default=r"E:\Code\Neron\ARES\Frames\Scenario3",
                    help="Directory containing 0000..0009.json")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1, help="Clips per batch; baseline uses 1")
    ap.add_argument("--radius", type=int, default=2, help="Positive radius in voxels around target center")
    ap.add_argument("--pos_weight", type=float, default=12.0, help="BCE positive class weight")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cpu")

    # Dataset & loader (baseline: single 10-frame clip)
    dataset = Sparse4DClipDataset(args.data_dir, radius_vox=args.radius)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=sparse_collate_fn,
        num_workers=0
    )

    model = Simple4DUNet(c_in=7, c_mid=32, dim=4)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Starting training...")
    # Outer progress bar over epochs
    epoch_iter = tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_iter:
        loss = train_one_epoch(model, loader, optimizer, device, pos_weight=args.pos_weight, show_progress=True)
        prec = evaluate(model, loader, device)
        # Use tqdm.write to avoid clobbering progress bars
        tqdm.write(f"Epoch {epoch:02d} | loss={loss:.4f} | pos-precision={prec:.4f}")

        # Save checkpoint each epoch
        # ckpt_path = os.path.join(args.save_dir, f"model_epoch_{epoch:02d}.pt")
        # torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_path)

    # Final save
    final_path = os.path.join(args.save_dir, "model_final.pt")
    torch.save({"epoch": args.epochs, "model_state": model.state_dict()}, final_path)
    print(f"Saved final checkpoint to: {final_path}")

    # Note: TorchScript export of ME models can be nontrivial due to SparseTensor types.
    # For C++ inference, we will prep a dedicated export step and a C++ runner that uses libtorch + ME.


if __name__ == "__main__":
    main()