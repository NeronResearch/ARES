import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '8')
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME


# -----------------------------
# Utils (match training)
# -----------------------------

def one_hot_motion(motion_type: int) -> np.ndarray:
    mapping = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4}
    idx = mapping.get(int(motion_type), 0)
    v = np.zeros(5, dtype=np.float32)
    v[idx] = 1.0
    return v


def load_clip_jsons(frame_dir: str, pattern: str = "000*.json"):
    files = sorted(glob.glob(os.path.join(frame_dir, pattern)))
    if len(files) != 10:
        raise RuntimeError(f"Expected 10 JSON files, found {len(files)} in {frame_dir}")
    return files


def build_inputs_from_jsons(frame_dir: str):
    """
    Returns:
      coords_xyz_it: [N,4] int32 (ix, iy, iz, it)
      feats:         [N,7] float32 (intensity_norm, 5x onehot(motion), t_norm)
      grid_info:     dict
    """
    files = load_clip_jsons(frame_dir, "000*.json")
    frames = []
    for p in files:
        with open(p, "r") as f:
            frames.append(json.load(f))

    grid_info = frames[0].get("grid_info", {})

    coords_list = []
    intens_list = []
    motion_oh_list = []
    tnorm_list = []

    # Gather per-frame voxel data exactly like training
    all_intens = []
    for it, fr in enumerate(frames):
        vox = fr.get("voxels", [])
        if not vox:
            arr_c = np.empty((0, 3), dtype=np.int32)
            intens = np.empty((0,), dtype=np.float32)
            motion_oh = np.empty((0, 5), dtype=np.float32)
        else:
            arr_c = np.array([v["coordinates"] for v in vox], dtype=np.int32)                 # [M,3]
            intens = np.array([v.get("intensity", 0.0) for v in vox], dtype=np.float32)       # [M]
            motion_oh = np.stack([one_hot_motion(v.get("motion_type", 0)) for v in vox], 0)   # [M,5]

        tcol = np.full((arr_c.shape[0], 1), it, dtype=np.int32)
        coords_list.append(np.hstack([arr_c, tcol]))  # [M,4]

        intens_list.append(intens)
        all_intens.append(intens)

        t_norm = np.full((arr_c.shape[0], 1), float(it) / 9.0, dtype=np.float32)
        tnorm_list.append(t_norm)
        motion_oh_list.append(motion_oh)

        print(f"{os.path.basename(files[it])} -> ({arr_c.shape[0]}, 3) vox")

    if len(coords_list) == 0:
        raise RuntimeError("No voxels found across frames.")

    coords_xyz_it = np.vstack(coords_list).astype(np.int32)             # [N,4]
    intens = np.concatenate(intens_list, axis=0).astype(np.float32)     # [N]
    motion_oh = np.vstack(motion_oh_list).astype(np.float32)            # [N,5]
    t_norm = np.vstack(tnorm_list).astype(np.float32)                   # [N,1]

    # Same normalization as training
    if intens.size > 1:
        mu, sigma = float(np.mean(intens)), float(np.std(intens) + 1e-6)
        intens_norm = (intens - mu) / sigma
    else:
        intens_norm = intens

    feats = np.concatenate([
        intens_norm.reshape(-1, 1),  # 1
        motion_oh,                   # 5
        t_norm                       # 1
    ], axis=1).astype(np.float32)    # -> [N,7]

    print(f"Loaded {len(files)} frames, {coords_xyz_it.shape[0]} voxels total")
    return coords_xyz_it, feats, grid_info


# -----------------------------
# Model (match training)
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
        self.down1 = ME.MinkowskiConvolution(c_mid, c_mid * 2, kernel_size=2, stride=(2, 2, 2, 2), dimension=dim)
        self.enc2 = ResidualBlock(c_mid * 2, dim=dim)
        self.down2 = ME.MinkowskiConvolution(c_mid * 2, c_mid * 4, kernel_size=2, stride=(2, 2, 2, 1), dimension=dim)
        self.bottleneck = ResidualBlock(c_mid * 4, dim=dim)
        # Decoder
        self.up1 = ME.MinkowskiConvolutionTranspose(c_mid * 4, c_mid * 2, kernel_size=2, stride=(2, 2, 2, 1), dimension=dim)
        self.dec1 = ResidualBlock(c_mid * 2, dim=dim)
        self.up2 = ME.MinkowskiConvolutionTranspose(c_mid * 2, c_mid, kernel_size=2, stride=(2, 2, 2, 2), dimension=dim)
        self.dec2 = ResidualBlock(c_mid, dim=dim)
        # Head
        self.head = ME.MinkowskiConvolution(c_mid, 1, kernel_size=1, stride=1, dimension=dim)

    def forward(self, x: ME.SparseTensor):
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))
        u1 = self.up1(x3)
        d1 = self.dec1(u1)
        u2 = self.up2(d1)
        d2 = self.dec2(u2)
        logits = self.head(d2)
        return logits

def _infer_on_files(model: Simple4DUNet, device, frame_files, out_path: str):
    """Run inference on an explicit ordered list of frame JSON paths and save to out_path (.npz).
    frame_files: list of file paths in order (len = clip_len)
    out_path: path to .npz file to write
    """
    import json
    import numpy as np
    import torch
    import MinkowskiEngine as ME

    all_coords = []
    all_feats = []

    for t, path in enumerate(frame_files):
        with open(path, "r") as f:
            d = json.load(f)
        vox = d.get("voxels", [])
        if len(vox) == 0:
            # skip empty frames (keeps behavior from original infer)
            print(f"{os.path.basename(path)} -> 0 voxels (skipped)")
            continue

        # (x, y, z)
        coords_xyz = np.array([v["coordinates"] for v in vox], dtype=np.int32)

        # intensity
        intens = np.array([v.get("intensity", 0.0) for v in vox], dtype=np.float32)
        mu, sigma = np.mean(intens), np.std(intens) + 1e-6
        intens_norm = (intens - mu) / sigma
        intens_norm = intens_norm.reshape(-1, 1)

        # one-hot motion type (5-dim)
        def one_hot_motion(m):
            mapping = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4}
            idx = mapping.get(int(m), 0)
            vec = np.zeros(5, dtype=np.float32)
            vec[idx] = 1.0
            return vec
        motion_oh = np.stack([one_hot_motion(v.get("motion_type", 0)) for v in vox], axis=0)

        # normalized time index (0..1 across the selected clip)
        t_norm = np.full((len(vox), 1), float(t) / max(1, len(frame_files) - 1), dtype=np.float32)

        # feature vector: [intensity_norm, 5x onehot(motion), t_norm] → 7D
        feats = np.concatenate([intens_norm, motion_oh, t_norm], axis=1)

        # 4D coords: [x, y, z, t]
        tcol = np.full((coords_xyz.shape[0], 1), t, dtype=np.int32)
        coords_xyz_it = np.hstack([coords_xyz, tcol])

        print(f"{os.path.basename(path)} -> {coords_xyz_it.shape[0]} voxels")
        all_coords.append(coords_xyz_it)
        all_feats.append(feats)

    if not all_coords:
        raise RuntimeError("No valid voxel data loaded for this clip.")

    coords_np = np.vstack(all_coords).astype(np.int32)   # [N,4]
    feats_np = np.vstack(all_feats).astype(np.float32)   # [N,7]
    print(f"Loaded {len(frame_files)} frames, {coords_np.shape[0]} total voxels -> shape {coords_np.shape}")

    # ----------------------------
    # Prepare SparseTensor
    # ----------------------------
    batch_col = np.zeros((coords_np.shape[0], 1), dtype=np.int32)   # batch index 0
    coords_me = np.hstack([batch_col, coords_np])                   # [N,5] = (b,x,y,z,t)

    coords_t = torch.from_numpy(coords_me)
    feats_t = torch.from_numpy(feats_np)
    st = ME.SparseTensor(features=feats_t, coordinates=coords_t, device=device)
    print(f"SparseTensor built: coords D={st.D}, features={st.F.shape}")

    # ----------------------------
    # Inference
    # ----------------------------
    with torch.no_grad():
        out = model(st)
        logits = out.F
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path,
             coords=coords_np,
             feats=feats_np,
             probs=probs)
    print(f"Inference complete. Saved to {out_path}")
    print(f"Pred stats: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}, std={probs.std():.4f}")


def infer(model_path: str, frame_dir: str, out_dir: str):
    """Run sliding windows of fixed length 10 across the JSON files in frame_dir.
    For each contiguous window of 10 frames, run inference and save output as [last_frame_basename].npz in out_dir.
    """
    import os
    import glob
    import torch

    clip_len = 10
    device = torch.device("cpu")

    # Load model once
    model = Simple4DUNet(c_in=7, c_mid=32, dim=4).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    frame_files_all = sorted(glob.glob(os.path.join(frame_dir, "*.json")))
    if len(frame_files_all) < clip_len:
        raise RuntimeError(f"Not enough frames for sliding window: found {len(frame_files_all)}, need {clip_len}")

    os.makedirs(out_dir, exist_ok=True)
    n_windows = len(frame_files_all) - clip_len + 1
    print(f"Sliding mode (fixed clip_len={clip_len}): {n_windows} windows over {len(frame_files_all)} frames")
    for i in range(n_windows):
        window = frame_files_all[i:i+clip_len]
        last_basename = os.path.splitext(os.path.basename(window[-1]))[0]
        out_path = os.path.join(out_dir, f"{last_basename}.npz")
        print(f"Processing window {i+1}/{n_windows}: {os.path.basename(window[0])} -> {os.path.basename(window[-1])}")
        _infer_on_files(model, device, window, out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model_final.pt")
    ap.add_argument("--frames", required=True, help="Directory containing frame JSON files (e.g. 0000.json, 0001.json, ...)")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()
    infer(args.model, args.frames, args.out)