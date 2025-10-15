import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import time
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

    all_intens = []

    for it, fr in enumerate(frames):
        vox = fr.get("voxels", [])
        if not vox:
            arr_c = np.empty((0, 3), dtype=np.int32)
            intens = np.empty((0,), dtype=np.float32)
            motion_oh = np.empty((0, 5), dtype=np.float32)
        else:
            arr_c = np.array([v["coordinates"] for v in vox], dtype=np.int32)
            intens = np.array([v.get("intensity", 0.0) for v in vox], dtype=np.float32)
            motion_oh = np.stack([one_hot_motion(v.get("motion_type", 0)) for v in vox], 0)

        tcol = np.full((arr_c.shape[0], 1), it, dtype=np.int32)
        coords_list.append(np.hstack([arr_c, tcol]))
        intens_list.append(intens)
        all_intens.append(intens)

        t_norm = np.full((arr_c.shape[0], 1), float(it) / 9.0, dtype=np.float32)
        tnorm_list.append(t_norm)
        motion_oh_list.append(motion_oh)

        print(f"{os.path.basename(files[it])} -> ({arr_c.shape[0]}, 3) vox")

    if len(coords_list) == 0:
        raise RuntimeError("No voxels found across frames.")

    coords_xyz_it = np.vstack(coords_list).astype(np.int32)
    intens = np.concatenate(intens_list, axis=0).astype(np.float32)
    motion_oh = np.vstack(motion_oh_list).astype(np.float32)
    t_norm = np.vstack(tnorm_list).astype(np.float32)

    # Normalize intensity across all voxels
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

def infer(model_path: str, frame_dir: str, out_dir: str, end_frame: int = None, clip_len: int = 10):
    import os
    import json
    import glob
    import numpy as np
    import torch
    import MinkowskiEngine as ME
    device = torch.device("cpu")

    total_start = time.perf_counter()

    # ----------------------------
    # Load frame JSONs
    # ----------------------------
    with Timer("Frame loading / frame selection"):
        if end_frame is not None:
            if end_frame < 1:
                raise RuntimeError(f"--frame must be >= 1, got {end_frame}")
            start_frame = max(1, end_frame - clip_len + 1)
            desired = [os.path.join(frame_dir, f"{i:04d}.json") for i in range(start_frame, end_frame + 1)]
            missing = [p for p in desired if not os.path.isfile(p)]
            if missing:
                raise RuntimeError(f"Missing frame JSON files for requested range {start_frame}-{end_frame}: {missing}")
            frame_files = desired
            print(f"Selected frames {start_frame:04d}.json - {end_frame:04d}.json ({len(frame_files)} files)")
        else:
            frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.json")))
            if len(frame_files) == 0:
                raise RuntimeError(f"No .json files found in {frame_dir}")
            print(f"Found {len(frame_files)} frames")

        all_coords = []
        all_feats = []
        for t, path in enumerate(frame_files):
            with open(path, "r") as f:
                d = json.load(f)
            vox = d.get("voxels", [])
            if len(vox) == 0:
                continue
            coords_xyz = np.array([v["coordinates"] for v in vox], dtype=np.int32)
            intens = np.array([v.get("intensity", 0.0) for v in vox], dtype=np.float32)
            mu, sigma = np.mean(intens), np.std(intens) + 1e-6
            intens_norm = (intens - mu) / sigma
            intens_norm = intens_norm.reshape(-1, 1)

            def one_hot_motion(m):
                mapping = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4}
                idx = mapping.get(int(m), 0)
                vec = np.zeros(5, dtype=np.float32)
                vec[idx] = 1.0
                return vec

            motion_oh = np.stack([one_hot_motion(v.get("motion_type", 0)) for v in vox], axis=0)
            t_norm = np.full((len(vox), 1), float(t) / max(1, len(frame_files) - 1), dtype=np.float32)
            feats = np.concatenate([intens_norm, motion_oh, t_norm], axis=1)
            tcol = np.full((coords_xyz.shape[0], 1), t, dtype=np.int32)
            coords_xyz_it = np.hstack([coords_xyz, tcol])
            print(f"{os.path.basename(path)} -> {coords_xyz_it.shape[0]} voxels")
            all_coords.append(coords_xyz_it)
            all_feats.append(feats)

        if not all_coords:
            raise RuntimeError("No valid voxel data loaded.")

        coords_np = np.vstack(all_coords).astype(np.int32)
        feats_np = np.vstack(all_feats).astype(np.float32)
        print(f"Loaded {len(frame_files)} frames, {coords_np.shape[0]} total voxels -> shape {coords_np.shape}")
    # end frame loading timer

    # ----------------------------
    # Prepare SparseTensor
    # ----------------------------
    with Timer("SparseTensor construction"):
        batch_col = np.zeros((coords_np.shape[0], 1), dtype=np.int32)
        coords_me = np.hstack([batch_col, coords_np])
        coords_t = torch.from_numpy(coords_me)
        feats_t = torch.from_numpy(feats_np)
        st = ME.SparseTensor(features=feats_t, coordinates=coords_t, device=device)
        print(f"SparseTensor built: coords D={st.D}, features={st.F.shape}")
    # end sparse tensor timer

    # ----------------------------
    # Load model
    # ----------------------------
    with Timer("Model loading"):
        model = Simple4DUNet(c_in=7, c_mid=32, dim=4).to(device)
        ckpt = torch.load(model_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=True)
        model.eval()
    # end model loading timer

    # ----------------------------
    # Inference
    # ----------------------------
    with Timer("Inference"):
        with torch.no_grad():
            out = model(st)
            logits = out.F
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    # end inference timer

    # ----------------------------
    # Output saving
    # ----------------------------
    with Timer("Output saving"):
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, "predictions_clip.npz"),
                 coords=coords_np,
                 feats=feats_np,
                 probs=probs)
        print(f"Inference complete. Saved to {out_dir}/predictions_clip.npz")
        print(f"Pred stats: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}, std={probs.std():.4f}")
    # end output timer

    total_elapsed = time.perf_counter() - total_start
    print(f"[TIMING] Total inference time: {total_elapsed:.3f}s")

# Simple Timer context manager
class Timer:
    def __init__(self, name="stage"):
        self.name = name
        self.start = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        print(f"[TIMING] {self.name}: {self.elapsed:.3f}s")
        return False  # Do not suppress exceptions

# Entry point
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model_final.pt")
    ap.add_argument("--frames", required=True, help="Directory containing 0000..0009.json")
    ap.add_argument("--frame", type=int, default=10, help="(optional) end frame index (1-based). If provided, the clip will be the previous --clip-length frames ending at this frame. Example: --frame 10 grabs 0001.json..0010.json")
    ap.add_argument("--clip-length", type=int, default=10, help="Number of frames to include in clip when using --frame (default: 10)")
    ap.add_argument("--out", required=True, help="Output directory")

    args = ap.parse_args()
    infer(args.model, args.frames, args.out, end_frame=args.frame, clip_len=args.clip_length)