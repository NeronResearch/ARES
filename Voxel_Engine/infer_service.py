import mmap, numpy as np, signal, zmq, json, torch, os, MinkowskiEngine as ME
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from time import time

# ----------------------------------------------------
# Graceful shutdown
# ----------------------------------------------------
RUNNING = True
def handle_sigint(sig, frame):
    global RUNNING
    print("\n[Python] Inference service shutting down...")
    RUNNING = False
signal.signal(signal.SIGINT, handle_sigint)

# ----------------------------------------------------
# ZeroMQ setup
# ----------------------------------------------------
ctx = zmq.Context()
socket = ctx.socket(zmq.REP)
socket.bind("tcp://127.0.0.1:5555")
print("[Python] Inference service started on tcp://127.0.0.1:5555")

# ----------------------------------------------------
# Model definitions
# ----------------------------------------------------
class ConvBNReLU(torch.nn.Module):
    def __init__(self, c_in, c_out, dim=4, ks=3, stride=1, dilation=1):
        super().__init__()
        self.block = torch.nn.Sequential(
            ME.MinkowskiConvolution(in_channels=c_in, out_channels=c_out, kernel_size=ks,
                                    stride=stride, dilation=dilation, dimension=dim),
            ME.MinkowskiBatchNorm(c_out),
            ME.MinkowskiReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, c, dim=4):
        super().__init__()
        self.conv1 = ConvBNReLU(c, c, dim=dim)
        self.conv2 = ME.MinkowskiConvolution(c, c, kernel_size=3, stride=1, dimension=dim)
        self.bn2 = ME.MinkowskiBatchNorm(c)
        self.relu = ME.MinkowskiReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out = out + x
        return self.relu(out)

class Simple4DUNet(torch.nn.Module):
    def __init__(self, c_in=7, c_mid=32, dim=4):
        super().__init__()
        self.dim = dim
        # Encoder
        self.stem = ConvBNReLU(c_in, c_mid, dim=dim)
        self.enc1 = ResidualBlock(c_mid, dim=dim)
        self.down1 = ME.MinkowskiConvolution(c_mid, c_mid * 2, kernel_size=2, stride=(2,2,2,2), dimension=dim)
        self.enc2 = ResidualBlock(c_mid * 2, dim=dim)
        self.down2 = ME.MinkowskiConvolution(c_mid * 2, c_mid * 4, kernel_size=2, stride=(2,2,2,1), dimension=dim)
        self.bottleneck = ResidualBlock(c_mid * 4, dim=dim)
        # Decoder
        self.up1 = ME.MinkowskiConvolutionTranspose(c_mid * 4, c_mid * 2, kernel_size=2, stride=(2,2,2,1), dimension=dim)
        self.dec1 = ResidualBlock(c_mid * 2, dim=dim)
        self.up2 = ME.MinkowskiConvolutionTranspose(c_mid * 2, c_mid, kernel_size=2, stride=(2,2,2,2), dimension=dim)
        self.dec2 = ResidualBlock(c_mid, dim=dim)
        # Head
        self.head = ME.MinkowskiConvolution(c_mid, 1, kernel_size=1, stride=1, dimension=dim)

    def forward(self, x):
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

# ----------------------------------------------------
# Load model weights
# ----------------------------------------------------
model_path = "model_final.pt"
device = torch.device("cpu")
model = Simple4DUNet(c_in=7, c_mid=32, dim=4).to(device)
ckpt = torch.load(model_path, map_location=device)
state = ckpt.get("model_state", ckpt)
model.load_state_dict(state, strict=True)
model.eval()
print(f"[Python] Loaded model from {model_path}")

# ----------------------------------------------------
# Shared memory helper
# ----------------------------------------------------
def open_shm(name, size, dtype):
    f = mmap.mmap(-1, size, name)
    arr = np.ndarray(shape=(size // np.dtype(dtype).itemsize,), dtype=dtype, buffer=f)
    return arr, f

# ----------------------------------------------------
# Main service loop
# ----------------------------------------------------
while RUNNING:
    try:
        msg = socket.recv(flags=zmq.NOBLOCK)
        payload = json.loads(msg.decode())

        coords_name = payload["coords_shm"]
        feats_name  = payload["feats_shm"]
        out_name    = payload["out_shm"]
        ncoords, coord_dim = payload["coords_shape"]
        _, nfeats = payload["feats_shape"]

        coords_bytes = ncoords * coord_dim * 4
        feats_bytes  = ncoords * nfeats * 4

        coords_np, coords_map = open_shm(coords_name, coords_bytes, np.int32)
        feats_np, feats_map   = open_shm(feats_name, feats_bytes, np.float32)

        coords = torch.from_numpy(coords_np.reshape(ncoords, coord_dim)).to(device)
        feats  = torch.from_numpy(feats_np.reshape(ncoords, nfeats)).to(device)

        # Add batch column for ME SparseTensor
        batch_col = torch.zeros((ncoords, 1), dtype=torch.int32, device=device)
        coords_me = torch.cat([batch_col, coords], dim=1)

        st = ME.SparseTensor(features=feats, coordinates=coords_me, device=device)

        with torch.no_grad():
            start = time()
            out = model(st)
            logits = out.F
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            dur = (time() - start) * 1000

        print(f"[Python] Inference OK ({ncoords} voxels) in {dur:.2f} ms")

        out_bytes = ncoords * 4
        out_map = mmap.mmap(-1, out_bytes, out_name)
        np.ndarray((ncoords,), dtype=np.float32, buffer=out_map)[:] = probs.astype(np.float32)

        socket.send_json({"status": "ok", "nbytes": out_bytes})

    except zmq.Again:
        continue
    except Exception as e:
        socket.send_json({"status": "error", "message": str(e)})
