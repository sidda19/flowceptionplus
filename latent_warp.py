import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2

# ============================================================
# Usage
# python latent_warp.py latent_a.pt forward_flow.npy warped_latent.pt
# ============================================================

if len(sys.argv) != 4:
    print("Usage:")
    print("python latent_warp.py <latent.pt> <flow.npy> <output.pt>")
    exit()

LATENT_PATH = sys.argv[1]
FLOW_PATH = sys.argv[2]
OUTPUT_PATH = sys.argv[3]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Load Latent
# ============================================================

latent = torch.load(LATENT_PATH, map_location=DEVICE).to(DEVICE)

print("Latent Shape :", latent.shape)

# ============================================================
# Load Flow
# ============================================================

flow = np.load(FLOW_PATH)

print("Original Flow Shape :", flow.shape)

# ============================================================
# Resize Flow to Latent Resolution
# ============================================================

latent_h = latent.shape[2]
latent_w = latent.shape[3]

orig_h, orig_w = flow.shape[:2]

flow = cv2.resize(flow, (latent_w, latent_h))

# Convert pixel displacement to latent displacement
flow[..., 0] *= latent_w / orig_w
flow[..., 1] *= latent_h / orig_h

flow = torch.from_numpy(flow).float()
flow = flow.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

print("Latent Flow Shape :", flow.shape)

# ============================================================
# Build Sampling Grid
# ============================================================

B, C, H, W = latent.shape

yy, xx = torch.meshgrid(
    torch.arange(H, device=DEVICE),
    torch.arange(W, device=DEVICE),
    indexing="ij"
)

xx = xx.float().unsqueeze(0).expand(B, -1, -1)
yy = yy.float().unsqueeze(0).expand(B, -1, -1)

xx = xx + flow[:, 0]
yy = yy + flow[:, 1]

xx = 2.0 * xx / (W - 1) - 1.0
yy = 2.0 * yy / (H - 1) - 1.0

grid = torch.stack((xx, yy), dim=-1)

print("Grid Shape :", grid.shape)

# ============================================================
# Warp Latent
# ============================================================

warped_latent = F.grid_sample(
    latent,
    grid,
    mode="bilinear",
    padding_mode="border",
    align_corners=True,
)

print("Warped Latent Shape :", warped_latent.shape)

# ============================================================
# Save
# ============================================================

torch.save(warped_latent.cpu(), OUTPUT_PATH)

print(f"\nSaved: {OUTPUT_PATH}")