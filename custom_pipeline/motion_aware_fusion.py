import argparse
import numpy as np
import torch
import torch.nn.functional as F

# =====================================================
# Arguments
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument("latent_original")
parser.add_argument("latent_warped")
parser.add_argument("motion_dir")
parser.add_argument("output")

args = parser.parse_args()

# =====================================================
# Load Latents
# =====================================================

latent_original = torch.load(args.latent_original)
latent_warped = torch.load(args.latent_warped)

print("Original :", latent_original.shape)
print("Warped   :", latent_warped.shape)

device = latent_original.device

# =====================================================
# Load Motion Masks
# =====================================================

background_mask = np.load(
    f"{args.motion_dir}/background_mask.npy"
).astype(np.float32)

foreground_mask = np.load(
    f"{args.motion_dir}/foreground_mask.npy"
).astype(np.float32)

residual_mask = np.load(
    f"{args.motion_dir}/residual_mask.npy"
).astype(np.float32)

# =====================================================
# Convert to tensors
# =====================================================

background_mask = torch.from_numpy(background_mask)
foreground_mask = torch.from_numpy(foreground_mask)
residual_mask = torch.from_numpy(residual_mask)

# =====================================================
# Resize masks to latent resolution
# =====================================================

latent_h = latent_original.shape[2]
latent_w = latent_original.shape[3]


def resize_mask(mask):

    mask = mask.unsqueeze(0).unsqueeze(0)

    mask = F.interpolate(
        mask,
        size=(latent_h, latent_w),
        mode="nearest",
    )

    mask = mask.expand(
        -1,
        latent_original.shape[1],
        -1,
        -1,
    )

    return mask.to(device)


background_mask = resize_mask(background_mask)
foreground_mask = resize_mask(foreground_mask)
residual_mask = resize_mask(residual_mask)

# =====================================================
# Motion-aware Fusion
# =====================================================

background = background_mask * latent_warped

foreground = foreground_mask * (
    0.7 * latent_warped +
    0.3 * latent_original
)

residual = residual_mask * latent_original

fused_latent = background + foreground + residual

print("Fused :", fused_latent.shape)

# =====================================================
# Save
# =====================================================

torch.save(
    fused_latent.cpu(),
    args.output,
)

print("\nSaved :", args.output)git ls-tree HEAD custom_pipeline
