import os
import argparse
import torch
import cv2
import numpy as np

from diffusers import AutoencoderKL


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    required=True
)

parser.add_argument(
    "--output_dir",
    required=True
)

parser.add_argument(
    "--vae",
    default="/home/project/FlowceptionPlus/models/stable_diffusion/sd15"
)

parser.add_argument(
    "--device",
    default="cuda:1"
)

args = parser.parse_args()


# ============================================================
# Configuration
# ============================================================

LATENT_SCALE = 0.18215

device = torch.device(
    args.device if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Header
# ============================================================

print("=" * 70)
print("FLOWCEPTION+ INTERPOLATION LATENT DECODING")
print("=" * 70)

print("Device :", device)
print("VAE    :", args.vae)
print("Input  :", args.input)
print("Output :", args.output_dir)

print("=" * 70)


# ============================================================
# Check
# ============================================================

if not os.path.exists(args.input):

    raise FileNotFoundError(
        args.input
    )

os.makedirs(
    args.output_dir,
    exist_ok=True
)


# ============================================================
# Load VAE
# ============================================================

print("\nLoading VAE...")

vae = AutoencoderKL.from_pretrained(
    args.vae,
    subfolder="vae"
).to(device)

vae.eval()

print("VAE Loaded Successfully.")


# ============================================================
# Load latent
# ============================================================

latents = torch.load(
    args.input,
    map_location="cpu"
).float()


print(
    "\nOriginal latent shape:",
    latents.shape
)


# ============================================================
# Normalize
# ============================================================

if latents.ndim == 4:

    # [4,T,H,W]
    latents = latents.unsqueeze(0)

elif latents.ndim != 5:

    raise RuntimeError(
        f"Expected [4,T,H,W] or "
        f"[B,4,T,H,W], got {latents.shape}"
    )


B, C, T, H, W = latents.shape


print("\nNormalized latent shape:")
print(latents.shape)


if C != 4:
    raise RuntimeError(
        f"Expected 4 channels, got {C}"
    )


# ============================================================
# GPU
# ============================================================

latents = latents.to(
    device,
    dtype=torch.float32
)


# ============================================================
# Decode
# ============================================================

print("\n" + "=" * 70)
print("DECODING FRAMES")
print("=" * 70)


with torch.no_grad():

    for t in range(T):

        print(
            f"Decoding frame {t:02d}/{T-1:02d}"
        )

        latent_frame = latents[
            :,
            :,
            t
        ]

        latent_frame = (
            latent_frame / LATENT_SCALE
        )

        decoded = vae.decode(
            latent_frame
        ).sample

        decoded = (
            decoded.clamp(-1, 1) + 1
        ) / 2

        image = (
            decoded[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        image = (
            image * 255
        ).round().astype(np.uint8)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        output_path = os.path.join(
            args.output_dir,
            f"frame_{t:02d}.png"
        )

        cv2.imwrite(
            output_path,
            image
        )

        print("Saved:", output_path)


print("\n" + "=" * 70)
print("DECODING COMPLETED")
print("=" * 70)

print("Frames :", T)
print("Output :", args.output_dir)

print("=" * 70)