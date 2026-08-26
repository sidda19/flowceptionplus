import os
import argparse
import torch
import cv2
from diffusers import AutoencoderKL


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Decode SD-U-Net refined interpolation latents"
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to denoised_predicted_latents.pt"
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory where decoded frames will be saved"
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda:1",
    help="Device to use, e.g. cuda:0, cuda:1, cpu"
)

args = parser.parse_args()


# ============================================================
# Configuration
# ============================================================

VAE_PATH = (
    "/home/project/FlowceptionPlus/"
    "models/stable_diffusion/sd15"
)

LATENT_SCALE = 0.18215

DEVICE = (
    args.device
    if args.device.startswith("cuda") and torch.cuda.is_available()
    else "cpu"
)

INPUT_LATENTS = os.path.abspath(args.input)
OUTPUT_DIR = os.path.abspath(args.output_dir)


# ============================================================
# Header
# ============================================================

print("=" * 70)
print("FLOWCEPTION+ SD-U-NET INTERPOLATION DECODING")
print("=" * 70)

print("Device :", DEVICE)
print("VAE    :", VAE_PATH)
print("Input  :", INPUT_LATENTS)
print("Output :", OUTPUT_DIR)

print("=" * 70)


# ============================================================
# Check input
# ============================================================

if not os.path.exists(INPUT_LATENTS):

    raise FileNotFoundError(
        f"\nSD-denoised latent file not found:\n"
        f"{INPUT_LATENTS}"
    )

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Load VAE
# ============================================================

print("\nLoading VAE...")

vae = AutoencoderKL.from_pretrained(
    VAE_PATH,
    subfolder="vae"
).to(DEVICE)

vae.eval()

print("VAE Loaded Successfully.")


# ============================================================
# Load SD-U-Net denoised latents
# ============================================================

print("\nLoading SD-denoised latents...")

latents = torch.load(
    INPUT_LATENTS,
    map_location="cpu"
)

print(
    "Original latent shape:",
    latents.shape
)

print(
    "Original dtype:",
    latents.dtype
)


# ============================================================
# Normalize shape
#
# Supported:
#
# [4, T, H, W]
# [1, 4, T, H, W]
#
# Expected for current project:
#
# [4, 16, 32, 32]
# or
# [1, 4, 16, 32, 32]
# ============================================================

if latents.ndim == 4:

    if latents.shape[0] != 4:

        raise RuntimeError(
            "\nInvalid 4D latent shape.\n"
            "Expected [4,T,H,W], got "
            f"{tuple(latents.shape)}"
        )

    # Add batch dimension
    latents = latents.unsqueeze(0)


elif latents.ndim == 5:

    if latents.shape[1] != 4:

        raise RuntimeError(
            "\nInvalid 5D latent shape.\n"
            "Expected [B,4,T,H,W], got "
            f"{tuple(latents.shape)}"
        )


else:

    raise RuntimeError(
        "\nInvalid latent dimensions.\n"
        "Expected [4,T,H,W] or [B,4,T,H,W].\n"
        f"Got {tuple(latents.shape)}"
    )


# ============================================================
# Shape information
# ============================================================

B, C, T, H, W = latents.shape

print("\nNormalized latent shape:")
print(latents.shape)

print("Batch   :", B)
print("Channels:", C)
print("Frames  :", T)
print("Height  :", H)
print("Width   :", W)


# ============================================================
# Sanity checks
# ============================================================

if B != 1:

    raise RuntimeError(
        f"Expected batch size 1, got {B}"
    )

if C != 4:

    raise RuntimeError(
        f"Expected 4 latent channels, got {C}"
    )

if H != 32 or W != 32:

    raise RuntimeError(
        "Expected latent resolution 32x32, "
        f"got {H}x{W}"
    )


# ============================================================
# Move to GPU
# ============================================================

latents = latents.to(
    DEVICE,
    dtype=torch.float32
)


# ============================================================
# Decode
# ============================================================

print()
print("=" * 70)
print("DECODING SD-REFINED FRAMES")
print("=" * 70)


with torch.no_grad():

    for frame_idx in range(T):

        print(
            f"\nDecoding frame "
            f"{frame_idx:02d}/{T - 1:02d}"
        )

        # ----------------------------------------------------
        # Select one frame
        #
        # [1,4,T,32,32]
        #       ↓
        # [1,4,32,32]
        # ----------------------------------------------------

        latent_frame = latents[
            :,
            :,
            frame_idx,
            :,
            :
        ]

        print(
            "Latent frame shape:",
            latent_frame.shape
        )

        # ----------------------------------------------------
        # Undo Stable Diffusion latent scaling
        # ----------------------------------------------------

        latent_frame = (
            latent_frame / LATENT_SCALE
        )

        # ----------------------------------------------------
        # VAE decode
        # ----------------------------------------------------

        decoded = vae.decode(
            latent_frame
        ).sample

        print(
            "Decoded tensor:",
            decoded.shape
        )

        # ----------------------------------------------------
        # [-1,1] -> [0,1]
        # ----------------------------------------------------

        decoded = (
            decoded.clamp(-1, 1) + 1
        ) / 2

        # ----------------------------------------------------
        # Tensor -> NumPy
        #
        # [1,3,256,256]
        #       ↓
        # [256,256,3]
        # ----------------------------------------------------

        image = (
            decoded[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        image = (
            image * 255
        ).round().astype("uint8")

        # ----------------------------------------------------
        # RGB -> BGR
        # ----------------------------------------------------

        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------
        # Save frame
        # ----------------------------------------------------

        output_path = os.path.join(
            OUTPUT_DIR,
            f"frame_{frame_idx:02d}.png"
        )

        cv2.imwrite(
            output_path,
            image_bgr
        )

        print(
            "Saved:",
            output_path
        )


# ============================================================
# Finished
# ============================================================

print()
print("=" * 70)
print("SD-U-NET DECODING COMPLETED")
print("=" * 70)

print("Total frames:", T)
print("Output:", OUTPUT_DIR)

print("=" * 70)
