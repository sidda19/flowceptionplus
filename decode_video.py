import os
import argparse
import torch
import cv2
import numpy as np
from diffusers import AutoencoderKL

# =====================================================
# Arguments
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument("latent_path")
parser.add_argument("output_dir")
parser.add_argument("output_video")

args = parser.parse_args()

# =====================================================
# Configuration
# =====================================================

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

VAE_PATH = "../models/stable_diffusion/sd15/vae"

OUTPUT_DIR = args.output_dir
VIDEO_NAME = args.output_video

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# Load VAE
# =====================================================

print("Loading VAE...")

vae = AutoencoderKL.from_pretrained(
    VAE_PATH,
    torch_dtype=torch.float16
).to(DEVICE)

vae.eval()

print("VAE Loaded.")

# =====================================================
# Load Latent
# =====================================================

video_latent = torch.load(
    args.latent_path,
    map_location=DEVICE
).to(
    DEVICE,
    dtype=torch.float16
)

print("Video Latent :", video_latent.shape)

B, C, T, H, W = video_latent.shape

# =====================================================
# Decode Frames
# =====================================================

decoded_frames = []

with torch.no_grad():

    for i in range(T):

        latent = video_latent[:, :, i]

        latent = latent / 0.18215

        image = vae.decode(latent).sample

        image = (image.clamp(-1, 1) + 1) / 2

        image = image[0].permute(1, 2, 0)

        image = image.cpu().numpy()

        image = (image * 255).astype(np.uint8)

        decoded_frames.append(image)

        frame_name = os.path.join(
            OUTPUT_DIR,
            f"frame_{i:03d}.png"
        )

        cv2.imwrite(
            frame_name,
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )

print(f"Saved {len(decoded_frames)} frames.")

# =====================================================
# Create MP4
# =====================================================

height, width = decoded_frames[0].shape[:2]

writer = cv2.VideoWriter(
    VIDEO_NAME,
    cv2.VideoWriter_fourcc(*"mp4v"),
    8,
    (width, height)
)

for frame in decoded_frames:

    writer.write(
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    )

writer.release()

print("Saved Video :", VIDEO_NAME)
print("Frames Saved :", OUTPUT_DIR)