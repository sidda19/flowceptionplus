import sys
import torch
import cv2
from diffusers import AutoencoderKL
import os

import cv2
import numpy as np
import torch

from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms

# --------------------------------------------------
# Usage
# python encode_frame.py frame_a.png latent_a.pt
# python encode_frame.py frame_b.png latent_b.pt
# --------------------------------------------------

if len(sys.argv) != 3:
    print("Usage:")
    print("python encode_frame.py <input_image> <output_latent>")
    exit()

IMAGE_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

# --------------------------------------------------
# Configuration
# --------------------------------------------------

VAE_PATH = "/home/project/FlowceptionPlus/models/stable_diffusion/sd15"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load VAE
# --------------------------------------------------

vae = AutoencoderKL.from_pretrained(
    VAE_PATH,
    subfolder="vae",
).to(DEVICE)
vae.eval()

print("VAE Loaded Successfully!")

# --------------------------------------------------
# Load Image
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))

image = torch.from_numpy(image).float()
image = image.permute(2, 0, 1).unsqueeze(0)

image = image / 255.0
image = image * 2 - 1
image = image.to(DEVICE)

print("Input image shape:", image.shape)

# --------------------------------------------------
# Encode
# --------------------------------------------------

with torch.no_grad():
    latent = vae.encode(image).latent_dist.mode()
    latent = latent * 0.18215

print("Latent shape:", latent.shape)

# --------------------------------------------------
# Save Latent
# --------------------------------------------------

torch.save(latent.cpu(), OUTPUT_PATH)

print(f"Saved latent to: {OUTPUT_PATH}")

# --------------------------------------------------
# Optional Reconstruction Check
# --------------------------------------------------

with torch.no_grad():
    decoded = vae.decode(latent / 0.18215).sample

decoded = (decoded.clamp(-1, 1) + 1) / 2
decoded = decoded[0].permute(1, 2, 0).cpu().numpy()
decoded = (decoded * 255).astype(np.uint8)

recon_name = OUTPUT_PATH.replace(".pt", "_reconstructed.png")

cv2.imwrite(
    recon_name,
    cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR)
)

print(f"Saved reconstruction to: {recon_name}")