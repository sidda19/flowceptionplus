import torch
import argparse

from diffusers import (
    DDIMScheduler,
    UNet2DConditionModel,
    MotionAdapter,
)
from diffusers.models.unets.unet_motion_model import UNetMotionModel

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda:1"

MODEL_PATH = "/home/project/FlowceptionPlus/models/stable_diffusion/sd15"

MOTION_ADAPTER_PATH = "/home/project/FlowceptionPlus/models/animatediff_new/motion_adapter"
parser = argparse.ArgumentParser()

parser.add_argument("video_latent")
parser.add_argument("output")

args = parser.parse_args()

LATENT_PATH = args.video_latent
SAVE_PATH = args.output

NUM_STEPS = 25

DTYPE = torch.float16

torch.cuda.empty_cache()

# ============================================================
# Scheduler
# ============================================================

scheduler = DDIMScheduler.from_pretrained(
    MODEL_PATH,
    subfolder="scheduler",
)

scheduler.set_timesteps(NUM_STEPS)

# ============================================================
# Motion Adapter
# ============================================================

print("Loading Motion Adapter...")

motion_adapter = MotionAdapter.from_pretrained(
    MOTION_ADAPTER_PATH,
    torch_dtype=DTYPE,
).to(DEVICE)

motion_adapter.eval()

print("Motion Adapter Loaded.")

# ============================================================
# SD UNet
# ============================================================

print("Loading SD UNet...")

unet = UNet2DConditionModel.from_pretrained(
    MODEL_PATH,
    subfolder="unet",
    torch_dtype=DTYPE,
).to(DEVICE)

unet.eval()

print("UNet Loaded.")

# ============================================================
# Convert to AnimateDiff Motion UNet
# ============================================================

print("Creating Motion UNet...")

motion_unet = UNetMotionModel.from_unet2d(
    unet,
    motion_adapter,
).to(DEVICE)

motion_unet.eval()

del unet
torch.cuda.empty_cache()

print("Motion UNet Ready.")

# ============================================================
# Load Video Latent
# ============================================================

video_latent = torch.load(
    LATENT_PATH,
    map_location=DEVICE,
).to(
    DEVICE,
    dtype=DTYPE,
)

print("Video Latent :", video_latent.shape)

B, C, T, H, W = video_latent.shape

latents = (
    video_latent
    .permute(0, 2, 1, 3, 4)
    .reshape(B * T, C, H, W)
)

print("Flattened :", latents.shape)

# ============================================================
# Empty Prompt Embedding
# ============================================================

encoder_hidden_states = torch.zeros(
    (
        B * T,
        77,
        768,
    ),
    device=DEVICE,
    dtype=DTYPE,
)

# ============================================================
# Moderate Noise
# ============================================================

noise = torch.randn_like(latents)

start_index = len(scheduler.timesteps) // 2

start_timestep = scheduler.timesteps[start_index]

latents = scheduler.add_noise(
    latents,
    noise,
    start_timestep,
)

print("Starting timestep :", start_timestep.item())

# ============================================================
# Denoising
# ============================================================

timesteps = scheduler.timesteps[start_index:]

for i, timestep in enumerate(timesteps):

    latent_model_input = latents.unsqueeze(2)

    latent_model_input = scheduler.scale_model_input(
        latent_model_input,
        timestep,
    )

    with torch.no_grad():

        noise_pred = motion_unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

    noise_pred = noise_pred.squeeze(2)

    latents = scheduler.step(
        noise_pred,
        timestep,
        latents,
    ).prev_sample

    if i % 5 == 0:
        print(
            f"Step {i+1}/{len(timesteps)} | "
            f"Timestep {timestep.item()}"
        )

# ============================================================
# Restore Video Shape
# ============================================================

denoised = (
    latents
    .reshape(B, T, C, H, W)
    .permute(0, 2, 1, 3, 4)
)

print("Output :", denoised.shape)

torch.save(
    denoised.cpu(),
    SAVE_PATH,
)

print("\nSaved :", SAVE_PATH)