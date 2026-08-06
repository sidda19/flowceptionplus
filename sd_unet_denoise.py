import argparse
import torch
from diffusers import (
    UNet2DConditionModel,
    DDIMScheduler,
)

# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("video_latent")
parser.add_argument("output")

parser.add_argument(
    "--steps",
    type=int,
    default=25,
)

args = parser.parse_args()

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "/home/project/FlowceptionPlus/models/stable_diffusion/sd15"

NUM_STEPS = args.steps

DTYPE = torch.float16

# ============================================================
# Scheduler
# ============================================================

scheduler = DDIMScheduler.from_pretrained(
    MODEL_PATH,
    subfolder="scheduler",
)

scheduler.set_timesteps(NUM_STEPS)

# ============================================================
# Load UNet
# ============================================================

print("Loading Stable Diffusion UNet...")

unet = UNet2DConditionModel.from_pretrained(
    MODEL_PATH,
    subfolder="unet",
    torch_dtype=DTYPE,
).to(DEVICE)

unet.eval()

print("UNet Loaded.")

# ============================================================
# Load Video Latent
# ============================================================

video_latent = torch.load(
    args.video_latent,
    map_location=DEVICE,
).to(
    DEVICE,
    dtype=DTYPE,
)

print("Video Latent:", video_latent.shape)

B, C, T, H, W = video_latent.shape

latents = (
    video_latent
    .permute(0, 2, 1, 3, 4)
    .reshape(B * T, C, H, W)
)

print("Flattened:", latents.shape)

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
# Moderate Noise Initialization
# ============================================================

noise = torch.randn_like(latents)

start_index = len(scheduler.timesteps) // 2

start_timestep = scheduler.timesteps[start_index]

latents = scheduler.add_noise(
    latents,
    noise,
    start_timestep,
)

print("Starting timestep:", start_timestep.item())

# ============================================================
# DDIM Denoising
# ============================================================

timesteps = scheduler.timesteps[start_index:]

for i, timestep in enumerate(timesteps):

    latent_model_input = scheduler.scale_model_input(
        latents,
        timestep,
    )

    with torch.no_grad():

        noise_pred = unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

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

print("Output:", denoised.shape)

# ============================================================
# Save
# ============================================================

torch.save(
    denoised.cpu(),
    args.output,
)

print("\nSaved:")
print(args.output)