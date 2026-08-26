import torch

from diffusers import (
    DDIMScheduler,
    UNet2DConditionModel,
    MotionAdapter,
)
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.unets.unet_motion_model import UNetMotionModel

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda:1"

MODEL_PATH = "/home/project/FlowceptionPlus/models/stable_diffusion/sd15"

MOTION_ADAPTER_PATH = "/home/project/FlowceptionPlus/models/animatediff_new/motion_adapter"

LATENT_PATH = "/home/project/FlowceptionPlus/custom_pipeline/video_latent.pt"

SAVE_PATH = "/home/project/FlowceptionPlus/custom_pipeline/denoised_latent.pt"

NUM_STEPS =30

DTYPE = torch.float16

torch.cuda.empty_cache()

# ============================================================
# Tokenizer + Text Encoder
# ============================================================

print("Loading Tokenizer...")

tokenizer = CLIPTokenizer.from_pretrained(
    MODEL_PATH,
    subfolder="tokenizer",
)

print("Loading Text Encoder...")

text_encoder = CLIPTextModel.from_pretrained(
    MODEL_PATH,
    subfolder="text_encoder",
    torch_dtype=DTYPE,
).to(DEVICE)

text_encoder.eval()

print("Text Encoder Loaded.")

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
# CLIP Prompt Embedding
# ============================================================

prompt = "a high quality video of a person performing tai chi outdoors"

text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="pt",
)

text_input = {
    k: v.to(DEVICE)
    for k, v in text_input.items()
}

with torch.no_grad():
    encoder_hidden_states = text_encoder(
        **text_input
    ).last_hidden_state

encoder_hidden_states = encoder_hidden_states.repeat(
    B * T,
    1,
    1,
)

print("Prompt Embedding :", encoder_hidden_states.shape)

# ============================================================
# Moderate Noise
# ============================================================

noise = torch.randn_like(latents)

start_index = int(len(scheduler.timesteps) * 0.75)

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
