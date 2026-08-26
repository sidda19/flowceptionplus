import torch

# ============================================================
# Configuration
# ============================================================

LATENT_A = "latent_a.pt"
LATENT_B = "latent_b.pt"
FUSED_LATENT = "fused_latent.pt"

OUTPUT = "video_latent.pt"

VIDEO_LENGTH = 16

# ============================================================
# Load Latents
# ============================================================

latent_a = torch.load(LATENT_A)
latent_b = torch.load(LATENT_B)
fused = torch.load(FUSED_LATENT)

print("Latent A :", latent_a.shape)
print("Fused    :", fused.shape)
print("Latent B :", latent_b.shape)

assert latent_a.shape == fused.shape == latent_b.shape

# ------------------------------------------------------------
# Convert to video format
# ------------------------------------------------------------

latent_a = latent_a.unsqueeze(2)
fused = fused.unsqueeze(2)
latent_b = latent_b.unsqueeze(2)

frames = []

for i in range(VIDEO_LENGTH):

    alpha = i / (VIDEO_LENGTH - 1)

    if alpha <= 0.5:

        t = alpha / 0.5

        frame = (1 - t) * latent_a + t * fused

    else:

        t = (alpha - 0.5) / 0.5

        frame = (1 - t) * fused + t * latent_b

    frames.append(frame)

video_latent = torch.cat(frames, dim=2)

print("\nVideo Latent:", video_latent.shape)

torch.save(video_latent, OUTPUT)

print("\nSaved:", OUTPUT)

print("\nFrame Statistics")
print("--------------------------------------")

for i in range(VIDEO_LENGTH):

    print(
        f"Frame {i:02d}",
        "Mean:",
        round(video_latent[:,:,i].mean().item(),4),
        "Std:",
        round(video_latent[:,:,i].std().item(),4)
    )

print("--------------------------------------")