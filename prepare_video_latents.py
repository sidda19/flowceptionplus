import argparse
import torch

# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("latent_a")
parser.add_argument("latent_b")
parser.add_argument("fused_latent")
parser.add_argument("output")

parser.add_argument(
    "--video_length",
    type=int,
    default=16,
)

args = parser.parse_args()

VIDEO_LENGTH = args.video_length

# ============================================================
# Load Latents
# ============================================================

latent_a = torch.load(args.latent_a)
latent_b = torch.load(args.latent_b)
fused = torch.load(args.fused_latent)

print("Latent A :", latent_a.shape)
print("Fused    :", fused.shape)
print("Latent B :", latent_b.shape)

assert latent_a.shape == fused.shape == latent_b.shape

# ============================================================
# Add temporal dimension
# ============================================================

latent_a = latent_a.unsqueeze(2)
fused = fused.unsqueeze(2)
latent_b = latent_b.unsqueeze(2)

# ============================================================
# Create Video Latent
# ============================================================

B, C, _, H, W = latent_a.shape

video_latent = torch.empty(
    (
        B,
        C,
        VIDEO_LENGTH,
        H,
        W,
    ),
    dtype=latent_a.dtype,
)

for i in range(VIDEO_LENGTH):

    alpha = i / (VIDEO_LENGTH - 1)

    if alpha <= 0.5:

        t = alpha / 0.5

        frame = (1 - t) * latent_a + t * fused

    else:

        t = (alpha - 0.5) / 0.5

        frame = (1 - t) * fused + t * latent_b

    video_latent[:, :, i] = frame.squeeze(2)

print("\nVideo Latent:", video_latent.shape)

torch.save(video_latent, args.output)

print("\nSaved:", args.output)

# ============================================================
# Statistics
# ============================================================

print("\nFrame Statistics")
print("--------------------------------------")

for i in range(VIDEO_LENGTH):

    print(
        f"Frame {i:02d}",
        "Mean:",
        round(video_latent[:, :, i].mean().item(), 4),
        "Std:",
        round(video_latent[:, :, i].std().item(), 4),
    )

print("--------------------------------------")