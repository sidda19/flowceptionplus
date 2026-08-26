import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from interpolation_net import InterpolationNet


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Flowception+ prediction-only interpolation inference"
)

parser.add_argument(
    "--input_dir",
    type=str,
    required=True,
    help="Input sample directory containing latents and masks"
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Trained InterpolationNet checkpoint"
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Output path for predicted_latents.pt"
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda:1",
    help="cuda:0, cuda:1, or cpu"
)

args = parser.parse_args()


# ============================================================
# Device
# ============================================================

if (
    args.device.startswith("cuda")
    and torch.cuda.is_available()
):
    device = torch.device(args.device)
else:
    print("CUDA unavailable or CPU requested.")
    device = torch.device("cpu")


# ============================================================
# Absolute paths
# ============================================================

INPUT_DIR = os.path.abspath(args.input_dir)
CHECKPOINT_PATH = os.path.abspath(args.checkpoint)
OUTPUT_PATH = os.path.abspath(args.output)


# ============================================================
# Header
# ============================================================

print("=" * 70)
print("FLOWCEPTION+ PREDICTION-ONLY INTERPOLATION")
print("=" * 70)

print("Input      :", INPUT_DIR)
print("Checkpoint :", CHECKPOINT_PATH)
print("Output     :", OUTPUT_PATH)
print("Device     :", device)

print("=" * 70)


# ============================================================
# Check paths
# ============================================================

if not os.path.isdir(INPUT_DIR):
    raise FileNotFoundError(
        f"Input directory does not exist:\n{INPUT_DIR}"
    )

if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint does not exist:\n{CHECKPOINT_PATH}"
    )


# ============================================================
# Required input files
# ============================================================

required_files = {

    "latent_a":
    os.path.join(
        INPUT_DIR,
        "latent_a.pt"
    ),

    "latent_b":
    os.path.join(
        INPUT_DIR,
        "latent_b.pt"
    ),

    "fused_latent":
    os.path.join(
        INPUT_DIR,
        "fused_latent.pt"
    ),

    "background_mask":
    os.path.join(
        INPUT_DIR,
        "background_mask.npy"
    ),

    "foreground_mask":
    os.path.join(
        INPUT_DIR,
        "foreground_mask.npy"
    ),

    "residual_mask":
    os.path.join(
        INPUT_DIR,
        "residual_mask.npy"
    ),
}


missing = []

for name, path in required_files.items():

    if not os.path.isfile(path):
        missing.append(
            f"{name}: {path}"
        )


if missing:

    raise FileNotFoundError(
        "\nMissing required input files:\n"
        + "\n".join(missing)
    )


# ============================================================
# Load latent helper
# ============================================================

def load_latent(path, name):

    latent = torch.load(
        path,
        map_location="cpu"
    ).float()

    print(
        f"{name} original shape:",
        tuple(latent.shape)
    )

    # [1,4,32,32] -> [4,32,32]
    if (
        latent.ndim == 4
        and latent.shape[0] == 1
    ):
        latent = latent.squeeze(0)

    if latent.shape != (4, 32, 32):

        raise RuntimeError(
            f"\nInvalid {name} shape.\n"
            f"Expected: (4, 32, 32)\n"
            f"Got     : {tuple(latent.shape)}"
        )

    return latent


# ============================================================
# Load mask helper
# ============================================================

def load_mask(path, name):

    mask = np.load(path)

    mask = torch.from_numpy(
        mask
    ).float()

    print(
        f"{name} original shape:",
        tuple(mask.shape)
    )

    # Handle possible shapes:
    # [H,W]
    # [1,H,W]
    # [H,W,1]

    if mask.ndim == 3:

        if mask.shape[0] == 1:

            mask = mask.squeeze(0)

        elif mask.shape[-1] == 1:

            mask = mask.squeeze(-1)

        else:

            raise RuntimeError(
                f"\nUnsupported {name} shape: "
                f"{tuple(mask.shape)}"
            )

    if mask.ndim != 2:

        raise RuntimeError(
            f"\nExpected 2D mask for {name}, "
            f"got {tuple(mask.shape)}"
        )

    # Resize to latent resolution
    if mask.shape != (32, 32):

        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(32, 32),
            mode="nearest"
        ).squeeze(0).squeeze(0)

    return mask


# ============================================================
# Load inputs
# ============================================================

print("\nLoading input latents...")

latent_a = load_latent(
    required_files["latent_a"],
    "latent_a"
)

latent_b = load_latent(
    required_files["latent_b"],
    "latent_b"
)

fused_latent = load_latent(
    required_files["fused_latent"],
    "fused_latent"
)


print("\nLoading motion masks...")

background_mask = load_mask(
    required_files["background_mask"],
    "background_mask"
)

foreground_mask = load_mask(
    required_files["foreground_mask"],
    "foreground_mask"
)

residual_mask = load_mask(
    required_files["residual_mask"],
    "residual_mask"
)


# ============================================================
# Add batch dimension
# ============================================================

latent_a = (
    latent_a
    .unsqueeze(0)
    .to(device)
)

latent_b = (
    latent_b
    .unsqueeze(0)
    .to(device)
)

fused_latent = (
    fused_latent
    .unsqueeze(0)
    .to(device)
)

background_mask = (
    background_mask
    .unsqueeze(0)
    .to(device)
)

foreground_mask = (
    foreground_mask
    .unsqueeze(0)
    .to(device)
)

residual_mask = (
    residual_mask
    .unsqueeze(0)
    .to(device)
)


# ============================================================
# Print final shapes
# ============================================================

print("\nFinal input shapes:")

print(
    "latent_a       :",
    tuple(latent_a.shape)
)

print(
    "latent_b       :",
    tuple(latent_b.shape)
)

print(
    "fused_latent   :",
    tuple(fused_latent.shape)
)

print(
    "background_mask:",
    tuple(background_mask.shape)
)

print(
    "foreground_mask:",
    tuple(foreground_mask.shape)
)

print(
    "residual_mask  :",
    tuple(residual_mask.shape)
)


# ============================================================
# Load model
# ============================================================

print("\nLoading InterpolationNet...")

model = InterpolationNet()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)


# ============================================================
# Handle checkpoint formats
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# Remove DataParallel prefix if present
state_dict = {

    key.replace("module.", ""): value

    for key, value in state_dict.items()
}


model.load_state_dict(
    state_dict
)

model.to(device)
model.eval()

print(
    "InterpolationNet loaded successfully."
)


# ============================================================
# Prediction
# ============================================================

print("\nRunning prediction...")

with torch.no_grad():

    prediction = model(

        latent_a,

        latent_b,

        fused_latent,

        background_mask,

        foreground_mask,

        residual_mask,
    )


print(
    "Prediction shape:",
    tuple(prediction.shape)
)


# ============================================================
# Validate prediction
# ============================================================

if prediction.ndim != 5:

    raise RuntimeError(
        f"\nExpected 5D prediction "
        f"[B,4,T,32,32].\n"
        f"Got: {tuple(prediction.shape)}"
    )


if prediction.shape[0] != 1:

    raise RuntimeError(
        f"Expected batch size 1, "
        f"got {prediction.shape[0]}"
    )


if prediction.shape[1] != 4:

    raise RuntimeError(
        f"Expected 4 latent channels, "
        f"got {prediction.shape[1]}"
    )


if prediction.shape[-2:] != (32, 32):

    raise RuntimeError(
        f"Expected latent size 32x32, "
        f"got {tuple(prediction.shape[-2:])}"
    )


# ============================================================
# Save prediction
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)


prediction_to_save = (
    prediction
    .squeeze(0)
    .detach()
    .cpu()
)


torch.save(
    prediction_to_save,
    OUTPUT_PATH
)


print("\n" + "=" * 70)
print("PREDICTION COMPLETED")
print("=" * 70)

print(
    "Saved:",
    OUTPUT_PATH
)

print(
    "Saved shape:",
    tuple(prediction_to_save.shape)
)

print("=" * 70)