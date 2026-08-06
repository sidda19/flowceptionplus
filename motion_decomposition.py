import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Arguments
# ==========================================================

parser = argparse.ArgumentParser()

parser.add_argument("forward_flow")
parser.add_argument("output_dir")

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ==========================================================
# Load Data
# ==========================================================

forward_flow = np.load(args.forward_flow)

occlusion_path = os.path.join(args.output_dir, "occlusion_mask.npy")

if os.path.exists(occlusion_path):
    occlusion_mask = np.load(occlusion_path).astype(np.uint8)
else:
    # fallback if occlusion mask not available
    occlusion_mask = np.zeros(
        forward_flow.shape[:2],
        dtype=np.uint8,
    )

print("Forward Flow :", forward_flow.shape)
print("Occlusion    :", occlusion_mask.shape)

# ==========================================================
# Motion Magnitude
# ==========================================================

flow_x = forward_flow[:, :, 0]
flow_y = forward_flow[:, :, 1]

motion = np.sqrt(flow_x ** 2 + flow_y ** 2)

motion = cv2.normalize(
    motion,
    None,
    0,
    255,
    cv2.NORM_MINMAX,
).astype(np.uint8)

motion = cv2.GaussianBlur(
    motion,
    (5, 5),
    0,
)

# ==========================================================
# Foreground
# ==========================================================

_, foreground_mask = cv2.threshold(
    motion,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)

kernel = np.ones((5, 5), np.uint8)

foreground_mask = cv2.morphologyEx(
    foreground_mask,
    cv2.MORPH_OPEN,
    kernel,
)

foreground_mask = cv2.morphologyEx(
    foreground_mask,
    cv2.MORPH_CLOSE,
    kernel,
)

# ==========================================================
# Background
# ==========================================================

background_mask = cv2.bitwise_not(
    foreground_mask
)

# ==========================================================
# Residual
# ==========================================================

residual_mask = (
    occlusion_mask * 255
).astype(np.uint8)

background_mask[residual_mask > 0] = 0

# ==========================================================
# Boolean masks
# ==========================================================

foreground_bool = foreground_mask > 0
background_bool = background_mask > 0
residual_bool = residual_mask > 0

# ==========================================================
# Save NPY
# ==========================================================

np.save(
    os.path.join(args.output_dir, "foreground_mask.npy"),
    foreground_bool,
)

np.save(
    os.path.join(args.output_dir, "background_mask.npy"),
    background_bool,
)

np.save(
    os.path.join(args.output_dir, "residual_mask.npy"),
    residual_bool,
)

# ==========================================================
# Save Images
# ==========================================================

cv2.imwrite(
    os.path.join(args.output_dir, "foreground_mask.png"),
    foreground_mask,
)

cv2.imwrite(
    os.path.join(args.output_dir, "background_mask.png"),
    background_mask,
)

cv2.imwrite(
    os.path.join(args.output_dir, "residual_mask.png"),
    residual_mask,
)

# ==========================================================
# Visualization
# ==========================================================

plt.figure(figsize=(15, 4))

plt.subplot(1, 4, 1)
plt.imshow(motion, cmap="gray")
plt.title("Motion")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(background_mask, cmap="gray")
plt.title("Background")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(foreground_mask, cmap="gray")
plt.title("Foreground")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(residual_mask, cmap="gray")
plt.title("Residual")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        args.output_dir,
        "motion_decomposition.png",
    )
)

plt.close()

print("\nMotion decomposition completed.")
print("Saved:")
print(os.path.join(args.output_dir, "background_mask.npy"))
print(os.path.join(args.output_dir, "foreground_mask.npy"))
print(os.path.join(args.output_dir, "residual_mask.npy"))
print(os.path.join(args.output_dir, "motion_decomposition.png"))