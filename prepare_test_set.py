import os
import random
import shutil

# ==========================================================
# Configuration
# ==========================================================

DATASET_ROOT = "/mnt/zone_a/FlowceptionPlus/datasets/TaichiHD/taichi-256/frames/train"

OUTPUT_ROOT = "./test_set"

NUM_VIDEOS = 30

FRAME_GAP = 50

SEED = 42

# ==========================================================
# Reproducibility
# ==========================================================

random.seed(SEED)

# ==========================================================
# Create output folder
# ==========================================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ==========================================================
# Collect usable videos
# ==========================================================

videos = []

for video in sorted(os.listdir(DATASET_ROOT)):

    video_path = os.path.join(DATASET_ROOT, video)

    if not os.path.isdir(video_path):
        continue

    frames = sorted(
        [
            f
            for f in os.listdir(video_path)
            if f.endswith(".png")
        ]
    )

    if len(frames) > FRAME_GAP + 1:

        videos.append(
            (
                video,
                video_path,
                frames,
            )
        )

print(f"Usable videos : {len(videos)}")

# ==========================================================
# Select videos
# ==========================================================

selected = random.sample(
    videos,
    min(NUM_VIDEOS, len(videos)),
)

print(f"Selected {len(selected)} videos.")

# ==========================================================
# Copy
# ==========================================================

for idx, (video_name, video_path, frames) in enumerate(selected):

    start = random.randint(
        0,
        len(frames) - FRAME_GAP - 1,
    )

    end = start + FRAME_GAP

    sample_dir = os.path.join(
        OUTPUT_ROOT,
        f"video_{idx+1:03d}",
    )

    gt_dir = os.path.join(
        sample_dir,
        "ground_truth",
    )

    os.makedirs(gt_dir, exist_ok=True)

    frame_a = frames[start]
    frame_b = frames[end]

    shutil.copy(
        os.path.join(video_path, frame_a),
        os.path.join(sample_dir, "frame_a.png"),
    )

    shutil.copy(
        os.path.join(video_path, frame_b),
        os.path.join(sample_dir, "frame_b.png"),
    )

    # Copy every frame between A and B (inclusive)

    for k in range(start, end + 1):

        shutil.copy(
            os.path.join(video_path, frames[k]),
            os.path.join(
                gt_dir,
                f"frame{k-start:03d}.png",
            ),
        )

    print(
        f"video_{idx+1:03d}"
        f"  frames {start} -> {end}"
    )

print("\nFinished.")
print("Saved dataset to:")
print(OUTPUT_ROOT)