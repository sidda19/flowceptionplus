import os
import subprocess
import sys

# ==========================================================
# Configuration
# ==========================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

PYTHON = sys.executable

PIPELINE = os.path.join(
    PROJECT_DIR,
    "run_flowception_pipeline.py",
)

TEST_SET = os.path.join(
    PROJECT_DIR,
    "test_set",
)

RESULTS = os.path.join(
    PROJECT_DIR,
    "results",
)

os.makedirs(RESULTS, exist_ok=True)

# ==========================================================
# Collect videos
# ==========================================================

videos = sorted([
    d
    for d in os.listdir(TEST_SET)
    if os.path.isdir(os.path.join(TEST_SET, d))
])

print(f"Found {len(videos)} videos")

# ==========================================================
# Run Pipeline
# ==========================================================

for idx, video in enumerate(videos):

    print("\n")
    print("=" * 70)
    print(f"{idx+1}/{len(videos)} : {video}")
    print("=" * 70)

    video_dir = os.path.join(TEST_SET, video)

    frame_a = os.path.join(video_dir, "frame_a.png")
    frame_b = os.path.join(video_dir, "frame_b.png")

    output = os.path.join(RESULTS, video)

    os.makedirs(output, exist_ok=True)

    command = [
        PYTHON,
        PIPELINE,
        frame_a,
        frame_b,
        output,
    ]

    try:

        subprocess.run(
            command,
            check=True,
        )

    except subprocess.CalledProcessError:

        print(f"\nFAILED : {video}")

        continue

print("\n")
print("=" * 70)
print("ALL VIDEOS FINISHED")
print("=" * 70)