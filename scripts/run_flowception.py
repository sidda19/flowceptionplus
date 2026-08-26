import subprocess
import sys
import os

# ==========================================================
# Python executable (same conda environment)
# ==========================================================

PYTHON = sys.executable

print("=" * 70)
print("FLOWCEPTION PIPELINE")
print("=" * 70)
print("Python executable:")
print(PYTHON)

# ==========================================================
# Project directory
# ==========================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# ==========================================================
# Pipeline
# ==========================================================

steps = [

    ("Encode Frame A",
     [PYTHON, "encode_frame.py",
      "frame_a.png",
      "latent_a.pt"]),

    ("Encode Frame B",
     [PYTHON, "encode_frame.py",
      "frame_b.png",
      "latent_b.pt"]),

    ("RAFT Optical Flow",
     [PYTHON,
      "raft_inference.py",
      "frame_a.png",
      "frame_b.png",
      "forward_flow.npy",
      "backward_flow.npy",
      "flow.png"]),

    ("Motion Decomposition",
     [PYTHON,
      "motion_decomposition.py"]),

    ("Warp Latent",
     [PYTHON,
      "latent_warp.py","latent_a.pt","forward_flow.npy","warped_latent.pt"]),

    ("Motion-aware Fusion",
     [PYTHON,
      "motion_aware_fusion.py"]),

    ("Prepare Video Latent",
     [PYTHON,
      "prepare_vedio_latents.py"]),

    ("AnimateDiff Denoising",
     [PYTHON,
      "animatediff_denoise_best.py"]),

    ("Decode Video",
     [PYTHON,
      "decode_all_frames.py"]),

    ("Create Video",
     [PYTHON,
      "frames_to_video.py"]),
]

# ==========================================================
# Execute
# ==========================================================

for idx, (name, command) in enumerate(steps, start=1):

    print("\n" + "=" * 70)
    print(f"STEP {idx}/{len(steps)} : {name}")
    print("=" * 70)
    print("Command:")
    print(" ".join(command))
    print()

    try:

        subprocess.run(
            command,
            cwd=PROJECT_DIR,
            check=True
        )

        print(f"\n{name} completed successfully.")

    except subprocess.CalledProcessError as e:

        print("\n" + "=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)
        print(f"Step: {name}")
        print(f"Return Code: {e.returncode}")
        sys.exit(1)

print("\n" + "=" * 70)
print("FLOWCEPTION PIPELINE FINISHED SUCCESSFULLY")
print("=" * 70)
