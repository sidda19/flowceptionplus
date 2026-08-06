import subprocess
import sys
import os

# ==========================================================
# Arguments
# ==========================================================

if len(sys.argv) != 4:

    print("Usage:")
    print("python run_flowception_pipeline.py frameA frameB output_folder")
    sys.exit(1)

FRAME_A = os.path.abspath(sys.argv[1])
FRAME_B = os.path.abspath(sys.argv[2])
OUTPUT_DIR = os.path.abspath(sys.argv[3])

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

# ==========================================================
# Script Paths
# ==========================================================

ENCODE = os.path.join(PROJECT_DIR, "encode_frame.py")
RAFT = os.path.join(PROJECT_DIR, "raft_inference.py")
MOTION = os.path.join(PROJECT_DIR, "motion_decomposition.py")
WARP = os.path.join(PROJECT_DIR, "latent_warp.py")
FUSION = os.path.join(PROJECT_DIR, "motion_aware_fusion.py")
PREPARE = os.path.join(PROJECT_DIR, "prepare_video_latents.py")
ANIMATEDIFF = os.path.join(PROJECT_DIR, "animatediff_denoise.py")
DECODE = os.path.join(PROJECT_DIR, "decode_video.py")

# ==========================================================
# Output Files
# ==========================================================

LATENT_A = os.path.join(OUTPUT_DIR, "latent_a.pt")
LATENT_B = os.path.join(OUTPUT_DIR, "latent_b.pt")

FORWARD_FLOW = os.path.join(OUTPUT_DIR, "forward_flow.npy")
BACKWARD_FLOW = os.path.join(OUTPUT_DIR, "backward_flow.npy")
FLOW_IMAGE = os.path.join(OUTPUT_DIR, "flow.png")

WARPED_LATENT = os.path.join(OUTPUT_DIR, "warped_latent.pt")
FUSED_LATENT = os.path.join(OUTPUT_DIR, "fused_latent.pt")

VIDEO_LATENT = os.path.join(OUTPUT_DIR, "video_latent.pt")
DENOISED_LATENT = os.path.join(OUTPUT_DIR, "denoised_latent.pt")

FRAME_FOLDER = os.path.join(OUTPUT_DIR, "decoded_frames")
VIDEO_FILE = os.path.join(OUTPUT_DIR, "generated.mp4")

print("=" * 70)
print("FLOWCEPTION")
print("=" * 70)
print("Python :", PYTHON)
print("Frame A:", FRAME_A)
print("Frame B:", FRAME_B)
print("Output :", OUTPUT_DIR)
print("=" * 70)

steps = [

    (
        "Encode Frame A",
        [
            PYTHON,
            ENCODE,
            FRAME_A,
            LATENT_A,
        ],
    ),

    (
        "Encode Frame B",
        [
            PYTHON,
            ENCODE,
            FRAME_B,
            LATENT_B,
        ],
    ),

    (
        "RAFT Optical Flow",
        [
            PYTHON,
            RAFT,
            FRAME_A,
            FRAME_B,
            FORWARD_FLOW,
            BACKWARD_FLOW,
            FLOW_IMAGE,
        ],
    ),

    (
        "Motion Decomposition",
        [
            PYTHON,
            MOTION,
            FORWARD_FLOW,
            OUTPUT_DIR,
        ],
    ),

    (
        "Warp Latent",
        [
            PYTHON,
            WARP,
            LATENT_A,
            FORWARD_FLOW,
            WARPED_LATENT,
        ],
    ),

    (
        "Motion Fusion",
        [
            PYTHON,
            FUSION,
            LATENT_A,
            WARPED_LATENT,
            OUTPUT_DIR,
            FUSED_LATENT,
        ],
    ),

    (
        "Prepare Video Latent",
        [
            PYTHON,
            PREPARE,
            LATENT_A,
            LATENT_B,
            FUSED_LATENT,
            VIDEO_LATENT,
        ],
    ),

    (
        "AnimateDiff Denoising",
        [
            PYTHON,
            ANIMATEDIFF,
            VIDEO_LATENT,
            DENOISED_LATENT,
        ],
    ),

    (
        "Decode Video",
        [
            PYTHON,
            DECODE,
            DENOISED_LATENT,
            FRAME_FOLDER,
            VIDEO_FILE,
        ],
    ),
]

# ==========================================================
# Execute
# ==========================================================

for i, (name, command) in enumerate(steps):

    print("\n")
    print("=" * 70)
    print(f"STEP {i+1}/{len(steps)} : {name}")
    print("=" * 70)

    try:

        subprocess.run(
            command,
            cwd=PROJECT_DIR,
            check=True,
        )

    except subprocess.CalledProcessError:

        print("\nFAILED STEP :", name)
        print("Command:")
        print(" ".join(command))
        raise

print("\n")
print("=" * 70)
print("FLOWCEPTION PIPELINE FINISHED")
print("=" * 70)
print("Results saved to:")
print(OUTPUT_DIR)