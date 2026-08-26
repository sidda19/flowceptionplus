import os
import sys
import subprocess
import argparse


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Run complete Flowception+ trained inference pipeline"
)

parser.add_argument(
    "--results_root",
    type=str,
    required=True,
    help="Input root containing video_001, video_002, ..."
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Trained InterpolationNet checkpoint"
)

parser.add_argument(
    "--output_root",
    type=str,
    required=True,
    help="Root directory for final outputs"
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda:1",
    help="cuda:0, cuda:1, or cpu"
)

parser.add_argument(
    "--skip_existing",
    action="store_true",
    help="Skip samples whose final_frames already exist"
)

args = parser.parse_args()


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PYTHON = sys.executable


RESULTS_ROOT = os.path.abspath(
    args.results_root
)

CHECKPOINT = os.path.abspath(
    args.checkpoint
)

OUTPUT_ROOT = os.path.abspath(
    args.output_root
)


# ============================================================
# Script paths
# ============================================================

INTERPOLATION_INFERENCE = os.path.join(
    PROJECT_DIR,
    "interpolation_inference.py"
)

DECODE_INTERPOLATION = os.path.join(
    PROJECT_DIR,
    "decode_interpolation.py"
)

SD_RUNNER = os.path.join(
    PROJECT_DIR,
    "run_sd_denoise_interpolation.py"
)

DECODE_SD = os.path.join(
    PROJECT_DIR,
    "decode_sd_interpolation.py"
)


# ============================================================
# Check paths
# ============================================================

if not os.path.isdir(RESULTS_ROOT):

    raise FileNotFoundError(
        f"\nInput results root does not exist:\n"
        f"{RESULTS_ROOT}"
    )


if not os.path.isfile(CHECKPOINT):

    raise FileNotFoundError(
        f"\nCheckpoint not found:\n"
        f"{CHECKPOINT}"
    )


required_scripts = [

    INTERPOLATION_INFERENCE,

    DECODE_INTERPOLATION,

    SD_RUNNER,

    DECODE_SD,

]


for script in required_scripts:

    if not os.path.isfile(script):

        raise FileNotFoundError(
            f"\nRequired script not found:\n"
            f"{script}"
        )


os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


# ============================================================
# Header
# ============================================================

print()
print("=" * 75)
print("FLOWCEPTION+ COMPLETE TRAINED INFERENCE PIPELINE")
print("=" * 75)

print("Project       :", PROJECT_DIR)
print("Input root    :", RESULTS_ROOT)
print("Checkpoint    :", CHECKPOINT)
print("Output root   :", OUTPUT_ROOT)
print("Python        :", PYTHON)
print("Device        :", args.device)

print("=" * 75)


# ============================================================
# Find video/sample folders
# ============================================================

samples = sorted([

    name

    for name in os.listdir(RESULTS_ROOT)

    if os.path.isdir(
        os.path.join(
            RESULTS_ROOT,
            name
        )
    )

])


if len(samples) == 0:

    raise RuntimeError(
        f"No sample directories found in:\n"
        f"{RESULTS_ROOT}"
    )


print(
    f"\nFound {len(samples)} input samples."
)


# ============================================================
# Required prediction inputs
# ============================================================

required_inputs = [

    "latent_a.pt",

    "latent_b.pt",

    "fused_latent.pt",

    "background_mask.npy",

    "foreground_mask.npy",

    "residual_mask.npy",

]


# ============================================================
# STEP 1 + STEP 2
#
# Predict and decode all videos
# ============================================================

for index, sample in enumerate(
    samples,
    start=1
):

    print()
    print("=" * 75)

    print(
        f"SAMPLE {index}/{len(samples)} : "
        f"{sample}"
    )

    print("=" * 75)


    # --------------------------------------------------------
    # Input folder
    # --------------------------------------------------------

    input_folder = os.path.join(
        RESULTS_ROOT,
        sample
    )


    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    output_folder = os.path.join(
        OUTPUT_ROOT,
        sample
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )


    final_frames_dir = os.path.join(
        output_folder,
        "final_frames"
    )


    # --------------------------------------------------------
    # Skip completed samples
    # --------------------------------------------------------

    if (
        args.skip_existing
        and os.path.isdir(final_frames_dir)
    ):

        frames = [

            f

            for f in os.listdir(final_frames_dir)

            if f.endswith(".png")

        ]


        if len(frames) > 0:

            print(
                "\nSkipping completed sample."
            )

            print(
                "Final frames:",
                final_frames_dir
            )

            continue


    # --------------------------------------------------------
    # Check required inputs
    # --------------------------------------------------------

    missing = [

        filename

        for filename in required_inputs

        if not os.path.isfile(
            os.path.join(
                input_folder,
                filename
            )
        )

    ]


    if missing:

        print(
            "\nSkipping sample due to missing inputs:"
        )

        for filename in missing:

            print(
                "  -",
                filename
            )

        continue


    # ========================================================
    # STEP 1
    # InterpolationNet prediction
    # ========================================================

    prediction_file = os.path.join(
        output_folder,
        "predicted_latents.pt"
    )


    print()
    print("-" * 75)
    print("STEP 1 : TRAINED INTERPOLATION PREDICTION")
    print("-" * 75)


    subprocess.run(

        [

            PYTHON,

            INTERPOLATION_INFERENCE,

            "--input_dir",
            input_folder,

            "--checkpoint",
            CHECKPOINT,

            "--output",
            prediction_file,

            "--device",
            args.device,

        ],

        cwd=PROJECT_DIR,

        check=True

    )


    if not os.path.isfile(prediction_file):

        raise RuntimeError(
            "\nPrediction completed but output "
            "was not created:\n"
            f"{prediction_file}"
        )


    # ========================================================
    # STEP 2
    # Decode predicted latent frames
    # ========================================================

    predicted_frames_dir = os.path.join(
        output_folder,
        "predicted_frames"
    )


    print()
    print("-" * 75)
    print("STEP 2 : DECODE PREDICTED LATENTS")
    print("-" * 75)


    subprocess.run(

        [

            PYTHON,

            DECODE_INTERPOLATION,

            "--input",
            prediction_file,

            "--output_dir",
            predicted_frames_dir,

            "--device",
            args.device,

        ],

        cwd=PROJECT_DIR,

        check=True

    )


    print(
        "\nPredicted frames saved to:"
    )

    print(
        predicted_frames_dir
    )


# ============================================================
# STEP 3
# SD-U-Net refinement
# ============================================================

print()
print("=" * 75)
print("STEP 3 : SD-U-NET REFINEMENT")
print("=" * 75)


subprocess.run(

    [

        PYTHON,

        SD_RUNNER,

        "--results_root",
        OUTPUT_ROOT,

        "--steps",
        "25",

        "--seed",
        "42",

    ],

    cwd=PROJECT_DIR,

    check=True

)


# ============================================================
# STEP 4
# Decode all denoised outputs
# ============================================================

print()
print("=" * 75)
print("STEP 4 : DECODE SD-U-NET OUTPUTS")
print("=" * 75)


output_samples = sorted([

    name

    for name in os.listdir(OUTPUT_ROOT)

    if os.path.isdir(
        os.path.join(
            OUTPUT_ROOT,
            name
        )
    )

])


for index, sample in enumerate(
    output_samples,
    start=1
):

    print()
    print("-" * 75)

    print(
        f"DECODING {index}/{len(output_samples)} : "
        f"{sample}"
    )

    print("-" * 75)


    sample_folder = os.path.join(
        OUTPUT_ROOT,
        sample
    )


    denoised_latent = os.path.join(
        sample_folder,
        "denoised_predicted_latents.pt"
    )


    final_frames = os.path.join(
        sample_folder,
        "final_frames"
    )


    if not os.path.isfile(denoised_latent):

        print(
            "Skipping: denoised output not found."
        )

        continue


    subprocess.run(

        [

            PYTHON,

            DECODE_SD,

            "--input",
            denoised_latent,

            "--output_dir",
            final_frames,

            "--device",
            args.device,

        ],

        cwd=PROJECT_DIR,

        check=True

    )


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 75)
print("ALL FLOWCEPTION+ TRAINED PIPELINE STAGES COMPLETED")
print("=" * 75)

print()
print("Final output root:")
print(
    OUTPUT_ROOT
)

print()
print("Expected structure:")
print()

print(
    "trained_results/"
)

print(
    "  video_001/"
)

print(
    "    predicted_latents.pt"
)

print(
    "    predicted_frames/"
)

print(
    "      frame_00.png"
)

print(
    "      ..."
)

print(
    "    denoised_predicted_latents.pt"
)

print(
    "    final_frames/"
)

print(
    "      frame_00.png"
)

print(
    "      ..."
)

print()

print("=" * 75)