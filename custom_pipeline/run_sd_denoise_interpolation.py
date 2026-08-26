import os
import subprocess
import sys
import argparse


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Run SD-U-Net refinement on predicted interpolation latents"
)

parser.add_argument(
    "--results_root",
    type=str,
    required=True,
    help="Root directory containing predicted_latents.pt files"
)

parser.add_argument(
    "--steps",
    type=int,
    default=25
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

args = parser.parse_args()


# ============================================================
# Configuration
# ============================================================

PYTHON = sys.executable

RESULTS = os.path.abspath(
    args.results_root
)

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SCRIPT = os.path.join(
    PROJECT_DIR,
    "sd_unet_denoise.py"
)


# ============================================================
# Checks
# ============================================================

if not os.path.isdir(RESULTS):

    raise FileNotFoundError(
        f"Results directory does not exist:\n{RESULTS}"
    )


if not os.path.isfile(SCRIPT):

    raise FileNotFoundError(
        f"SD U-Net script not found:\n{SCRIPT}"
    )


# ============================================================
# Find samples
# ============================================================

samples = sorted([

    d

    for d in os.listdir(RESULTS)

    if os.path.isdir(
        os.path.join(
            RESULTS,
            d
        )
    )

])


print("=" * 70)
print("FLOWCEPTION+ SD-U-NET REFINEMENT")
print("=" * 70)

print("Results :", RESULTS)
print("Samples :", len(samples))
print("Steps   :", args.steps)
print("Seed    :", args.seed)

print("=" * 70)


# ============================================================
# Process samples
# ============================================================

for idx, sample in enumerate(
    samples,
    start=1
):

    print()
    print("=" * 70)

    print(
        f"SAMPLE {idx}/{len(samples)} : "
        f"{sample}"
    )

    print("=" * 70)


    folder = os.path.join(
        RESULTS,
        sample
    )


    input_latent = os.path.join(
        folder,
        "predicted_latents.pt"
    )


    output_latent = os.path.join(
        folder,
        "denoised_predicted_latents.pt"
    )


    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.isfile(input_latent):

        print(
            "Skipping: predicted_latents.pt not found."
        )

        continue


    # --------------------------------------------------------
    # Run SD U-Net
    # --------------------------------------------------------

    subprocess.run(

        [

            PYTHON,

            SCRIPT,

            input_latent,

            output_latent,

            "--steps",
            str(args.steps),

            "--seed",
            str(args.seed),

        ],

        check=True,

        cwd=PROJECT_DIR
    )


    if not os.path.isfile(output_latent):

        raise RuntimeError(
            "SD U-Net completed but output was not created:\n"
            f"{output_latent}"
        )


    print(
        "\nCompleted:",
        sample
    )


# ============================================================
# Finished
# ============================================================

print()
print("=" * 70)
print("ALL  U-NET REFINEMENTS COMPLETED")
print("=" * 70)