import os
import subprocess
import sys

PYTHON = sys.executable

RESULTS = "./results"

SCRIPT = "sd_unet_denoise.py"

videos = sorted([
    d for d in os.listdir(RESULTS)
    if os.path.isdir(os.path.join(RESULTS, d))
])

print("Videos :", len(videos))

for video in videos:

    folder = os.path.join(RESULTS, video)

    input_latent = os.path.join(folder, "video_latent.pt")

    output_latent = os.path.join(folder, "denoised_latent_sd.pt")

    print("=" * 60)
    print(video)
    print("=" * 60)

    subprocess.run(
        [
            PYTHON,
            SCRIPT,
            input_latent,
            output_latent,
        ],
        check=True,
    )

print("\nFinished.")