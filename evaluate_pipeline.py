import os
import cv2
import lpips
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

# ==========================================================
# Configuration
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = lpips.LPIPS(net="alex").to(DEVICE)

GT_ROOT = "./test_set"
RESULT_ROOT = "./results"

PIPELINES = {
    "AnimateDiff": "decoded_frames",
    "SDUNet": "decoded_frames_sd",
}

# ==========================================================
# Helper Functions
# ==========================================================

def load_image(path):

    img = cv2.imread(path)

    if img is None:
        raise RuntimeError(f"Cannot read {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0

    return img


def lpips_score(img1, img2):

    t1 = (
        torch.from_numpy(img1)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )

    t2 = (
        torch.from_numpy(img2)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        score = loss_fn(
            t1 * 2 - 1,
            t2 * 2 - 1,
        )

    return score.item()


# ==========================================================
# Evaluate
# ==========================================================

for pipeline_name, pred_folder in PIPELINES.items():

    print("\n" + "=" * 70)
    print(f"Evaluating {pipeline_name}")
    print("=" * 70)

    rows = []

    videos = sorted([
        v for v in os.listdir(GT_ROOT)
        if os.path.isdir(os.path.join(GT_ROOT, v))
    ])

    for video in tqdm(videos):

        gt_dir = os.path.join(
            GT_ROOT,
            video,
            "ground_truth"
        )

        pred_dir = os.path.join(
            RESULT_ROOT,
            video,
            pred_folder
        )

        if not os.path.exists(pred_dir):
            print(f"Skipping {video} (missing {pred_folder})")
            continue

        gt_frames = sorted([
            f for f in os.listdir(gt_dir)
            if f.endswith(".png")
        ])

        pred_frames = sorted([
            f for f in os.listdir(pred_dir)
            if f.endswith(".png")
        ])

        n = min(
            len(gt_frames),
            len(pred_frames)
        )

        if n == 0:
            continue

        psnr_all = []
        ssim_all = []
        lpips_all = []

        for i in range(n):

            gt = load_image(
                os.path.join(
                    gt_dir,
                    gt_frames[i]
                )
            )

            pred = load_image(
                os.path.join(
                    pred_dir,
                    pred_frames[i]
                )
            )

            psnr = peak_signal_noise_ratio(
                gt,
                pred,
                data_range=1.0
            )

            ssim = structural_similarity(
                gt,
                pred,
                channel_axis=2,
                data_range=1.0
            )

            lp = lpips_score(
                gt,
                pred
            )

            psnr_all.append(psnr)
            ssim_all.append(ssim)
            lpips_all.append(lp)

        rows.append({

            "video": video,

            "frames": n,

            "PSNR": np.mean(psnr_all),

            "SSIM": np.mean(ssim_all),

            "LPIPS": np.mean(lpips_all),

        })

    df = pd.DataFrame(rows)

    csv_name = f"evaluation_{pipeline_name}.csv"

    df.to_csv(
        csv_name,
        index=False
    )

    print("\nPer-video Results")
    print(df)

    print("\nAverage Metrics")
    print(df[["PSNR", "SSIM", "LPIPS"]].mean())

    print(f"\nSaved -> {csv_name}")

print("\nEvaluation Finished.")