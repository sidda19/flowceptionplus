
import os
import cv2
import lpips
import torch
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Generalized Flowception+ image/video evaluation"
)

parser.add_argument(
    "--gt_root",
    type=str,
    required=True,
    help="Root directory containing video_xxx/ground_truth/"
)

parser.add_argument(
    "--results_root",
    type=str,
    required=True,
    help="Root directory containing video_xxx/<prediction_folder>/"
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="./evaluation_results",
    help="Directory where evaluation CSV files will be saved"
)

parser.add_argument(
    "--gt_folder",
    type=str,
    default="ground_truth",
    help="Ground-truth frame folder inside each video directory"
)

parser.add_argument(
    "--pipeline",
    action="append",
    required=True,
    help=(
        "Pipeline in NAME=FOLDER format. "
        "Can be specified multiple times. "
        "Example: --pipeline InterpolationNet=predicted_frames "
        "--pipeline SDUNet=final_frames"
    )
)

parser.add_argument(
    "--match_mode",
    type=str,
    choices=[
        "index",
        "evenly_spaced",
    ],
    default="index",
    help=(
        "Frame matching strategy. "
        "'index' compares GT frame 0..N with prediction frame 0..N. "
        "'evenly_spaced' samples GT frames uniformly when prediction "
        "has fewer frames."
    )
)

parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Device for LPIPS, e.g. cuda:1, cuda:0, or cpu"
)

args = parser.parse_args()


# ============================================================
# Device
# ============================================================

if args.device is not None:

    if (
        args.device.startswith("cuda")
        and torch.cuda.is_available()
    ):
        DEVICE = args.device
    else:
        print(
            "Requested CUDA unavailable. "
            "Using CPU."
        )
        DEVICE = "cpu"

else:

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# Absolute paths
# ============================================================

GT_ROOT = os.path.abspath(
    args.gt_root
)

RESULT_ROOT = os.path.abspath(
    args.results_root
)

OUTPUT_DIR = os.path.abspath(
    args.output_dir
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Parse pipelines
# ============================================================

PIPELINES = {}

for pipeline in args.pipeline:

    if "=" not in pipeline:

        raise ValueError(
            "\nInvalid pipeline format:\n"
            f"{pipeline}\n\n"
            "Expected:\n"
            "NAME=FOLDER\n"
            "Example:\n"
            "InterpolationNet=predicted_frames"
        )

    name, folder = pipeline.split(
        "=",
        1
    )

    name = name.strip()
    folder = folder.strip()

    if not name or not folder:

        raise ValueError(
            f"Invalid pipeline specification: {pipeline}"
        )

    PIPELINES[name] = folder


# ============================================================
# Header
# ============================================================

print()
print("=" * 75)
print("FLOWCEPTION+ GENERALIZED EVALUATION")
print("=" * 75)

print("Ground truth root :", GT_ROOT)
print("Results root      :", RESULT_ROOT)
print("Output directory  :", OUTPUT_DIR)
print("Device            :", DEVICE)
print("GT folder         :", args.gt_folder)
print("Match mode        :", args.match_mode)

print()
print("Pipelines:")

for name, folder in PIPELINES.items():

    print(
        f"  {name} -> {folder}"
    )

print("=" * 75)


# ============================================================
# Check paths
# ============================================================

if not os.path.isdir(GT_ROOT):

    raise FileNotFoundError(
        f"\nGround-truth root does not exist:\n{GT_ROOT}"
    )


if not os.path.isdir(RESULT_ROOT):

    raise FileNotFoundError(
        f"\nResults root does not exist:\n{RESULT_ROOT}"
    )


# ============================================================
# Load LPIPS
# ============================================================

print()
print("Loading LPIPS model...")

loss_fn = lpips.LPIPS(
    net="alex"
).to(DEVICE)

loss_fn.eval()

print("LPIPS loaded successfully.")


# ============================================================
# Helper: Load image
# ============================================================

def load_image(path):

    img = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if img is None:

        raise RuntimeError(
            f"Cannot read image:\n{path}"
        )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = (
        img.astype(
            np.float32
        ) / 255.0
    )

    return img


# ============================================================
# Helper: LPIPS
# ============================================================

def lpips_score(
    img1,
    img2
):

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
            t1 * 2.0 - 1.0,
            t2 * 2.0 - 1.0
        )

    return float(
        score.item()
    )


# ============================================================
# Helper: Find PNG frames
# ============================================================

def get_frames(folder):

    if not os.path.isdir(folder):

        return []

    frames = sorted([

        f

        for f in os.listdir(folder)

        if f.lower().endswith(".png")

    ])

    return frames


# ============================================================
# Helper: Match frames
# ============================================================

def match_frames(
    gt_frames,
    pred_frames,
    mode
):

    if len(gt_frames) == 0:
        return []

    if len(pred_frames) == 0:
        return []


    # --------------------------------------------------------
    # INDEX MODE
    #
    # Same behavior as the old evaluator:
    #
    # GT:
    #   0 1 2 3 4 ...
    #
    # Prediction:
    #   0 1 2 3 4 ...
    #
    # --------------------------------------------------------

    if mode == "index":

        n = min(
            len(gt_frames),
            len(pred_frames)
        )

        return [

            (
                gt_frames[i],
                pred_frames[i]
            )

            for i in range(n)

        ]


    # --------------------------------------------------------
    # EVENLY SPACED MODE
    #
    # Example:
    #
    # GT = 50 frames
    # Prediction = 16 frames
    #
    # Select 16 GT frames distributed across
    # the complete 50-frame sequence.
    # --------------------------------------------------------

    if mode == "evenly_spaced":

        n = min(
            len(gt_frames),
            len(pred_frames)
        )

        if n == 1:

            indices = [0]

        else:

            indices = np.linspace(
                0,
                len(gt_frames) - 1,
                n
            ).round().astype(int)

        return [

            (
                gt_frames[
                    int(gt_index)
                ],
                pred_frames[
                    pred_index
                ]
            )

            for pred_index, gt_index
            in enumerate(indices)

        ]


    raise ValueError(
        f"Unknown match mode: {mode}"
    )


# ============================================================
# Get videos
# ============================================================

videos = sorted([

    v

    for v in os.listdir(GT_ROOT)

    if os.path.isdir(
        os.path.join(
            GT_ROOT,
            v
        )
    )

])


if len(videos) == 0:

    raise RuntimeError(
        f"No video directories found in:\n{GT_ROOT}"
    )


print()
print(
    f"Found {len(videos)} ground-truth videos."
)


# ============================================================
# Evaluation
# ============================================================

all_pipeline_summaries = []


for pipeline_name, pred_folder in PIPELINES.items():

    print()
    print("=" * 75)

    print(
        f"EVALUATING : {pipeline_name}"
    )

    print(
        f"Prediction folder : {pred_folder}"
    )

    print("=" * 75)


    rows = []


    for video in tqdm(
        videos,
        desc=pipeline_name
    ):

        # ----------------------------------------------------
        # Ground truth directory
        # ----------------------------------------------------

        gt_dir = os.path.join(
            GT_ROOT,
            video,
            args.gt_folder
        )


        # ----------------------------------------------------
        # Prediction directory
        # ----------------------------------------------------

        pred_dir = os.path.join(
            RESULT_ROOT,
            video,
            pred_folder
        )


        # ----------------------------------------------------
        # Check directories
        # ----------------------------------------------------

        if not os.path.isdir(gt_dir):

            print(
                f"\nSkipping {video}: "
                "ground-truth directory missing."
            )

            continue


        if not os.path.isdir(pred_dir):

            print(
                f"\nSkipping {video}: "
                f"prediction directory missing: "
                f"{pred_dir}"
            )

            continue


        # ----------------------------------------------------
        # Get frames
        # ----------------------------------------------------

        gt_frames = get_frames(
            gt_dir
        )

        pred_frames = get_frames(
            pred_dir
        )


        if len(gt_frames) == 0:

            print(
                f"\nSkipping {video}: "
                "no GT frames."
            )

            continue


        if len(pred_frames) == 0:

            print(
                f"\nSkipping {video}: "
                "no predicted frames."
            )

            continue


        # ----------------------------------------------------
        # Match frames
        # ----------------------------------------------------

        matches = match_frames(
            gt_frames,
            pred_frames,
            args.match_mode
        )


        if len(matches) == 0:

            print(
                f"\nSkipping {video}: "
                "no matching frames."
            )

            continue


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        psnr_all = []
        ssim_all = []
        lpips_all = []


        # ----------------------------------------------------
        # Evaluate frames
        # ----------------------------------------------------

        for gt_name, pred_name in matches:

            gt_path = os.path.join(
                gt_dir,
                gt_name
            )

            pred_path = os.path.join(
                pred_dir,
                pred_name
            )


            gt = load_image(
                gt_path
            )

            pred = load_image(
                pred_path
            )


            # ------------------------------------------------
            # Image shape check
            # ------------------------------------------------

            if gt.shape != pred.shape:

                raise RuntimeError(
                    "\nImage shape mismatch!\n"
                    f"Video      : {video}\n"
                    f"GT frame   : {gt_name}\n"
                    f"Pred frame : {pred_name}\n"
                    f"GT shape   : {gt.shape}\n"
                    f"Pred shape : {pred.shape}\n"
                )


            # ------------------------------------------------
            # PSNR
            # ------------------------------------------------

            psnr = peak_signal_noise_ratio(
                gt,
                pred,
                data_range=1.0
            )


            # ------------------------------------------------
            # SSIM
            # ------------------------------------------------

            ssim = structural_similarity(
                gt,
                pred,
                channel_axis=2,
                data_range=1.0
            )


            # ------------------------------------------------
            # LPIPS
            # ------------------------------------------------

            lp = lpips_score(
                gt,
                pred
            )


            psnr_all.append(
                psnr
            )

            ssim_all.append(
                ssim
            )

            lpips_all.append(
                lp
            )


        # ----------------------------------------------------
        # Per-video result
        # ----------------------------------------------------

        rows.append({

            "video":
                video,

            "gt_frames":
                len(gt_frames),

            "pred_frames":
                len(pred_frames),

            "evaluated_frames":
                len(matches),

            "match_mode":
                args.match_mode,

            "PSNR":
                float(
                    np.mean(
                        psnr_all
                    )
                ),

            "SSIM":
                float(
                    np.mean(
                        ssim_all
                    )
                ),

            "LPIPS":
                float(
                    np.mean(
                        lpips_all
                    )
                ),

        })


    # ========================================================
    # Create DataFrame
    # ========================================================

    df = pd.DataFrame(
        rows
    )


    if df.empty:

        print(
            f"\nNo valid results for {pipeline_name}."
        )

        continue


    # ========================================================
    # Save per-video CSV
    # ========================================================

    safe_name = (
        pipeline_name
        .replace(" ", "_")
        .replace("/", "_")
    )


    csv_path = os.path.join(
        OUTPUT_DIR,
        f"evaluation_{safe_name}.csv"
    )


    df.to_csv(
        csv_path,
        index=False
    )


    # ========================================================
    # Average metrics
    # ========================================================

    average_psnr = df[
        "PSNR"
    ].mean()

    average_ssim = df[
        "SSIM"
    ].mean()

    average_lpips = df[
        "LPIPS"
    ].mean()


    # ========================================================
    # Print results
    # ========================================================

    print()
    print("=" * 75)

    print(
        f"PER-VIDEO RESULTS : {pipeline_name}"
    )

    print("=" * 75)

    print(
        df.to_string(
            index=False
        )
    )


    print()
    print(
        f"AVERAGE METRICS : {pipeline_name}"
    )

    print(
        f"PSNR  : {average_psnr:.6f}"
    )

    print(
        f"SSIM  : {average_ssim:.6f}"
    )

    print(
        f"LPIPS : {average_lpips:.6f}"
    )


    print()
    print(
        "Saved:"
    )

    print(
        csv_path
    )


    # ========================================================
    # Store overall summary
    # ========================================================

    all_pipeline_summaries.append({

        "pipeline":
            pipeline_name,

        "videos_evaluated":
            len(df),

        "total_evaluated_frames":
            int(
                df[
                    "evaluated_frames"
                ].sum()
            ),

        "average_PSNR":
            average_psnr,

        "average_SSIM":
            average_ssim,

        "average_LPIPS":
            average_lpips,

    })


# ============================================================
# Overall summary
# ============================================================

if len(all_pipeline_summaries) > 0:

    summary_df = pd.DataFrame(
        all_pipeline_summaries
    )


    summary_path = os.path.join(
        OUTPUT_DIR,
        "evaluation_summary.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False
    )


    print()
    print("=" * 75)
    print("OVERALL EVALUATION SUMMARY")
    print("=" * 75)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved summary:"
    )

    print(
        summary_path
    )


# ============================================================
# Finished
# ============================================================

print()
print("=" * 75)
print("EVALUATION COMPLETED")
print("=" * 75)

print(
    "Ground truth :",
    GT_ROOT
)

print(
    "Results      :",
    RESULT_ROOT
)

print(
    "Evaluation   :",
    OUTPUT_DIR
)

print("=" * 75)
