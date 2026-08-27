# FlowceptionPlus: Motion-Aware Video Frame Interpolation

A Flowception-inspired video frame interpolation pipeline that combines **RAFT optical flow, motion decomposition, motion-aware latent fusion, and a trained InterpolationNet** to generate intermediate video frames.

The project progressively evolves from a Flowception-inspired latent interpolation pipeline into a **motion-aware and trained latent-space interpolation system**.

---

## Overview

Video frame interpolation aims to generate intermediate frames between two consecutive input frames.

Our approach uses **optical-flow-based motion information** extracted from the input frames and performs interpolation primarily in the **latent space**.

### Overall Pipeline

```text
Frame A + Frame B
       │
       ▼
RAFT Optical Flow
       │
       ├── Forward Flow
       └── Backward Flow
       │
       ▼
Occlusion / Motion Analysis
       │
       ▼
Motion Decomposition
       │
       ├── Background Mask
       ├── Foreground Mask
       └── Residual Mask
       │
       ▼
Motion-Aware Latent Fusion
       │
       ▼
Fused Latent Representation
       │
       ▼
Trained InterpolationNet
       │
       ▼
16 Intermediate Latent Frames
       │
       ▼
Latent Decoding
       │
       ▼
SD-U-Net Refinement
       │
       ▼
Final Interpolated Frames
```

---

## Key Components

### 1. RAFT Optical Flow

**RAFT** is used to estimate motion between the two input frames.

The system obtains:

* Forward optical flow
* Backward optical flow
* Motion information for subsequent decomposition

Using bidirectional optical flow allows the pipeline to identify different motion regions and improve the latent initialization.

---

### 2. Motion Decomposition

Instead of treating the entire image as a single motion region, the estimated motion is decomposed into three regions:

```text
Background Motion
Foreground Motion
Residual Motion
```

These regions are represented using motion masks:

```text
background_mask.npy
foreground_mask.npy
residual_mask.npy
```

The three regions represent:

| Region         | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| **Background** | Generally smoother or global scene motion                                   |
| **Foreground** | Object-level motion                                                         |
| **Residual**   | Difficult regions such as occlusions, inconsistent flow, and complex motion |

This decomposition allows the interpolation system to treat different motion patterns separately.

---

## 3. Motion-Aware Latent Fusion

The endpoint frames are first represented in latent space.

The latent representations are then combined with the motion information obtained from the optical-flow and decomposition stages.

The resulting files include:

```text
latent_a.pt
latent_b.pt
fused_latent.pt
```

The `fused_latent` provides a motion-aware representation that is used as input to the interpolation network.

---

## 4. InterpolationNet

A dedicated **InterpolationNet** is used to predict intermediate latent representations.

### Inputs

The network receives:

```text
latent_a
latent_b
fused_latent
background_mask
foreground_mask
residual_mask
```

### Output

The network predicts a tensor with shape:

```text
[B, 4, 16, 32, 32]
```

where:

* `B` = batch size
* `4` = latent channels
* `16` = number of intermediate frames
* `32 × 32` = latent spatial resolution

Therefore, the trained model generates **16 intermediate latent frames**.

The best trained checkpoint is expected at:

```text
checkpoints/interpolation_net_best.pt
```

Large model checkpoints can be excluded from GitHub, while the expected checkpoint location is documented separately.

---

# Training Pipeline

Training data is generated from motion-aware latent representations.

Important training scripts include:

```text
prepare_interpolation_dataset.py
prepare_interpolation_training_data.py
interpolation_dataset.py
train_interpolation_net.py
interpolation_net.py
```

### Training Flow

```text
Video Samples
      │
      ▼
Motion / Latent Preparation
      │
      ▼
Training Samples
      │
      ▼
InterpolationNet Training
      │
      ▼
Best Model Checkpoint
```

---

# Trained Inference

During inference, the trained model predicts intermediate frames **without requiring ground-truth intermediate frames**.

The main inference script is:

```text
interpolation_inference.py
```

### Inference Flow

```text
Input Latents + Motion Masks
             │
             ▼
     Trained InterpolationNet
             │
             ▼
Predicted Intermediate Latents
```

The predicted latent tensor is stored as:

```text
predicted_latents.pt
```

### Running the Trained Pipeline

For processing multiple video directories:

```bash
python run_trained_pipeline.py \
    --results_root /path/to/results_v4 \
    --checkpoint /path/to/checkpoints/interpolation_net_best.pt \
    --output_root /path/to/trained_results \
    --device cuda:1
```

---

# Latent Decoding

The predicted latent representations are converted back into RGB frames using:

```text
decode_interpolation.py
```

The decoded frames are stored in:

```text
predicted_frames/
```

---

# SD-U-Net Refinement

An additional **Stable Diffusion U-Net-based refinement stage** is implemented to improve the predicted latent representations.

### Refinement Pipeline

```text
predicted_latents.pt
        │
        ▼
SD-U-Net Denoising
        │
        ▼
denoised_predicted_latents.pt
        │
        ▼
Latent Decoding
        │
        ▼
final_frames/
```

Main scripts:

```text
run_sd_denoise_interpolation.py
sd_unet_denoise.py
decode_sd_interpolation.py
```

The refinement stage is an additional processing step rather than the core interpolation network.

---

# Experimental Progression

The project was developed progressively through multiple experimental stages.

## Initial Pipeline

The initial implementation established the basic **Flowception-inspired latent interpolation workflow**.

## Motion-Aware Improvements

The pipeline was then enhanced with:

* Bidirectional RAFT optical flow
* Motion decomposition
* Background / foreground / residual masks
* Motion-aware latent fusion
* Flow-based latent initialization

Intermediate experimental results were stored in:

```text
results/
results_v2/
results_v3/
```

`results_v3` served as the stronger pre-training baseline.

## Trained Interpolation

The next stage introduced the trained `InterpolationNet`.

The final large-scale experimental outputs were generated under:

```text
results_v4/
```

The trained model was evaluated on **30 test videos**.

---

# Evaluation

The generated frames are compared against ground-truth frames using three image-quality metrics.

### PSNR

**Peak Signal-to-Noise Ratio**

Higher values indicate better reconstruction quality.

### SSIM

**Structural Similarity Index**

Higher values indicate greater structural similarity to the ground truth.

### LPIPS

**Learned Perceptual Image Patch Similarity**

Lower values generally indicate better perceptual similarity.

Evaluation is implemented in:

```text
evaluate_pipeline.py
```

---

# Dataset and Evaluation Setup

The evaluation was performed on:

```text
30 test videos
```

Each ground-truth video contains approximately:

```text
51 frames
```

The trained interpolation model generates:

```text
16 intermediate frames
```

Therefore, evaluation matches the **16 generated frames against 16 evenly spaced ground-truth frames** from each sequence.

> **Important:** This is an experimental limitation. The evaluation does not compare against every ground-truth frame in the approximately 51-frame sequences.

---

# Results

The trained `InterpolationNet` achieved approximately:

| Metric    | Average Performance |
| --------- | ------------------: |
| **PSNR**  |              ~20 dB |
| **SSIM**  |               ~0.59 |
| **LPIPS** |               ~0.32 |

Performance varies across individual videos.

The best-performing samples achieved:

```text
PSNR > 24 dB
```

with the strongest example reaching approximately:

```text
25.18 dB PSNR
```

The experiments also included comparisons with earlier **AnimateDiff** and **SD-U-Net-based approaches**.

---

# Repository Structure

```text
custom_pipeline/
│
├── motion_aware_fusion.py
├── prediction_video_latents.py
│
├── interpolation_net.py
├── interpolation_inference.py
├── interpolation_dataset.py
├── train_interpolation_net.py
│
├── run_trained_pipeline.py
│
├── run_sd_denoise_interpolation.py
├── sd_unet_denoise.py
│
├── decode_interpolation.py
├── decode_sd_interpolation.py
│
├── evaluate_pipeline.py
│
├── prepare_interpolation_dataset.py
├── prepare_interpolation_training_data.py
├── prepare_trained_interpolation_inputs.py
│
├── checkpoints/
│   └── README.md
│
├── configs/
│
├── archive/
│
└── README.md
```

---

# Reproducibility

The complete pipeline is separated into independent stages:

```text
Motion Extraction
       │
       ▼
Motion Decomposition
       │
       ▼
Latent Preparation
       │
       ▼
Training
       │
       ▼
Trained Inference
       │
       ▼
Refinement
       │
       ▼
Decoding
       │
       ▼
Evaluation
```

This modular structure makes it possible to independently test:

* Motion processing
* Motion decomposition
* Latent fusion
* Interpolation network
* Refinement
* Decoding
* Evaluation

Large datasets, generated outputs, latent tensors, videos, and model weights are excluded from version control using `.gitignore`.

---

# Limitations

The current experimental setup has several limitations:

1. The trained model generates 16 intermediate frames, while the ground-truth sequences contain approximately 51 frames.
2. Evaluation therefore uses 16 evenly spaced ground-truth frames instead of evaluating every ground-truth frame.
3. The model and large pretrained components were not trained completely from scratch.
4. Experiments were performed on a relatively small evaluation subset of 30 videos.
5. Motion decomposition depends on estimated optical flow and can be affected by inaccurate flow, occlusions, and complex motion.
6. The SD-U-Net refinement stage introduces additional computational cost.
7. This project is **Flowception-inspired** rather than a complete reproduction of the original Flowception training system.

---

# Future Improvements

Potential improvements include:

* Training on a larger number of videos
* Generating more intermediate frames
* Improving motion decomposition
* Better occlusion handling
* More sophisticated flow-guided latent initialization
* Joint training of interpolation and refinement components
* More extensive ablation studies
* Evaluation using additional video-specific metrics such as temporal consistency

---

# Acknowledgements

This project is inspired by the ideas presented in:

**Flowception: Temporally Expansive Flow Matching for Video Generation**

The implementation also makes use of:

* RAFT optical flow
* Diffusion-based latent representations
* SD-U-Net-based latent refinement

This repository contains our **experimental FlowceptionPlus implementation and extensions**, rather than an official reproduction of the original Flowception implementation.
