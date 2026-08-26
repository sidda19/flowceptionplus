# FlowceptionPlus

### A Latent-Space Video Frame Interpolation Pipeline using RAFT, Stable Diffusion, and AnimateDiff

## Project Overview

This project implements a Flow-Guided Video Frame Interpolation pipeline by combining:

- RAFT for optical flow estimation.
- Occlusion-aware motion decomposition.
- Latent warping and motion-aware fusion.
- Stable Diffusion VAE for latent encoding and decoding.
- AnimateDiff for temporal latent denoising.

The proposed pipeline takes two keyframes as input, estimates optical flow using RAFT, performs occlusion-aware motion decomposition, constructs motion-guided latent representations, refines them using AnimateDiff, and reconstructs high-quality interpolated video frames through Stable Diffusion's VAE.

## Pipeline

```text
Input Frames
      │
      ▼
Frame Extraction
      │
      ▼
VAE Encoding
      │
      ▼
RAFT Optical Flow
      │
      ▼
Occlusion Estimation
      │
      ▼
Motion Decomposition
      │
      ▼
Latent Warping
      │
      ▼
Motion-aware Fusion
      │
      ▼
Video Latent Preparation
      │
      ▼
AnimateDiff Denoising
      │
      ▼
VAE Decoding
      │
      ▼
Interpolated Video
```
## Repository Structure

```text
FlowceptionPlus/
│
├── custom_pipeline/        # Main interpolation pipeline
├── flowception/            # Original Flowception framework
├── RAFT/                   # RAFT optical flow implementation
├── models/                 # Stable Diffusion & AnimateDiff models
├── scripts/     # Archived helper and experimental scripts           
├── README.md
├── .gitignore
└── freeze.txt
```
## Core Pipeline Components

| Script | Purpose |
|--------|---------|
| `extract_pipeline_frames.py` | Selects input frames from a video. |
| `encode_frame.py` | Encodes input images into Stable Diffusion latent space. |
| `raft_inference.py` | Computes forward and backward optical flow using RAFT. |
| `occlusion_estimation.py` | Estimates occluded regions from bidirectional optical flow. |
| `motion_decomposition.py` | Generates background, foreground, and residual motion masks. |
| `latent_warp.py` | Warps latent features using optical flow. |
| `motion_aware_fusion.py` | Fuses original and warped latents using motion masks. |
| `prepare_video_latents.py` | Builds the intermediate latent video sequence. |
| `animatediff_denoise.py` | Refines latent video using AnimateDiff. |
| `decode_video.py` | Decodes the refined latent sequence into video frames and an output video. |
| `run_flowception_pipeline.py` | Executes the complete interpolation pipeline. |
## Installation

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd FlowceptionPlus
```

### Create and activate the environment

```bash
conda create -n flowception_final python=3.10
conda activate flowception_final
```

### Download required models

Before running the pipeline, download or place the required pretrained models in the appropriate directories:

- Stable Diffusion v1.5
- AnimateDiff Motion Adapter
- RAFT pretrained weights (`raft-things.pth`)

Ensure the directory structure matches the paths used in the project.
## Usage

Run the complete interpolation pipeline using:

```bash
python custom_pipeline/run_flowception_pipeline.py \
    <frame_a.png> \
    <frame_b.png> \
    <output_directory>
```

Example:

```bash
python custom_pipeline/run_flowception_pipeline.py \
    frame_a.png \
    frame_b.png \
    results/
```

The pipeline automatically performs:

1. Frame encoding
2. Optical flow estimation (RAFT)
3. Occlusion estimation
4. Motion decomposition
5. Latent warping
6. Motion-aware fusion
7. Video latent preparation
8. AnimateDiff denoising
9. Video decoding


## Results

The implemented pipeline successfully performs:

- Optical flow estimation using RAFT.
- Occlusion-aware motion estimation.
- Motion decomposition into background, foreground, and residual regions.
- Latent-space warping and motion-aware fusion.
- Video latent sequence generation.
- Temporal latent refinement using AnimateDiff.
- Reconstruction of smooth interpolated video frames.

### Generated Outputs

The pipeline generates the following outputs:

- Forward and backward optical flow
- Occlusion masks
- Background, foreground, and residual masks
- Warped latent representations
- Motion-aware fused latent
- Video latent sequence
- Denoised latent sequence
- Decoded video frames
- Final interpolated video
## Technologies Used

- Python
- PyTorch
- Hugging Face Diffusers
- Stable Diffusion v1.5
- AnimateDiff
- RAFT
- OpenCV
- NumPy
- Matplotlib
## Acknowledgements

This project builds upon several outstanding open-source projects:

- Flowception
- RAFT
- Stable Diffusion
- AnimateDiff

Their publicly available implementations and pretrained models made this work possible.
## License

This repository is intended for research and educational purposes.

Please refer to the original licenses of Flowception, RAFT, Stable Diffusion, and AnimateDiff for the respective components incorporated into this project.