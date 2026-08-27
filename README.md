# FlowceptionPlus: Motion-Aware Video Frame Interpolation

A Flowception-inspired video frame interpolation pipeline that combines optical-flow-based motion analysis, latent-space fusion, a trained interpolation network, and optional SD-U-Net refinement.

The project was developed progressively from an initial Flowception-inspired interpolation pipeline into a motion-aware and trainable interpolation system.

---

## 1. Project Overview

Video frame interpolation aims to generate intermediate frames between two consecutive frames.

Given:

```text
Frame A                         Frame B
   │                               │
   └─────────── Motion ────────────┘
                    ↓
            Intermediate Frames
