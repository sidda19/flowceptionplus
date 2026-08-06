import os
import sys

import scipy
print("=" * 60)

import cv2
import torch
import numpy as np
from argparse import Namespace

############################################################
# Usage
#
# python raft_inference.py frame1.png frame2.png forward.npy backward.npy flow.png
############################################################

if len(sys.argv) != 6:
    print("Usage:")
    print("python raft_inference.py <frame1> <frame2> <forward_flow.npy> <backward_flow.npy> <flow_visualization.png>")
    exit()

FRAME1 = sys.argv[1]
FRAME2 = sys.argv[2]
FORWARD_OUT = sys.argv[3]
BACKWARD_OUT = sys.argv[4]
FLOW_IMAGE_OUT = sys.argv[5]

############################################################
# Add RAFT/core
############################################################

RAFT_ROOT = os.path.abspath("../RAFT")
sys.path.append(os.path.join(RAFT_ROOT, "core"))

from raft import RAFT
from utils.utils import InputPadder
from utils.flow_viz import flow_to_image

############################################################
# Configuration
############################################################

MODEL = os.path.join(RAFT_ROOT, "models", "raft-things.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

args = Namespace(
    small=False,
    mixed_precision=False,
    alternate_corr=False,
    dropout=0,
)

############################################################
# Load Model
############################################################

model = RAFT(args)

checkpoint = torch.load(MODEL, map_location=DEVICE)

checkpoint = {
    k.replace("module.", ""): v
    for k, v in checkpoint.items()
}

model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("RAFT Loaded.")

############################################################
# Read Images
############################################################

image1 = cv2.imread(FRAME1)
image2 = cv2.imread(FRAME2)

if image1 is None:
    raise FileNotFoundError(FRAME1)

if image2 is None:
    raise FileNotFoundError(FRAME2)

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

image1 = torch.from_numpy(image1).permute(2,0,1).float()[None].to(DEVICE)
image2 = torch.from_numpy(image2).permute(2,0,1).float()[None].to(DEVICE)

############################################################
# Pad
############################################################

padder = InputPadder(image1.shape)
image1, image2 = padder.pad(image1, image2)

############################################################
# Forward Flow
############################################################

with torch.no_grad():
    _, forward_flow = model(
        image1,
        image2,
        iters=20,
        test_mode=True
    )

############################################################
# Backward Flow
############################################################

with torch.no_grad():
    _, backward_flow = model(
        image2,
        image1,
        iters=20,
        test_mode=True
    )

############################################################
# Convert
############################################################

forward_flow = (
    forward_flow[0]
    .permute(1,2,0)
    .cpu()
    .numpy()
)

backward_flow = (
    backward_flow[0]
    .permute(1,2,0)
    .cpu()
    .numpy()
)

############################################################
# Save
############################################################

np.save(FORWARD_OUT, forward_flow)
np.save(BACKWARD_OUT, backward_flow)

flow_img = flow_to_image(forward_flow)

cv2.imwrite(
    FLOW_IMAGE_OUT,
    cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)
)

print("--------------------------------")
print("Forward :", forward_flow.shape)
print("Backward:", backward_flow.shape)
print("Saved:", FORWARD_OUT)
print("Saved:", BACKWARD_OUT)
print("Saved:", FLOW_IMAGE_OUT)
print("--------------------------------")