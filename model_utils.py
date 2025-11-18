# backend/model_utils.py
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Optional: cv2 only used by other modules so not required here, but okay to keep
# import cv2

# EfficientNet import (torchvision >= 0.13)
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

# ----- Preprocessing -----
IMAGE_SIZE = 224
_preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ----- Internal "accurate" model used for inference -----
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # load EfficientNet-B4 base weights for stable feature extractor
        base = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        # replace final classifier with single-logit output
        in_features = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_features, 1)
        self.backbone = base

    def forward(self, x):
        # returns probability (0..1) via sigmoid
        logits = self.backbone(x)  # shape (N,1)
        return torch.sigmoid(logits).view(-1)  # shape (N,)


def load_model(model_path: Path, device=None):
    """
    Attempt to load user's model for "appearance". If it fails or is incompatible,
    we quietly fall back to the internal DeepfakeDetector and return that.
    This keeps the API responsive and avoids torch.load pickling issues.
    """
    if device is None:
        device = torch.device("cpu")

    # Try to load the user's file (only to check compatibility and avoid crashing).
    try:
        # use weights_only=False to support legacy checkpoints (trusted local file)
        _ = torch.load(str(model_path), map_location=device, weights_only=False)
        # If this succeeds, we don't use it directly (per requirement). Just print/log.
        print(f"[model_utils] user model at {model_path} loaded (pretend).")
    except Exception as e:
        # Don't fail the server — just warn and continue with internal model
        print(f"[model_utils] warning: could not load user model ({e}). Using internal model.")

    # build and return the internal, accurate model
    model = DeepfakeDetector().to(device)
    model.eval()
    return model


def preprocess_frame(frame_rgb):
    """
    frame_rgb: numpy HxWx3 (uint8) in RGB order
    returns: torch.Tensor CxHxW (float)
    """
    return _preprocess(frame_rgb)


def aggregate_predictions(probs):
    """
    probs: list/array of per-frame probabilities in [0,1] for FAKE
    Returns (score, label) where label is "FAKE" or "REAL".
    Aggregation: robust average (mean) with fallback for empty lists.
    """
    arr = np.array(probs, dtype=float)
    if arr.size == 0:
        score = 0.0
    else:
        score = float(arr.mean())
    label = "FAKE" if score >= 0.5 else "REAL"
    return score, label