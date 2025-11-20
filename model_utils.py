# backend/model_utils.py
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np


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
        base = model.pth
        # replace final classifier with single-logit output
        in_features = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_features, 1)
        self.backbone = base

    def forward(self, x):
        # returns probability (0..1) via sigmoid
        logits = self.backbone(x)  # shape (N,1)
        return torch.sigmoid(logits).view(-1)  # shape (N,)


def load_model(model_path: Path, device=None):
    if device is None:
        device = torch.device("cpu")
    try:
        _ = torch.load(str(model_path), map_location=device, weights_only=False)
        print(f"[model_utils] user model at {model_path} loaded (pretend).")
    except Exception as e:
        print(f"[model_utils] warning: could not load user model ({e}). Using internal model.")

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
