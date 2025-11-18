# backend/app.py
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import os
import torch
import cv2
import numpy as np

from model_utils import load_model, preprocess_frame, aggregate_predictions

# Config
UPLOAD_DIR = Path("./tmp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_UPLOAD_BYTES = 600 * 1024 * 1024  # 600 MB
SAMPLE_FRAMES = 32
BATCH_SIZE = 16  # if you have GPU, increasing this helps

app = FastAPI()

# CORS - allow local testing from file:// or localhost:3000 etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("./model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_PATH, device=DEVICE)


def allowed_file_ext(filename: str) -> bool:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return ext in {"mp4", "mov", "avi", "mkv", "webm"}


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    # Basic validation
    if not allowed_file_ext(video.filename):
        raise HTTPException(status_code=400, detail="Unsupported file extension")

    # Save uploaded stream to temp file
    uid = uuid.uuid4().hex
    temp_path = UPLOAD_DIR / f"{uid}_{video.filename}"
    size = 0
    try:
        with open(temp_path, "wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File too large")
                f.write(chunk)

        # Extract frames (sample evenly up to SAMPLE_FRAMES)
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open uploaded video")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            cap.release()
            raise HTTPException(status_code=400, detail="Empty or unsupported video")

        # compute evenly spaced indices
        indices = np.linspace(0, max(frame_count - 1, 0), num=min(SAMPLE_FRAMES, frame_count), dtype=int)
        indices_set = set(indices.tolist())

        frames_rgb = []
        idx = 0
        success = True
        while success and len(frames_rgb) < len(indices_set):
            success, frame = cap.read()
            if not success:
                break
            if idx in indices_set:
                # convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_rgb.append(frame_rgb)
            idx += 1
        cap.release()

        if len(frames_rgb) == 0:
            raise HTTPException(status_code=400, detail="No frames extracted")

        # Preprocess & run inference in batches
        tensors = [preprocess_frame(f) for f in frames_rgb]  # CPU tensors
        probs = []

        # Move batches to device
        for i in range(0, len(tensors), BATCH_SIZE):
            batch = torch.stack(tensors[i:i+BATCH_SIZE]).to(DEVICE)  # (N,C,H,W)
            with torch.no_grad():
                out = model(batch)  # expects output shape (N,) with probabilities
                # ensure CPU numpy floats
                out_np = out.detach().cpu().numpy().astype(float)
                probs.extend(out_np.tolist())

        score, label = aggregate_predictions(probs)

        return JSONResponse(content={"prediction": label, "score": float(score)})

    finally:
        # cleanup temp file
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)