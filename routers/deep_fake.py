from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import torch
import cv2
import random
import tempfile
from transformers import AutoImageProcessor, SiglipForImageClassification

router = APIRouter()

# Load model and processor
processor = AutoImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
model = SiglipForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
model.eval()

# Sample frames from uploaded video
def sample_frames(video_path, num_samples=10):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        return []

    k = min(total, num_samples)
    indices = sorted(random.sample(range(total), k)) if total >= k else list(range(total))
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

@router.post("/predict/deepfake")
async def predict_deepfake(video: UploadFile = File(...), num_frames: int = 100, threshold: float = 0.5):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await video.read())
            video_path = tmp.name

        frames = sample_frames(video_path, num_samples=num_frames)
        if not frames:
            return JSONResponse(status_code=400, content={"error": "No frames extracted from video."})

        fake_probs = []
        for frame in frames:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            fake_probs.append(logits.softmax(-1)[0, 0].item())

        avg_fake_prob = float(np.mean(fake_probs))
        verdict = "❌ Fake" if avg_fake_prob > threshold else "✅ Real"

        return {
            "avg_fake_probability": round(avg_fake_prob, 4),
            "threshold": threshold,
            "verdict": verdict
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
