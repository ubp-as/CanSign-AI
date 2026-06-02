import os
import sys
import pathlib

# ── Fix import path ───────────────────────────────────────────────────────────
# __file__ = /app/app/app.py  →  .parent.parent = /app  (project root)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import io
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Try importing from canadian_labels.py if it exists (local dev),
# otherwise fall back to the inlined dict (HF Spaces / Docker).
try:
    from canadian_labels import CANADIAN_SIGNS
except ModuleNotFoundError:
    CANADIAN_SIGNS = {
        0: "Speed Limit 20",         1: "Speed Limit 30",          2: "Speed Limit 50",
        3: "Speed Limit 60",         4: "Speed Limit 70",          5: "Speed Limit 80",
        6: "End Speed Limit 80",     7: "Speed Limit 100",         8: "Speed Limit 120",
        9: "No Passing",            10: "No Passing (Trucks)",    11: "Right of Way at Intersection",
       12: "Priority Road",         13: "Yield",                  14: "Stop",
       15: "No Vehicles",           16: "No Trucks",              17: "No Entry",
       18: "General Caution",       19: "Dangerous Curve Left",   20: "Dangerous Curve Right",
       21: "Double Curve",          22: "Bumpy Road",             23: "Slippery Road",
       24: "Road Narrows Right",    25: "Road Work",              26: "Traffic Signals",
       27: "Pedestrian Crossing",   28: "Children Crossing",      29: "Bicycles Crossing",
       30: "Ice / Snow",            31: "Wild Animals Crossing",  32: "End of All Restrictions",
       33: "Turn Right Ahead",      34: "Turn Left Ahead",        35: "Ahead Only",
       36: "Go Straight or Right",  37: "Go Straight or Left",    38: "Keep Right",
       39: "Keep Left",             40: "Roundabout Mandatory",   41: "End of No Passing",
       42: "End of No Passing (Trucks)",
    }

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']  = '-1'

app = FastAPI(title="CanSign-AI API")

# ── Load Model ────────────────────────────────────────────────────────────────
MODEL_CANDIDATES = [
    PROJECT_ROOT / "traffic_sign_model.keras",
    PROJECT_ROOT / "traffic_sign_model.h5",
    pathlib.Path("traffic_sign_model.keras"),
    pathlib.Path("traffic_sign_model.h5"),
]

model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
if model_path is None:
    raise RuntimeError("❌ traffic_sign_model.keras not found. Place it in the project root.")

print(f"✅ Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# ── Preprocessing (HSV crop + resize + normalize — matches predict_image.py) ──
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image format")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, np.array([0,   70,  50]), np.array([10,  255, 255])),  # Red 1
        cv2.inRange(hsv, np.array([170, 70,  50]), np.array([180, 255, 255])),  # Red 2
        cv2.inRange(hsv, np.array([20,  80,  80]), np.array([35,  255, 255])),  # Yellow
        cv2.inRange(hsv, np.array([100, 70,  70]), np.array([120, 255, 255])),  # Blue
    ]
    mask = cv2.bitwise_or(
        cv2.bitwise_or(masks[0], masks[1]),
        cv2.bitwise_or(masks[2], masks[3])
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    cropped = img
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c        = max(contours, key=cv2.contourArea)
        area     = cv2.contourArea(c)
        img_area = img.shape[0] * img.shape[1]
        if 0.05 < (area / img_area) < 0.80:
            x, y, w, h = cv2.boundingRect(c)
            pad = int(max(w, h) * 0.1)
            x1, y1 = max(0, x - pad),                max(0, y - pad)
            x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            cropped = img[y1:y2, x1:x2]

    resized = cv2.resize(cropped, (32, 32))
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)


def is_out_of_distribution(predictions: np.ndarray) -> bool:
    """
    Reject images that are clearly not traffic signs.
      - top1 >= 0.50  : model must prefer at least one class
      - gap  >= 0.30  : top prediction must lead the runner-up
      - entropy < 1.5 : predictions shouldn't be nearly uniform
    """
    sorted_preds = np.sort(predictions)[::-1]
    top1 = float(sorted_preds[0])
    top2 = float(sorted_preds[1]) if len(sorted_preds) > 1 else 0.0
    entropy = float(-np.sum(predictions * np.log(predictions + 1e-10)))

    if top1 < 0.50:       return True
    if (top1 - top2) < 0.30: return True
    if entropy > 1.5:     return True
    return False


# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        img_tensor  = preprocess_image(image_bytes)
        preds       = model.predict(img_tensor, verbose=0)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    top3_idx = np.argsort(preds)[::-1][:3]
    top_conf = float(preds[top3_idx[0]])

    if is_out_of_distribution(preds):
        return JSONResponse({
            "top_prediction": "Not a recognized traffic sign",
            "confidence": round(top_conf * 100, 1),
            "top3": [{"rank": 1, "class_id": -1, "sign_name": "Not a recognized traffic sign", "confidence": round(top_conf * 100, 1)}]
        })

    top3 = []
    for rank, idx in enumerate(top3_idx, start=1):
        top3.append({
            "rank":       rank,
            "class_id":   int(idx),
            "sign_name":  CANADIAN_SIGNS.get(int(idx), f"Class {idx}"),
            "confidence": round(float(preds[idx]) * 100, 1)
        })

    return JSONResponse({
        "top_prediction": top3[0]["sign_name"],
        "confidence":     top3[0]["confidence"],
        "top3":           top3
    })


# ── Serve Frontend ────────────────────────────────────────────────────────────
STATIC_DIR = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
