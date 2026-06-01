import os
import io
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import pathlib

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = FastAPI(title="CanSign-AI API")

# ── Load Model ────────────────────────────────────────────────────────────────
# Search common paths for the model file
MODEL_CANDIDATES = [
    pathlib.Path("traffic_sign_model.keras"),
    pathlib.Path("traffic_sign_model.h5"),
    pathlib.Path(__file__).parent.parent / "traffic_sign_model.keras",
    pathlib.Path(__file__).parent.parent / "traffic_sign_model.h5"
]

model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
if model_path is None:
    raise RuntimeError("❌ traffic_sign_model.keras not found. Download it to the project root.")

print(f"✅ Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# ── Label Mapping (Inlined for HF Spaces portability) ─────────────────────────
CANADIAN_SIGNS = {
    0: "Speed Limit 20", 1: "Speed Limit 30", 2: "Speed Limit 50", 3: "Speed Limit 60",
    4: "Speed Limit 70", 5: "Speed Limit 80", 6: "End Speed Limit 80", 7: "Speed Limit 100",
    8: "Speed Limit 120", 9: "No Passing", 10: "No Passing (Trucks)", 11: "Right of Way at Intersection",
    12: "Priority Road", 13: "Yield", 14: "Stop", 15: "No Vehicles", 16: "No Trucks", 17: "No Entry",
    18: "General Caution", 19: "Dangerous Curve Left", 20: "Dangerous Curve Right", 21: "Double Curve",
    22: "Bumpy Road", 23: "Slippery Road", 24: "Road Narrows Right", 25: "Road Work", 26: "Traffic Signals",
    27: "Pedestrian Crossing", 28: "Children Crossing", 29: "Bicycles Crossing", 30: "Ice / Snow",
    31: "Wild Animals Crossing", 32: "End of All Restrictions", 33: "Turn Right Ahead", 34: "Turn Left Ahead",
    35: "Ahead Only", 36: "Go Straight or Right", 37: "Go Straight or Left", 38: "Keep Right", 39: "Keep Left",
    40: "Roundabout Mandatory", 41: "End of No Passing", 42: "End of No Passing (Trucks)"
}

# ── Preprocessing (matches training pipeline exactly: resize + normalize only) ─
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image format")
    resized = cv2.resize(img, (32, 32))
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)


def is_out_of_distribution(preds: np.ndarray) -> bool:
    """Reject if top confidence is low OR predictions are too spread out.
    Uses numpy only — no scipy needed.

    Thresholds explained:
    - top_conf >= 0.92: model must be strongly committed to ONE class
    - entropy < 0.8: very little probability mass on other classes
    - top2_gap >= 0.70: top prediction must dominate over #2 by a wide margin
    All three must pass. A real sign typically scores 0.95+, entropy ~0.1, gap ~0.9.
    A random photo spreads probability across many classes even if one "wins".
    """
    top_conf = float(np.max(preds))
    if top_conf < 0.92:
        return True

    eps = 1e-9
    entropy = float(-np.sum(preds * np.log(preds + eps)))
    if entropy > 0.8:
        return True

    # Gap between #1 and #2 must be large
    sorted_preds = np.sort(preds)[::-1]
    top2_gap = float(sorted_preds[0] - sorted_preds[1])
    if top2_gap < 0.70:
        return True

    return False

# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        img_tensor = preprocess_image(image_bytes)
        preds = model.predict(img_tensor, verbose=0)[0]
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
            "rank": rank,
            "class_id": int(idx),
            "sign_name": CANADIAN_SIGNS.get(int(idx), f"Class {idx}"),
            "confidence": round(float(preds[idx]) * 100, 1)
        })

    return JSONResponse({
        "top_prediction": top3[0]["sign_name"],
        "confidence": top3[0]["confidence"],
        "top3": top3
    })

# ── Serve Frontend ────────────────────────────────────────────────────────────
# Resolve static path robustly for both local & HF Spaces
STATIC_DIR = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
