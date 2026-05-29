# app/app.py
# FastAPI backend for Traffic Sign Classifier
#
# Usage:
#   pip install -r requirements_app.txt
#   uvicorn app.app:app --reload
#
# Then open http://localhost:8000

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

# ── Labels ────────────────────────────────────────────────────────────────────
CANADIAN_SIGNS = {i: f"Class {i}" for i in range(43)}
CANADIAN_SIGNS.update({
    0: "Speed Limit 20", 1: "Speed Limit 30", 2: "Speed Limit 50",
    3: "Speed Limit 60", 4: "Speed Limit 70", 5: "Speed Limit 80",
    6: "End Speed Limit 80", 7: "Speed Limit 100", 8: "Speed Limit 120",
    9: "No Passing", 10: "No Passing (Trucks)",
    11: "Right of Way at Intersection", 12: "Priority Road", 13: "Yield",
    14: "Stop", 15: "No Vehicles", 16: "No Trucks", 17: "No Entry",
    18: "General Caution", 19: "Dangerous Curve Left",
    20: "Dangerous Curve Right", 21: "Double Curve", 22: "Bumpy Road",
    23: "Slippery Road", 24: "Road Narrows Right", 25: "Road Work",
    26: "Traffic Signals", 27: "Pedestrian Crossing",
    28: "Children Crossing", 29: "Bicycles Crossing", 30: "Ice / Snow",
    31: "Wild Animals Crossing", 32: "End of All Restrictions",
    33: "Turn Right Ahead", 34: "Turn Left Ahead", 35: "Ahead Only",
    36: "Go Straight or Right", 37: "Go Straight or Left",
    38: "Keep Right", 39: "Keep Left", 40: "Roundabout Mandatory",
    41: "End of No Passing", 42: "End of No Passing (Trucks)",
})

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Traffic Sign Classifier")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

MODEL = None

@app.on_event("startup")
def load_model():
    global MODEL
    for path in ["traffic_sign_model.keras", "traffic_sign_model.h5",
                 "app/traffic_sign_model.keras", "app/traffic_sign_model.h5"]:
        if os.path.exists(path):
            print(f"✅ Loading model from {path}")
            MODEL = tf.keras.models.load_model(path)
            return
    print("⚠️  No model file found — place traffic_sign_model.keras in project root.")

@app.get("/")
def index():
    return FileResponse("app/static/index.html")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Place traffic_sign_model.keras in project root.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((32, 32))
        img_array = np.array(img, dtype="float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    preds = MODEL.predict(img_array, verbose=0)[0]
    top3_indices = np.argsort(preds)[::-1][:3]

    results = [
        {
            "rank": int(rank + 1),
            "class_id": int(idx),
            "sign_name": CANADIAN_SIGNS.get(int(idx), f"Class {idx}"),
            "confidence": float(round(preds[idx] * 100, 2)),
        }
        for rank, idx in enumerate(top3_indices)
    ]

    return {
        "top_prediction": results[0]["sign_name"],
        "confidence": results[0]["confidence"],
        "top3": results,
    }

