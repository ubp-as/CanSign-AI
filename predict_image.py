# predict_image.py
# Run this LOCALLY after downloading traffic_sign_model.keras from Colab.
# Usage: python predict_image.py path/to/image.jpg

import os
import sys
import numpy as np
import cv2
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from canadian_labels import CANADIAN_SIGNS

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = 'traffic_sign_model.keras'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'traffic_sign_model.h5'
    if not os.path.exists(MODEL_PATH):
        print("❌ No model found. Download traffic_sign_model.keras from Colab/Hugging Face first.")
        sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)

def prepare_image(img_path, target_size=(32, 32)):
    """Auto-detect, crop, and preprocess exactly like training pipeline."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {img_path}")

    # 1. HSV color masking for traffic signs (Red, Yellow, Blue, White-ish)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])),   # Red 1
        cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255])), # Red 2
        cv2.inRange(hsv, np.array([20, 80, 80]), np.array([35, 255, 255])),   # Yellow
        cv2.inRange(hsv, np.array([100, 70, 70]), np.array([120, 255, 255]))  # Blue
    ]
    mask = cv2.bitwise_or(cv2.bitwise_or(masks[0], masks[1]), cv2.bitwise_or(masks[2], masks[3]))

    # 2. Clean mask (remove noise, fill gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 3. Find largest plausible sign contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cropped = img_bgr  # fallback
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        img_area = img_bgr.shape[0] * img_bgr.shape[1]
        
        # Only crop if sign is 5%–80% of frame (filters tiny noise or full backgrounds)
        if 0.05 < (area / img_area) < 0.8:
            x, y, w, h = cv2.boundingRect(c)
            pad = int(max(w, h) * 0.1)  # 10% padding
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(img_bgr.shape[1], x + w + pad), min(img_bgr.shape[0], y + h + pad)
            cropped = img_bgr[y1:y2, x1:x2]

    # 4. Resize & normalize EXACTLY like training (cv2.resize + /255.0)
    resized = cv2.resize(cropped, target_size)
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else input("📷 Enter path to image: ").strip()
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        sys.exit(1)

    print(f"\n🔍 Predicting: {img_path}")
    pred = model.predict(prepare_image(img_path), verbose=0)[0]

    top3_idx = np.argsort(pred)[::-1][:3]
    top_conf = pred[top3_idx[0]] * 100

    print(f"\n{'─'*45}")
    print(f"{'Rank':<6} {'Sign':<30} {'Confidence':>10}")
    print(f"{'─'*45}")

    if top_conf < 50:
        # Graceful fallback for ambiguous/uncropped images
        print(f"{'#1':<6} {'Unclear / Not a recognized sign':<30} {'':>10}")
        print(f"{'─'*45}\n💡 Tip: Upload a photo where the sign is clearly visible and centered.")
    else:
        for rank, idx in enumerate(top3_idx, start=1):
            name = CANADIAN_SIGNS.get(int(idx), f"Class {idx}")
            confidence = pred[idx] * 100
            marker = "  ← TOP PICK" if rank == 1 else ""
            print(f"#{rank:<5} {name:<30} {confidence:>8.1f}%{marker}")

    print(f"{'─'*45}\n")
