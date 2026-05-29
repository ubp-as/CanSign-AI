# predict_image.py
# Run this LOCALLY after downloading traffic_sign_model.keras from Colab.
#
# Usage:
#   python predict_image.py path/to/image.jpg

import os
# Suppress CUDA/GPU warnings — we're running inference on CPU locally
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']  = '-1'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import sys
from canadian_labels import CANADIAN_SIGNS

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = 'traffic_sign_model.keras'
if not os.path.exists(MODEL_PATH):
    # fallback to .h5 if someone still has the old format
    MODEL_PATH = 'traffic_sign_model.h5'
    if not os.path.exists(MODEL_PATH):
        print("❌ No model found. Download traffic_sign_model.keras from Colab first.")
        sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)


def prepare_image(img_path, target_size=(32, 32)):
    """Load, resize, and normalize a single image for prediction."""
    img       = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else input("📷 Enter path to image: ").strip()

    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        sys.exit(1)

    print(f"\n🔍 Predicting: {img_path}")
    pred = model.predict(prepare_image(img_path), verbose=0)[0]

    top3 = np.argsort(pred)[::-1][:3]

    print(f"\n{'─'*45}")
    print(f"{'Rank':<6} {'Sign':<30} {'Confidence':>10}")
    print(f"{'─'*45}")

    for rank, idx in enumerate(top3, start=1):
        name       = CANADIAN_SIGNS.get(int(idx), f"Class {idx}")
        confidence = pred[idx] * 100
        marker     = "  ← TOP PICK" if rank == 1 else ""
        print(f"#{rank:<5} {name:<30} {confidence:>8.1f}%{marker}")

    print(f"{'─'*45}\n")
