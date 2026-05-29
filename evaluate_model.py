# evaluate_model.py
# Run this LOCALLY after downloading traffic_sign_model.keras from Colab.
#
# Usage:
#   python evaluate_model.py

import os
# Suppress CUDA/GPU warnings — running inference on CPU locally
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']  = '-1'

import numpy as np
import tensorflow as tf
from data_preprocessing import load_tfds_data
from canadian_labels import CANADIAN_SIGNS

# ── Load model ────────────────────────────────────────────────────────────────
print("📂 Loading model...")
MODEL_PATH = 'traffic_sign_model.keras'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'traffic_sign_model.h5'
    if not os.path.exists(MODEL_PATH):
        print("❌ No model found. Download traffic_sign_model.keras from Colab first.")
        raise SystemExit(1)

model = tf.keras.models.load_model(MODEL_PATH)

# ── Load data ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, num_classes = load_tfds_data()

# ── Overall accuracy ──────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\n📊 Test accuracy : {test_acc:.4f}")
print(f"📉 Test loss     : {test_loss:.4f}")

# ── Per-class breakdown ───────────────────────────────────────────────────────
y_pred        = model.predict(X_test, verbose=0)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test,  axis=1)

print(f"\n{'─'*50}")
print(f"{'Class':<6} {'Sign Name':<32} {'Acc':>6}")
print(f"{'─'*50}")

for class_id in range(num_classes):
    mask = y_true_labels == class_id
    if mask.sum() == 0:
        continue
    acc  = (y_pred_labels[mask] == class_id).mean()
    name = CANADIAN_SIGNS.get(class_id, f"Class {class_id}")
    flag = "  ⚠️ " if acc < 0.80 else ""
    print(f"{class_id:<6} {name:<32} {acc:>5.1%}{flag}")

print(f"{'─'*50}")
