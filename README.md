# 🚦 Traffic-Sign-Classifier

**[Try the live demo →](https://huggingface.co/spaces/ubp-as/traffic-sign-classifier)**

A CNN-based traffic sign classifier trained on 50,000+ images across 43 Canadian road sign classes. Achieves **99%+ test accuracy**. Includes a drag-and-drop web demo — upload any traffic sign photo and get an instant prediction with confidence scores.

![Web Demo](preview.png)

---

## How it works

Images are resized to 32×32, normalized, and passed through a custom CNN with two convolutional blocks, batch normalization, and dropout. The model outputs a probability across all 43 sign classes and returns the top 3 predictions. Trained on a T4 GPU in Google Colab in under 20 minutes.

---

## Features

- 🖼️ **Drag-and-drop web interface** — upload any traffic sign photo, get an instant prediction
- 📊 **Top 3 predictions** with confidence bars so you can see how certain the model is
- ⚡ **FastAPI backend** — clean REST API, returns JSON, easy to integrate into other projects
- 🇨🇦 **Mapped to Canadian signs** — all 43 GTSRB classes labeled with Canadian road sign names
- 🚫 **No horizontal flip augmentation** — traffic signs are not symmetric, so this was intentionally excluded during training

---

## Run locally

```bash
git clone https://github.com/ubp-as/CanSign-AI.git
cd CanSign-AI
pip install -r requirements_app.txt
```

Download `traffic_sign_model.keras` from the [Hugging Face Space](https://huggingface.co/spaces/ubp-as/CanSign-AI), place it in the project root, then:

```bash
uvicorn app.app:app --reload
```

Open **http://localhost:8000** — or predict a single image via CLI:

```bash
python predict_image.py path/to/sign.jpg
```

---

## REST API

```
POST /predict
Content-Type: multipart/form-data

{
  "top_prediction": "Stop",
  "confidence": 99.7,
  "top3": [
    { "rank": 1, "class_id": 14, "sign_name": "Stop",     "confidence": 99.7 },
    { "rank": 2, "class_id": 17, "sign_name": "No Entry", "confidence": 0.2  },
    { "rank": 3, "class_id": 13, "sign_name": "Yield",    "confidence": 0.1  }
  ]
}
```

---

## Dataset & Model

- **Dataset:** GTSRB (German Traffic Sign Recognition Benchmark) — 50,000+ images, 43 classes
- **Architecture:** Custom CNN — 2 conv blocks, BatchNormalization, Dropout, Dense classifier
- **Training:** tf.data pipeline with augmentation (rotation, zoom, brightness — no horizontal flip)
- **Hardware:** Google Colab T4 GPU
- **Accuracy: 99%+** on held-out test set

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | Custom CNN (TensorFlow / Keras) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / JavaScript |
| Training | Google Colab T4 GPU |
| Dataset | GTSRB (43 classes, 50,000+ images) |
| Hosting | Hugging Face Spaces |
