# 🚦 Canadian Traffic Sign Classifier

**[Try the live demo →](https://huggingface.co/spaces/ubp-as/traffic-sign-classifier)**

A CNN-based traffic sign classifier trained on 50,000+ images across 43 Canadian road sign classes. Achieves **99%+ test accuracy**. Includes a drag-and-drop web demo — upload any traffic sign photo and get an instant prediction with confidence scores.

![Web Demo](app/static/preview.png)

---

## How it works

Images are resized to 32×32, normalized, and passed through a custom CNN with two convolutional blocks, batch normalization, and dropout. The model outputs a probability across all 43 sign classes and returns the top 3 predictions. Trained on a T4 GPU in Google Colab in under 20 minutes.

---

## Run locally

**1. Clone the repo and install dependencies**
```bash
git clone https://github.com/ubp-as/Canadian-Traffic-Sign-Classifier.git
cd Canadian-Traffic-Sign-Classifier
pip install -r requirements_app.txt
```

**2. Download the model**

Download `traffic_sign_model.keras` from the [Hugging Face repo](https://huggingface.co/spaces/ubp-as/traffic-sign-classifier/tree/main) and place it in the project root.

**3. Start the web app**
```bash
uvicorn app.app:app --reload
```
Open **http://localhost:8000** and drag in any traffic sign image.

**4. Or use the CLI**
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

## Project Structure

```
├── app/
│   ├── app.py              # FastAPI backend
│   └── static/
│       └── index.html      # Web frontend
├── canadian_labels.py      # Class ID → sign name mapping
├── cnn_model.py            # CNN architecture
├── data_preprocessing.py   # Data loading and augmentation
├── predict_image.py        # CLI prediction tool
├── train_colab.ipynb       # Training notebook (Colab)
├── requirements.txt        # ML dependencies
└── requirements_app.txt    # Web app dependencies
```
