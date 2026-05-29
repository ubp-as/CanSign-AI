# 🚦 Canadian Traffic Sign Classifier

A CNN-based traffic sign classifier trained on the GTSRB dataset, mapped to Canadian road signs. Includes a live web demo you can run locally or deploy for free.

## Project Structure

```
├── canadian_labels.py          # Maps class IDs to Canadian sign names
├── cnn_model.py                # CNN architecture
├── data_preprocessing.py       # Data loading and augmentation
├── evaluate_model.py           # Per-class accuracy breakdown (run locally)
├── predict_image.py            # Predict a single image via CLI
├── train_colab.ipynb           # Training notebook — run on Google Colab GPU
├── requirements.txt            # Local ML dependencies
├── requirements_app.txt        # Web app dependencies
├── traffic_sign_model.keras    # Trained model (download from Colab after training)
└── app/
    ├── app.py                  # FastAPI backend (REST API)
    └── static/
        └── index.html          # Web frontend — drag & drop image → prediction
```

## Workflow

### 1. Train on Google Colab (free GPU)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `train_colab.ipynb`
3. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`
4. Run all cells: `Runtime → Run all`
5. Download `traffic_sign_model.keras` when training completes
6. Place it in this project folder (root, next to `app/`)

### 2. Install local dependencies
```bash
pip install -r requirements.txt
pip install -r requirements_app.txt
```

### 3. Run the web demo
```bash
uvicorn app.app:app --reload
```
Then open **http://localhost:8000** — drag and drop any traffic sign image to get a prediction with confidence scores.

### 4. CLI tools
```bash
# Evaluate model accuracy on test set
python evaluate_model.py

# Predict a single image
python predict_image.py path/to/sign.jpg
```

## REST API

The backend exposes one endpoint:

```
POST /predict
Content-Type: multipart/form-data
Body: file=<image>

Response:
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

## Deploy for Free (Hugging Face Spaces)

1. Create a free account at [huggingface.co](https://huggingface.co)
2. New Space → SDK: **Docker**
3. Upload all project files including `traffic_sign_model.keras`
4. Add a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements_app.txt
EXPOSE 7860
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "7860"]
```
5. Your live demo URL: `https://huggingface.co/spaces/YOUR_USERNAME/traffic-sign-classifier`

## Dataset
- **GTSRB** (German Traffic Sign Recognition Benchmark)
- 50,000+ images across 43 classes
- German signs are visually identical to Canadian signs for most classes

## Model
- Custom CNN: 2 convolutional blocks + BatchNormalization + Dropout
- tf.data pipeline with augmentation (rotation, zoom, brightness)
- Trained on Google Colab T4 GPU
- **Test accuracy: 99%+**
