# 🚦 Canadian Traffic Sign Classifier

A CNN-based traffic sign classifier trained on the GTSRB dataset, mapped to Canadian road signs.

## Project Structure

```
├── canadian_labels.py      # Maps class IDs to Canadian sign names
├── cnn_model.py            # CNN architecture
├── data_preprocessing.py   # Loads GTSRB via tensorflow_datasets
├── evaluate_model.py       # Per-class accuracy breakdown (run locally)
├── predict_image.py        # Predict a single image (run locally)
├── train_colab.ipynb       # Training notebook — run on Google Colab GPU
├── requirements.txt        # Local dependencies
└── traffic_sign_model.h5   # Trained model (download from Colab after training)
```

## Workflow

### 1. Train on Google Colab (free GPU)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `train_colab.ipynb`
3. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`
4. Run all cells: `Runtime → Run all`
5. Download `traffic_sign_model.h5` when training completes
6. Place it in this project folder

### 2. Install local dependencies
```bash
pip install -r requirements.txt
```

### 3. Evaluate the model
```bash
python evaluate_model.py
```

### 4. Predict a single image
```bash
python predict_image.py path/to/sign.jpg
```

## Dataset
- **GTSRB** (German Traffic Sign Recognition Benchmark)
- 50,000+ images across 43 classes
- Loaded automatically via `tensorflow_datasets` — no manual download needed
- German signs are visually identical to Canadian signs for most classes

## Model
- Custom CNN with 2 convolutional blocks + BatchNormalization
- Data augmentation (rotation, zoom, brightness — no horizontal flip)
- Class weights to handle dataset imbalance
- Expected accuracy: ~95%+ on test set
