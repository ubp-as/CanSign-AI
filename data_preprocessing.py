# data_preprocessing.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE    = (32, 32)
NUM_CLASSES = 43


def load_data(data_dir):
    images, labels = [], []

    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)

        if not os.path.isdir(label_path):
            continue
        try:
            class_id = int(label)
        except ValueError:
            print(f"⚠️  Skipping non-numeric folder: '{label}'")
            continue

        img_files = [f for f in os.listdir(label_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm'))]

        for img_name in img_files:
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(class_id)

    if len(images) == 0:
        raise ValueError(
            f"❌ No images found in '{data_dir}'.\n"
            "Run download_gtsrb.py first to populate the dataset folder."
        )

    X = np.array(images, dtype='float32') / 255.0
    y = np.array(labels)
    print(f"✅ Loaded {len(X)} images across {len(np.unique(y))} classes.")
    return X, y


def load_tfds_data(data_dir='traffic_sign'):
    """Main entry point used by evaluate_model.py"""
    X, y_raw = load_data(data_dir)
    num_classes = int(max(y_raw) + 1)
    print(f"🔢 Detected {num_classes} classes (0 to {int(max(y_raw))}).")

    y = to_categorical(y_raw, num_classes=num_classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_raw
    )
    print(f"📐 Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, num_classes


def get_augmented_generator(X_train, y_train, batch_size=32):
    """Augmentation — no horizontal flip, signs are not symmetric."""
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=False,
    )
    datagen.fit(X_train)
    return datagen.flow(X_train, y_train, batch_size=batch_size)
