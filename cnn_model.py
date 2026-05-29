# cnn_model.py
# CNN architecture for traffic sign classification.
#
# Fix: Uses tf.keras.Input() as first layer instead of passing
# input_shape to Conv2D directly, which suppresses the Keras warning.

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam


def create_cnn_model(input_shape=(32, 32, 3), num_classes=43):
    inputs = Input(shape=input_shape)

    # ── Block 1 ──────────────────────────────────────────────────────────────
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)   # 32x32 → 16x16
    x = Dropout(0.25)(x)

    # ── Block 2 ──────────────────────────────────────────────────────────────
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)   # 16x16 → 8x8
    x = Dropout(0.25)(x)

    # ── Classifier ───────────────────────────────────────────────────────────
    x = Flatten()(x)                        # 8x8x64 = 4096 features
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
