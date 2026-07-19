"""Week 3 Custom CNN baseline model for casting defect detection."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04, fill_mode="reflect"),
            layers.RandomZoom(0.08, fill_mode="reflect"),
            layers.RandomContrast(0.10),
            layers.RandomBrightness(0.08, value_range=(0, 255)),
        ],
        name="augmentation",
    )

def build_custom_cnn(image_size=(224, 224)):
    inputs = keras.Input(shape=image_size + (3,), name="image")
    x = make_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(x)

    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="probability_defective")(x)

    model = keras.Model(inputs, outputs, name="Week3_CustomCNN_Baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="defect_precision"),
            keras.metrics.Recall(name="defect_recall"),
            keras.metrics.AUC(name="roc_auc", curve="ROC"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model

def callbacks_for_week3(model_dir, report_dir):
    return [
        keras.callbacks.ModelCheckpoint(
            model_dir / "week3_custom_cnn_best.keras",
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(report_dir / "week3_custom_cnn_training_log.csv"),
    ]
