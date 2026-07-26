"""Week 4 Xception transfer-learning model for casting defect detection."""

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

def load_xception_backbone(image_size=(299, 299)):
    try:
        backbone = keras.applications.Xception(
            include_top=False,
            weights="imagenet",
            input_shape=image_size + (3,),
            pooling="avg",
            name="xception",
        )
    except Exception:
        backbone = keras.applications.Xception(
            include_top=False,
            weights=None,
            input_shape=image_size + (3,),
            pooling="avg",
            name="xception",
        )
    return backbone

def build_xception_transfer_model(image_size=(299, 299), learning_rate=1e-3):
    inputs = keras.Input(shape=image_size + (3,), name="image")
    x = make_augmentation()(inputs)
    x = keras.applications.xception.preprocess_input(x)

    backbone = load_xception_backbone(image_size=image_size)
    backbone.trainable = False

    x = backbone(x, training=False)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dense(256, activation="relu", name="head_dense_256")(x)
    x = layers.Dropout(0.35, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="probability_defective")(x)

    model = keras.Model(inputs, outputs, name="Week4_Xception_TransferLearning")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
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

def callbacks_for_week4(model_dir, report_dir):
    return [
        keras.callbacks.ModelCheckpoint(
            model_dir / "week4_xception_best.keras",
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
        keras.callbacks.CSVLogger(report_dir / "week4_xception_training_log.csv"),
    ]
