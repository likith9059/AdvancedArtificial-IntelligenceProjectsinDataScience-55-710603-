"""Week 7 fresh Xception training utilities."""

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

def build_fresh_xception_model(image_size=(299, 299), learning_rate=1e-3):
    inputs = keras.Input(shape=image_size + (3,), name="image")
    x = make_augmentation()(inputs)
    x = keras.applications.xception.preprocess_input(x)

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

    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dense(256, activation="relu", name="head_dense_256")(x)
    x = layers.Dropout(0.35, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="probability_defective")(x)

    model = keras.Model(inputs, outputs, name="Week7_Fresh_Xception")
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

def unfreeze_last_xception_layers(model, trainable_fraction=0.20):
    backbone = model.get_layer("xception")
    backbone.trainable = True
    fine_tune_at = int(len(backbone.layers) * (1 - trainable_fraction))

    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False

    trainable_count = sum(1 for layer in backbone.layers if layer.trainable)
    return backbone, fine_tune_at, trainable_count
