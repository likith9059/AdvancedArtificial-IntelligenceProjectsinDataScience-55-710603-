"""Week 5 model comparison utilities."""
from tensorflow import keras
from tensorflow.keras import layers

MODEL_CONFIGS = {
    "MobileNetV2": (keras.applications.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input),
    "EfficientNetB0": (keras.applications.EfficientNetB0, keras.applications.efficientnet.preprocess_input),
    "DenseNet121": (keras.applications.DenseNet121, keras.applications.densenet.preprocess_input),
    "Xception": (keras.applications.Xception, keras.applications.xception.preprocess_input),
}

def make_augmentation():
    return keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.04), layers.RandomZoom(0.08), layers.RandomContrast(0.10)], name="augmentation")

def build_transfer_model(model_name, image_size=(224,224), learning_rate=1e-3):
    model_class, preprocess = MODEL_CONFIGS[model_name]
    inputs = keras.Input(shape=image_size + (3,), name="image")
    x = make_augmentation()(inputs)
    x = preprocess(x)
    try:
        backbone = model_class(include_top=False, weights="imagenet", input_shape=image_size + (3,), pooling="avg", name=model_name.lower())
    except Exception:
        backbone = model_class(include_top=False, weights=None, input_shape=image_size + (3,), pooling="avg", name=model_name.lower())
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="probability_defective")(x)
    model = keras.Model(inputs, outputs, name=f"Week5_{model_name}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy", keras.metrics.Precision(name="defect_precision"), keras.metrics.Recall(name="defect_recall"), keras.metrics.AUC(name="roc_auc"), keras.metrics.AUC(name="pr_auc", curve="PR")])
    return model
