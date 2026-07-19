"""Week 3 preprocessing utilities for casting defect detection."""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

LABEL_TO_ID = {"ok_front": 0, "def_front": 1}
ID_TO_LABEL = {0: "ok_front", 1: "def_front"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def discover_class_pairs(root: Path):
    pairs = []
    for def_dir in root.rglob("def_front"):
        parent = def_dir.parent
        if (parent / "ok_front").is_dir():
            pairs.append(parent)
    return sorted(set(pairs))

def select_dataset_source(dataset_root: Path):
    class_pairs = discover_class_pairs(dataset_root)
    if not class_pairs:
        raise FileNotFoundError(f"No valid class-pair folder found below {dataset_root}")
    raw_512_pairs = [p for p in class_pairs if "512" in str(p).lower()]
    return raw_512_pairs[0] if raw_512_pairs else class_pairs[0]

def create_manifest(selected_source: Path):
    records = []
    for label_name, label_id in LABEL_TO_ID.items():
        class_dir = selected_source / label_name
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image = cv2.imread(str(path))
                records.append({
                    "path": str(path),
                    "filename": path.name,
                    "label_name": label_name,
                    "label": label_id,
                    "readable": image is not None,
                    "height": None if image is None else image.shape[0],
                    "width": None if image is None else image.shape[1],
                })
    manifest = pd.DataFrame(records)
    return manifest[manifest["readable"]].reset_index(drop=True)

def create_stratified_splits(manifest, seed=123, train_fraction=0.70, validation_fraction=0.15, test_fraction=0.15):
    train_df, temp_df = train_test_split(
        manifest,
        train_size=train_fraction,
        stratify=manifest["label"],
        random_state=seed,
    )

    validation_ratio_from_temp = validation_fraction / (validation_fraction + test_fraction)

    validation_df, test_df = train_test_split(
        temp_df,
        train_size=validation_ratio_from_temp,
        stratify=temp_df["label"],
        random_state=seed,
    )

    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(drop=True)

def decode_and_resize(path, label, image_size=(224, 224)):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size, antialias=True)
    image = tf.cast(image, tf.float32)
    return image, tf.cast(label, tf.float32)

def make_dataset(dataframe, image_size=(224, 224), batch_size=32, training=False, seed=123):
    paths = dataframe["path"].astype(str).to_numpy()
    labels = dataframe["label"].astype(np.float32).to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(len(dataframe), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(lambda p, y: decode_and_resize(p, y, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def calculate_class_weights(train_df):
    values = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].to_numpy(),
    )
    return {0: float(values[0]), 1: float(values[1])}
