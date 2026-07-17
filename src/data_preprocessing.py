"""Week 2 data preprocessing and EDA utilities.

Project: Automated Casting Defect Detection Using Deep Learning and Computer Vision
Purpose: Reusable helper functions for image inventory, metadata checks and duplicate detection.
"""

from pathlib import Path
import hashlib
import cv2
import pandas as pd
import numpy as np

CLASS_NAMES = ["def_front", "ok_front"]

def collect_image_records(root_dir, split_name, class_names=None):
    """Create image inventory records from a directory split."""
    if class_names is None:
        class_names = CLASS_NAMES

    root_dir = Path(root_dir)
    records = []

    for label in class_names:
        label_dir = root_dir / label
        if not label_dir.exists():
            continue

        for img_path in label_dir.glob("*"):
            if img_path.is_file():
                records.append({
                    "split": split_name,
                    "label": label,
                    "path": str(img_path),
                    "filename": img_path.name,
                    "extension": img_path.suffix.lower()
                })

    return records

def inspect_image(path):
    """Return readability and dimension metadata for one image."""
    img = cv2.imread(str(path))
    if img is None:
        return {
            "readable": False,
            "height": np.nan,
            "width": np.nan,
            "channels": np.nan
        }

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) == 3 else 1

    return {
        "readable": True,
        "height": h,
        "width": w,
        "channels": c
    }

def file_hash(path, block_size=65536):
    """Generate MD5 hash for exact duplicate detection."""
    hasher = hashlib.md5()
    with open(path, "rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()

def build_inventory(train_dir, test_dir, class_names=None):
    """Build full image inventory dataframe for train and test folders."""
    records = []
    records.extend(collect_image_records(train_dir, "train", class_names))
    records.extend(collect_image_records(test_dir, "test", class_names))
    return pd.DataFrame(records)

def add_image_metadata(df):
    """Add readability, height, width and channel metadata to an image dataframe."""
    meta = df["path"].apply(lambda p: pd.Series(inspect_image(p)))
    return pd.concat([df, meta], axis=1)

def add_duplicate_hashes(df):
    """Add MD5 hash column for readable images."""
    out = df.copy()
    out["md5_hash"] = out["path"].apply(file_hash)
    return out

def find_duplicates(df_with_hash):
    """Return rows that belong to duplicate hash groups."""
    return (
        df_with_hash.groupby("md5_hash")
        .filter(lambda x: len(x) > 1)
        .sort_values(["md5_hash", "split", "label"])
    )
