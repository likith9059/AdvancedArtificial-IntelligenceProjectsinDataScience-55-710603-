from pathlib import Path

DATA_ROOT = Path("/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
CLASS_NAMES = ["def_front", "ok_front"]
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32
SEED = 123
