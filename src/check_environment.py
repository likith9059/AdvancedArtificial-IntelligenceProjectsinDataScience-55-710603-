from config import DATA_ROOT, TRAIN_DIR, TEST_DIR, CLASS_NAMES

def check(path, name):
    print(f"{name}: {'FOUND' if path.exists() else 'MISSING'} -> {path}")

print("Week 1 Environment Check")
check(DATA_ROOT, "Dataset root")
check(TRAIN_DIR, "Train folder")
check(TEST_DIR, "Test folder")

for split_name, split_dir in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
    for cls in CLASS_NAMES:
        check(split_dir / cls, f"{split_name}/{cls}")
