from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path


MODEL_PATH = Path("casting_defect_model.keras")
BACKUP_PATH = Path("casting_defect_model_before_fix.keras")
TEMP_PATH = Path("casting_defect_model_repaired.tmp.keras")


def repair_config(value):
    """Recursively remove unsupported RandomRotation.value_range entries."""
    changes = 0

    if isinstance(value, dict):
        if value.get("class_name") == "RandomRotation":
            layer_config = value.get("config")
            if isinstance(layer_config, dict) and "value_range" in layer_config:
                layer_config.pop("value_range", None)
                changes += 1

        for child in value.values():
            changes += repair_config(child)

    elif isinstance(value, list):
        for child in value:
            changes += repair_config(child)

    return changes


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"{MODEL_PATH.name} was not found. Run this script from the Flask project folder."
        )

    if not zipfile.is_zipfile(MODEL_PATH):
        raise ValueError(
            f"{MODEL_PATH.name} is not a valid Keras v3 .keras archive."
        )

    if BACKUP_PATH.exists():
        raise FileExistsError(
            f"{BACKUP_PATH.name} already exists. Remove or rename it before running again."
        )

    shutil.copy2(MODEL_PATH, BACKUP_PATH)

    with zipfile.ZipFile(MODEL_PATH, "r") as source:
        members = {
            info.filename: (info, source.read(info.filename))
            for info in source.infolist()
        }

    if "config.json" not in members:
        raise KeyError("config.json was not found inside the .keras archive.")

    config_info, config_bytes = members["config.json"]
    config = json.loads(config_bytes.decode("utf-8"))
    changes = repair_config(config)

    if changes == 0:
        raise RuntimeError(
            "No RandomRotation value_range entry was found. "
            "The model may have a different compatibility problem."
        )

    repaired_config = json.dumps(
        config,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    with zipfile.ZipFile(
        TEMP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as target:
        for filename, (info, data) in members.items():
            if filename == "config.json":
                data = repaired_config

            new_info = zipfile.ZipInfo(
                filename=info.filename,
                date_time=info.date_time,
            )
            new_info.comment = info.comment
            new_info.extra = info.extra
            new_info.internal_attr = info.internal_attr
            new_info.external_attr = info.external_attr
            new_info.create_system = info.create_system
            new_info.compress_type = zipfile.ZIP_DEFLATED

            target.writestr(new_info, data)

    TEMP_PATH.replace(MODEL_PATH)

    print(f"Repair completed successfully.")
    print(f"Removed unsupported value_range from {changes} RandomRotation layer(s).")
    print(f"Repaired model: {MODEL_PATH.resolve()}")
    print(f"Backup model:   {BACKUP_PATH.resolve()}")


if __name__ == "__main__":
    main()
