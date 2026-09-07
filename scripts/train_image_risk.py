import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, IMAGE_SIZE, MODEL_DIR

MANIFEST = DATA_DIR / "satellite_intensity_manifest.csv"
REQUIRED_COLUMNS = {"image_path", "wind_speed_kt", "pressure_hpa"}


def load_manifest():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or [])
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    images, targets = [], []
    for row in rows:
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = DATA_DIR / image_path
        try:
            target = [float(row["wind_speed_kt"]), float(row["pressure_hpa"])]
        except (TypeError, ValueError):
            continue
        if image_path.is_file() and np.all(np.isfinite(target)):
            images.append(str(image_path))
            targets.append(target)
    if len(images) < 20:
        raise ValueError(f"Need at least 20 verified image rows; found {len(images)}")
    return images, np.asarray(targets, dtype=np.float32)


def load_image(path):
    import tensorflow as tf
    image = tf.keras.utils.load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    return tf.keras.utils.img_to_array(image)


def train():
    paths, targets = load_manifest()

    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    images = np.asarray([load_image(path) for path in paths], dtype=np.float32)
    rng = np.random.default_rng(123)
    order = rng.permutation(len(images))
    split = max(1, int(len(order) * 0.2))
    validation_order, training_order = order[:split], order[split:]
    target_mean = targets[training_order].mean(axis=0)
    target_std = targets[training_order].std(axis=0)
    target_std[target_std == 0] = 1.0

    x_train = preprocess_input(images[training_order])
    x_validation = preprocess_input(images[validation_order])
    y_train = (targets[training_order] - target_mean) / target_std
    y_validation = (targets[validation_order] - target_mean) / target_std

    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base.trainable = False
    output = Dense(2, activation="linear")(GlobalAveragePooling2D()(base.output))
    model = Model(base.input, output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=20,
        batch_size=min(16, len(x_train)),
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=1,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "cyclone_image_risk.keras")
    with (MODEL_DIR / "image_risk_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "task": "supervised_image_to_wind_pressure_regression",
            "manifest": str(MANIFEST),
            "target_names": ["wind_speed_kt", "pressure_hpa"],
            "target_mean": target_mean.tolist(),
            "target_std": target_std.tolist(),
            "training_rows": len(training_order),
            "validation_rows": len(validation_order),
        }, handle, indent=2)
    print(f"Saved supervised image-risk model using {len(paths)} verified rows.")


if __name__ == "__main__":
    try:
        train()
    except (FileNotFoundError, ValueError) as error:
        print(f"Image-risk training not started: {error}")
        raise SystemExit(1)
