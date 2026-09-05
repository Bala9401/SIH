import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, IMAGE_SIZE, BATCH_SIZE


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(value))


def evaluate_cnn(metrics_dir):
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    dataset_dir = DATA_DIR / "processed_satellite"
    model_path = MODEL_DIR / "cyclone_cnn.keras"
    if not dataset_dir.exists() or not model_path.exists():
        return {"available": False, "reason": "processed satellite dataset or CNN model not found"}
    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_dir, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=True)
    model = tf.keras.models.load_model(model_path)
    labels, predictions = [], []
    for images, batch_labels in dataset:
        outputs = model.predict(preprocess_input(tf.cast(images, tf.float32)), verbose=0)
        labels.extend(batch_labels.numpy().tolist())
        predictions.extend(np.argmax(outputs, axis=1).tolist())
    matrix = confusion_matrix(labels, predictions).tolist()
    with open(metrics_dir / "cnn_confusion_matrix.json", "w") as file:
        json.dump(matrix, file, indent=2)
    return {
        "available": True,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": matrix,
        "task": "satellite_product_classification; intensity_labels_unavailable"
    }


def evaluate_lstm(metrics_dir):
    import joblib
    import tensorflow as tf
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    artifacts = [MODEL_DIR / "cyclone_lstm.keras", MODEL_DIR / "scaler.pkl",
                 DATA_DIR / "processed" / "track_sequences.npz",
                 DATA_DIR / "processed" / "cyclone_tracks.json",
                 DATA_DIR / "processed" / "ibtracs_metadata.json"]
    if not all(path.exists() for path in artifacts):
        return {"available": False, "reason": "LSTM artifacts are incomplete"}
    model = tf.keras.models.load_model(artifacts[0])
    scaler = joblib.load(artifacts[1])
    sequences = np.load(artifacts[2])
    predictions = model.predict(sequences["X_test"], verbose=0)
    actual = scaler.inverse_transform(sequences["y_test"])
    predicted = scaler.inverse_transform(predictions)
    names = ["latitude", "longitude", "wind", "pressure"]
    metrics = {f"{name}_mae": float(mean_absolute_error(actual[:, i], predicted[:, i])) for i, name in enumerate(names)}
    for i, name in enumerate(names):
        metrics[f"{name}_rmse"] = float(math.sqrt(mean_squared_error(actual[:, i], predicted[:, i])))

    with open(artifacts[4], "r") as file:
        metadata = json.load(file)
    with open(artifacts[3], "r") as file:
        tracks = json.load(file)
    test_storms = set(metadata.get("test_storms", []))
    eligible_test_storms = set()
    errors = {12: [], 24: [], 36: [], 48: []}
    for storm_id in test_storms:
        points = tracks.get(storm_id, [])
        if len(points) <= 6:
            continue
        # Evaluate one complete 48-hour forecast origin for every eligible unseen storm.
        for end in [len(points) - 16]:
            if end < 6:
                continue
            eligible_test_storms.add(storm_id)
            history = points[end - 6:end]
            raw = np.asarray([[p["lat"], p["lon"], p.get("wind"), p.get("pressure")] for p in history], dtype=float)
            if not np.isfinite(raw).all():
                continue
            sequence = scaler.transform(raw).reshape(1, 6, 4)
            rollout = []
            for _ in range(16):
                next_scaled = model(sequence, training=False).numpy()[0]
                next_raw = scaler.inverse_transform(next_scaled.reshape(1, -1))[0]
                rollout.append(next_raw)
                sequence = np.concatenate([sequence[:, 1:, :], next_scaled.reshape(1, 1, 4)], axis=1)
            for horizon in errors:
                index = horizon // 3 - 1
                target_index = end + index
                if target_index < len(points):
                    target = points[target_index]
                    errors[horizon].append(float(haversine_km(rollout[index][0], rollout[index][1], target["lat"], target["lon"])))
    horizon_stats = {}
    for horizon, values in errors.items():
        horizon_stats[str(horizon)] = {
            "valid_forecasts": len(values),
            "mean_error_km": float(np.mean(values)) if values else None,
            "median_error_km": float(np.median(values)) if values else None,
            "p75_error_km": float(np.percentile(values, 75)) if values else None,
        }
        metrics[f"track_error_km_{horizon}h"] = horizon_stats[str(horizon)]["mean_error_km"]
    uncertainty = {
        "label": "Model-derived prototype uncertainty corridor based on held-out errors.",
        "method": "75th percentile Haversine error from all eligible unseen test storms",
        "horizons": horizon_stats,
    }
    with open(metrics_dir / "track_uncertainty.json", "w") as file:
        json.dump(uncertainty, file, indent=2)
    metrics["available"] = True
    metrics["test_storms_available"] = len(test_storms)
    metrics["test_storms_used"] = len(eligible_test_storms)
    metrics["track_error_statistics"] = horizon_stats
    return metrics


def evaluate():
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    try:
        cnn = evaluate_cnn(metrics_dir)
    except Exception as error:
        cnn = {"available": False, "reason": str(error)}
    try:
        lstm = evaluate_lstm(metrics_dir)
    except Exception as error:
        lstm = {"available": False, "reason": str(error)}
    with open(metrics_dir / "cnn_metrics.json", "w") as file:
        json.dump(cnn, file, indent=2)
    with open(metrics_dir / "lstm_metrics.json", "w") as file:
        json.dump(lstm, file, indent=2)
    print(json.dumps({"cnn": cnn, "lstm": lstm}, indent=2))


if __name__ == "__main__":
    evaluate()
