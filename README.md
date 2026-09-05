# AI Cyclone Identification and Early Warning Prototype

This SIH 2026 prototype uses INSAT-3D satellite products and NOAA IBTrACS North Indian Ocean best-track data. It provides satellite-product classification, meteorological track/wind/pressure forecasting, a model-derived prototype uncertainty corridor, and explainable rule-based coastal risk scoring.

> Prototype AI prediction only. This is not an official warning system. Follow IMD, NDMA, and local authority advisories.

## What the current data supports

- **A. Satellite product classification:** MobileNetV2 classifies the three available image product folders: `insat3d_for_reference_ds`, `insat3d_ir_cyclone_ds`, and `insat3d_raw_cyclone_ds`.
- **B. Meteorological forecasting:** The LSTM predicts `LAT`, `LON`, `WMO_WIND`, and `WMO_PRES` from IBTrACS. Forecasts are generated every 3 hours through `T+48h`.
- **C. Prototype uncertainty corridor:** `results/metrics/track_uncertainty.json` stores mean, median, and 75th-percentile held-out Haversine errors for each horizon. The dashboard uses the 75th percentile. This is not a probabilistic meteorological cone.
- **D. Rule-based coastal risk:** Uses Haversine distance to documented Indian coastal reference points plus wind, pressure, and forecast trend. It is a prototype and does not use a coastline shapefile or population grid.
- **Unsupported capabilities:** True satellite-image intensity classification and aligned multi-source feature fusion are not implemented because the available image dataset lacks verified intensity labels and aligned multi-channel samples.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The pinned environment is Python 3.12 with Flask 3.0, TensorFlow 2.20, NumPy 1.26, Pandas 2.1, and scikit-learn 1.3.

## Dataset layout

```text
data/
  ibtracs/ibtracs.NI.list.v04r00.csv
  satellite/
    insat3d_for_reference_ds/
    insat3d_ir_cyclone_ds/
    insat3d_raw_cyclone_ds/
```

IBTrACS must provide numeric `LAT`, `LON`, `WMO_WIND`, and `WMO_PRES` rows. Rows missing any of these four values are excluded rather than filled with invented measurements.

## Complete pipeline

Run from the project directory:

```powershell
python scripts\inspect_dataset.py
python scripts\preprocess_images.py
python scripts\preprocess_ibtracs.py
python scripts\train_cnn.py
python scripts\train_lstm.py
python scripts\evaluate_models.py
python app.py
```

Open `http://127.0.0.1:5000/dashboard`.

`preprocess_ibtracs.py` first splits by storm ID into training, validation, and final-test storms; fits `models/scaler.pkl` only on training storms; stores raw values in `data/processed/cyclone_tracks.json`; and writes scaled fit/validation/test sequences to `track_sequences.npz`. The predictor applies that same scaler once to raw observations and inverse-transforms predictions back to real units.

## Evaluation outputs

- `results/metrics/cnn_metrics.json`: accuracy, precision, recall, F1, and confusion matrix for the seeded validation partition.
- `results/metrics/cnn_confusion_matrix.json`: CNN confusion matrix.
- `results/metrics/lstm_metrics.json`: latitude/longitude/wind/pressure MAE and RMSE, number of test storms used, and per-horizon valid forecast count, mean, median, and 75th-percentile Haversine error.
- `results/metrics/track_uncertainty.json`: held-out-test mean, median, and 75th-percentile error statistics. The 75th percentile is used by the prototype corridor.

Metrics are generated only by `scripts/evaluate_models.py`; the API reads these files directly at `/api/model-metrics`.

## API

- `GET /api/status`
- `GET /api/cyclones`
- `POST /predict/image` with multipart field `file`
- `POST /predict/track` with JSON `{ "cyclone_id": "..." }`
- `GET /api/risk`
- `GET /api/model-metrics`

Successful track responses include `forecast_horizon_hours: 48`, `forecast_step_hours: 3`, pressure fields, and `uncertainty_radius_km` where evaluation data is available. Errors use `{ "success": false, "error": "..." }` on upload and prediction routes.

## SIH demonstration

1. Run the complete pipeline once and start Flask.
2. Open the dashboard and select a cyclone from the IBTrACS list.
3. Show the historical track, 16 forecast points through `T+48h`, wind/pressure values, and the prototype uncertainty corridor.
4. Upload an INSAT image and explain that the current CNN identifies the source product, not intensity.
5. Show the generated metrics and held-out storm split in the `results/metrics` files.
6. Show risk factors and distance to the coastal reference geometry.
7. State the limitations clearly: no verified image intensity labels, no aligned multi-channel fusion, prototype coastline geometry, and large long-horizon track errors in the current evaluation.
