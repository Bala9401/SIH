# 🌪️ AI-Based Cyclone Early Warning System

## Project Overview
This project is an academic prototype for the Smart India Hackathon (SIH). It implements an **AI Cyclone Early Warning System** that utilizes satellite imagery and historical tracking data to predict cyclone behavior and assess risk. 

**AI Pipeline:**
Satellite Image → CNN → Classification → IBTrACS → LSTM → Track Prediction → Risk Assessment → Dashboard

**⚠️ Disclaimer:** This is an academic prototype created for demonstration purposes only. It is **not** an official meteorological warning system. Always rely on official agencies like the India Meteorological Department (IMD) for actual weather warnings.

## Features
- **MobileNetV2 CNN** for satellite image classification (INSAT-3D product types).
- **LSTM** for cyclone track prediction (latitude, longitude, wind speed, pressure).
- **Rule-based risk assessment** providing explainable scoring and severity levels.
- **Interactive web dashboard** built with Flask, featuring Leaflet.js maps and Chart.js visualizations.
- **Demo mode** featuring real historical data from Cyclone Fani (2019) when models are untrained.

## Architecture

```text
data/
 ├── ibtracs/                # Historical tracking dataset (CSV)
 ├── satellite/              # Training images for CNN
 ├── testing_images/         # Holdout test images
 models/                     # Saved H5 models and scalers
 static/                     # CSS, JS, Maps
 templates/                  # HTML Dashboard views
 app.py                      # Flask web server
 train_all.py                # Full pipeline training script
 train_cnn.py                # CNN training logic
 train_lstm.py               # LSTM training logic
 data_processor.py           # Data cleaning & prep
 requirements.txt            # Python dependencies
 setup.bat                   # Windows installation script
 README.md                   # Project documentation
```

## Dataset Information

### Satellite Images
- **Source:** INSAT-3D satellite products.
- **Types:** Reference images, infrared scans, and raw cyclone datasets.
- **Volume:** ~419 images across 3 distinct classes.
- **Format:** JPEG/PNG formats, resized internally to 224x224 pixels.

### IBTrACS Historical Data
- **Source:** International Best Track Archive for Climate Stewardship (IBTrACS) v4, North Indian Ocean subset (`ibtracs.NI.list.v04r00.csv`).
- **Volume:** ~60,677 observations detailing ~1,785 unique cyclones.
- **Key Features:** SID (Storm ID), NAME, LAT, LON, WMO_WIND, WMO_PRES, ISO_TIME.

## Installation

You can set up the environment using our automated script or manually.

```bash
# Option 1: Use setup.bat (Windows)
setup.bat

# Option 2: Manual Installation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Placement

To train the models from scratch, ensure your datasets are placed exactly as follows:

1. **Satellite Images:** Place image class folders inside `data/satellite/` (e.g., `data/satellite/reference/`, `data/satellite/infrared/`).
2. **Historical Tracks:** Place the CSV file at `data/ibtracs/ibtracs.NI.list.v04r00.csv`.

## Training

To train the complete AI pipeline, run:

```bash
python train_all.py
```
This script automates an 8-step pipeline:
1. Validating dataset directories.
2. Preprocessing satellite images.
3. Training the CNN model.
4. Saving CNN artifacts.
5. Cleaning and scaling IBTrACS data.
6. Generating sequence data for LSTM.
7. Training the LSTM model.
8. Saving LSTM models and scalers.

## Running the Application

Once installed (and optionally trained), start the dashboard:

```bash
python app.py
```
Then, open your browser and navigate to: http://localhost:5000

## CNN Model
- **Architecture:** Transfer learning via MobileNetV2 with ImageNet pretrained weights.
- **Custom Head:** GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(Softmax).
- **Callbacks:** EarlyStopping and ModelCheckpoint.
- **Purpose:** Classifies the satellite product type (e.g., IR, Visible, Reference) rather than cyclone intensity.

## LSTM Model
- **Architecture:** Sequential LSTM network for time-series forecasting.
- **Inputs:** 6-step sequences (18 hours of context) of `[lat, lon, wind, pressure]`.
- **Outputs:** Next predicted step of `[lat, lon, wind, pressure]`.
- **Forecast Horizon:** Recursive prediction generating a 48-hour forecast (16 steps × 3 hours).
- **Preprocessing:** Features are normalized using `MinMaxScaler`.

## Risk Assessment
Calculates a comprehensive danger score based on a weighted multi-factor system:
- **Satellite Context (30%)**: Based on CNN evaluation of current imagery.
- **Wind Speed (25%)**: Current/predicted maximum sustained winds.
- **Pressure (15%)**: Central barometric pressure drops.
- **Coastal Proximity (15%)**: Distance to the nearest landmass (using a prototype Indian coastline calculation).
- **Intensity Trend (10%)**: Rate of intensification over the last 12 hours.
- **Uncertainty (5%)**: Confidence intervals of model outputs.

**Risk Levels:**
- 🟢 **LOW** (< 25)
- 🟡 **MODERATE** (25 - 50)
- 🟠 **HIGH** (50 - 75)
- 🔴 **VERY HIGH** (> 75)

## Dashboard Features
- **Key Performance Indicators (KPIs):** Real-time metrics at a glance.
- **Interactive Map:** Leaflet.js map displaying historical tracks alongside AI-predicted paths.
- **Visual Analytics:** Chart.js graphs illustrating trends in wind, pressure, latitude, and longitude.
- **Early Warning Panel:** Real-time risk level display with actionable recommended safety measures.
- **Model Metrics Panel:** Transparent display of CNN and LSTM evaluation metrics.
- **Satellite Upload:** Upload custom satellite imagery to get instant CNN classifications.

## Demo Mode
If you run `app.py` without training the models first, **Demo Mode** activates automatically. It utilizes historical tracking data from **Cyclone Fani (2019)** to demonstrate the UI capabilities. All demo data is clearly labeled to distinguish it from live predictions.

## Model results and reproducibility

Run preprocessing before training; it writes `data/processed/ibtracs_ni_processed.csv`, storm-disjoint sequences, and `results/ibtracs_data_quality_report.json`.

```bash
python scripts/preprocess_ibtracs.py
python scripts/train_lstm.py
python scripts/evaluate_models.py
python app.py
```

TensorFlow is optional for the web interface. If it is unavailable, the dashboard explicitly uses a persistence baseline; it does not present synthetic tracks as model forecasts. Train/test storm IDs are kept separate and the scaler is fitted only on training storms.

### Satellite data limitation (important)

The numeric values in `data/satellite/insat_3d_ds - Sheet.csv` have no verified meteorological meaning. They are never used as wind, pressure, category, or intensity targets. Current satellite functionality is image validation, 224×224 preprocessing, gallery/product exploration, and optional **satellite product classification** (`reference`, `infrared`, `raw`). It is not cyclone detection or intensity estimation.

For a scientifically valid satellite detection/intensity model, each image needs a verified manifest with: `image_path,product,cyclone_id,cyclone_name,timestamp_utc,latitude,longitude,wind_speed_kt,pressure_hpa,storm_category,source`, plus documented provenance and split rules that prevent the same storm appearing in both train and test.

## Historical model performance (superseded)

The values below are retained only as historical project notes. Do not treat them as current results unless reproduced from the saved artifacts and evaluation command above.

### CNN Performance
- **Accuracy:** 100%
- **Precision:** 100% | **Recall:** 100% | **F1 Score:** 100%
- *Note:* This task represents satellite product classification (distinguishing between very distinct image types), which accounts for the perfect accuracy, rather than complex cyclone intensity estimation.

### LSTM Performance
- **Latitude MAE:** 0.76° | **Longitude MAE:** 1.13°
- **Wind Speed MAE:** 6.4 knots | **Pressure MAE:** 3.9 hPa
- **Track Displacement Error:** ~229 km (12h) | ~379 km (24h) | ~712 km (48h)

## Troubleshooting

- **TensorFlow Installation Issues:** Ensure you are using a 64-bit version of Python. If you have an older CPU, you may need a specific TensorFlow version that doesn't require AVX instructions.
- **Missing Datasets:** Ensure you have correctly placed `ibtracs.NI.list.v04r00.csv` and the `satellite/` subdirectories exactly as specified in the "Dataset Placement" section.
- **Port Conflicts:** If `http://localhost:5000` is already in use, change the port in `app.py` by modifying `app.run(port=5001)`.
- **Memory Issues (OOM):** If the LSTM or CNN training crashes due to RAM limitations, reduce the `batch_size` in the training scripts.

## Limitations
- The satellite dataset trains the CNN to provide product classification, **not** cyclone intensity or structural analysis.
- LSTM track prediction errors naturally increase with the forecast horizon (especially beyond 24 hours).
- The risk assessment module uses prototype-level logic and simplified coastline boundaries.
- **Not an official meteorological warning system.**
- The interactive map requires an active internet connection to load Leaflet map tiles.

## SIH 2-Minute Demo Steps
1. Open `http://localhost:5000`
2. Click **'Launch Dashboard'**
3. Highlight the project title and the prominent **disclaimer**.
4. Select a cyclone from the dropdown menu (e.g., **Fani**).
5. Click **'Load / Predict Track'**.
6. Show the historical track plotted on the map.
7. Show the AI-predicted track (represented by the orange dashed line).
8. Scroll down to show the dynamic wind/pressure and location charts.
9. Show the computed risk assessment panel and recommended actions.
10. Upload a satellite image from `data/satellite/`.
11. Show the resulting CNN classification output.
12. Show the transparency of model metrics on the UI.
13. Conclude by reiterating: *"This is an AI prototype – official warnings must come from the IMD."*

## Scientific Integrity Notice
- All displayed metrics are real, trained values derived from our evaluation scripts.
- No fabricated accuracy or predictive track data is used.
- Historical datasets and AI-generated predictions are explicitly labeled in the UI.
- This system is strictly an academic exploration and **not** a substitute for official meteorological warnings.

## License
Academic / Research prototype for Smart India Hackathon (SIH).
