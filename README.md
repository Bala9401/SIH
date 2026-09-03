# AI-Based Cyclone Identification, Classification, Track Prediction and Early Warning System

> **DISCLAIMER: Prototype AI prediction only. Not an official meteorological warning system. Always follow official advisories from the India Meteorological Department (IMD) and National Disaster Management Authority (NDMA).**

[![Python 3.12](https://img.shields.io/badge/Python-3.12.1-blue.svg)](https://www.python.org/)
[![Flask 3.0](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
[![Bootstrap 5](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)](https://getbootstrap.com/)
[![Leaflet.js](https://img.shields.io/badge/Leaflet-1.9-brightgreen.svg)](https://leafletjs.com/)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-yellow.svg)](https://www.chartjs.org/)

An end-to-end Artificial Intelligence system developed for the **Smart India Hackathon (SIH)**. The platform ingests satellite imagery and historical meteorological records to identify cyclones, estimate their intensity, forecast future trajectories with cones of uncertainty, assess coastal risk levels, and broadcast early warnings via an interactive web dashboard.

---

## Table of Contents
1. [Project Title](#1-project-title)
2. [Problem Statement](#2-problem-statement)
3. [Proposed Solution](#3-proposed-solution)
4. [Objectives](#4-objectives)
5. [System Architecture](#5-system-architecture)
6. [Technology Stack](#6-technology-stack)
7. [Dataset Sources](#7-dataset-sources)
8. [Dataset Placement Instructions](#8-dataset-placement-instructions)
9. [Environment Setup](#9-environment-setup)
10. [Training Commands](#10-training-commands)
11. [Running Commands](#11-running-commands)
12. [Dashboard Usage Guide](#12-dashboard-usage-guide)
13. [API Documentation](#13-api-documentation)
14. [CNN Architecture (Identification & Intensity)](#14-cnn-architecture-identification--intensity)
15. [LSTM Architecture (Track Prediction)](#15-lstm-architecture-track-prediction)
16. [Risk Assessment Methodology](#16-risk-assessment-methodology)
17. [Evaluation Metrics](#17-evaluation-metrics)
18. [System Limitations](#18-system-limitations)
19. [Future Improvements](#19-future-improvements)
20. [SIH 2-Minute Demo Procedure](#20-sih-2-minute-demo-procedure)
21. [Disclaimer](#21-disclaimer)

---

## 1. Project Title
**AI-Based Cyclone Identification, Classification, Track Prediction and Early Warning System**  
*A Deep Learning and Geospatial Early Warning Platform for Disaster Preparedness in the North Indian Ocean Basin.*

---

## 2. Problem Statement
Tropical cyclones originating in the **Bay of Bengal** and the **Arabian Sea** regularly strike coastal India, Bangladesh, and neighboring regions, causing catastrophic loss of life, destruction of infrastructure, coastal flooding, and severe economic setbacks.

Traditional Numerical Weather Prediction (NWP) models require multi-million-dollar high-performance computing (HPC) clusters, take several hours to assimilate data and output forecast updates, and struggle with fast-developing Rapid Intensification (RI) events. Furthermore, local disaster management authorities (SDMA/DDMA) often lack simple, intuitive AI tools to rapidly analyze satellite images, visualize forecast cones of uncertainty, and generate immediate coastal risk advisories.

There is an urgent need for an agile, lightweight, AI-driven decision support system capable of:
- Instantaneous satellite image recognition and intensity classification.
- Sequence-based trajectory and barometric pressure forecasting within seconds.
- Quantitative coastal threat scoring and multi-tier early warning generation.

---

## 3. Proposed Solution
We propose an end-to-end intelligent platform that bridges the gap between raw meteorological data and on-the-ground disaster response:

1. **Computer Vision (CNN)**: Uses a transfer-learned MobileNetV2 architecture to identify cyclone cloud formations from satellite imagery and categorize intensity based on India Meteorological Department (IMD) standards.
2. **Sequential Trajectory Forecasting (LSTM)**: Employs a multi-layer Long Short-Term Memory network trained on historical NOAA IBTrACS data to predict the cyclone's future coordinates (Latitude/Longitude), central barometric pressure, and wind speed for +12h, +24h, +36h, and +48h.
3. **Cone of Uncertainty Generation**: Dynamically constructs expanding spatial probability bounds around forecast coordinates to communicate track variance to emergency planners.
4. **Multi-Factor Coastal Risk Engine**: Evaluates a weighted risk index (0–100) combining sustained wind velocity, central pressure drop, coastal proximity, and landfall translation speed.
5. **Interactive Geospatial Dashboard**: A responsive web interface powered by Leaflet.js, Bootstrap 5, and Chart.js providing real-time visual tracking, satellite image upload, historical replay, and automated warning bulletins.
6. **Zero-Setup Demo Fallback**: Operates in **Demo Mode** with built-in synthetic inference if datasets or pre-trained model weights are not present, ensuring reliable presentations.

---

## 4. Objectives
The project satisfies the following 10 core engineering and meteorological objectives:

1. **Multi-Source Ingestion**: Ingest satellite imagery (JPEG/PNG) and tabular atmospheric track records (NOAA IBTrACS CSV).
2. **Automated Image Preprocessing**: Standardize, crop, normalize, and augment satellite images with corrupted file handling.
3. **Deep Learning Classification**: Train a lightweight MobileNetV2 CNN classifier achieving high accuracy on cyclone vs. non-cyclone images and IMD storm stages.
4. **Temporal Track Modeling**: Implement an LSTM network capable of multi-step sequence-to-point forecasting of storm coordinates and intensity.
5. **Probabilistic Uncertainty Modeling**: Mathematically calculate and display the expanding forecast "Cone of Uncertainty" along the predicted trajectory.
6. **Rule-Based Coastal Risk Scoring**: Formulate a composite 0–100 risk index that translates complex meteorology into actionable alert tiers (Green, Yellow, Orange, Red).
7. **Geospatial GIS Visualization**: Provide an interactive Leaflet map rendering historical tracks, current positions, forecast paths, coastal threat zones, and landfall radii.
8. **Real-Time Analytical Plotting**: Render dynamic Chart.js visualizations for central pressure depletion and wind speed escalation trends.
9. **RESTful Web Service**: Expose clean Flask API endpoints (`/api/classify`, `/api/predict_track`, `/api/risk_assessment`) for third-party disaster management tool integration.
10. **Offline & Edge Capability**: Optimize the model runtime to execute on standard CPU laptops without requiring external cloud GPUs during field deployment.

---

## 5. System Architecture

The following diagram illustrates the complete dataflow and functional components of the system:

```
+-----------------------------------------------------------------------------------+
|                                  DATA INGESTION                                   |
|  +-------------------------------------+   +------------------------------------+ |
|  |     Satellite Imagery (Kaggle)      |   |   NOAA IBTrACS Historical CSVs     | |
|  |    (Visible / Infrared / Water)     |   | (Lat, Lon, Pressure, Wind, Time)   | |
|  +------------------+------------------+   +-----------------+------------------+ |
+---------------------|----------------------------------------|--------------------+
                      v                                        v
+-----------------------------------------------------------------------------------+
|                             PREPROCESSING PIPELINE                                |
|  +-------------------------------------+   +------------------------------------+ |
|  |   preprocess_images.py              |   |   preprocess_ibtracs.py            | |
|  |   - Resize (224x224x3)              |   |   - Filter North Indian (NI) Basin | |
|  |   - MinMax Normalization [0, 1]     |   |   - Interpolate Missing Readings   | |
|  |   - Data Augmentation & Cleaning    |   |   - Sliding Window Sequence Slicing| |
|  +------------------+------------------+   +-----------------+------------------+ |
+---------------------|----------------------------------------|--------------------+
                      v                                        v
+-----------------------------------------------------------------------------------+
|                               AI ENGINE & MODELS                                  |
|  +-------------------------------------+   +------------------------------------+ |
|  |     CNN Classifier (MobileNetV2)    |   |         LSTM Sequence Model        | |
|  |  Input: 224x224 Satellite Image     |   |  Input: Past 4-8 timesteps [t-n..t]| |
|  |  Output: Cyclone Class & Confidence |   |  Output: [Lat, Lon, Wind, Pres]t+1 | |
|  |  Weights: models/cyclone_cnn.keras  |   |  Weights: models/track_lstm.keras  | |
|  +------------------+------------------+   +-----------------+------------------+ |
+---------------------|----------------------------------------|--------------------+
                      \                                        /
                       \                                      /
                        v                                    v
+-----------------------------------------------------------------------------------+
|                            COASTAL RISK ENGINE                                    |
|   - Wind Speed Severity (35%)      - Central Pressure Deficit (25%)               |
|   - Distance to Coastline (25%)    - Forward Translation Speed (15%)              |
|   Output: Composite Risk Score (0-100) -> Alert Level: GREEN / YELLOW / ORANGE / RED |
+-------------------------------------+---------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                              FLASK REST API & BACKEND                             |
|             routes: /api/status | /api/classify | /api/predict_track              |
|                     /api/risk_assessment | /api/historical_storms                 |
|             Fallback: Automatic Synthetic Inferences if models absent             |
+-------------------------------------+---------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|                           WEB PRESENTATION DASHBOARD                              |
|  +--------------------+  +----------------------+  +----------------------------+ |
|  |   Leaflet.js Map   |  |   Chart.js Trends    |  |   Bootstrap 5 Command UI   | |
|  | - Storm Coordinates|  | - Pressure profile   |  | - Image drag & drop upload | |
|  | - Forecast track   |  | - Wind speed profile |  | - Risk alert badges        | |
|  | - Uncertainty Cone |  | - Historical overlay |  | - Actionable advisories    | |
|  +--------------------+  +----------------------+  +----------------------------+ |
+-----------------------------------------------------------------------------------+
```

---

## 6. Technology Stack

| Layer | Technologies | Details & Purpose |
|---|---|---|
| **Programming Language** | Python 3.12.1 | Core computational language for modeling and backend |
| **Backend Framework** | Flask 3.0.0, Werkzeug | Lightweight RESTful microservice and HTTP server |
| **Deep Learning** | TensorFlow 2.15.0, Keras | MobileNetV2 transfer learning and Multi-Layer LSTM |
| **Data Science & ML** | Scikit-learn 1.3.2, NumPy 1.26.2, Pandas 2.1.4 | Matrix processing, scaling, CSV analytics |
| **Image Processing** | OpenCV (headless 4.8.1.78), Pillow 10.1.0 | Image resizing, color space conversion, corrupt image checks |
| **Visualization Backend** | Matplotlib 3.8.2, Seaborn 0.13.0 | Offline metric evaluation and loss curve generation |
| **Frontend UI** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3 | Responsive modern disaster management dashboard |
| **Geospatial Mapping** | Leaflet.js 1.9.4, OpenStreetMap Tiles | Interactive mapping, track polyline rendering, uncertainty cones |
| **Client-Side Charts** | Chart.js 4.4 | Central pressure depletion and wind speed forecasting charts |
| **Automation** | Windows Batch Scripts (`setup.bat`, `run.bat`) | One-click installation and server execution |

---

## 7. Dataset Sources

1. **Satellite Imagery**:
   - **Name**: The Cyclone Image Dataset / Tropical Cyclone Satellite Imagery.
   - **Source**: Kaggle ([Search Cyclone Satellite Image](https://www.kaggle.com/datasets/search?q=cyclone+satellite+image)).
   - **Description**: Thousands of visible and infrared satellite snapshots of tropical storms, categorized into storm and non-storm classes or IMD intensity stages.

2. **Historical Best-Track Data**:
   - **Name**: NOAA IBTrACS (International Best Track Archive for Climate Stewardship) v04.
   - **Source**: NOAA National Centers for Environmental Information ([NOAA IBTrACS Access](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/)).
   - **File**: `ibtracs.NI.list.v04r00.csv` (North Indian Ocean basin covering Bay of Bengal and Arabian Sea from 1848 to present).

---

## 8. Dataset Placement Instructions

Place all downloaded datasets inside the `data/` directory according to the following structure:

```
cyclone_ai_system/
└── data/
    ├── README.md
    ├── satellite/
    │   ├── Cyclone/            <-- Place cyclone satellite images here
    │   │   ├── cyclone_001.jpg
    │   │   └── ...
    │   └── Non-Cyclone/        <-- Place non-cyclone images here
    │       ├── clear_001.jpg
    │       └── ...
    └── ibtracs/
        └── ibtracs.NI.list.v04r00.csv   <-- Place NOAA IBTrACS CSV here
```

> **Note**: For full details and alternative download links, consult [data/README.md](file:///c:/Users/Balaganesh/OneDrive/Desktop/SIH/cyclone_ai_system/data/README.md).

---

## 9. Environment Setup

### Method 1: Automated Setup (Recommended)
Double-click `setup.bat` or execute it from the Windows Command Prompt/PowerShell:

```cmd
setup.bat
```

This automated script will:
1. Create a clean Python virtual environment (`venv`).
2. Activate the virtual environment.
3. Upgrade pip and install all pinned packages from `requirements.txt`.
4. Create required directories (`data/`, `models/`, `uploads/`, `results/`).

### Method 2: Manual Installation
If you prefer manual control:

```powershell
# Navigate to project directory
cd c:\Users\Balaganesh\OneDrive\Desktop\SIH\cyclone_ai_system

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary folders
mkdir data\satellite, data\ibtracs, data\processed, models, uploads, results\plots, results\metrics, results\predictions -Force
```

---

## 10. Training Commands

> [!NOTE]
> Training is **optional**. If you skip training, the system will seamlessly run in **Demo Mode** using built-in synthetic inference.

To train the models on your downloaded datasets:

### Train All Models Sequentially
```powershell
venv\Scripts\activate
python train_all.py
```

### Train CNN Model Only
```powershell
python train_cnn.py
```
*Preprocesses images in `data/satellite/`, builds the MobileNetV2 transfer learning model, saves weights to `models/cyclone_cnn.keras`, and outputs evaluation curves to `results/plots/`.*

### Train LSTM Track Model Only
```powershell
python train_lstm.py
```
*Parses `data/ibtracs/ibtracs.NI.list.v04r00.csv`, builds normalized temporal sequence arrays, trains the 2-layer LSTM model, saves weights to `models/track_lstm.keras`, and generates track error metrics in `results/metrics/`.*

---

## 11. Running Commands

### Start the Application
Double-click `run.bat` or run:

```powershell
venv\Scripts\activate
python app.py
```

The Flask server will start on:
```
http://127.0.0.1:5000/
```

Open your web browser (Chrome, Edge, Firefox) and navigate to `http://127.0.0.1:5000/` to access the dashboard.

---

## 12. Dashboard Usage Guide

1. **System Overview Panel**:
   - The top banner shows operational status (Normal, Active Cyclone Watch, or Demo Mode).
   - High-level metric cards display Active Cyclones, Highest Current Wind Speed, Minimum Central Pressure, and Prevailing Alert Level.

2. **Satellite Image Classifier**:
   - Navigate to the **Satellite Analysis** tab.
   - Drag and drop a satellite snapshot (JPEG/PNG) or click to browse.
   - Click **Classify Image**.
   - Within seconds, the dashboard displays:
     - Identification: Cyclone Detected or Normal Weather.
     - IMD Intensity Class: (e.g., *Very Severe Cyclonic Storm*).
     - Classification Confidence percentage (e.g., *94.8%*).
     - Estimated maximum sustained wind speed (knots / km/h).

3. **Geospatial Track & Trajectory Map**:
   - Navigate to the **Live Track & Prediction** view.
   - The Leaflet map displays:
     - **Blue Markers & Line**: Historical path of the storm (past 24–48 hours).
     - **Pulsing Red Marker**: Current estimated eye location.
     - **Dashed Red Line**: AI-predicted trajectory for +12h, +24h, +36h, +48h.
     - **Translucent Shaded Polygon**: Expanding **Cone of Uncertainty** showing potential track dispersion.
     - **Yellow/Red Rings**: Coastal alert zones within the landfall radius.

4. **Atmospheric Trend Analytics**:
   - View dynamic Chart.js charts:
     - **Pressure Trend Chart**: Barometric pressure drops (hPa) indicating storm intensification.
     - **Wind Velocity Chart**: Forecasted sustained wind speeds over the next 48 hours.

5. **Early Warning & Advisory Panel**:
   - View calculated **Risk Score (0–100)** and color-coded alert badge (Green/Yellow/Orange/Red).
   - Read automated emergency advisories recommending specific disaster mitigation actions:
     - Fishermen warning to return to harbor.
     - Coastal low-lying area evacuation alerts.
     - Power and telecommunication disruption preparations.
   - Click **Export Advisory Bulletin** to download or print an emergency bulletin.

---

## 13. API Documentation

The system provides RESTful JSON endpoints for disaster management software integrations:

### 1. System Health & Status
- **Route**: `GET /api/status`
- **Description**: Returns server status, mode (live vs demo), and model availability.
- **Sample Response**:
  ```json
  {
    "status": "online",
    "mode": "demo",
    "cnn_model_loaded": true,
    "lstm_model_loaded": true,
    "timestamp": "2026-09-03T20:56:40Z",
    "disclaimer": "Prototype AI prediction only. Not an official meteorological warning system."
  }
  ```

### 2. Classify Satellite Image
- **Route**: `POST /api/classify`
- **Description**: Upload a satellite image for CNN classification.
- **Form Data**: `file` (image/jpeg, image/png)
- **Sample Response**:
  ```json
  {
    "success": true,
    "is_cyclone": true,
    "class_name": "Very Severe Cyclonic Storm",
    "confidence": 0.942,
    "estimated_wind_knots": 85,
    "estimated_wind_kmh": 157.4,
    "disclaimer": "Prototype AI prediction only. Not an official meteorological warning system."
  }
  ```

### 3. Predict Cyclone Track
- **Route**: `POST /api/predict_track`
- **Description**: Ingests past sequential coordinates and outputs multi-step forecasted trajectory with uncertainty bounds.
- **Request Body**:
  ```json
  {
    "storm_id": "NI202301",
    "past_points": [
      {"lat": 14.2, "lon": 85.0, "wind_knots": 50, "pressure_hpa": 992},
      {"lat": 14.9, "lon": 85.4, "wind_knots": 60, "pressure_hpa": 984},
      {"lat": 15.6, "lon": 85.9, "wind_knots": 75, "pressure_hpa": 974},
      {"lat": 16.4, "lon": 86.3, "wind_knots": 85, "pressure_hpa": 962}
    ]
  }
  ```
- **Sample Response**:
  ```json
  {
    "success": true,
    "forecast": [
      {"step_hours": 12, "lat": 17.3, "lon": 86.8, "wind_knots": 90, "pressure_hpa": 954, "uncertainty_radius_km": 45},
      {"step_hours": 24, "lat": 18.2, "lon": 87.2, "wind_knots": 95, "pressure_hpa": 948, "uncertainty_radius_km": 85},
      {"step_hours": 36, "lat": 19.1, "lon": 87.5, "wind_knots": 85, "pressure_hpa": 958, "uncertainty_radius_km": 130},
      {"step_hours": 48, "lat": 20.0, "lon": 87.7, "wind_knots": 70, "pressure_hpa": 970, "uncertainty_radius_km": 180}
    ],
    "estimated_landfall": {
      "predicted": true,
      "region": "Odisha - West Bengal Coast",
      "eta_hours": 42
    },
    "disclaimer": "Prototype AI prediction only. Not an official meteorological warning system."
  }
  ```

### 4. Coastal Risk Assessment
- **Route**: `POST /api/risk_assessment`
- **Description**: Computes composite risk index and alert tier.
- **Request Body**:
  ```json
  {
    "wind_speed_kmh": 160,
    "pressure_hpa": 950,
    "distance_to_coast_km": 110,
    "forward_speed_kmh": 18
  }
  ```
- **Sample Response**:
  ```json
  {
    "success": true,
    "risk_score": 84.5,
    "alert_level": "RED",
    "threat_category": "Extremely Severe Coastal Threat",
    "advisories": [
      "Immediate suspension of all fishing and maritime activities.",
      "Initiate mandatory evacuation for vulnerable coastal settlements within 50 km.",
      "Activate emergency power backups and storm surge shelters."
    ],
    "disclaimer": "Prototype AI prediction only. Not an official meteorological warning system."
  }
  ```

### 5. Historical Storms Directory
- **Route**: `GET /api/historical_storms`
- **Description**: Returns list of prominent historical North Indian Ocean cyclones available for demonstration replay (e.g., FANI, AMPHAN, BIPARJOY, HUDHUD).

---

## 14. CNN Architecture (Identification & Intensity)

The vision component utilizes **MobileNetV2** pre-trained on ImageNet as a transfer learning feature extractor:

```
[Input Image: 224 x 224 x 3]
           |
[MobileNetV2 Base (Pre-trained ImageNet, Frozen)]
  - Depthwise Separable Convolutions
  - Inverted Residual Blocks (Bottlenecks)
  - Output Feature Maps: 7 x 7 x 1280
           |
[GlobalAveragePooling2D] -> Vector of 1280 features
           |
[BatchNormalization]
           |
[Dense Layer: 128 units, ReLU activation]
           |
[Dropout: 0.30] (Regularization to prevent overfitting)
           |
[Output Dense: Softmax Activation (C classes)]
```

### Why MobileNetV2?
- **Lightweight Model Size**: Model weight footprint is ~14 MB, fitting easily in memory.
- **Fast CPU Inference**: Predicts within 60–100 ms on a standard laptop CPU without dedicated GPU acceleration.
- **Low Memory Overhead**: Suitable for field deployment on edge laptops or portable disaster command tablets.
- **Robust Transfer Learning**: Generalizes well to cloud textures, spiral bands, and eye formations.

---

## 15. LSTM Architecture (Track Prediction)

Cyclonic trajectories are dynamic time-series influenced by atmospheric steering currents, Coriolis acceleration, and barometric gradients. We model this with a stacked Long Short-Term Memory (LSTM) network:

```
[Input Sequence: (Timesteps = 4 to 8, Features = 4)]
Features per timestep: [Latitude, Longitude, Wind Speed, Pressure]
                       |
[LSTM Layer 1: 64 Units, return_sequences=True]
                       |
[Dropout: 0.20]
                       |
[LSTM Layer 2: 32 Units, return_sequences=False]
                       |
[Dense Layer: 32 Units, ReLU]
                       |
[Output Dense: 4 Units, Linear Activation]
Predicts: [Latitude(t+1), Longitude(t+1), Wind(t+1), Pressure(t+1)]
```

### Multi-Step Trajectory Rollout
To predict tracks across +12h, +24h, +36h, and +48h, the model uses an **autoregressive rolling sequence generator**:
1. Ingest past steps $[t-3, t-2, t-1, t]$.
2. Predict step $t+1$.
3. Append $t+1$ to sequence, drop $t-3$, and re-predict for $t+2$.
4. Calculate expanding uncertainty radius:
   $$R_t = R_0 + k \cdot t$$
   where $R_0 \approx 35\text{ km}$ and $k \approx 3.0\text{ km/hour}$, creating the Cone of Uncertainty.

---

## 16. Risk Assessment Methodology

The coastal danger level is computed via a multi-factor rule-based algorithm:

### Multi-Factor Weights
$$\text{Risk Score} = (0.35 \times S_{\text{wind}}) + (0.25 \times S_{\text{pressure}}) + (0.25 \times S_{\text{coast}}) + (0.15 \times S_{\text{speed}})$$

1. **Wind Severity Score ($S_{\text{wind}}$)**:
   - Wind $< 62\text{ km/h}$: Low ($0 - 30$)
   - Wind $62 - 118\text{ km/h}$ (Cyclonic Storm): Moderate ($31 - 65$)
   - Wind $> 118\text{ km/h}$ (Severe Cyclone): Severe ($66 - 100$)
2. **Pressure Deficit Score ($S_{\text{pressure}}$)**:
   - Evaluated as central pressure drops below standard ambient pressure ($1013\text{ hPa}$).
   - Deeper drops ($< 960\text{ hPa}$) indicate violent storm surges and higher threat.
3. **Coastal Proximity Score ($S_{\text{coast}}$)**:
   - Distance $> 300\text{ km}$: Threat is minimal ($0 - 20$).
   - Distance $100 - 300\text{ km}$: Threat elevated ($21 - 60$).
   - Distance $< 100\text{ km}$: Immediate landfall danger ($61 - 100$).
4. **Translation Speed Factor ($S_{\text{speed}}$)**:
   - Very slow-moving cyclones ($< 10\text{ km/h}$) prolong rainfall and surge damage over coastal districts.

### Alert Classification Tiers

| Composite Score | Alert Tier | Color | Operational Meaning & Directives |
|---|---|---|---|
| **0 – 39** | **GREEN** | Green | **Watch / Normal**: Low threat. Routine monitoring. Maritime advisories. |
| **40 – 59** | **YELLOW** | Yellow | **Advisory / Alert**: Developing threat. Coast guard and ports placed on standby. |
| **60 – 79** | **ORANGE** | Orange | **Warning**: Severe threat within 24–36 hrs. Evacuation preparations active. |
| **80 – 100** | **RED** | Red | **Evacuation / Danger**: Imminent catastrophic landfall. Mandatory coastal evacuation. |

### IMD Intensity Scale Reference
- **Depression (D)**: Wind $31 - 49\text{ km/h}$
- **Deep Depression (DD)**: Wind $50 - 61\text{ km/h}$
- **Cyclonic Storm (CS)**: Wind $62 - 88\text{ km/h}$
- **Severe Cyclonic Storm (SCS)**: Wind $89 - 117\text{ km/h}$
- **Very Severe Cyclonic Storm (VSCS)**: Wind $118 - 166\text{ km/h}$
- **Extremely Severe Cyclonic Storm (ESCS)**: Wind $167 - 221\text{ km/h}$
- **Super Cyclonic Storm (SuCS)**: Wind $\ge 222\text{ km/h}$

---

## 17. Evaluation Metrics

The system's machine learning components are evaluated using standard quantitative metrics:

### CNN Classification Metrics
- **Accuracy**: Fraction of correct cyclone presence and intensity determinations.
- **Precision, Recall, & F1-Score**: Weighted multi-class evaluation to ensure rare super-cyclone events are not misclassified as low-grade storms.
- **Confusion Matrix**: Heatmap visualization tracking category cross-talk.

### LSTM Track Metrics
- **Mean Absolute Error (MAE)**: Measured separately in degrees latitude and longitude.
- **Haversine Distance Error**: Great-circle geographical displacement in kilometers:
  $$d = 2r \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
- **24-Hour Track Error Target**: $< 120\text{ km}$ on historical validation test sets.
- **Root Mean Squared Error (RMSE)**: On barometric pressure and wind velocity estimations.

---

## 18. System Limitations

To maintain scientific integrity and realistic expectations, we explicitly document the following 13 system limitations:

1. **2D Surface Limitation**: Does not model vertical atmospheric wind shear, vertical vorticity, or 3D tropospheric moisture flux.
2. **Absence of Ocean Heat Content (OHC)**: Track and intensity do not incorporate real-time bathymetry or sea surface temperature (SST) anomalies.
3. **Satellite Cloud Top Occlusion**: Visible and infrared imagery can only view cloud tops; eye wall dynamics obscured by cirrus canopies may degrade classification.
4. **Resolution Variance**: Satellite images from disparate sources (INSAT, GOES, Himawari) exhibit varying spatial and radiometric resolutions.
5. **Class Imbalance in Historical Data**: Extreme Super Cyclones (e.g., 1999 Odisha) are rare in the historical record compared to frequent minor depressions.
6. **Simplified Landfall Decay**: The prototype uses generalized empirical pressure recovery rather than explicit topography-driven friction models upon landfall.
7. **Storm Surge Physics**: Storm surge height is estimated via heuristic proxy rather than full 2D hydrodynamic numerical solvers (such as SLOSH or ADCIRC).
8. **Temporal Sampling Discrepancies**: NOAA IBTrACS data is sampled at 3-hour or 6-hour intervals; rapid micro-scale course deviations between intervals are smoothed out.
9. **Single-Station Sensor Offline Tolerance**: If local weather radar fails, the system relies exclusively on orbital satellite imagery.
10. **Hardware Resource Constraints**: Training complex deep networks on consumer laptops is constrained by RAM and CPU thermals.
11. **Internet Dependency for GIS Base Maps**: Rendering dynamic OpenStreetMap tiles in Leaflet requires internet access (unless offline mbtiles are pre-cached).
12. **Rule-Based Risk Boundary Conditions**: The coastal risk index is rule-based and does not yet account for local micro-infrastructure (e.g., concrete vs. thatched housing).
13. **Prototype Advisory Limitation**: Not connected directly to the Global Telecommunication System (GTS) of WMO.

---

## 19. Future Improvements

Planned future iterations for this platform include:

1. **Automated INSAT-3D/3DR API Feeds**: Direct real-time polling of ISRO MOSDAC satellite feeds every 15 minutes.
2. **Physics-Informed Neural Networks (PINNs)**: Incorporating atmospheric Navier-Stokes fluid dynamic constraints into the loss function.
3. **Multi-Modal Transformer Architecture**: Fusing satellite imagery and sequential track data into a single vision-temporal Transformer.
4. **Hydrodynamic Storm Surge Integration**: Coupling the forecast tracks directly with hydrodynamic models to predict coastal inundation down to street level.
5. **Automated Multi-Lingual SMS/IVR Alert Broadcasting**: Sending automated localized voice/SMS alerts in Odia, Bengali, Tamil, Telugu, and Hindi to coastal populations.
6. **Drone/UAV Post-Landfall Damage Mapping**: Ingesting aerial drone photos post-cyclone for automated infrastructure damage tagging.
7. **Edge Hardware Deployment**: Packaging the inference pipeline as an offline container for Raspberry Pi 5 or NVIDIA Jetson edge nodes in remote shelters.
8. **Ensemble Numerical Assimilation**: Blending AI predictions with GFS and ECMWF numerical model outputs for hybrid ensemble forecasting.

---

## 20. SIH Demo Procedure (2-Minute Script)

Follow this 10-step script during your Smart India Hackathon presentation for maximum impact:

| Time | Step | Action & Narration | Screen / Feature |
|---|---|---|---|
| **0:00 - 0:15** | **1. Welcome & Dashboard Launch** | Open `http://127.0.0.1:5000/`. Highlight the clean interface, active status indicator, and real-time metric cards. | Home Overview Panel |
| **0:15 - 0:30** | **2. Problem Statement** | Mention that Indian coastal states face lethal cyclones and traditional NWP models take hours to run on supercomputers. Introduce our instant AI system. | Problem Summary Card |
| **0:30 - 0:45** | **3. Satellite Image Upload** | Click on the **Satellite Analysis** tab. Drag and drop a sample cyclone satellite image from `data/sample_images/` or desktop. | Upload Modal |
| **0:45 - 0:55** | **4. Instant CNN Classification** | Click **Classify**. Show sub-second inference: *“Cyclone Detected: Very Severe Cyclonic Storm (Confidence: 94.2%)”*. | CNN Results Card |
| **0:55 - 1:10** | **5. Trajectory & Track Prediction** | Switch to the **Live Track Map**. Point out past historical points (blue) and AI predicted points (dashed red) moving toward the coast. | Leaflet GIS Map |
| **1:10 - 1:25** | **6. Cone of Uncertainty** | Point out the shaded translucent polygon expanding along the path. Explain how uncertainty grows over +12h, +24h, +36h, +48h. | Spatial Uncertainty Cone |
| **1:25 - 1:40** | **7. Coastal Threat & Alert Meter** | Show the dynamic Risk Score (e.g., *84.5 - RED ALERT*). Explain the multi-factor scoring (wind, pressure, distance to shore). | Risk Engine Gauge |
| **1:40 - 1:50** | **8. Atmospheric Trend Charts** | Scroll down to the Chart.js visualizers. Show the rapid pressure drop curve and sustained wind velocity escalation. | Trend Analytics Graphs |
| **1:50 - 1:55** | **9. Automated Advisory Generation** | Click **Generate Advisory Report**. Display the automated directives for fishermen, coastal evacuations, and port closures. | Advisory Modal |
| **1:55 - 2:00** | **10. Concluding Statement** | Emphasize speed, lightweight CPU execution, and the mandatory AI Prototype Disclaimer. Conclude on time! | Final Summary & Disclaimer |

---

## 21. Disclaimer

> **IMPORTANT NOTICE**:  
> This software is an experimental prototype developed solely for educational, research, and hackathon evaluation purposes as part of the Smart India Hackathon. It is **NOT** certified or approved as an operational meteorological early warning system.
>
> In any real-world cyclone scenario, all emergency response personnel, government bodies, and citizens must strictly heed the official warnings, tracks, and bulletins issued by the **India Meteorological Department (IMD)**, **National Disaster Management Authority (NDMA)**, and relevant state disaster authorities. The developers and contributors assume no legal liability for actions taken based on inferences generated by this software.

---
*Built with passion for disaster resilience and public safety.*
