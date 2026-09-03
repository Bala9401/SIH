import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Demo Mode is enabled until both trained model artifacts are available.
DEMO_MODE = not all((MODEL_DIR / name).exists() for name in (
    "cyclone_cnn.keras", "cyclone_lstm.keras", "scaler.pkl"
))

# CNN Config
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 2 # Placeholder, will be auto-detected

# LSTM Config
SEQUENCE_LENGTH = 6
LSTM_UNITS = 64
LSTM_EPOCHS = 50

# Risk thresholds (wind speed in knots)
RISK_THRESHOLDS = {
    'LOW': 34,
    'MODERATE': 47,
    'HIGH': 63,
    'VERY_HIGH': 90
}

# Demo cyclone data (Cyclone Fani 2019 - North Indian Ocean)
DEMO_CYCLONE_DATA = [
    {"lat": 5.2, "lon": 88.5, "wind": 25, "pressure": 1004, "time": "2019-04-26T00:00:00Z"},
    {"lat": 5.7, "lon": 88.3, "wind": 30, "pressure": 1000, "time": "2019-04-26T06:00:00Z"},
    {"lat": 6.2, "lon": 88.1, "wind": 35, "pressure": 998, "time": "2019-04-26T12:00:00Z"},
    {"lat": 6.9, "lon": 87.8, "wind": 45, "pressure": 994, "time": "2019-04-27T00:00:00Z"},
    {"lat": 7.9, "lon": 87.5, "wind": 55, "pressure": 990, "time": "2019-04-27T12:00:00Z"},
    {"lat": 8.7, "lon": 87.2, "wind": 65, "pressure": 986, "time": "2019-04-28T00:00:00Z"},
    {"lat": 10.3, "lon": 86.6, "wind": 80, "pressure": 978, "time": "2019-04-29T00:00:00Z"},
    {"lat": 11.9, "lon": 86.1, "wind": 95, "pressure": 970, "time": "2019-04-30T00:00:00Z"},
    {"lat": 13.9, "lon": 85.6, "wind": 105, "pressure": 962, "time": "2019-05-01T00:00:00Z"},
    {"lat": 15.6, "lon": 85.1, "wind": 115, "pressure": 954, "time": "2019-05-02T00:00:00Z"},
    {"lat": 17.6, "lon": 84.8, "wind": 115, "pressure": 946, "time": "2019-05-02T12:00:00Z"},
    {"lat": 19.6, "lon": 85.7, "wind": 100, "pressure": 962, "time": "2019-05-03T03:00:00Z"}, # Landfall
]

# Flask Config
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

DISCLAIMER_TEXT = "Prototype AI prediction only. Not an official meteorological warning system."
DISCLAIMER = DISCLAIMER_TEXT
CLASS_NAMES = ["No_Cyclone", "Cyclone"] # Placeholder
