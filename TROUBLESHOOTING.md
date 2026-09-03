# Troubleshooting & FAQ Guide - AI Cyclone Early Warning System

> **Disclaimer**: Prototype AI prediction only. Not an official meteorological warning system.

This guide provides immediate, step-by-step solutions for common issues, runtime warnings, environment mismatches, and configuration adjustments encountered when setting up, training, or running the AI Cyclone Early Warning System.

---

## Quick Diagnostic Checklist
Before troubleshooting specific errors, run through this quick 3-point check:
1. **Virtual Environment Active?** Your terminal prompt should start with `(venv)`. If not, run `venv\Scripts\activate`.
2. **Dependencies Installed?** Run `pip list` and ensure packages like `flask`, `tensorflow`, `numpy`, and `pandas` are listed.
3. **Correct Directory?** Ensure your working directory is `c:\Users\Balaganesh\OneDrive\Desktop\SIH\cyclone_ai_system\`.

---

## Table of Problems & Solutions

| Category | Issue / Symptom | Quick Solution Summary |
|---|---|---|
| **Python & Packages** | [1. Python Version Incompatibility](#1-python-version-incompatibility) | Use Python 3.10 – 3.12 (specifically 3.12.1 recommended) |
| **TensorFlow** | [2. TensorFlow Installation Fails](#2-tensorflow-installation-fails) | Install `tensorflow-cpu` or verify pip version |
| **Dependencies** | [3. ModuleNotFoundError / Import Errors](#3-modulenotfounderror--import-errors) | Run `pip install -r requirements.txt` inside active venv |
| **Datasets** | [4. Missing Dataset Error](#4-missing-dataset-error) | Use built-in DEMO MODE or download as per data/README.md |
| **Data Ingestion** | [5. Wrong CSV Columns in IBTrACS](#5-wrong-csv-columns-in-ibtracs) | Pipeline auto-adapts; ensure North Indian basin CSV is used |
| **Image Pipeline** | [6. Corrupted or Unreadable Images](#6-corrupted-or-unreadable-images) | Preprocessing automatically filters bad images |
| **Models** | [7. Model File Not Found (.keras)](#7-model-file-not-found-keras) | Fallback to DEMO MODE or run `python train_all.py` |
| **Server & Networking**| [8. Port 5000 Already in Use](#8-port-5000-already-in-use) | Change port in `config.py` or kill conflicting process |
| **Flask Server** | [9. Flask Won't Start](#9-flask-wont-start) | Check venv activation and Python traceback output |
| **Hardware & RAM** | [10. Out of Memory (OOM) Errors](#10-out-of-memory-oom-errors) | Reduce `BATCH_SIZE` in `config.py` from 32 to 16 or 8 |
| **Performance** | [11. Slow Model Training](#11-slow-model-training) | Reduce `EPOCHS` in `config.py` or enable transfer layer freezing |
| **Web UI** | [12. Map Not Loading (Blank Grey Map)](#12-map-not-loading-blank-grey-map) | Check internet connection for OpenStreetMap tile fetching |
| **Web UI** | [13. Charts Not Showing](#13-charts-not-showing) | Check browser JavaScript console (F12) for syntax or CDN issues |
| **File Uploads** | [14. Image Upload Fails / HTTP 500](#14-image-upload-fails--http-500) | Verify the `uploads/` directory exists and has write permissions |

---

## Detailed Solutions

### 1. Python Version Incompatibility
- **Symptom**: `ERROR: Could not find a version that satisfies the requirement tensorflow` or syntax errors during module initialization.
- **Cause**: TensorFlow 2.15.0 requires Python 3.9 through 3.12. Older versions (e.g., Python 3.8) or very new experimental releases (e.g., Python 3.13) may lack pre-built wheels on Windows.
- **Solution**:
  1. Check your Python version:
     ```powershell
     python --version
     ```
     Target version is **Python 3.12.1**.
  2. If you have multiple Python versions installed, invoke the specific launcher:
     ```powershell
     py -3.12 -m venv venv
     venv\Scripts\activate
     pip install -r requirements.txt
     ```

---

### 2. TensorFlow Installation Fails
- **Symptom**: `pip install tensorflow` freezes, fails with C++ build errors, or complains about missing wheel files.
- **Cause**: Large download timeouts, missing Microsoft Visual C++ Redistributable, or GPU driver conflicts on Windows.
- **Solution**:
  1. Upgrade pip first:
     ```powershell
     python -m pip install --upgrade pip
     ```
  2. Install the lightweight CPU-only build:
     ```powershell
     pip install tensorflow-cpu==2.15.0
     ```
  3. Ensure the [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) is installed on your Windows machine.

---

### 3. ModuleNotFoundError / Import Errors
- **Symptom**: `ModuleNotFoundError: No module named 'flask'` or `No module named 'PIL'`.
- **Cause**: Packages were installed into the global Python interpreter rather than the virtual environment, or the virtual environment was not activated.
- **Solution**:
  1. Activate your virtual environment:
     ```cmd
     call venv\Scripts\activate.bat
     ```
     *(In PowerShell: `.\venv\Scripts\Activate.ps1`)*
  2. If PowerShell throws an execution policy error:
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```
  3. Reinstall dependencies:
     ```powershell
     pip install -r requirements.txt
     ```

---

### 4. Missing Dataset Error
- **Symptom**: `FileNotFoundError: data/satellite directory empty` or `ibtracs.csv not found`.
- **Cause**: You haven't downloaded Kaggle satellite imagery or NOAA IBTrACS data yet.
- **Solution**:
  - **Option A (Instant Hackathon Demo)**: You do **not** need the full dataset to run the project. The system has built-in **DEMO MODE**. Launch the application directly:
    ```powershell
    python app.py
    ```
    The application will automatically detect missing training data and activate synthetic fallback streams for demonstration.
  - **Option B (Full Training)**: Follow the instructions in [data/README.md](file:///c:/Users/Balaganesh/OneDrive/Desktop/SIH/cyclone_ai_system/data/README.md) to place Kaggle images in `data/satellite/` and NOAA CSV in `data/ibtracs/`.

---

### 5. Wrong CSV Columns in IBTrACS
- **Symptom**: `KeyError: 'USA_PRES'` or missing latitude/longitude columns when processing historical data.
- **Cause**: NOAA IBTrACS CSVs sometimes have different column headers across versions (e.g., `WMO_PRES` vs `USA_PRES`, or header comment lines starting with `#` or units in row 2).
- **Solution**:
  - The preprocessing module (`preprocess_ibtracs.py`) is designed with auto-adaptive column discovery. It automatically inspects the CSV header and falls back through candidate column names:
    - Latitude: `['LAT', 'latitude', 'lat']`
    - Longitude: `['LON', 'longitude', 'lon']`
    - Pressure: `['WMO_PRES', 'USA_PRES', 'PRES', 'pressure']`
    - Wind: `['WMO_WIND', 'USA_WIND', 'WIND', 'wind_speed']`
  - If using custom CSVs, ensure your file contains these basic columns or run:
    ```powershell
    python -c "import pandas as pd; df=pd.read_csv('data/ibtracs/ibtracs.NI.list.v04r00.csv', nrows=5); print(df.columns.tolist()[:15])"
    ```

---

### 6. Corrupted or Unreadable Images
- **Symptom**: `PIL.UnidentifiedImageError: cannot identify image file` or `cv2.error: empty image buffer`.
- **Cause**: Kaggle image downloads occasionally contain 0-byte truncated files or unsupported formats disguised as `.jpg`.
- **Solution**:
  - The preprocessing script (`preprocess_images.py`) wraps image ingestion in a `try...except` block:
    ```python
    try:
        with Image.open(img_path) as img:
            img.verify() # Validates integrity
    except Exception:
        print(f"Skipping corrupted image: {img_path}")
    ```
  - Corrupt files are automatically flagged and skipped without terminating the training pipeline.

---

### 7. Model File Not Found (.keras)
- **Symptom**: `Warning: models/cyclone_cnn.keras not found. Starting in DEMO MODE.`
- **Cause**: You ran `app.py` before running `train_all.py`, or weights have not been exported yet.
- **Solution**:
  - This is an **intended feature**, not a fatal crash. The dashboard will function seamlessly in **Demo Mode** using realistic sample inferences.
  - To generate local trained model weights:
    ```powershell
    python train_all.py
    ```
    This generates `models/cyclone_cnn.keras` and `models/track_lstm.keras`.

---

### 8. Port 5000 Already in Use
- **Symptom**: `OSError: [Errno 10048] Only one usage of each socket address (protocol/network address/port) is normally permitted` or `Address already in use`.
- **Cause**: Another background process (AirPlay, prior Flask instance, or local development server) is occupying port 5000.
- **Solution**:
  - **Method 1: Change Port in `config.py` or command line**:
    Edit `app.py` or run:
    ```powershell
    python app.py --port 5050
    ```
    Then browse to `http://127.0.0.1:5050/`.
  - **Method 2: Kill Existing Process on Port 5000 in Windows**:
    ```powershell
    # Find process ID (PID)
    netstat -ano | findstr :5000
    # Kill the process (replace <PID> with number from last column)
    taskkill /PID <PID> /F
    ```

---

### 9. Flask Won't Start
- **Symptom**: Running `python app.py` terminates immediately with code 1 and no web page loads.
- **Cause**: Syntax error, unhandled exception in import block, or misconfigured environment variables.
- **Solution**:
  1. Run `python app.py` directly in PowerShell (not double-clicking `.bat`) to keep the terminal window open and inspect the traceback:
     ```powershell
     python app.py
     ```
  2. Verify your Flask installation:
     ```powershell
     python -c "import flask; print(flask.__version__)"
     ```
     Should print `3.0.0`.

---

### 10. Out of Memory (OOM) Errors
- **Symptom**: `ResourceExhaustedError: OOM when allocating tensor` or Python process unexpectedly killed.
- **Cause**: Machine RAM or CPU memory saturated by large image batches during CNN training.
- **Solution**:
  1. Open `config.py`.
  2. Reduce the `BATCH_SIZE` setting:
     ```python
     # Default: BATCH_SIZE = 32
     BATCH_SIZE = 16  # Or 8 for laptops with 8GB RAM
     ```
  3. Ensure other heavy applications (browser tabs, video games) are closed during training.

---

### 11. Slow Model Training
- **Symptom**: Epoch 1/20 takes > 15 minutes on a laptop CPU.
- **Cause**: Large dataset size, un-quantized image processing, or running without transfer layer caching.
- **Solution**:
  1. In `config.py`, reduce `EPOCHS`:
     ```python
     CNN_EPOCHS = 5      # Quick training for demonstration
     LSTM_EPOCHS = 10
     ```
  2. Verify that MobileNetV2 base layers remain frozen (`layer.trainable = False`), ensuring only the top classification head weights are updated during training.
  3. Limit the satellite training subset to 1,000–2,000 images for fast convergence.

---

### 12. Map Not Loading (Blank Grey Map)
- **Symptom**: The Leaflet map container displays a solid grey canvas with zoom buttons, but no geographical continents or coastlines.
- **Cause**: OpenStreetMap tile layer requests require active internet access. If the laptop is offline in a hackathon venue without Wi-Fi, tile images cannot download.
- **Solution**:
  1. Connect to Wi-Fi or mobile hotspot.
  2. Ensure your firewall or proxy isn't blocking outgoing requests to `https://*.tile.openstreetmap.org/`.
  3. Alternatively, clear your browser cache and refresh (`Ctrl + F5`).

---

### 13. Charts Not Showing
- **Symptom**: Track line appears on the map, but the barometric pressure and wind velocity charts below remain blank.
- **Cause**: Browser blocked CDN script loading, or JavaScript execution encountered an uncaught error.
- **Solution**:
  1. Open Developer Tools in your browser by pressing `F12`.
  2. Select the **Console** tab and check for red error messages.
  3. If CDN assets for Chart.js are blocked by an offline network, ensure local vendor fallback scripts exist in `static/vendor/` or reconnect to internet.
  4. Verify that data passed to `/api/predict_track` contains valid numeric arrays.

---

### 14. Image Upload Fails / HTTP 500
- **Symptom**: Uploading a satellite image returns `500 Internal Server Error` or `FileNotFoundError: uploads/xyz.jpg`.
- **Cause**: The `uploads/` staging folder does not exist or lacks write permissions.
- **Solution**:
  1. Create the `uploads/` directory manually:
     ```powershell
     mkdir uploads 2>nul
     ```
  2. Ensure the file being uploaded is a valid image extension (`.jpg`, `.jpeg`, `.png`, `.bmp`).
  3. Ensure file size does not exceed `MAX_CONTENT_LENGTH` (default: 16 MB).

---

## Still Encountering Issues?

If your issue is not listed above:
1. Verify the project directory matches: `c:\Users\Balaganesh\OneDrive\Desktop\SIH\cyclone_ai_system\`
2. Check the console logs where `python app.py` is running; all API errors print complete diagnostic tracebacks.
3. Remember that the system is fully capable of running and demonstrating in **DEMO MODE** for presentation purposes even if datasets or models are not yet trained.

---

> **Disclaimer**: Prototype AI prediction only. Not an official meteorological warning system.
