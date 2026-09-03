# Dataset Setup Guide - AI Cyclone Early Warning System

> **Disclaimer**: Prototype AI prediction only. Not an official meteorological warning system.

This guide provides comprehensive, step-by-step instructions for acquiring and placing the datasets required for training and evaluating the AI Cyclone Early Warning System.

---

## ⚡ Important Note: Demo Mode

> [!NOTE]
> **You do NOT need to download datasets to test or demo this application!**
> The system comes with built-in **DEMO MODE**. If no trained models or datasets are detected, the application automatically uses realistic synthetic data and simulated model inferences for demonstration, testing, and Hackathon presentations.

If you wish to train the CNN and LSTM models on real-world historical data, follow the instructions below.

---

## 1. Satellite Imagery Dataset (TheCycloneImageDataset)

The satellite image dataset is used to train the Convolutional Neural Network (CNN - MobileNetV2) for cyclone identification and intensity classification.

### Source Details
- **Platform**: Kaggle
- **Dataset Name**: The Cyclone Image Dataset (or Tropical Cyclone Satellite Images)
- **URL**: [https://www.kaggle.com/datasets/search?q=cyclone+satellite+image](https://www.kaggle.com/datasets/search?q=cyclone+satellite+image) / [https://www.kaggle.com/datasets/kmader/cyclone-wildfire-runway-images](https://www.kaggle.com/datasets/kmader/cyclone-wildfire-runway-images)
- **Primary Alternative**: [Kaggle Tropical Cyclone Intensity Dataset](https://www.kaggle.com/datasets) or NOAA NESDIS satellite imagery archives.

### Download Steps
1. Create a free account or log in at [Kaggle](https://www.kaggle.com).
2. Navigate to the Cyclone Image Dataset page.
3. Click the **Download (ZIP)** button.
4. Extract the contents of the downloaded `.zip` archive.
5. Place the image categories into the project folder at:
   ```
   cyclone_ai_system/data/satellite/
   ```

### Supported Organization Formats
The training pipeline automatically scans `data/satellite/`. You can organize the images in either of the following two standard formats:

**Option A: Subfolder per Class (Recommended)**
```
data/satellite/
├── Cyclone/
│   ├── cyclone_001.jpg
│   ├── cyclone_002.jpg
│   └── ...
└── Non-Cyclone/
    ├── no_cyclone_001.jpg
    ├── no_cyclone_002.jpg
    └── ...
```

**Option B: Intensity Category Folders**
```
data/satellite/
├── Depression/
├── Deep_Depression/
├── Cyclonic_Storm/
├── Severe_Cyclonic_Storm/
├── Very_Severe_Cyclonic_Storm/
└── Super_Cyclone/
```

*Note: The image preprocessor (`preprocess_images.py`) will automatically resize images to 224x224 pixels, normalize pixel intensities to `[0, 1]`, and skip any corrupted files.*

---

## 2. Track & Intensity Historical Dataset (NOAA IBTrACS)

The track forecasting module uses NOAA's International Best Track Archive for Climate Stewardship (IBTrACS) dataset to train the Long Short-Term Memory (LSTM) recurrent network for trajectory (latitude/longitude) and central pressure forecasting.

### Source Details
- **Provider**: National Oceanic and Atmospheric Administration (NOAA) / NCEI
- **Archive Name**: IBTrACS v04 (International Best Track Archive for Climate Stewardship)
- **Official Portal**: [https://www.ncei.noaa.gov/products/international-best-track-archive](https://www.ncei.noaa.gov/products/international-best-track-archive)
- **Direct Data Download Page**: [https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/)

### Recommended File: North Indian Ocean (NI) Basin
Because the Smart India Hackathon project specifically targets the Indian subcontinent (Bay of Bengal and Arabian Sea), download the basin-specific subset:
- **Target File**: `ibtracs.NI.list.v04r00.csv` (North Indian Basin, ~10–25 MB)
- **Alternative (Full Global)**: `ibtracs.ALL.list.v04r00.csv` (~300+ MB, contains all global oceanic basins)

### Download Steps
1. Visit the [NOAA IBTrACS CSV Access Page](https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/).
2. Locate and click `ibtracs.NI.list.v04r00.csv`.
3. Save the downloaded CSV file directly into:
   ```
   cyclone_ai_system/data/ibtracs/
   ```
4. If the downloaded file is named `ibtracs.NI.list.v04r00.csv` or `ibtracs.csv`, our dataset inspector automatically recognizes it.

### Key IBTrACS Columns Used
- `SID`: Unique storm identifier
- `NAME`: Cyclone name (e.g., FANI, AMPHAN, BIPARJOY)
- `ISO_TIME`: UTC timestamp of the observation (standard 3-hour or 6-hour intervals)
- `LAT`: Latitude in decimal degrees
- `LON`: Longitude in decimal degrees
- `WMO_WIND` / `USA_WIND`: Maximum sustained wind speed in knots
- `WMO_PRES` / `USA_PRES`: Minimum central barometric pressure in hPa/mbar

---

## 3. Expected Folder Structure After Placement

Once both datasets are placed, your `cyclone_ai_system/data` directory tree should look like this:

```
cyclone_ai_system/
└── data/
    ├── README.md
    ├── satellite/
    │   ├── Cyclone/
    │   │   ├── img_001.jpg
    │   │   └── ...
    │   └── Non-Cyclone/
    │       ├── img_101.jpg
    │       └── ...
    ├── ibtracs/
    │   └── ibtracs.NI.list.v04r00.csv
    └── processed/
        ├── cnn_processed/       (Generated during preprocessing)
        └── lstm_sequences.npy    (Generated during preprocessing)
```

---

## 4. Verification Check

To check if your datasets are correctly positioned and readable by the data ingestion pipeline, run:

```bash
# Activate virtual environment
venv\Scripts\activate

# Run dataset inspection and preprocessing verification
python -c "import os; print('Satellite exists:', os.path.exists('data/satellite')); print('IBTrACS exists:', os.path.exists('data/ibtracs'))"
```

If files are missing or incomplete, the training scripts will display descriptive warnings and switch to generating synthetic samples so that pipeline execution never crashes.

---

> **Disclaimer**: Prototype AI prediction only. Not an official meteorological warning system.
