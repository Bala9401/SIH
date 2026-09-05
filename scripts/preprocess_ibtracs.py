import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, SEQUENCE_LENGTH, MODEL_DIR

def preprocess():
    print("Preprocessing IBTrACS data...")
    try:
        ibtracs_dir = DATA_DIR / "ibtracs"
        csv_files = list(ibtracs_dir.glob("*.csv")) if ibtracs_dir.exists() else []
        if not csv_files:
            print("No IBTrACS CSV found.")
            return

        df = pd.read_csv(csv_files[0], low_memory=False, skiprows=[1])
        
        if 'BASIN' in df.columns:
            df = df[df['BASIN'] == 'NI']
            
        cols_to_keep = ['SID', 'ISO_TIME', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES']
        available_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[available_cols].dropna()

        df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
        df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
        for column in ['WMO_WIND', 'WMO_PRES']:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')
            
        df = df.dropna().sort_values(by=['SID', 'ISO_TIME'])
        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
        df = df.dropna(subset=['ISO_TIME'])

        features = ['LAT', 'LON', 'WMO_WIND', 'WMO_PRES']
        # Pressure and wind are required model targets; do not invent missing values.
        df = df.dropna(subset=features).sort_values(by=['SID', 'ISO_TIME'])
        storm_ids = sorted(df['SID'].unique())
        if len(storm_ids) < 2:
            print("At least two storms with complete four-feature observations are required.")
            return
        train_ids, test_ids = train_test_split(storm_ids, test_size=0.2, random_state=42)
        train_ids, test_ids = set(train_ids), set(test_ids)
        fit_ids, validation_ids = train_test_split(sorted(train_ids), test_size=0.2, random_state=42)
        fit_ids, validation_ids = set(fit_ids), set(validation_ids)

        scaler = MinMaxScaler()
        scaler.fit(df[df['SID'].isin(train_ids)][features])

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
        for sid, group in df.groupby('SID'):
            raw_vals = group[features].to_numpy(dtype=np.float32)
            vals = scaler.transform(raw_vals)
            for i in range(len(vals) - SEQUENCE_LENGTH):
                if sid in fit_ids:
                    target = (X_train, y_train)
                elif sid in validation_ids:
                    target = (X_val, y_val)
                else:
                    target = (X_test, y_test)
                target[0].append(vals[i:i+SEQUENCE_LENGTH])
                target[1].append(vals[i+SEQUENCE_LENGTH])
                
        if not X_train or not X_val or not X_test:
            print("Not enough sequences generated.")
            return

        X_train, y_train = np.asarray(X_train), np.asarray(y_train)
        X_val, y_val = np.asarray(X_val), np.asarray(y_val)
        X_test, y_test = np.asarray(X_test), np.asarray(y_test)

        proc_dir = DATA_DIR / "processed"
        proc_dir.mkdir(parents=True, exist_ok=True)
        np.savez(proc_dir / "track_sequences.npz", X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)

        tracks = {}
        for sid, group in df.groupby('SID'):
            points = []
            for _, row in group.iterrows():
                point = {
                    "time": row['ISO_TIME'].isoformat(),
                    "lat": float(row['LAT']),
                    "lon": float(row['LON'])
                }
                if 'WMO_WIND' in df.columns:
                    point['wind'] = float(row['WMO_WIND'])
                if 'WMO_PRES' in df.columns:
                    point['pressure'] = float(row['WMO_PRES'])
                points.append(point)
            if len(points) >= SEQUENCE_LENGTH:
                tracks[str(sid)] = points
        with open(proc_dir / "cyclone_tracks.json", "w") as f:
            json.dump(tracks, f)

        with open(proc_dir / "ibtracs_metadata.json", "w") as f:
            json.dump({"num_train_sequences": len(X_train), "num_validation_sequences": len(X_val),
                       "num_test_sequences": len(X_test), "features": features,
                       "train_storms": sorted(fit_ids), "validation_storms": sorted(validation_ids),
                       "test_storms": sorted(test_ids), "scaler_fit": "training_storms_only"}, f, indent=2)
            
        print("Processed track sequences saved.")
    except Exception as e:
        print(f"Error processing IBTrACS: {e}")

if __name__ == "__main__":
    preprocess()
