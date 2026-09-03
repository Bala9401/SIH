import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

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
            
        cols_to_keep = ['SID', 'ISO_TIME', 'LAT', 'LON', 'WMO_WIND']
        available_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[available_cols].dropna()

        df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
        df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
        if 'WMO_WIND' in df.columns:
            df['WMO_WIND'] = pd.to_numeric(df['WMO_WIND'], errors='coerce')
            
        df = df.dropna().sort_values(by=['SID', 'ISO_TIME'])
        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
        df = df.dropna(subset=['ISO_TIME'])

        features = ['LAT', 'LON']
        if 'WMO_WIND' in df.columns:
            features.append('WMO_WIND')

        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        X, y = [], []
        for sid, group in df.groupby('SID'):
            vals = group[features].values
            for i in range(len(vals) - SEQUENCE_LENGTH):
                X.append(vals[i:i+SEQUENCE_LENGTH])
                y.append(vals[i+SEQUENCE_LENGTH])
                
        if len(X) == 0:
            print("Not enough sequences generated.")
            return

        X = np.array(X)
        y = np.array(y)

        proc_dir = DATA_DIR / "processed"
        proc_dir.mkdir(parents=True, exist_ok=True)
        np.savez(proc_dir / "track_sequences.npz", X=X, y=y)

        tracks = {}
        display_columns = ['SID', 'ISO_TIME', 'LAT', 'LON'] + (['WMO_WIND'] if 'WMO_WIND' in df.columns else [])
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
                points.append(point)
            if len(points) >= SEQUENCE_LENGTH:
                tracks[str(sid)] = points
        with open(proc_dir / "cyclone_tracks.json", "w") as f:
            json.dump(tracks, f)

        with open(proc_dir / "ibtracs_metadata.json", "w") as f:
            json.dump({"num_sequences": len(X), "features": features}, f)
            
        print("Processed track sequences saved.")
    except Exception as e:
        print(f"Error processing IBTrACS: {e}")

if __name__ == "__main__":
    preprocess()
