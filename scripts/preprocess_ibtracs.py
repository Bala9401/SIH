"""Create storm-safe, auditable IBTrACS features and forecasting sequences."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

WIND_COLUMNS = ("WMO_WIND", "USA_WIND", "NEWDELHI_WIND")
PRESSURE_COLUMNS = ("WMO_PRES", "USA_PRES", "NEWDELHI_PRES")
FEATURE_COLUMNS = ["LAT", "LON", "wind_kt", "pressure_hpa"]

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in kilometres."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))

def _choose_source(frame, columns, value_name):
    numeric = pd.DataFrame({c: pd.to_numeric(frame[c], errors="coerce") for c in columns if c in frame})
    values = numeric.bfill(axis=1).iloc[:, 0] if not numeric.empty else pd.Series(np.nan, index=frame.index)
    source = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in columns:
        if column in numeric: source = source.mask(source.isna() & numeric[column].notna(), column)
    return values.rename(value_name), source.rename(f"{value_name}_source")

def clean_ibtracs(csv_path):
    raw = pd.read_csv(csv_path, low_memory=False, skiprows=[1])
    missing = {"SID", "ISO_TIME", "LAT", "LON"} - set(raw.columns)
    if missing: raise ValueError(f"IBTrACS is missing required columns: {sorted(missing)}")
    report = {"input_rows": int(len(raw)), "input_columns": list(raw.columns)}
    df = raw.copy()
    if "BASIN" in df: df = df[df.BASIN.astype(str).str.strip().eq("NI")].copy()
    report["north_indian_ocean_rows"] = int(len(df))
    df["ISO_TIME"] = pd.to_datetime(df.ISO_TIME, utc=True, errors="coerce")
    for col in ("LAT", "LON", "DIST2LAND", "LANDFALL", "STORM_SPEED", "STORM_DIR"):
        if col in df: df[col] = pd.to_numeric(df[col], errors="coerce")
    df["wind_kt"], df["wind_kt_source"] = _choose_source(df, WIND_COLUMNS, "wind_kt")
    df["pressure_hpa"], df["pressure_hpa_source"] = _choose_source(df, PRESSURE_COLUMNS, "pressure_hpa")
    before = len(df)
    df = df.dropna(subset=["SID", "ISO_TIME", "LAT", "LON"])
    df = df[df.LAT.between(-90, 90) & df.LON.between(-180, 360)]
    df = df[df.wind_kt.isna() | df.wind_kt.between(0, 250)]
    df = df[df.pressure_hpa.isna() | df.pressure_hpa.between(800, 1100)]
    df = df.sort_values(["SID", "ISO_TIME"]).drop_duplicates(["SID", "ISO_TIME"], keep="last").copy()
    report.update({"removed_invalid_or_duplicate_rows": int(before-len(df)), "usable_rows": int(len(df)), "storms": int(df.SID.nunique()), "time_range_utc": [df.ISO_TIME.min().isoformat(), df.ISO_TIME.max().isoformat()]})
    g = df.groupby("SID", group_keys=False)
    df["previous_lat"], df["previous_lon"] = g.LAT.shift(), g.LON.shift()
    df["elapsed_hours"] = g.ISO_TIME.diff().dt.total_seconds()/3600
    distance = haversine_km(df.previous_lat, df.previous_lon, df.LAT, df.LON)
    df["movement_speed_kmh"] = distance/df.elapsed_hours
    df.loc[(df.elapsed_hours <= 0) | (df.elapsed_hours > 48), "movement_speed_kmh"] = np.nan
    df["movement_direction_deg"] = (np.degrees(np.arctan2(df.LON-df.previous_lon, df.LAT-df.previous_lat))+360)%360
    df["wind_change_kt"], df["pressure_change_hpa"] = g.wind_kt.diff(), g.pressure_hpa.diff()
    df["rolling_wind_kt"] = g.wind_kt.transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["rolling_pressure_hpa"] = g.pressure_hpa.transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["month"] = df.ISO_TIME.dt.month
    df["season"] = np.select([df.month.isin([12,1,2]), df.month.isin([3,4,5]), df.month.isin([6,7,8,9])], ["winter","pre_monsoon","monsoon"], default="post_monsoon")
    df["month_sin"], df["month_cos"] = np.sin(2*np.pi*df.month/12), np.cos(2*np.pi*df.month/12)
    df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*df.ISO_TIME.dt.hour/24), np.cos(2*np.pi*df.ISO_TIME.dt.hour/24)
    df["landfall_indicator"] = (df.LANDFALL.fillna(0)>0).astype(int) if "LANDFALL" in df else 0
    return df, report

def build_sequences(df, lookback):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    complete = df.dropna(subset=FEATURE_COLUMNS).copy(); storms = sorted(complete.SID.unique())
    train, test = train_test_split(storms, test_size=.2, random_state=42); train, validation = train_test_split(train, test_size=.2, random_state=42)
    scaler = MinMaxScaler().fit(complete[complete.SID.isin(train)][FEATURE_COLUMNS])
    split = {"train":set(train), "validation":set(validation), "test":set(test)}; buckets={k:([],[]) for k in split}
    for sid, group in complete.groupby("SID", sort=False):
        vals=scaler.transform(group[FEATURE_COLUMNS].to_numpy(float)); bucket=next(k for k,v in split.items() if sid in v)
        for i in range(len(vals)-lookback): buckets[bucket][0].append(vals[i:i+lookback]); buckets[bucket][1].append(vals[i+lookback])
    arrays={}
    for k,(x,y) in buckets.items(): arrays[f"X_{k}"],arrays[f"y_{k}"]=np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32)
    return arrays, scaler, {f"{k}_storms":sorted(v) for k,v in split.items()}

def preprocess():
    cleaned, report = clean_ibtracs(config.DATA_DIR/"ibtracs"/"ibtracs.NI.list.v04r00.csv")
    out=config.DATA_DIR/"processed"; out.mkdir(parents=True,exist_ok=True); cleaned.to_csv(out/"ibtracs_ni_processed.csv",index=False)
    arrays, scaler, split=build_sequences(cleaned,config.SEQUENCE_LENGTH); np.savez_compressed(out/"track_sequences.npz",**arrays)
    import joblib; config.MODEL_DIR.mkdir(parents=True,exist_ok=True); joblib.dump(scaler,config.MODEL_DIR/"scaler.pkl")
    tracks={}; names={}
    for sid, group in cleaned.groupby("SID",sort=False):
        points=[{"time":r.ISO_TIME.isoformat(),"lat":float(r.LAT),"lon":float(r.LON),"wind":None if pd.isna(r.wind_kt) else float(r.wind_kt),"pressure":None if pd.isna(r.pressure_hpa) else float(r.pressure_hpa),"distance_to_land_km":None if pd.isna(r.get("DIST2LAND")) else float(r.DIST2LAND),"landfall":int(r.landfall_indicator)} for _,r in group.iterrows()]
        if len(points)>=config.SEQUENCE_LENGTH: tracks[str(sid)]=points
        named=group.NAME.dropna().astype(str).str.strip() if "NAME" in group else pd.Series(dtype=str)
        if not named.empty: names[str(sid)]=named.iloc[0]
    (out/"cyclone_tracks.json").write_text(json.dumps(tracks),encoding="utf-8")
    report.update({"feature_columns":FEATURE_COLUMNS,"lookback":config.SEQUENCE_LENGTH,"sequence_counts":{k:int(v.shape[0]) for k,v in arrays.items() if k.startswith("X_")},"source_priority":{"wind":list(WIND_COLUMNS),"pressure":list(PRESSURE_COLUMNS)},"split":split,"missing_wind":int(cleaned.wind_kt.isna().sum()),"missing_pressure":int(cleaned.pressure_hpa.isna().sum())})
    (out/"ibtracs_metadata.json").write_text(json.dumps({**report,"storm_names":names,"scaler_fit":"training storms only"},indent=2),encoding="utf-8")
    config.RESULTS_DIR.mkdir(parents=True,exist_ok=True); (config.RESULTS_DIR/"ibtracs_data_quality_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2)); return report

if __name__=="__main__": preprocess()
