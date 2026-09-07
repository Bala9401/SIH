import os
import json
import numpy as np

try:
    from tensorflow.keras.models import load_model
    import joblib
except ImportError:
    pass

import config
from prediction.cyclone_matcher import CycloneMatcher

class CycloneTrackPredictor:
    def __init__(self):
        self.demo_mode = True
        self.model = None
        self.scaler = None
        self.sequence_length = getattr(config, 'SEQUENCE_LENGTH', 6)
        self.cyclone_matcher = CycloneMatcher()

        try:
            model_path = os.path.join(config.MODEL_DIR, "cyclone_lstm.keras")
            scaler_path = os.path.join(config.MODEL_DIR, "scaler.pkl")

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                output_size = int(self.model.output_shape[-1])
                self.demo_mode = not (getattr(self.scaler, 'n_features_in_', 0) == 4 and output_size == 4)
        except Exception as e:
            print(f"Failed to load track predictor model: {e}")
            self.demo_mode = True

    def get_historical_track(self, cyclone_id=None):
        try:
            processed_data_path = os.path.join(config.DATA_DIR, "processed", "cyclone_tracks.json")
            if os.path.exists(processed_data_path):
                with open(processed_data_path, 'r') as f:
                    data = json.load(f)
                
                if cyclone_id and cyclone_id in data:
                    return data[cyclone_id]
        except Exception as e:
            print(f"Error loading historical track: {e}")

        if self.demo_mode and cyclone_id == "FANI2019":
            return getattr(config, 'DEMO_CYCLONE_DATA', [])
        return []

    def _load_tracks(self):
        processed_data_path = os.path.join(config.DATA_DIR, "processed", "cyclone_tracks.json")
        if not os.path.exists(processed_data_path):
            return {}
        with open(processed_data_path, 'r') as f:
            return json.load(f)

    def identify_cyclone(self, image_analysis):
        tracks = self._load_tracks()
        mapping_result = self.cyclone_matcher.match(image_analysis, tracks)
        if mapping_result.get('matched'):
            if mapping_result['cyclone_id'] not in tracks:
                return {"matched": False, "reason": "Verified mapping points to an unavailable IBTrACS track."}
            return mapping_result
        # A satellite product image cannot be assigned to an IBTrACS storm from
        # timestamp/position proximity alone.  That would make the subsequent
        # LSTM forecast look image-derived when it is not.  The UI therefore
        # offers the user a clearly labelled manual storm selection instead.
        return {
            "matched": False,
            "reason": "No verified image-to-cyclone mapping. LSTM track forecast requires a verified cyclone identity and historical sequence; use Advanced / Manual Analysis for a historical storm."
        }

    def get_available_cyclones(self):
        if self.demo_mode:
            return [{"id": "FANI2019", "name": "Fani (2019)"}]
            
        try:
            processed_data_path = os.path.join(config.DATA_DIR, "processed", "cyclone_tracks.json")
            if os.path.exists(processed_data_path):
                with open(processed_data_path, 'r') as f:
                    data = json.load(f)
                metadata_path = os.path.join(config.DATA_DIR, "processed", "ibtracs_metadata.json")
                names = {}
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        names = json.load(f).get('storm_names', {})
                # Anonymous early-era IBTrACS records are valid observations but
                # make a human-selection UI misleading. They remain in model
                # training data, while the dashboard offers named storms only.
                return [
                    {"id": cid, "name": name}
                    for cid, name in names.items()
                    if cid in data and name not in (None, '', 'NOT_NAMED')
                ]
        except Exception:
            pass
            
        return [{"id": "DEMO01", "name": "Demo Cyclone"}]

    def get_cyclone_name(self, cyclone_id):
        if not cyclone_id:
            return None
        metadata_path = os.path.join(config.DATA_DIR, "processed", "ibtracs_metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                name = json.load(f).get('storm_names', {}).get(cyclone_id)
                return name if name not in (None, '', 'NOT_NAMED') else None
        except (OSError, json.JSONDecodeError):
            return None

    def predict_track(self, recent_track, steps=16, allow_baseline=True):
        if not allow_baseline and (self.demo_mode or not recent_track or len(recent_track) < self.sequence_length):
            return []

        if self.demo_mode or not recent_track or len(recent_track) < self.sequence_length:
            # A missing model must not produce a plausible-looking fabricated
            # storm path.  Persistence is an explicit, reproducible baseline.
            predictions = []
            if not recent_track:
                return predictions
                
            last_point = recent_track[-1]
            current_lat = last_point.get('lat', 15.0)
            current_lon = last_point.get('lon', 85.0)
            current_wind = last_point.get('wind', 50)
            current_pressure = last_point.get('pressure')
            
            for i in range(1, steps + 1):
                
                predictions.append({
                    "time": f"T+{i*3}h", "time_offset": i * 3,
                    "lat": round(current_lat, 2),
                    "lon": round(current_lon, 2),
                    "wind_estimated": round(current_wind, 1),
                    "pressure_estimated": current_pressure,
                    "uncertainty_radius_km": None,
                    "demo_mode": True, "forecast_method": "persistence_baseline"
                })
            return predictions

        try:
            features = []
            feature_count = getattr(self.scaler, 'n_features_in_', 3)
            for point in recent_track[-self.sequence_length:]:
                values = [point.get('lat', 0), point.get('lon', 0),
                          point.get('wind', point.get('wind_speed', 0)), point.get('pressure')]
                features.append(values[:feature_count])
                
            input_seq = np.array(features)
            
            if self.scaler:
                original_shape = input_seq.shape
                flat_seq = input_seq.reshape(-1, original_shape[-1])
                scaled_seq = self.scaler.transform(flat_seq)
                input_seq = scaled_seq.reshape(1, original_shape[0], original_shape[1])
            else:
                input_seq = np.expand_dims(input_seq, axis=0)

            predictions = []
            uncertainty_path = os.path.join(config.RESULTS_DIR, "metrics", "track_uncertainty.json")
            uncertainty = {}
            if os.path.exists(uncertainty_path):
                with open(uncertainty_path, 'r') as uncertainty_file:
                    uncertainty = json.load(uncertainty_file)
            uncertainty_horizons = uncertainty.get('horizons', {})
            current_seq = input_seq.copy()
            
            for i in range(steps):
                pred = self.model.predict(current_seq)[0]
                
                if self.scaler:
                    pred_unscaled = self.scaler.inverse_transform(pred.reshape(1, -1))[0]
                else:
                    pred_unscaled = pred
                    
                pred_lat, pred_lon = pred_unscaled[:2]
                pred_wind = pred_unscaled[2] if len(pred_unscaled) > 2 else None
                pred_pressure = pred_unscaled[3] if len(pred_unscaled) > 3 else None
                
                predictions.append({
                    "time": f"T+{(i+1)*3}h", "time_offset": (i + 1) * 3,
                    "lat": float(pred_lat),
                    "lon": float(pred_lon),
                    "wind_estimated": float(pred_wind) if pred_wind is not None else None,
                    "pressure_estimated": float(pred_pressure) if pred_pressure is not None else None,
                    "uncertainty_radius_km": (uncertainty_horizons.get(str((i + 1) * 3), {})
                                               .get('p75_error_km')),
                    "demo_mode": False
                })
                
                current_seq = np.roll(current_seq, -1, axis=1)
                current_seq[0, -1] = pred
                
            return predictions
        except Exception as e:
            print(f"Error predicting track: {e}")
            if not allow_baseline:
                return []
            demo_pred = []
            last_point = recent_track[-1]
            current_lat = last_point.get('lat', 15.0)
            current_lon = last_point.get('lon', 85.0)
            current_wind = last_point.get('wind', 50)
            
            for i in range(1, steps + 1):
                
                demo_pred.append({
                    "time": f"T+{i*3}h",
                    "lat": round(current_lat, 2),
                    "lon": round(current_lon, 2),
                    "wind_estimated": round(current_wind, 1),
                    "pressure_estimated": last_point.get('pressure'),
                    "uncertainty_radius_km": None,
                    "demo_mode": True,
                    "error": str(e), "forecast_method": "persistence_baseline"
                })
            return demo_pred
