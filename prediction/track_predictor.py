import os
import json
import numpy as np

try:
    from tensorflow.keras.models import load_model
    import joblib
except ImportError:
    pass

import config

class CycloneTrackPredictor:
    def __init__(self):
        self.demo_mode = True
        self.model = None
        self.scaler = None
        self.sequence_length = getattr(config, 'SEQUENCE_LENGTH', 6)

        try:
            model_path = os.path.join(config.MODEL_DIR, "cyclone_lstm.keras")
            scaler_path = os.path.join(config.MODEL_DIR, "scaler.pkl")

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                self.demo_mode = False
        except Exception as e:
            print(f"Failed to load track predictor model: {e}")
            self.demo_mode = True

    def get_historical_track(self, cyclone_id=None):
        if self.demo_mode:
            return getattr(config, 'DEMO_CYCLONE_DATA', [])
        
        try:
            processed_data_path = os.path.join(config.DATA_DIR, "processed", "cyclone_tracks.json")
            if os.path.exists(processed_data_path):
                with open(processed_data_path, 'r') as f:
                    data = json.load(f)
                
                if cyclone_id and cyclone_id in data:
                    return data[cyclone_id]
                else:
                    if data:
                        return list(data.values())[0]
                    return []
        except Exception as e:
            print(f"Error loading historical track: {e}")
            
        return getattr(config, 'DEMO_CYCLONE_DATA', [])

    def get_available_cyclones(self):
        if self.demo_mode:
            return [{"id": "FANI2019", "name": "Fani (2019)"}]
            
        try:
            processed_data_path = os.path.join(config.DATA_DIR, "processed", "cyclone_tracks.json")
            if os.path.exists(processed_data_path):
                with open(processed_data_path, 'r') as f:
                    data = json.load(f)
                return [{"id": cid, "name": cid} for cid in data.keys()]
        except Exception:
            pass
            
        return [{"id": "DEMO01", "name": "Demo Cyclone"}]

    def predict_track(self, recent_track, steps=4):
        if self.demo_mode or not recent_track or len(recent_track) < self.sequence_length:
            predictions = []
            if not recent_track:
                return predictions
                
            last_point = recent_track[-1]
            current_lat = last_point.get('lat', 15.0)
            current_lon = last_point.get('lon', 85.0)
            current_wind = last_point.get('wind', 50)
            
            for i in range(1, steps + 1):
                new_lat = current_lat + 0.5 * i
                new_lon = current_lon - 0.2 * i + 0.05 * (i**2)
                new_wind = current_wind + 5 * i
                
                predictions.append({
                    "time": f"T+{i*3}h",
                    "lat": round(new_lat, 2),
                    "lon": round(new_lon, 2),
                    "wind_estimated": round(new_wind, 1),
                    "demo_mode": True
                })
            return predictions

        try:
            features = []
            feature_count = getattr(self.scaler, 'n_features_in_', 3)
            for point in recent_track[-self.sequence_length:]:
                values = [point.get('lat', 0), point.get('lon', 0), point.get('wind', 0)]
                if feature_count > 3:
                    values.append(point.get('pressure', 1000))
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
                    "time": f"T+{(i+1)*3}h",
                    "lat": float(pred_lat),
                    "lon": float(pred_lon),
                    "wind_estimated": float(pred_wind) if pred_wind is not None else None,
                    "pressure_estimated": float(pred_pressure) if pred_pressure is not None else None,
                    "demo_mode": False
                })
                
                current_seq = np.roll(current_seq, -1, axis=1)
                current_seq[0, -1] = pred
                
            return predictions
        except Exception as e:
            print(f"Error predicting track: {e}")
            demo_pred = []
            last_point = recent_track[-1]
            current_lat = last_point.get('lat', 15.0)
            current_lon = last_point.get('lon', 85.0)
            current_wind = last_point.get('wind', 50)
            
            for i in range(1, steps + 1):
                new_lat = current_lat + 0.5 * i
                new_lon = current_lon - 0.2 * i + 0.05 * (i**2)
                new_wind = current_wind + 5 * i
                
                demo_pred.append({
                    "time": f"T+{i*3}h",
                    "lat": round(new_lat, 2),
                    "lon": round(new_lon, 2),
                    "wind_estimated": round(new_wind, 1),
                    "demo_mode": True,
                    "error": str(e)
                })
            return demo_pred
