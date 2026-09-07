import os
import json
import re
from fractions import Fraction
import numpy as np

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except ImportError:
    pass

import config

class CycloneImagePredictor:
    def __init__(self):
        self.demo_mode = True
        self.model = None
        self.class_names = getattr(config, 'CLASS_NAMES', ['Cyclone', 'No Cyclone'])
        self.task = "unknown"
        self.intensity_classification_available = False
        self.image_size = getattr(config, 'IMAGE_SIZE', 224)
        self.image_risk_model = None
        self.image_risk_metadata = None

        try:
            model_path = os.path.join(config.MODEL_DIR, "cyclone_cnn.keras")
            metadata_path = os.path.join(config.MODEL_DIR, "metadata.json")
            risk_model_path = os.path.join(config.MODEL_DIR, "cyclone_image_risk.keras")
            risk_metadata_path = os.path.join(config.MODEL_DIR, "image_risk_metadata.json")

            if os.path.exists(model_path):
                self.model = load_model(model_path)
                self.demo_mode = False
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if "class_names" in metadata:
                        self.class_names = metadata["class_names"]
                    self.task = metadata.get("task", "unknown")
                    self.intensity_classification_available = bool(metadata.get("intensity_labels_available", False))

            if os.path.exists(risk_model_path) and os.path.exists(risk_metadata_path):
                self.image_risk_model = load_model(risk_model_path)
                with open(risk_metadata_path, 'r') as f:
                    self.image_risk_metadata = json.load(f)
        except Exception as e:
            print(f"Failed to load image predictor model: {e}")
            self.demo_mode = True

    def preprocess_image(self, image_path):
        try:
            img = load_img(image_path, target_size=(self.image_size, self.image_size))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array.astype(np.float32))
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def validate_image(self, image_path):
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                width, height = image.size
                if width < 64 or height < 64:
                    return False, "Image is too small for satellite analysis."
                return True, None
        except Exception as error:
            return False, str(error)

    def _extract_image_metadata(self, image_path, filename=None):
        metadata = {
            "image_filename": filename or os.path.basename(image_path),
            "timestamp": None,
            "latitude": None,
            "longitude": None,
            "cyclone_id": None,
        }
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                exif = image.getexif()
                metadata["timestamp"] = exif.get(36867) or exif.get(306)
                gps = exif.get_ifd(34853) if hasattr(exif, 'get_ifd') else None
                if gps and 2 in gps and 4 in gps:
                    def to_decimal(value):
                        parts = [float(Fraction(item)) for item in value]
                        return parts[0] + parts[1] / 60 + parts[2] / 3600
                    metadata["latitude"] = to_decimal(gps[2]) * (-1 if gps.get(1) == 'S' else 1)
                    metadata["longitude"] = to_decimal(gps[4]) * (-1 if gps.get(3) == 'W' else 1)
        except Exception:
            pass
        return metadata

    def _contextual_risk_score(self, class_index, probabilities):
        """Return a bounded context score; product classes are not intensity labels."""
        product_context = {
            "insat3d_for_reference_ds": 15.0,
            "insat3d_ir_cyclone_ds": 35.0,
            "insat3d_raw_cyclone_ds": 40.0,
        }
        class_name = self.class_names[class_index] if class_index < len(self.class_names) else ""
        base_score = product_context.get(class_name, 0.0)
        confidence = float(probabilities[class_index]) if len(probabilities) > class_index else 0.0
        score = base_score * (0.5 + 0.5 * confidence)
        return round(min(45.0, max(0.0, score)), 1)

    def _risk_level(self, score):
        if score < 25:
            return "LOW"
        if score < 50:
            return "MODERATE"
        if score < 75:
            return "HIGH"
        return "VERY HIGH"

    def _supervised_risk(self, image_array):
        # The bundled image CSV has undocumented numeric labels.  Until a
        # verified manifest is supplied, any image-to-wind/pressure model is
        # scientifically invalid and must never be served.
        return None
        if self.image_risk_model is None or not self.image_risk_metadata:
            return None
        prediction = self.image_risk_model.predict(image_array, verbose=0)[0]
        target_mean = self.image_risk_metadata.get("target_mean", [0.0, 0.0])
        target_std = self.image_risk_metadata.get("target_std", [1.0, 1.0])
        wind_speed = float(prediction[0] * target_std[0] + target_mean[0])
        pressure = float(prediction[1] * target_std[1] + target_mean[1])
        wind_score = min(100.0, max(0.0, (wind_speed - 20.0) / 80.0 * 100.0))
        pressure_score = min(100.0, max(0.0, (1010.0 - pressure) / 60.0 * 100.0))
        score = round((wind_score + pressure_score) / 2.0, 1)
        return {
            "image_risk_score": score,
            "image_risk_level": self._risk_level(score),
            "estimated_wind_speed_kt": round(wind_speed, 1),
            "estimated_pressure_hpa": round(pressure, 1),
            "risk_prediction_from_image_available": True,
            "image_risk_basis": "Supervised image-to-wind/pressure regression trained from the verified satellite manifest.",
        }

    def predict(self, image_path, filename=None):
        image_metadata = self._extract_image_metadata(image_path, filename)
        image_valid, validation_error = self.validate_image(image_path)
        if not image_valid:
            return {
                "prediction": "Unavailable",
                "confidence": 0.0,
                "class_name": "Unavailable",
                "class_index": -1,
                "all_probabilities": [],
                "task": "satellite_product_classification",
                "image_valid": False,
                "image_model_available": bool(self.model is not None),
                "cyclone_detected": None,
                "cyclone_name": None,
                "cyclone_name_available": False,
                "cyclone_detection_supported": False,
                "intensity_prediction_available": False,
                "risk_prediction_from_image_available": False,
                "image_risk_score": None,
                "image_risk_level": "UNAVAILABLE",
                "error": validation_error,
                "demo_mode": self.demo_mode
                ,**image_metadata
            }
        if self.demo_mode or self.model is None:
            return {
                "prediction": "Unavailable",
                "confidence": 0.0,
                "class_name": "Unavailable",
                "class_index": -1,
                "all_probabilities": [],
                "task": "satellite_product_classification",
                "image_valid": True,
                "image_model_available": False,
                "cyclone_detected": None,
                "cyclone_name": None,
                "cyclone_name_available": False,
                "cyclone_detection_supported": False,
                "intensity_classification_available": False,
                "intensity_prediction_available": False,
                "risk_prediction_from_image_available": False,
                "image_risk_score": None,
                "image_risk_level": "UNAVAILABLE",
                "demo_mode": True
                ,**image_metadata
            }

        try:
            img_array = self.preprocess_image(image_path)
            if img_array is None:
                raise ValueError("Image preprocessing failed")

            predictions = self.model.predict(img_array)[0]
            predicted_class_index = int(np.argmax(predictions))
            confidence = float(predictions[predicted_class_index])
            supervised_risk = self._supervised_risk(img_array)
            if supervised_risk:
                image_risk = supervised_risk
            else:
                image_risk = {
                    "image_risk_score": self._contextual_risk_score(predicted_class_index, predictions),
                    "image_risk_level": self._risk_level(self._contextual_risk_score(predicted_class_index, predictions)),
                    "risk_prediction_from_image_available": False,
                    "image_risk_basis": "Satellite product context only; verified intensity labels are unavailable.",
                }
            
            if predicted_class_index < len(self.class_names):
                class_name = self.class_names[predicted_class_index]
            else:
                class_name = f"Class {predicted_class_index}"

            return {
                "prediction": class_name,
                "confidence": confidence,
                "class_name": class_name,
                "class_index": predicted_class_index,
                "all_probabilities": [float(p) for p in predictions],
                "task": self.task,
                "image_valid": True,
                "image_model_available": True,
                "cyclone_detected": None,
                "cyclone_name": None,
                "cyclone_name_available": False,
                "cyclone_detection_supported": False,
                "intensity_classification_available": self.intensity_classification_available,
                "intensity_prediction_available": bool(supervised_risk),
                **image_risk,
                **image_metadata,
                "demo_mode": False
            }
        except Exception as e:
            print(f"Error predicting image: {e}")
            return {
                "prediction": "Error",
                "confidence": 0.0,
                "class_name": "Error",
                "class_index": -1,
                "all_probabilities": [],
                "task": "satellite_product_classification",
                "image_valid": True,
                "image_model_available": bool(self.model is not None),
                "cyclone_detected": None,
                "cyclone_name": None,
                "cyclone_name_available": False,
                "cyclone_detection_supported": False,
                "intensity_prediction_available": False,
                "risk_prediction_from_image_available": False,
                "image_risk_score": None,
                "image_risk_level": "UNAVAILABLE",
                "demo_mode": True,
                "error": str(e),
                **image_metadata
            }
