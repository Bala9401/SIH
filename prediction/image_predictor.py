import os
import json
import numpy as np

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
except ImportError:
    pass

import config

class CycloneImagePredictor:
    def __init__(self):
        self.demo_mode = True
        self.model = None
        self.class_names = getattr(config, 'CLASS_NAMES', ['Cyclone', 'No Cyclone'])
        self.image_size = getattr(config, 'IMAGE_SIZE', 224)

        try:
            model_path = os.path.join(config.MODEL_DIR, "cyclone_cnn.keras")
            metadata_path = os.path.join(config.MODEL_DIR, "metadata.json")

            if os.path.exists(model_path):
                self.model = load_model(model_path)
                self.demo_mode = False
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if "class_names" in metadata:
                        self.class_names = metadata["class_names"]
        except Exception as e:
            print(f"Failed to load image predictor model: {e}")
            self.demo_mode = True

    def preprocess_image(self, image_path):
        try:
            img = load_img(image_path, target_size=(self.image_size, self.image_size))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def predict(self, image_path):
        if self.demo_mode or self.model is None:
            return {
                "prediction": "Cyclone",
                "confidence": 0.87,
                "class_name": "Cyclone",
                "class_index": 1,
                "all_probabilities": [0.13, 0.87],
                "demo_mode": True
            }

        try:
            img_array = self.preprocess_image(image_path)
            if img_array is None:
                raise ValueError("Image preprocessing failed")

            predictions = self.model.predict(img_array)[0]
            predicted_class_index = int(np.argmax(predictions))
            confidence = float(predictions[predicted_class_index])
            
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
                "demo_mode": True,
                "error": str(e)
            }
