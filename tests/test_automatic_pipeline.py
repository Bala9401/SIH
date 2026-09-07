import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from PIL import Image

import app


class AutomaticPipelineTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    @staticmethod
    def image_file(filename):
        stream = io.BytesIO()
        Image.new("RGB", (64, 64), "black").save(stream, format="PNG")
        stream.seek(0)
        return stream, filename

    @staticmethod
    def image_result(filename):
        return {
            "prediction": "insat3d_ir_cyclone_ds",
            "class_name": "insat3d_ir_cyclone_ds",
            "confidence": 0.91,
            "all_probabilities": [0.02, 0.91, 0.07],
            "task": "satellite_product_classification",
            "image_filename": filename,
        }

    def test_new_image_replaces_previous_automatic_pipeline_state(self):
        image_predictor = Mock()
        image_predictor.class_names = ["reference", "infrared", "raw"]
        image_predictor.predict.side_effect = lambda path, filename: self.image_result(filename)

        track_predictor = Mock()
        track_predictor.sequence_length = 2
        track_predictor.demo_mode = False
        track_predictor.cyclone_matcher.mapping_path = "verified-mapping.csv"
        track_predictor.identify_cyclone.side_effect = [
            {"matched": True, "cyclone_id": "CYCLONE-A", "cyclone_name": "Alpha", "match_confidence": 0.98,
             "timestamp": "2024-01-01T00:00:00", "method": "verified_mapping_image_name", "source": "test"},
            {"matched": True, "cyclone_id": "CYCLONE-B", "cyclone_name": "Beta", "match_confidence": 0.97,
             "timestamp": "2024-02-01T00:00:00", "method": "verified_mapping_image_name", "source": "test"},
        ]
        track_predictor.get_track_through_observation.side_effect = [
            [{"time": "2024-01-01T00:00:00", "lat": 10, "lon": 80, "wind": 40, "pressure": 990},
             {"time": "2024-01-01T03:00:00", "lat": 11, "lon": 81, "wind": 45, "pressure": 988}],
            [{"time": "2024-02-01T00:00:00", "lat": 12, "lon": 82, "wind": 50, "pressure": 980},
             {"time": "2024-02-01T03:00:00", "lat": 13, "lon": 83, "wind": 55, "pressure": 978}],
        ]
        track_predictor.predict_track.side_effect = [
            [{"time_offset": 3, "lat": 12, "lon": 82, "wind_estimated": 46, "pressure_estimated": 987}],
            [{"time_offset": 3, "lat": 14, "lon": 84, "wind_estimated": 56, "pressure_estimated": 977}],
        ]
        risk_assessor = Mock()
        risk_assessor.assess_risk.return_value = {"available": True, "risk_level": "MODERATE", "risk_score": 40}

        with patch.object(app, "image_predictor", image_predictor), \
             patch.object(app, "track_predictor", track_predictor), \
             patch.object(app, "risk_assessor", risk_assessor):
            first = self.client.post("/api/analyze", data={"file": self.image_file("image-a.png")}, content_type="multipart/form-data")
            second = self.client.post("/api/analyze", data={"file": self.image_file("image-b.png")}, content_type="multipart/form-data")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json["cyclone"]["id"], "CYCLONE-A")
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json["cyclone"]["id"], "CYCLONE-B")
        self.assertEqual(second.json["match"]["method"], "verified_mapping_image_name")
        self.assertEqual(second.json["track"]["historical"][0]["lat"], 12)
        self.assertEqual(track_predictor.predict_track.call_args_list[1].args[0][0]["lat"], 12)
        self.assertEqual(len(track_predictor.identify_cyclone.call_args_list[1].args[0]["sha256"]), 64)

    def test_unmapped_image_stops_before_lstm_and_risk(self):
        image_predictor = Mock()
        image_predictor.class_names = []
        image_predictor.predict.return_value = self.image_result("unmapped.png")
        track_predictor = Mock()
        track_predictor.cyclone_matcher.mapping_path = "verified-mapping.csv"
        track_predictor.identify_cyclone.return_value = {
            "matched": False,
            "reason": "No verified cyclone identity could be determined for this satellite image.",
        }
        risk_assessor = Mock()

        with patch.object(app, "image_predictor", image_predictor), \
             patch.object(app, "track_predictor", track_predictor), \
             patch.object(app, "risk_assessor", risk_assessor):
            response = self.client.post("/api/analyze", data={"file": self.image_file("unmapped.png")}, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["cyclone"]["detected"])
        self.assertIn("No verified cyclone identity", response.json["cyclone"]["reason"])
        self.assertFalse(response.json["track"]["available"])
        track_predictor.get_historical_track.assert_not_called()
        track_predictor.predict_track.assert_not_called()
        risk_assessor.assess_risk.assert_not_called()

    @unittest.skipUnless(
        (Path(__file__).parents[1] / "models" / "cyclone_lstm.keras").exists(),
        "verified LSTM artifact is unavailable",
    )
    def test_three_real_mapped_images_recalculate_current_risk(self):
        project_root = Path(__file__).parents[1]
        expected = {
            "25.jpeg": "2013339N09084",
            "111.jpeg": "2013324N07103",
            "101.jpeg": "2013281N12098",
        }
        responses = []
        for image_name, cyclone_id in expected.items():
            image_path = next((project_root / "data" / "satellite").rglob(image_name))
            with image_path.open("rb") as image_file:
                response = self.client.post(
                    "/api/analyze",
                    data={"file": (image_file, image_name)},
                    content_type="multipart/form-data",
                )
            self.assertEqual(response.status_code, 200)
            payload = response.json
            self.assertEqual(payload["match"]["cyclone_id"], cyclone_id)
            self.assertTrue(payload["risk"]["available"])
            self.assertEqual(payload["risk"]["meteorological_factors"]["wind_speed"], payload["current_conditions"]["wind"])
            self.assertEqual(payload["risk"]["meteorological_factors"]["pressure"], payload["current_conditions"]["pressure"])
            self.assertEqual(payload["risk"]["risk_score"], payload["risk_assessment"]["risk_score"])
            responses.append(payload)

        scores = [payload["risk"]["risk_score"] for payload in responses]
        current_conditions = [
            (payload["cyclone"]["id"], payload["current_conditions"]["time"], payload["current_conditions"]["wind"], payload["current_conditions"]["pressure"])
            for payload in responses
        ]
        self.assertEqual(len(set(item[0] for item in current_conditions)), 3)
        self.assertGreater(len(set(scores)), 1)
        self.assertEqual(len(set(current_conditions)), 3)


if __name__ == "__main__":
    unittest.main()