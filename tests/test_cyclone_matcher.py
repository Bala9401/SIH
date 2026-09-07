import csv
import os
import tempfile
import unittest
from unittest.mock import patch

from prediction.cyclone_matcher import CycloneMatcher
from prediction.track_predictor import CycloneTrackPredictor


class CycloneMatcherTests(unittest.TestCase):
    def write_mapping(self, rows):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        self.addCleanup(lambda: os.unlink(handle.name))
        writer = csv.DictWriter(handle, fieldnames=sorted(CycloneMatcher.REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
        handle.close()
        return handle.name

    def test_exact_image_mapping_is_verified(self):
        path = self.write_mapping([{
            "image_name": "fani_001.jpg", "cyclone_id": "FANI-ID", "cyclone_name": "FANI",
            "timestamp": "2019-05-02T06:00:00", "latitude": "15.6", "longitude": "85.1",
            "wind_speed": "62", "pressure": "954", "source": "verified_test_source",
        }])
        result = CycloneMatcher(path).match({"image_filename": "fani_001.jpg"})
        self.assertTrue(result["matched"])
        self.assertEqual(result["cyclone_id"], "FANI-ID")
        self.assertEqual(result["method"], "verified_mapping_image_name")

    def test_unmapped_image_is_not_assigned(self):
        path = self.write_mapping([])
        result = CycloneMatcher(path).match({"image_filename": "unknown.jpg"})
        self.assertFalse(result["matched"])
        self.assertIn("No verified", result["reason"])

    def test_track_predictor_requires_verified_image_mapping(self):
        predictor = CycloneTrackPredictor()
        with patch.object(predictor, "_load_tracks", return_value={"STORM-1": []}), \
             patch.object(predictor.cyclone_matcher, "match", return_value={"matched": False}):
            result = predictor.identify_cyclone({
                "image_filename": "STORM-1.jpg",
                "timestamp": "2024-01-01T00:00:00",
                "latitude": 15.0,
                "longitude": 85.0,
            })
        self.assertFalse(result["matched"])
        self.assertIn("manual", result["reason"].lower())


if __name__ == "__main__":
    unittest.main()
