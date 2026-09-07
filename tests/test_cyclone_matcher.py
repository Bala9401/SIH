import csv
import hashlib
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
        writer = csv.DictWriter(handle, fieldnames=[
            "image_name", "sha256", "cyclone_id", "cyclone_name", "match_method", "match_confidence",
        ])
        writer.writeheader()
        writer.writerows(rows)
        handle.close()
        return handle.name

    def test_sha256_mapping_is_verified(self):
        path = self.write_mapping([{
            "image_name": "fani_001.jpg", "sha256": "a" * 64, "cyclone_id": "FANI-ID", "cyclone_name": "FANI",
            "match_method": "verified", "match_confidence": "1.0",
        }])
        result = CycloneMatcher(path).match({"sha256": "A" * 64})
        self.assertTrue(result["matched"])
        self.assertEqual(result["cyclone_id"], "FANI-ID")
        self.assertEqual(result["method"], "sha256")

    def test_unmapped_image_is_not_assigned(self):
        path = self.write_mapping([])
        result = CycloneMatcher(path).match({"sha256": "b" * 64})
        self.assertFalse(result["matched"])
        self.assertIn("not a verified dataset image", result["reason"])

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
        self.assertIn("not a verified dataset image", result["reason"])

    def test_two_hashes_select_different_cyclones(self):
        path = self.write_mapping([
            {"image_name": "a.jpg", "sha256": "a" * 64, "cyclone_id": "A", "cyclone_name": "ALPHA", "match_method": "verified", "match_confidence": "1.0"},
            {"image_name": "b.jpg", "sha256": "b" * 64, "cyclone_id": "B", "cyclone_name": "BETA", "match_method": "verified", "match_confidence": "1.0"},
        ])
        matcher = CycloneMatcher(path)
        self.assertEqual(matcher.match({"sha256": "a" * 64})["cyclone_id"], "A")
        self.assertEqual(matcher.match({"sha256": "b" * 64})["cyclone_id"], "B")


if __name__ == "__main__":
    unittest.main()
