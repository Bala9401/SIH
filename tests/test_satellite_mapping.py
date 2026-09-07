import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from prediction.cyclone_matcher import CycloneMatcher
from prediction.risk_assessment import CycloneRiskAssessor
from prediction.track_predictor import CycloneTrackPredictor
from scripts.build_satellite_mapping import build_mapping


class SatelliteMappingTests(unittest.TestCase):
    def make_project(self):
        root = Path(tempfile.mkdtemp())
        (root / "data" / "satellite").mkdir(parents=True)
        (root / "data" / "processed").mkdir(parents=True)
        return root

    def write_csv(self, path, fieldnames, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_empty_manifest_emits_no_mapping_rows(self):
        root = self.make_project()
        manifest = root / "data" / "satellite_intensity_manifest.csv"
        self.write_csv(manifest, ["image_path", "product", "cyclone_id", "cyclone_name", "timestamp_utc", "latitude", "longitude", "wind_speed_kt", "pressure_hpa", "storm_category", "source"], [])
        rows = build_mapping(root)
        self.assertEqual(rows, [])
        self.assertEqual(list(csv.DictReader((root / "data" / "satellite" / "satellite_cyclone_mapping.csv").open(encoding="utf-8"))), [])

    def test_complete_manifest_requires_exact_ibtracs_observation(self):
        root = self.make_project()
        image_path = root / "data" / "satellite" / "storm.jpg"
        image_path.write_bytes(b"verified image bytes")
        manifest_fields = ["image_path", "product", "cyclone_id", "cyclone_name", "timestamp_utc", "latitude", "longitude", "wind_speed_kt", "pressure_hpa", "storm_category", "source"]
        self.write_csv(root / "data" / "satellite_intensity_manifest.csv", manifest_fields, [{
            "image_path": "data/satellite/storm.jpg", "product": "infrared", "cyclone_id": "SID-1", "cyclone_name": "TEST",
            "timestamp_utc": "2020-01-01T00:00:00Z", "latitude": "10", "longitude": "80", "wind_speed_kt": "40",
            "pressure_hpa": "990", "storm_category": "storm", "source": "verified-test-manifest",
        }])
        track_fields = ["SID", "NAME", "ISO_TIME", "LAT", "LON", "wind_kt", "pressure_hpa"]
        self.write_csv(root / "data" / "processed" / "ibtracs_ni_processed.csv", track_fields, [{
            "SID": "SID-1", "NAME": "TEST", "ISO_TIME": "2020-01-01T00:00:00+00:00", "LAT": "10", "LON": "80", "wind_kt": "40", "pressure_hpa": "990",
        }])
        rows = build_mapping(root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["match_method"], "verified_manifest_ibtracs_exact_timestamp")
        self.assertEqual(rows[0]["source"].split(";")[0], "verified-test-manifest")
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        match = CycloneMatcher(root / "data" / "satellite" / "satellite_cyclone_mapping.csv").match({"sha256": image_hash})
        self.assertTrue(match["matched"])
        self.assertEqual(match["cyclone_id"], "SID-1")

    def test_risk_is_unavailable_without_current_observations(self):
        result = CycloneRiskAssessor().assess_risk(None, [], {}, None, None)
        self.assertFalse(result["available"])
        self.assertIsNone(result["risk_score"])

    def test_manual_lstm_mode_uses_real_historical_sequence(self):
        predictor = CycloneTrackPredictor()
        cyclone = predictor.get_available_cyclones()[0]
        historical = predictor.get_historical_track(cyclone["id"])
        forecast = predictor.predict_track(historical, allow_baseline=False)
        self.assertFalse(predictor.demo_mode)
        self.assertGreaterEqual(len(historical), predictor.sequence_length)
        self.assertEqual(len(forecast), 16)
        self.assertTrue(all(point.get("demo_mode") is False for point in forecast))


if __name__ == "__main__":
    unittest.main()