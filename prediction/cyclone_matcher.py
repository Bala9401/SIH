import csv
import hashlib
import os

import config


class CycloneMatcher:
    """Resolve uploaded image metadata through a verified mapping index only."""

    REQUIRED_COLUMNS = {
        "image_name", "sha256", "cyclone_id", "cyclone_name",
        "match_method", "match_confidence",
    }

    def __init__(self, mapping_path=None):
        self.mapping_path = mapping_path or os.path.join(
            config.DATA_DIR, "satellite", "satellite_cyclone_mapping.csv"
        )

    def _rows(self):
        if not os.path.exists(self.mapping_path):
            return []
        with open(self.mapping_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not self.REQUIRED_COLUMNS.issubset(fields):
                return []
            return [row for row in reader if row.get("sha256") and row.get("cyclone_id")]

    @staticmethod
    def sha256_file(image_path):
        digest = hashlib.sha256()
        with open(image_path, "rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def match(self, image_analysis, tracks=None):
        image_analysis = image_analysis or {}
        rows = self._rows()
        image_hash = str(image_analysis.get("sha256") or "").strip().lower()
        exact = [row for row in rows if row.get("sha256", "").strip().lower() == image_hash]
        if len(exact) == 1:
            row = exact[0]
            return self._result(row, 1.0, "sha256")
        if len(exact) > 1:
            return {"matched": False, "reason": "Multiple verified mappings exist for this SHA-256 hash."}
        return {
            "matched": False,
            "reason": "Uploaded image is not a verified dataset image. Automatic cyclone identification is unavailable for this image.",
        }

    @staticmethod
    def _result(row, confidence, method, time_hours=None, distance_km=None):
        result = {
            "matched": True,
            "cyclone_id": row["cyclone_id"],
            "cyclone_name": row.get("cyclone_name") or None,
            "timestamp": row.get("ibtracs_timestamp") or row.get("image_timestamp") or None,
            "latitude": CycloneMatcher._number(row.get("ibtracs_latitude") or row.get("image_latitude")),
            "longitude": CycloneMatcher._number(row.get("ibtracs_longitude") or row.get("image_longitude")),
            "match_confidence": confidence,
            "method": method,
            "source": row.get("source") or None,
            "match_method": row.get("match_method") or method,
            "sha256": row.get("sha256"),
            "wind_speed_kmh": CycloneMatcher._knots_to_kmh(row.get("ibtracs_wind_knots")),
            "pressure_hpa": CycloneMatcher._number(row.get("ibtracs_pressure_hpa")),
            "intensity": row.get("intensity") or None,
        }
        return result

    @staticmethod
    def _number(value):
        try:
            return float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _knots_to_kmh(value):
        knots = CycloneMatcher._number(value)
        return round(knots * 1.852, 2) if knots is not None else None
