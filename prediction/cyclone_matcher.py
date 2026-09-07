import csv
import math
import os
from datetime import datetime

import config


class CycloneMatcher:
    """Resolve uploaded image metadata through a verified mapping index only."""

    REQUIRED_COLUMNS = {
        "image_name", "cyclone_id", "cyclone_name", "timestamp",
        "latitude", "longitude", "wind_speed", "pressure", "match_method",
        "match_confidence", "source",
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
            if not self.REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
                return []
            return [row for row in reader if row.get("image_name") and row.get("cyclone_id")]

    @staticmethod
    def _same_image(left, right):
        return os.path.basename(str(left or "")).casefold() == os.path.basename(str(right or "")).casefold()

    @staticmethod
    def _parse_time(value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            try:
                return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                return None

    def match(self, image_analysis, tracks=None):
        image_analysis = image_analysis or {}
        rows = self._rows()
        image_name = image_analysis.get("image_filename")

        exact = [row for row in rows if self._same_image(row["image_name"], image_name)]
        if len(exact) == 1:
            row = exact[0]
            return self._result(row, 1.0, "verified_mapping_image_name")
        if len(exact) > 1:
            return {"matched": False, "reason": "Multiple verified mappings exist for this image name."}

        target_time = self._parse_time(image_analysis.get("timestamp"))
        latitude = image_analysis.get("latitude")
        longitude = image_analysis.get("longitude")
        if not rows or target_time is None or latitude is None or longitude is None:
            return {"matched": False, "reason": "No verified satellite-image-to-cyclone mapping available."}

        candidates = []
        for row in rows:
            row_time = self._parse_time(row.get("timestamp"))
            try:
                row_lat = float(row["latitude"])
                row_lon = float(row["longitude"])
            except (TypeError, ValueError):
                continue
            if row_time is None:
                continue
            time_hours = abs((row_time - target_time).total_seconds()) / 3600
            distance_km = math.hypot(row_lat - float(latitude), row_lon - float(longitude)) * 111
            if time_hours <= 3 and distance_km <= 300:
                candidates.append((time_hours, distance_km, row))
        if not candidates:
            return {"matched": False, "reason": "No verified mapping met the 3-hour and 300-km thresholds."}

        time_hours, distance_km, row = min(candidates, key=lambda item: (item[0], item[1], item[2]["cyclone_id"]))
        confidence = round(max(0.0, 1.0 - (time_hours / 3 + distance_km / 300) / 2), 3)
        return self._result(row, confidence, "verified_mapping_timestamp_position", time_hours, distance_km)

    @staticmethod
    def _result(row, confidence, method, time_hours=None, distance_km=None):
        result = {
            "matched": True,
            "cyclone_id": row["cyclone_id"],
            "cyclone_name": row.get("cyclone_name") or None,
            "timestamp": row.get("timestamp") or None,
            "latitude": float(row["latitude"]) if row.get("latitude") else None,
            "longitude": float(row["longitude"]) if row.get("longitude") else None,
            "match_confidence": confidence,
            "method": method,
            "source": row.get("source") or None,
        }
        if time_hours is not None:
            result["time_delta_hours"] = round(time_hours, 2)
        if distance_km is not None:
            result["distance_km"] = round(distance_km, 1)
        return result
