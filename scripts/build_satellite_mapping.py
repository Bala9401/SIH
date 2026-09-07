"""Build an auditable image-to-IBTrACS mapping from a verified manifest.

The satellite image collection bundled with this repository is a product
classifier dataset, not a geolocated observation archive. This script therefore
never infers storm identity from filenames, folder names, or classifier labels.
Rows are emitted only when the manifest contains complete provenance and an
exact cyclone/timestamp observation exists in the processed IBTrACS data.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_COLUMNS = [
    "image_name", "cyclone_id", "cyclone_name", "timestamp", "latitude",
    "longitude", "wind_speed", "pressure", "match_method",
    "match_confidence", "source",
]
MANIFEST_COLUMNS = {
    "image_path", "product", "cyclone_id", "cyclone_name", "timestamp_utc",
    "latitude", "longitude", "wind_speed_kt", "pressure_hpa", "source",
}
TRACK_COLUMNS = {"SID", "ISO_TIME", "LAT", "LON", "wind_kt", "pressure_hpa"}


def parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def image_exists(root: Path, image_path: str) -> bool:
    candidate = Path(image_path)
    return (root / candidate).is_file() or (root / "data" / "satellite" / candidate).is_file()


def build_mapping(project_root: Path, manifest_path: Path | None = None,
                  tracks_path: Path | None = None, output_path: Path | None = None) -> list[dict[str, str]]:
    manifest_path = manifest_path or project_root / "data" / "satellite_intensity_manifest.csv"
    tracks_path = tracks_path or project_root / "data" / "processed" / "ibtracs_ni_processed.csv"
    output_path = output_path or project_root / "data" / "satellite" / "satellite_cyclone_mapping.csv"

    manifest_rows = read_csv(manifest_path)
    tracks = read_csv(tracks_path)
    track_fields = set(tracks[0]) if tracks else set()
    if not MANIFEST_COLUMNS.issubset(set(manifest_rows[0]) if manifest_rows else set()):
        manifest_rows = []
    if tracks and not TRACK_COLUMNS.issubset(track_fields):
        tracks = []

    track_index = {}
    for track in tracks:
        timestamp = parse_timestamp(track.get("ISO_TIME", ""))
        if timestamp is not None:
            track_index[(track.get("SID", ""), timestamp)] = track

    mapping: list[dict[str, str]] = []
    for row in manifest_rows:
        required_values = [row.get(column, "").strip() for column in MANIFEST_COLUMNS]
        if not all(required_values) or not image_exists(project_root, row["image_path"]):
            continue
        timestamp = parse_timestamp(row["timestamp_utc"])
        track = track_index.get((row["cyclone_id"], timestamp)) if timestamp else None
        if track is None:
            continue
        if row["cyclone_name"].strip().casefold() != track.get("NAME", row["cyclone_name"]).strip().casefold():
            continue
        mapping.append({
            "image_name": Path(row["image_path"]).name,
            "cyclone_id": row["cyclone_id"],
            "cyclone_name": row["cyclone_name"],
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "latitude": track["LAT"],
            "longitude": track["LON"],
            "wind_speed": track["wind_kt"],
            "pressure": track["pressure_hpa"],
            "match_method": "verified_manifest_ibtracs_exact_timestamp",
            "match_confidence": "1.0",
            "source": f"{row['source']}; IBTrACS processed track: {tracks_path.relative_to(project_root)}",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(mapping)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    rows = build_mapping(args.project_root.resolve())
    print(f"Wrote {len(rows)} verified image-to-IBTrACS mappings.")
    if not rows:
        print("No rows emitted: complete, sourced image metadata and an exact IBTrACS observation are required.")


if __name__ == "__main__":
    main()