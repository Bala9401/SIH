import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, RESULTS_DIR


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
PRODUCT_DIRS = {
    "reference": DATA_DIR / "satellite" / "insat3d_for_reference_ds",
    "infrared": DATA_DIR / "satellite" / "insat3d_ir_cyclone_ds",
    "raw": DATA_DIR / "satellite" / "insat3d_raw_cyclone_ds",
}


def image_inventory():
    inventory = []
    unreadable = []
    for product, root in PRODUCT_DIRS.items():
        for path in root.rglob("*") if root.exists() else []:
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    exif = image.getexif()
                    inventory.append({"product": product, "path": str(path.relative_to(DATA_DIR)),
                        "filename": path.name, "stem": path.stem, "size": list(image.size),
                        "mode": image.mode, "exif_datetime": exif.get(306), "cyclone_id": None})
            except (OSError, ValueError) as error:
                unreadable.append({"path": str(path.relative_to(DATA_DIR)), "error": str(error)})
    return inventory, unreadable


def load_ibtracs():
    files = sorted((DATA_DIR / "ibtracs").glob("*.csv"))
    if not files:
        return pd.DataFrame()
    frame = pd.read_csv(files[0], low_memory=False, skiprows=[1])
    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], errors="coerce")
    for column in ["LAT", "LON", "WMO_WIND", "WMO_PRES"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["SID", "ISO_TIME"])


def analyze():
    images, unreadable = image_inventory()
    labels_path = DATA_DIR / "satellite" / "insat_3d_ds - Sheet.csv"
    labels = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame()
    label_map = {}
    if {"img_name", "label"}.issubset(labels.columns):
        label_map = {str(row.img_name): row.label for row in labels.itertuples()}
    for item in images:
        item["metadata_label"] = label_map.get(item["filename"])

    ibtracs = load_ibtracs()
    timestamped = [item for item in images if item["exif_datetime"]]
    exif_dates = []
    for item in timestamped:
        try:
            exif_dates.append(datetime.strptime(item["exif_datetime"], "%Y:%m:%d %H:%M:%S"))
        except ValueError:
            pass

    # Timestamp-only proximity is reported for audit, but never treated as a match:
    # no image contains a cyclone ID or a documented product-to-storm mapping.
    timestamp_candidates = 0
    timestamp_candidates_with_complete_meteorology = 0
    if not ibtracs.empty and exif_dates:
        observed = ibtracs["ISO_TIME"].dropna().tolist()
        complete_observed = ibtracs.dropna(subset=["WMO_WIND", "WMO_PRES"])["ISO_TIME"].tolist()
        for value in exif_dates:
            if any(abs((candidate.to_pydatetime() - value).total_seconds()) <= 3 * 3600 for candidate in observed):
                timestamp_candidates += 1
            if any(abs((candidate.to_pydatetime() - value).total_seconds()) <= 3 * 3600 for candidate in complete_observed):
                timestamp_candidates_with_complete_meteorology += 1

    common_stems = set()
    by_product = {}
    for product in PRODUCT_DIRS:
        stems = {item["stem"] for item in images if item["product"] == product}
        by_product[product] = len(stems)
        common_stems = stems if not common_stems else common_stems & stems

    report = {
        "image_count": len(images),
        "unreadable_image_count": len(unreadable),
        "unreadable_images": unreadable,
        "product_counts": dict(Counter(item["product"] for item in images)),
        "product_unique_stems": by_product,
        "aligned_product_stems": len(common_stems),
        "images_with_exif_datetime": len(timestamped),
        "exif_datetime_range": [min(exif_dates).isoformat(), max(exif_dates).isoformat()] if exif_dates else None,
        "metadata_csv": {
            "path": str(labels_path.relative_to(DATA_DIR)) if labels_path.exists() else None,
            "rows": len(labels),
            "columns": list(labels.columns),
            "numeric_label_values": sorted(pd.to_numeric(labels["label"], errors="coerce").dropna().unique().tolist()) if "label" in labels else [],
            "label_semantics_verified": False,
        },
        "ibtracs": {
            "rows": len(ibtracs),
            "storm_ids": int(ibtracs["SID"].nunique()) if not ibtracs.empty else 0,
            "time_range": [ibtracs["ISO_TIME"].min().isoformat(), ibtracs["ISO_TIME"].max().isoformat()] if not ibtracs.empty else None,
        },
        "timestamp_only_candidates_within_3h": timestamp_candidates,
        "timestamp_candidates_with_complete_wind_pressure_within_3h": timestamp_candidates_with_complete_meteorology,
        "reliable_image_ibtracs_matches": 0,
        "intensity_training_allowed": False,
        "reason": "Images have no cyclone ID and the numeric metadata labels have no documented units or provenance. Timestamp-only proximity is not sufficient for image-level IBTrACS labeling.",
        "recommendation": "Add a manifest containing image path, product, cyclone ID, UTC acquisition time, and verified wind/pressure source before training image intensity regression or classification.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "satellite_dataset_report.json"
    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    analyze()
