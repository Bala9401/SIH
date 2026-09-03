import os
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, RESULTS_DIR

def inspect_datasets():
    print("Inspecting datasets...")
    results = {"satellite": {}, "ibtracs": {}}
    
    sat_dir = DATA_DIR / "satellite"
    if sat_dir.exists():
        classes = [d.name for d in sat_dir.iterdir() if d.is_dir()]
        total_images = 0
        class_counts = {}
        for c in classes:
            count = sum(1 for path in (sat_dir / c).rglob("*")
                        if path.is_file() and path.suffix.lower() in {
                            ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"
                        })
            class_counts[c] = count
            total_images += count
        
        results["satellite"] = {
            "exists": True,
            "classes": classes,
            "total_images": total_images,
            "class_counts": class_counts
        }
    else:
        results["satellite"] = {"exists": False}

    ibtracs_dir = DATA_DIR / "ibtracs"
    csv_files = list(ibtracs_dir.glob("*.csv")) if ibtracs_dir.exists() else []
    if csv_files:
        try:
            import pandas as pd
            df = pd.read_csv(csv_files[0], low_memory=False, skiprows=[1])
            results["ibtracs"] = {
                "exists": True,
                "file": csv_files[0].name,
                "columns": list(df.columns),
                "num_observations": len(df),
                "num_cyclones": len(df['SID'].unique()) if 'SID' in df.columns else 0
            }
        except Exception as e:
            results["ibtracs"] = {"exists": True, "error": str(e)}
    else:
        results["ibtracs"] = {"exists": False}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "dataset_summary.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {RESULTS_DIR / 'dataset_summary.json'}")

if __name__ == "__main__":
    inspect_datasets()
