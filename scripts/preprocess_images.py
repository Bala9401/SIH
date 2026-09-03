import os
import sys
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, RESULTS_DIR, IMAGE_SIZE

def preprocess():
    print("Preprocessing satellite images...")
    try:
        raw_dir = DATA_DIR / "satellite"
        proc_dir = DATA_DIR / "processed_satellite"
        if not raw_dir.exists():
            print("No raw satellite images found.")
            return

        proc_dir.mkdir(parents=True, exist_ok=True)
        stats = {}
        
        classes = [d for d in raw_dir.iterdir() if d.is_dir()]
        for c in classes:
            c_proc = proc_dir / c.name
            c_proc.mkdir(exist_ok=True)
            files = [f for f in c.rglob("*") if f.is_file() and f.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"
            }]
            valid = 0
            for f in files:
                try:
                    img = Image.open(f).convert("RGB")
                    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                    output_path = c_proc / f.name
                    if output_path.exists():
                        output_path = c_proc / f"{f.parent.name}_{f.name}"
                    img.save(output_path)
                    valid += 1
                except Exception:
                    pass
            stats[c.name] = valid
            
        print("Preprocessing stats:", stats)
        
        if stats:
            plt.bar(stats.keys(), stats.values())
            plt.title("Class Distribution")
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(RESULTS_DIR / "class_distribution.png")
            print("Preprocessed images saved.")
    except Exception as e:
        print(f"Error preprocessing images: {e}")

if __name__ == "__main__":
    preprocess()
