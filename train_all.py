import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_script(script_name):
    print(f"\n{'='*50}\nRunning {script_name}\n{'='*50}")
    script_path = BASE_DIR / "scripts" / script_name
    try:
        result = subprocess.run([sys.executable, str(script_path)], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_name}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error with {script_name}: {e}")
        return False

def main():
    print("Starting AI Cyclone Early Warning System Pipeline")
    
    scripts = [
        ("Inspecting datasets", "inspect_dataset.py"),
        ("Preparing satellite images", "preprocess_images.py"),
        ("Training CNN", "train_cnn.py"),
        ("Preparing IBTrACS", "preprocess_ibtracs.py"),
        ("Training LSTM", "train_lstm.py"),
        ("Evaluating models", "evaluate_models.py")
    ]
    
    any_failed = False
    for i, (desc, script) in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] {desc}")
        success = run_script(script)
        if not success:
            print(f"Step [{i}/{len(scripts)}] failed.")
            any_failed = True
            
    if any_failed:
        print("\nSome steps failed. Ensure DEMO_MODE is True in config.py for the application to run with fallback data.")
    else:
        print("\nAll steps completed successfully.")

if __name__ == "__main__":
    main()
