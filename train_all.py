import os
import subprocess
import sys

def main():
    scripts = [
        "inspect_dataset.py",
        "analyze_satellite_dataset.py",
        "preprocess_images.py",
        "train_cnn.py",
        "preprocess_ibtracs.py",
        "train_lstm.py",
        "evaluate_models.py"
    ]

    total_scripts = len(scripts)
    success = []
    failed = []

    print(f"Starting execution of {total_scripts} scripts in sequence...")
    print("-" * 50)

    for i, script in enumerate(scripts, 1):
        print(f"[{i}/{total_scripts}] Running {script}...")
        
        if not os.path.exists(script):
            print(f"  ERROR: {script} not found in the current directory.")
            failed.append(script)
            continue
            
        try:
            result = subprocess.run([sys.executable, script], check=True)
            print(f"  SUCCESS: {script} completed.")
            success.append(script)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {script} encountered an error (Code {e.returncode}).")
            failed.append(script)
        print("-" * 50)

    print("\nFinal Summary:")
    print(f"Successfully ran: {len(success)} scripts.")
    for s in success:
        print(f"  - {s}")
    
    print(f"Failed to run: {len(failed)} scripts.")
    for s in failed:
        print(f"  - {s}")

if __name__ == "__main__":
    main()
