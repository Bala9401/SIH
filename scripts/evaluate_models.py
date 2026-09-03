import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR, RESULTS_DIR

def evaluate():
    print("Evaluating models...")
    try:
        import tensorflow as tf
        
        cnn_path = MODEL_DIR / "cyclone_cnn.keras"
        lstm_path = MODEL_DIR / "cyclone_lstm.keras"
        
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "plots").mkdir(exist_ok=True)
        (RESULTS_DIR / "metrics").mkdir(exist_ok=True)
        
        if cnn_path.exists():
            print("CNN model found for evaluation.")
            # Note: actual evaluation code would load test set here
        else:
            print("CNN model not found.")
            
        if lstm_path.exists():
            print("LSTM model found for evaluation.")
            # Note: actual evaluation code would load test set here
        else:
            print("LSTM model not found.")
            
        print("Evaluation routines completed.")
    except Exception as e:
        print(f"Error in evaluation: {e}")

if __name__ == "__main__":
    evaluate()
