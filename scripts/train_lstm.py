import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, SEQUENCE_LENGTH, LSTM_UNITS, LSTM_EPOCHS

def train_lstm():
    print("Training LSTM...")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        
        proc_dir = DATA_DIR / "processed"
        data_path = proc_dir / "track_sequences.npz"
        if not data_path.exists():
            print("Track sequences not found. Skipping LSTM training.")
            return
            
        data = np.load(data_path)
        X, y = data['X'], data['y']
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = Sequential([
            LSTM(LSTM_UNITS, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(y.shape[1])
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        callbacks = [EarlyStopping(patience=10)]
        
        print("Starting LSTM training...")
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=LSTM_EPOCHS, batch_size=32, callbacks=callbacks)
        
        model.save(MODEL_DIR / "cyclone_lstm.keras")
        print("LSTM training completed.")
        
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        metric_dir = RESULTS_DIR / "metrics"
        metric_dir.mkdir(exist_ok=True)
        with open(metric_dir / "lstm_metrics.json", "w") as f:
            json.dump({"loss": "mse"}, f)
    except ImportError:
        print("TensorFlow not installed. Skipping actual LSTM training.")
    except Exception as e:
        print(f"Error training LSTM: {e}")

if __name__ == "__main__":
    train_lstm()
