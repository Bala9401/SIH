import os
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, IMAGE_SIZE, BATCH_SIZE, EPOCHS

def train_cnn():
    print("Training CNN...")
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
        from tensorflow.keras.models import Model
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        
        proc_dir = DATA_DIR / "processed_satellite"
        if not proc_dir.exists() or not list(proc_dir.iterdir()):
            print("Processed satellite data not found. Skipping CNN training.")
            return
            
        train_ds = tf.keras.utils.image_dataset_from_directory(
            proc_dir, validation_split=0.2, subset="training", seed=123,
            image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            proc_dir, validation_split=0.2, subset="validation", seed=123,
            image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE
        )
        class_names = train_ds.class_names
        num_classes = len(class_names)

        if num_classes < 2:
            raise ValueError("CNN needs at least two labeled folders. Current satellite folders are product sources, not intensity labels.")

        train_ds = train_ds.map(lambda images, labels: (preprocess_input(tf.cast(images, tf.float32)), labels))
        val_ds = val_ds.map(lambda images, labels: (preprocess_input(tf.cast(images, tf.float32)), labels))
        
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        loss_fn = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        callbacks = [
            EarlyStopping(patience=5),
            ModelCheckpoint(filepath=str(MODEL_DIR / "cyclone_cnn.keras"), save_best_only=True)
        ]
        
        print("Starting CNN training...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
        
        with open(MODEL_DIR / "metadata.json", "w") as f:
            json.dump({"class_names": class_names, "task": "satellite_product_classification",
                       "preprocessing": "tensorflow.keras.applications.mobilenet_v2.preprocess_input",
                       "intensity_labels_available": False}, f, indent=2)
            
        print("CNN training completed.")
    except ImportError:
        print("TensorFlow not installed. Skipping actual CNN training.")
    except Exception as e:
        print(f"Error training CNN: {e}")

if __name__ == "__main__":
    train_cnn()
