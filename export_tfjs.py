import os
import json
import joblib
import keras
import glob
import tensorflowjs as tfjs

def load_model_and_scaler(model_dir):
    model_files = glob.glob(os.path.join(model_dir, 'model_*.h5'))
    if not model_files:
        print("No model file found.")
        return None, None
    model_path = max(model_files, key=os.path.getctime) # Use the latest trained model
    
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    if not os.path.exists(scaler_path):
        print("Scaler not found.")
        return None, None
        
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

model_dir = './model'
model, scaler = load_model_and_scaler(model_dir)

if model is not None:
    # 1. Export the Keras model to TensorFlow.js format
    tfjs_target_dir = './static/tfjs_model'
    os.makedirs(tfjs_target_dir, exist_ok=True)
    tfjs.converters.save_keras_model(model, tfjs_target_dir)
    print(f"Model exported to {tfjs_target_dir}")

    # 2. Export Scaler and LabelEncoder info
    label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    label_encoder = joblib.load(label_encoder_path)
    
    metadata = {
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        },
        "label_encoder": {
            "classes": label_encoder.classes_.tolist()
        }
    }
    
    with open('./static/model_meta.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Metadata (scaler & label encoder) exported to ./static/model_meta.json")
