import os
import json
import joblib
import keras
import glob

def load_model_and_scaler(model_dir):
    model_files = glob.glob(os.path.join(model_dir, 'model_*.h5'))
    if not model_files:
        print("No model file found.")
        return None, None
    model_path = max(model_files, key=os.path.getctime)
    
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
    # Extract weights
    weights = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            w, b = layer.get_weights()
            weights.append({
                "w": w.tolist(),
                "b": b.tolist(),
                "activation": layer.activation.__name__
            })

    label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    label_encoder = joblib.load(label_encoder_path)
    
    metadata = {
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        },
        "label_encoder": {
            "classes": label_encoder.classes_.tolist()
        },
        "layers": weights
    }
    
    os.makedirs('./static', exist_ok=True)
    with open('./static/model_meta.json', 'w') as f:
        json.dump(metadata, f)
        
    print("Exported pure JSON weights and metadata to ./static/model_meta.json")
