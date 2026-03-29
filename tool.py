import joblib, glob, os
scaler = joblib.load('model/scaler.pkl')
le = joblib.load('model/label_encoder.pkl')
latest_model = max(glob.glob('model/model_*.h5'), key=os.path.getctime)
print('MEAN:', scaler.mean_.tolist())
print('SCALE:', scaler.scale_.tolist())
print('CLASSES:', le.classes_.tolist())
print('LATEST_MODEL:', latest_model)
import subprocess
try:
    subprocess.run(['tensorflowjs_converter', '--input_format', 'keras', latest_model, 'static/tfjs_model'], check=True)
    print("CONVERSION SUCCESS")
except Exception as e:
    print("CONVERSION FAILED", str(e))
