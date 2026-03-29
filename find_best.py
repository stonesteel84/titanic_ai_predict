import json
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load model metadata
with open('./static/model_meta.json', 'r') as f:
    meta = json.load(f)

scaler_mean = np.array(meta['scaler']['mean'])
scaler_scale = np.array(meta['scaler']['scale'])
layers = meta['layers']

def predict(features):
    # Standard Scaling
    x = (np.array(features) - scaler_mean) / scaler_scale
    
    # Forward Pass
    for layer in layers:
        w = np.array(layer['w'])
        b = np.array(layer['b'])
        x = np.dot(x, w) + b
        if layer['activation'] == 'relu':
            x = relu(x)
        elif layer['activation'] == 'sigmoid':
            x = sigmoid(x)
    return x[0]

# Search for the best survival combination
best_prob = -1
best_combo = None

# Search Space
pclasses = [1, 2, 3]
sexes = [0, 1] # 0=Female, 1=Male (Based on previous encoder check)
ages = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80]
sibsp_vals = [0, 1, 2]
parch_vals = [0, 1, 2]
embarked_vals = [0, 1, 2] # C, Q, S

for p in pclasses:
    for s in sexes:
        for a in ages:
            for sib in sibsp_vals:
                for par in parch_vals:
                    for e in embarked_vals:
                        prob = predict([p, s, a, sib, par, e])
                        if prob > best_prob:
                            best_prob = prob
                            best_combo = {
                                "pclass": p,
                                "sex": "Female" if s == 0 else "Male",
                                "age": a,
                                "sibsp": sib,
                                "parch": par,
                                "embarked": meta['label_encoder']['classes'][e],
                                "prob": float(prob)
                            }

print(json.dumps(best_combo, indent=4))
