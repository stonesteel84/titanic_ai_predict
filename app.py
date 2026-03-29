from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import joblib
import pandas as pd
from predict import load_model_and_scaler, prepare_data_for_prediction

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model_dir = './model'
try:
    model, scaler = load_model_and_scaler(model_dir)
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
except Exception as e:
    print(f"Error loading model: {e}")
    model, scaler, label_encoder = None, None, None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/result", response_class=HTMLResponse)
async def get_result(
    request: Request,
    pclass: int,
    sex: int,
    age: float,
    sibsp: int,
    parch: int,
    embarked: str
):
    if not model:
        return templates.TemplateResponse("result.html", {"request": request, "error": "Model not loaded properly"})

    try:
        df = pd.DataFrame([{
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Embarked': embarked.upper()
        }])
        
        input_scaled = prepare_data_for_prediction(df, label_encoder, scaler)
        prediction = model.predict(input_scaled)
        prob = float(prediction[0][0])
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "survived": prob > 0.5,
            "probability": round(prob * 100, 2)
        })
    except Exception as e:
        return templates.TemplateResponse("result.html", {"request": request, "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
