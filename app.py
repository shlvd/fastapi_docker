from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib

app = FastAPI(title="Simple classifier")

model = load(pathlib.Path("model/heart-v1.joblib"))


class InputData(BaseModel):
    age: int = 64
    sex: int = 1
    cp: int = 3
    trestbps: int = 120
    chol: int = 267
    fbs: int = 0
    restecg: int = 0
    thalach: int = 99
    exang: int = 1
    oldpeak: float = 1.8
    slope: int = 1
    ca: int = 2
    thal: int = 2


class OutputData(BaseModel):
    score: float = 0.8032


@app.post('/score', response_model=OutputData)
def score(data: InputData):
    model_input = np.array([v for k,v in data.dict().items()]).reshape(1, -1)
    result = model.predict_proba(model_input)[:, -1]
    return {'score': result}






