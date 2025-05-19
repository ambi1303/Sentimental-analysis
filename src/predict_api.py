from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
preprocessor = joblib.load('data/preprocessor.joblib')
model = joblib.load('data/sentiment_model.joblib')

class Survey(BaseModel):
    Rating1: float
    Rating2: float
    Recommend: str
    Platform: str

@app.post("/predict")
def predict(s: Survey):
    df = pd.DataFrame([s.dict()])
    X = preprocessor.transform(df)
    pred = model.predict(X)[0]
    return {"sentiment": pred}
