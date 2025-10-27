import pickle

from typing import Dict, Any

from fastapi import FastAPI
from typing import Literal
from pydantic import BaseModel, Field
import uvicorn


# --- Define input schema ---
class LeadRecord(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


# --- Initialize app ---
app = FastAPI(title="leadscoring-prediction")


# --- Load pipeline ---
with open('pipeline_v1.bin','rb') as f_in:
    pipeline = pickle.load(f_in)


# --- Prediction function ---
def predict_single(record: dict) -> float:
    return float(pipeline.predict_proba([record])[0, 1])

# --- API endpoint ---
@app.post("/predict")
def predict(record: LeadRecord):
    prob = predict_single(record.model_dump())
    return {
        "leadscore_prob": prob,
        "convert": prob >= 0.5
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9696)



record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}