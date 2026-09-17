from pathlib import Path
import sys
import joblib
import pandas as pd
import torch
from torch import nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.cnn_models import build_cnn

ROOT = Path(__file__).resolve().parents[1]
PREPROCESSOR = joblib.load(ROOT / "model/preprocessor.joblib")


CHECKPOINT = torch.load(ROOT / "model/model_5l.pt", map_location="cpu")
MODEL = build_cnn(CHECKPOINT["input_length"], CHECKPOINT["output_dim"], CHECKPOINT["architecture"], CHECKPOINT["task"])
MODEL.load_state_dict(CHECKPOINT["state_dict"])
MODEL.eval()
app = FastAPI(title="Diabetes Prediction API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class DiabetesInput(BaseModel):
    pregnancies: int = Field(..., ge=0, le=30)
    glucose: float = Field(..., ge=0, le=500)
    blood_pressure: float = Field(..., ge=0, le=300)
    skin_thickness: float = Field(..., ge=0, le=150)
    insulin: float = Field(..., ge=0, le=1000)
    bmi: float = Field(..., ge=0, le=100)
    diabetes_pedigree_function: float = Field(..., ge=0, le=3)
    age: int = Field(..., ge=1, le=120)

class PredictionResponse(BaseModel):
    disease_probability: float
    prediction: int
    risk_level: str

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/predict/diabetes", response_model=PredictionResponse)
def predict(payload: DiabetesInput):
    values = pd.DataFrame([payload.model_dump()]).rename(columns={"pregnancies":"Pregnancies", "glucose":"Glucose", "blood_pressure":"BloodPressure", "skin_thickness":"SkinThickness", "insulin":"Insulin", "bmi":"BMI", "diabetes_pedigree_function":"DiabetesPedigreeFunction", "age":"Age"})
    transformed = PREPROCESSOR.transform(values)
    with torch.no_grad():
        probability = float(MODEL(torch.tensor(transformed, dtype=torch.float32)).item())
    return {"disease_probability": round(probability, 4), "prediction": int(probability >= .5), "risk_level": "Cao" if probability >= .7 else "Trung bình" if probability >= .4 else "Thấp"}
