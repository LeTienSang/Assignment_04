from pathlib import Path
import sys
import hashlib

import joblib
import pandas as pd
import numpy as np
import torch
from torch import nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.cnn_models import build_cnn

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
FEATURES = ["views", "cart_additions", "total_spent", "days_since_last_active"]

preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")


checkpoint = torch.load(MODEL_DIR / "model_5l.pt", map_location="cpu")
model = build_cnn(checkpoint["input_length"], checkpoint["output_dim"], checkpoint["architecture"], checkpoint["task"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()

app = FastAPI(title="Customer Behavior API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerBehavior(BaseModel):
    views: int = Field(..., ge=0, le=10000)
    cart_additions: int = Field(..., ge=0, le=1000)
    total_spent: float = Field(..., ge=0, le=1_000_000)
    days_since_last_active: int = Field(..., ge=0, le=3650)


class PredictionResponse(BaseModel):
    purchase_probability: float
    segment: str
    churn_risk: str


def classify_segment(probability: float) -> str:
    if probability >= 0.7:
        return "High intent"
    if probability >= 0.4:
        return "Consideration"
    return "Low intent"


def classify_churn(days: int) -> str:
    if days >= 45:
        return "High"
    if days >= 14:
        return "Medium"
    return "Low"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
@app.post("/predict/ecommerce-behavior", response_model=PredictionResponse)
def predict(customer: CustomerBehavior) -> PredictionResponse:
    values = pd.DataFrame([customer.model_dump()])[FEATURES]
    values["spend_per_view"] = values["total_spent"] / (values["views"] + 1)
    values["activity_score"] = values["views"] + values["cart_additions"] * 2 - values["days_since_last_active"] * .1
    tabular = preprocessor.transform(values[["views", "cart_additions", "total_spent", "days_since_last_active", "spend_per_view", "activity_score"]]).astype(np.float32)
    texts = values[["views", "cart_additions", "total_spent", "days_since_last_active", "spend_per_view", "activity_score"]].astype(str).agg("|".join, axis=1)
    embedding = np.array([[int.from_bytes(hashlib.sha256(f"{text}:{index}".encode()).digest()[:4], "big") / 2**32 for index in range(100)] for text in texts], dtype=np.float32)
    transformed = np.hstack([tabular, embedding])
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.tensor(transformed, dtype=torch.float32)), dim=1).numpy()[0]
    probability = float(probabilities[2] + probabilities[3])
    return PredictionResponse(
        purchase_probability=round(probability, 4),
        segment=classify_segment(probability),
        churn_risk=classify_churn(customer.days_since_last_active),
    )
