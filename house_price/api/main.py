from pathlib import Path
import sys
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
PREPROCESSOR = joblib.load(ROOT / "model/preprocessor.joblib")


CHECKPOINT = torch.load(ROOT / "model/model_5l.pt", map_location="cpu")
MODEL = build_cnn(CHECKPOINT["input_length"], CHECKPOINT["output_dim"], CHECKPOINT["architecture"], CHECKPOINT["task"])
MODEL.load_state_dict(CHECKPOINT["state_dict"])
MODEL.eval()
app = FastAPI(title="Vietnam Housing Price API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class HousePriceInput(BaseModel):
    address: str = Field(..., min_length=2)
    area: float = Field(..., gt=0, le=100000)
    frontage: float | None = Field(None, ge=0)
    access_road: float | None = Field(None, ge=0)
    floors: int | None = Field(None, ge=0, le=100)
    bedrooms: int | None = Field(None, ge=0, le=100)
    bathrooms: int | None = Field(None, ge=0, le=100)
    house_direction: str | None = None
    balcony_direction: str | None = None
    legal_status: str | None = None
    furniture_state: str | None = None

class PredictionResponse(BaseModel):
    predicted_price: float
    unit: str

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/predict/house-price", response_model=PredictionResponse)
def predict(payload: HousePriceInput):
    data = payload.model_dump()
    address = data.pop("address")
    row = {"Area": data.pop("area"), "Frontage": data.pop("frontage"), "Access Road": data.pop("access_road"), "Floors": data.pop("floors"), "Bedrooms": data.pop("bedrooms"), "Bathrooms": data.pop("bathrooms"), "District": address.rsplit(",", 1)[-1].strip(), "House direction": data.pop("house_direction"), "Balcony direction": data.pop("balcony_direction"), "Legal status": data.pop("legal_status"), "Furniture state": data.pop("furniture_state")}
    encoded = PREPROCESSOR.transform(pd.DataFrame([row]))
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    values = np.zeros((1, 220), dtype=np.float32)
    values[:, :min(encoded.shape[1], 220)] = encoded[:, :220]
    with torch.no_grad():
        value = float(MODEL(torch.tensor(values, dtype=torch.float32)).item())
    return {"predicted_price": round(value, 3), "unit": "tỷ VNĐ"}
