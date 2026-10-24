from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# Application setup
# --------------------------------------------------

app = FastAPI(
    title="Delivery Time Prediction API",
    description="API for predicting food delivery time.",
    version="1.0.0",
)

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

FEATURES = [
    "distance_km",
    "prep_time_min",
    "traffic_level",
    "rain",
]


# --------------------------------------------------
# Request format and validation
# --------------------------------------------------

class Order(BaseModel):
    distance_km: float = Field(gt=0, le=25)
    prep_time_min: float = Field(ge=0)
    traffic_level: int = Field(ge=1, le=3)
    rain: int = Field(ge=0, le=1)


class CompareOrders(BaseModel):
    order1: Order
    order2: Order


def predict_delivery(order: Order) -> float:
    """Return the predicted delivery time in minutes."""

    features = pd.DataFrame(
        [
            {
                "distance_km": order.distance_km,
                "prep_time_min": order.prep_time_min,
                "traffic_level": order.traffic_level,
                "rain": order.rain,
            }
        ],
        columns=FEATURES,
    )

    prediction = model.predict(features)[0]

    return round(float(prediction), 2)


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Delivery Time Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": "delivery-time-model",
        "model_version": "1.0.0",
        "model_type": type(model).__name__,
        "features": FEATURES,
        "max_distance_km": 25,
    }


@app.post("/predict")
def predict(order: Order):
    delivery_time = predict_delivery(order)

    return {
        "delivery_min": delivery_time,
        "model_version": "1.0.0",
    }


@app.post("/compare")
def compare_orders(request: CompareOrders):
    prediction1 = predict_delivery(request.order1)
    prediction2 = predict_delivery(request.order2)

    return {
        "order1_delivery_min": prediction1,
        "order2_delivery_min": prediction2,
        "faster_order": (
            "order1"
            if prediction1 < prediction2
            else "order2"
            if prediction2 < prediction1
            else "equal"
        ),
        "difference_min": round(
            abs(prediction1 - prediction2), 2
        ),
        "model_version": "1.0.0",
    }