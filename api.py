import pickle
import numpy as np
import logging
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ---------------------------------------------------
# Logging Configuration
# ---------------------------------------------------
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------
# Load Model and Scaler
# ---------------------------------------------------
try:
    model = pickle.load(open("models/best_model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    logging.info("Model and Scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
    raise e

# ---------------------------------------------------
# Initialize FastAPI App
# ---------------------------------------------------
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML-powered Fraud Detection System with Threshold Optimization",
    version="1.0"
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Only mount static if folder exists
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def serve_home():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
else:
    @app.get("/")
    def serve_home():
        return {"message": "Frontend not available"}
# ---------------------------------------------------
# Input Schema
# ---------------------------------------------------
class Transaction(BaseModel):
    transaction_amount: float = Field(..., gt=0)
    transaction_time: int = Field(..., ge=0)
    customer_age: int = Field(..., ge=18)
    merchant_id: int
    customer_location: int

# ---------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------
@app.post("/predict")
def predict(transaction: Transaction):

    try:
        # Convert input into model format
        data = np.array([[
            transaction.transaction_amount,
            transaction.transaction_time,
            transaction.customer_age,
            transaction.merchant_id,
            transaction.customer_location
        ]])

        # Scale input
        data_scaled = scaler.transform(data)

        # Predict probability
        prob = model.predict_proba(data_scaled)[0][1]

        logging.info(f"Fraud probability: {prob}")

        # Business threshold
        threshold = 0.3
        prediction = int(prob >= threshold)

        # Risk level interpretation
        if prob >= 0.7:
            risk = "High Risk"
        elif prob >= 0.3:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        return {
            "fraud_probability": float(prob),
            "prediction": prediction,
            "risk_level": risk,
            "threshold_used": threshold
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": "Prediction failed"}