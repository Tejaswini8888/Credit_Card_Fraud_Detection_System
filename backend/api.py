import pickle
import numpy as np
import logging
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------
# Load Model and Scaler (FIXED ✅)
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models", "fraud_model.pkl")
scaler_path = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# ---------------------------------------------------
# Database Setup
# ---------------------------------------------------

db_path = os.path.join(BASE_DIR, "fraud_system.db")
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    amount REAL,
    time INTEGER,
    age INTEGER,
    merchant_id INTEGER,
    probability REAL,
    risk_level TEXT,
    timestamp TEXT
)
""")
conn.commit()

# ---------------------------------------------------
# Initialize FastAPI
# ---------------------------------------------------

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="2.0"
)

from fastapi.responses import FileResponse
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
def serve_home():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))
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
# Fraud Explanation
# ---------------------------------------------------

def explain_fraud(transaction, probability):
    reasons = []

    if transaction.transaction_amount > 2000:
        reasons.append("Unusually high transaction amount")

    if transaction.transaction_time < 5:
        reasons.append("Transaction at unusual time")

    if transaction.customer_age < 21:
        reasons.append("Young account holder")

    if probability > 0.7:
        reasons.append("Model detected strong fraud pattern")

    if not reasons:
        reasons.append("No strong fraud indicators")

    return reasons

# ---------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        data = np.array([[
            transaction.transaction_amount,
            transaction.transaction_time,
            transaction.customer_age,
            transaction.merchant_id,
            transaction.customer_location
        ]])

        data_scaled = scaler.transform(data)
        prob = model.predict_proba(data_scaled)[0][1]

        # Rule-based adjustments
        if transaction.transaction_amount > 4000:
            prob += 0.4
        if transaction.transaction_time < 3:
            prob += 0.2
        if transaction.customer_age < 21:
            prob += 0.2

        prob = min(prob, 1.0)

        threshold = 0.3
        prediction = int(prob >= threshold)

        if prob > 0.65:
            risk = "High Risk"
        elif prob > 0.35:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        reasons = explain_fraud(transaction, prob)

        # Save to DB
        cursor.execute("""
        INSERT INTO transactions
        (amount, time, age, merchant_id, probability, risk_level, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            transaction.transaction_amount,
            transaction.transaction_time,
            transaction.customer_age,
            transaction.merchant_id,
            float(prob),
            risk,
            datetime.now().isoformat()
        ))
        conn.commit()

        return {
            "fraud_probability": float(prob),
            "prediction": prediction,
            "risk_level": risk,
            "fraud_reasons": reasons
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": str(e)}

# ---------------------------------------------------
# Dashboard API
# ---------------------------------------------------

@app.get("/dashboard")
def fraud_dashboard():
    try:
        cursor.execute("SELECT COUNT(*) FROM transactions")
        total_transactions = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM transactions WHERE risk_level='High Risk'")
        high_risk = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM transactions WHERE risk_level='Medium Risk'")
        medium_risk = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM transactions WHERE risk_level='Low Risk'")
        low_risk = cursor.fetchone()[0]

        return {
            "total_transactions": total_transactions,
            "high_risk": high_risk,
            "medium_risk": medium_risk,
            "low_risk": low_risk
        }

    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------
# Model Metrics
# ---------------------------------------------------

@app.get("/model-metrics")
def model_metrics():
    return {
        "confusion_matrix": [[75, 11], [1, 13]],
        "accuracy": 0.88,
        "precision_class0": 0.99,
        "recall_class0": 0.87,
        "precision_class1": 0.54,
        "recall_class1": 0.93,
        "roc_auc": 0.9551
    }