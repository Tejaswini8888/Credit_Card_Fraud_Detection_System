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
# Load Model and Scaler
# ---------------------------------------------------
model = pickle.load(open("models/fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------------------------------------------
# Database Setup
# ---------------------------------------------------
conn = sqlite3.connect("fraud_system.db", check_same_thread=False)
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def serve_home():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
else:
    @app.get("/")
    def serve_home():
        return {"message": "Frontend not available"}

@app.get("/fraud-dashboard")
def open_dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "dashboard.html"))


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

        # Add rule-based fraud indicators
        if transaction.transaction_amount > 4000:
            prob += 0.4

        if transaction.transaction_time < 3:
            prob += 0.2

        if transaction.customer_age < 21:
            prob += 0.2

        prob = min(prob, 1.0)


        threshold = 0.3
        prediction = int(prob >= threshold)

        # (better spread)
        if prob >= 0.8:
            risk = "High Risk"
        elif prob >= 0.2:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        reasons = explain_fraud(transaction, prob)

        # Save transaction to database
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
        return {"error": "Prediction failed"}





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