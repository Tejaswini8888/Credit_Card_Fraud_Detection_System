import os
import pickle
import numpy as np
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app import engine, SessionLocal, Base
from backend.app import models
from backend.app import UserCreate, Transaction
from backend.app import hash_password, verify_password, create_access_token
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Enterprise Fraud Monitoring System")

# Load model safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "fraud_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- USER ROUTES ----------------

@app.get("/")
def home():
    return {"message": "Enterprise Fraud Monitoring System Running"}
@app.post("/register")

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):

    existing_user = db.query(models.User).filter(
        models.User.username == user.username
    ).first()

    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed = hash_password(user.password)

    db_user = models.User(
        username=user.username,
        hashed_password=hashed
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return {"message": "User created"}

@app.post("/login")
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

# ---------------- PREDICT ROUTE ----------------

@app.post("/predict")
def predict(transaction: Transaction, db: Session = Depends(get_db)):

    data = np.array([[
        transaction.transaction_amount,
        transaction.transaction_time,
        transaction.customer_age,
        transaction.merchant_id,
        transaction.customer_location
    ]])

    data_scaled = scaler.transform(data)
    prob = model.predict_proba(data_scaled)[0][1]

    threshold = 0.3
    prediction = int(prob >= threshold)

    if prob >= 0.7:
        risk = "High Risk"
    elif prob >= 0.3:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    record = models.TransactionRecord(
        transaction_amount=transaction.transaction_amount,
        transaction_time=transaction.transaction_time,
        customer_age=transaction.customer_age,
        merchant_id=transaction.merchant_id,
        customer_location=transaction.customer_location,
        fraud_probability=float(prob),
        prediction=prediction,
        risk_level=risk
    )

    db.add(record)
    db.commit()

    return {
        "fraud_probability": float(prob),
        "prediction": prediction,
        "risk_level": risk
    }