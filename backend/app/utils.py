import joblib
import numpy as np

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_fraud(data):
    data = np.array(data).reshape(1, -1)
    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    return prediction, probability