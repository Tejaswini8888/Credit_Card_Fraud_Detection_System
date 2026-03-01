# 💳 Credit Card Fraud Detection System (Production-Ready)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![Deployment](https://img.shields.io/badge/Deployment-Railway-purple)
![Status](https://img.shields.io/badge/Status-Live-success)

---

## 🚀 Live Demo

🌐 **Frontend + API:**  
👉 https://credit-fraud-api.up.railway.app  

📘 **Swagger Docs:**  
👉 https://credit-fraud-api.up.railway.app/docs  

---


## 📊 Project Overview

A full-stack Machine Learning system that detects fraudulent credit card transactions in real time.

This project covers the **complete ML lifecycle**:

- 📊 Data Analysis
- ⚖ Class Imbalance Handling
- 🎯 Threshold Optimization
- 🧠 Model Selection
- 🚀 API Development
- 🐳 Docker Containerization
- ☁ Cloud Deployment
- 🎨 Frontend Integration

---

## 🏗 System Architecture


User (Frontend UI)\
↓\
FastAPI Backend (Docker)\
↓\
StandardScaler (Preprocessing)\
↓\
Logistic Regression Model\
↓\
Threshold Optimization (0.3)\
↓\
Fraud Risk Classification\
↓\
JSON Response


---

## 🧠 Business Objective

Fraud detection is an **imbalanced classification problem**.

### Risk Analysis

- ❌ False Negative → Fraud missed → Financial loss
- ❌ False Positive → Genuine blocked → Customer frustration

### Strategy

We optimized for **high Recall on Fraud class** while maintaining reasonable precision.

---

## 📈 Model Performance

| Metric | Fraud Class |
|--------|-------------|
| Recall | ~0.93 |
| Precision | ~0.54 |
| ROC-AUC | ~0.95 |

Threshold optimized at:


0.3


---

## 🛠 Tech Stack

### Backend
- Python
- FastAPI
- Scikit-learn
- Uvicorn

### Frontend
- HTML
- CSS
- Vanilla JavaScript

### Deployment
- Docker
- Docker Hub
- Railway

---

## 📂 Project Structure


Credit_Card_Fraud_Detection/

│\
├── api.py\
├── Dockerfile\
├── requirements.txt\
├── models/\
│ ├── best_model.pkl\
│ ├── scaler.pkl\
├── static/\
│ └── index.html\
├── notebooks/\
│ ├── 1_EDA.ipynb\
│ ├── 2_Modeling.ipynb\
└── README.md


---

## 🐳 Run Locally with Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```
Open:

http://localhost:8000

#### 🚀 Run Without Docker
pip install -r requirements.txt\
uvicorn api:app --reload

## 👩‍💻 Author

Tejaswini Madarapu

🔗 LinkedIn:
https://www.linkedin.com/in/tejaswini-madarapu/