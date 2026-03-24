# 💳 Credit Card Fraud Detection System (Production-Ready)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![Deployment](https://img.shields.io/badge/Deployment-Railway-purple)
![Status](https://img.shields.io/badge/Status-Live-success)

---

## 🚀 Live Demo

🌐 **Full Application (Frontend + API):**  
👉 https://credit-fraud-api.up.railway.app  

📘 **Interactive API Docs (Swagger):**  
👉 https://credit-fraud-api.up.railway.app/docs  

---

## 📊 Project Overview

A **production-ready end-to-end Machine Learning system** designed to detect fraudulent credit card transactions in real time.

This project demonstrates the **complete ML lifecycle from data to deployment**:

- 📊 Exploratory Data Analysis (EDA)
- ⚖ Handling severe class imbalance
- 🎯 Decision threshold optimization
- 🧠 Model training & evaluation
- 🚀 REST API development using FastAPI
- 🐳 Docker containerization
- ☁ Cloud deployment on Railway
- 🎨 Interactive frontend dashboard

---

## 🏗 System Architecture

User (Frontend UI)  
↓  
FastAPI Backend (Dockerized)  
↓  
StandardScaler (Preprocessing)  
↓  
Logistic Regression Model  
↓  
Threshold Optimization (0.3)  
↓  
Fraud Risk Classification  
↓  
JSON Response  

---

## 🧠 Business Objective

Fraud detection is a **highly imbalanced classification problem**, where:

- ❌ False Negative → Fraud goes undetected → Financial loss  
- ❌ False Positive → Legitimate transaction blocked → Poor user experience  

### 🎯 Approach

The model is optimized for:

- ✅ **High Recall on Fraud Class** (to minimize missed frauds)  
- ⚖ Balanced Precision to reduce false alarms  

---

## 📈 Model Performance

| Metric        | Value |
|--------------|------|
| Recall (Fraud) | ~0.93 |
| Precision (Fraud) | ~0.54 |
| ROC-AUC | ~0.95 |

### 🔍 Threshold Optimization
Default threshold (0.5) was adjusted to:


0.3


to improve fraud detection sensitivity.

---

## 🛠 Tech Stack

### 🔹 Backend
- Python
- FastAPI
- Scikit-learn
- Uvicorn

### 🔹 Frontend
- HTML
- CSS
- JavaScript (Vanilla)
- Chart.js

### 🔹 Deployment
- Docker
- Docker Hub
- Railway (Cloud Hosting)

---

## 📂 Project Structure


Credit_Card_Fraud_Detection/

│\
├── api.py\
├── Dockerfile\
├── requirements.txt\
├── models/
│ ├── best_model.pkl\
│ ├── scaler.pkl\
│\
├── static/\
│ └── index.html\
│
└── README.md


---

## 🐳 Run Locally with Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```
Open in browser:

http://localhost:8000


## 🚀 Run Without Docker

```
pip install -r requirements.txt
uvicorn api:app --reload
```

## ✨ Key Highlights
🚀 Real-time fraud prediction API\
🎯 Threshold tuning for business impact\
📊 Interactive dashboard visualization\
🐳 Fully containerized application\
☁ Deployed on cloud (Railway)\
💡 Designed with production readiness in mind


# 👩‍💻 Author

## Tejaswini Madarapu

🔗 LinkedIn:
https://www.linkedin.com/in/tejaswini-madarapu/