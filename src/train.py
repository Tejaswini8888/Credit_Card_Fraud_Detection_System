import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_model(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    model = LogisticRegression(max_iter=1000, class_weight='balanced')

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='recall',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test, threshold=0.3):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_probs))

def save_model(model, scaler):
    with open("models/fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)