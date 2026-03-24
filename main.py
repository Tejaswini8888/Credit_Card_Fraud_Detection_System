from src.preprocessing import load_data, split_data, scale_data
from src.train import train_model, evaluate_model, save_model

# Load
df = load_data("data/creditcard.csv")

# Split
X_train, X_test, y_train, y_test = split_data(df)

# Scale
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

# Train
best_model = train_model(X_train_scaled, y_train)

# Evaluate
evaluate_model(best_model, X_test_scaled, y_test, threshold=0.3)

# Save
save_model(best_model, scaler)