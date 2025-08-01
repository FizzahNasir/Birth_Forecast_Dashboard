import joblib
import pandas as pd

# Load model and features
model = joblib.load("best_model.pkl")
features = joblib.load("model_features.pkl")

print("✅ Model and features loaded successfully!")
print("Total features:", len(features))

# Create dummy test data
test_data = pd.DataFrame([[100, 5, 4, 1, 5000, 4800]], columns=features[:6])  # Replace with your actual feature names

# Predict
prediction = model.predict(test_data)
print("Predicted Births:", prediction)
