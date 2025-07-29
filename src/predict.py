import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Load data
X, y = fetch_california_housing(return_X_y=True)

# Load model
model = joblib.load("models/sklearn_model.joblib")

# Predict
y_pred = model.predict(X)
score = r2_score(y, y_pred)

print(f"[PREDICT] R² Score: {score:.6f}")
