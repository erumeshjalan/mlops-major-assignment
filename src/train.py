from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import os

# Load dataset
X, y = fetch_california_housing(return_X_y=True)

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict and evaluate
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R² Score: {r2:.6f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sklearn_model.joblib")
print("Model saved to models/sklearn_model.joblib")
