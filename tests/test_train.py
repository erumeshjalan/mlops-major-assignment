import os
import joblib
import subprocess
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Run training script before tests
def setup_module(module):
    subprocess.run(["python", "src/train.py"], check=True)

def test_model_exists():
    assert os.path.exists("models/sklearn_model.joblib"), "Model file not found!"

def test_model_accuracy():
    model = joblib.load("models/sklearn_model.joblib")
    X, y = fetch_california_housing(return_X_y=True)
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)
    assert score > 0.5, f"Model R² too low: {score}"
