\# MLOps Major Assignment

This repository contains the complete MLOps pipeline for a Linear Regression model using the California Housing dataset.



Includes:

\- Model training

\- Unit testing

\- Manual quantization

\- Dockerization

\- GitHub Actions CI/CD



All code is maintained on a single `main` branch as per assignment guidelines.



1. Repository Structure
Key Directories and Files:

src/: Contains main Python scripts
train.py: Model training
quantize.py: Model quantization
predict.py: Model inference
models/: Stores trained and quantized model files (created during workflow)
tests/: Contains unit tests (test_train.py)
Dockerfile: Defines container environment and workflow
requirements.txt: Python dependencies
README.md: Project overview
2. Workflow Breakdown
Dockerfile (CI/CD + Dockerization):

Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Train model at build time
RUN python src/train.py

CMD ["python", "src/predict.py"]
Explanation:

Installs dependencies
Copies code
Trains model during container build (outputs model to models/)
Default command runs prediction script
Testing (tests/test_train.py):

Python
def setup_module(module):
    subprocess.run(["python", "src/train.py"], check=True)

def test_model_exists():
    assert os.path.exists("models/sklearn_model.joblib")

def test_model_accuracy():
    model = joblib.load("models/sklearn_model.joblib")
    X, y = fetch_california_housing(return_X_y=True)
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)
    assert score > 0.5
Tests ensure model file is created and accuracy is above threshold.
Typical CI/CD Steps:

Install dependencies
Train model (train.py)
Quantize model (quantize.py)
Run unit tests
Build Docker image and run prediction (predict.py)
3. Code Analysis
Main Modules:

src/train.py: Loads California Housing data, trains a linear regression model, saves to models/sklearn_model.joblib.
Output: R² score, model file.
src/quantize.py: Loads trained model, quantizes weights (manual), saves quantized weights and bias, reconstructs and evaluates quantized model.
Output: Quantization report with error metrics, quantized model files.
src/predict.py: Loads model and dataset, runs prediction, prints R² score.
tests/test_train.py: Automates training and validates model existence and accuracy.
Code Example: Quantization Output

Python
print("==== Quantization Report ====")
print(f"Weight range: [{min_val:.6f}, {max_val:.6f}]")
print(f"Scale: {scale:.8f}")
print(f"Quantization Error (MAE): {mae:.8f}")
print(f"Preserved Bias: {bias:.6f}")
print(f"[QUANTIZED] R² Score: {r2:.6f}")
print("Quantization complete.")
Typical Outputs:

Model training: R² Score: 0.606123 
Quantization: Detailed report as above
Prediction: [PREDICT] R² Score: 0.597812 
Tests: Pass/fail for model file and accuracy
Summary:
This repository implements a full MLOps pipeline for linear regression: training, quantizing, testing, and Dockerizing, with code and logs at each step.
Let me know if you want a deeper dive into any file, logs, or outputs!


