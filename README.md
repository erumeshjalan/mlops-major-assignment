\# MLOps Major Assignment

This repository contains the complete MLOps pipeline for a Linear Regression model using the California Housing dataset.



Includes:

\- Model training

\- Unit testing

\- Manual quantization

\- Dockerization

\- GitHub Actions CI/CD



All code is maintained on a single `main` branch as per assignment guidelines.


**1. Repository Structure**
Key files and folders:

src/
train.py: Trains a linear regression model.
quantize.py: Quantizes model weights, saves quantized parameters, reconstructs and evaluates quantized model.
predict.py: Loads trained model and runs predictions.
models/
Stores trained (sklearn_model.joblib), quantized, and unquantized parameter files.
tests/
test_train.py: Validates model existence and accuracy.
Dockerfile: Builds the environment and runs training/prediction inside a container.
requirements.txt: Python dependencies.
README.md: Full overview and documentation.

**2. Workflow Breakdown**
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/127fd74d-01da-47ae-9129-b3e8efb63063" />


Dockerfile installs dependencies, copies code, trains the model at build time, and runs prediction on startup.
CI/CD Steps:
Install dependencies
Train model (train.py)
Quantize model (quantize.py)
Run unit tests (test_train.py)
Build Docker image and run prediction (predict.py)

**Model Training**

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c6757081-7357-4d61-a2be-320ae4fade26" />

Inputs: California Housing dataset.
Outputs: Trained model file (models/sklearn_model.joblib), R² score printed.

**Manual Quantization**

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e42bfb94-d1ff-4774-8135-02555efd5ae7" />

Inputs: Trained model weights and bias.
Outputs: Quantized weights, quantization report, reconstructed model evaluation.

**Prediction**

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/857cac7d-2ca4-4edd-84cb-d5c9f97a1704" />

**Testing**

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e19d39c3-2afa-465a-b15c-4b6f8481e75b" />

**Typical Outputs:**
**Training:**
R² Score: 0.606123
Model saved to models/sklearn_model.joblib

**Quantization:**
Weight range: [min_val, max_val]
Scale: scale_value
Quantization Error (MAE): mae_value
Preserved Bias: bias_value
[QUANTIZED] R² Score: r2_value
Quantization complete.

**Prediction:**
[PREDICT] R² Score: 0.597812
