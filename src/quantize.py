import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_absolute_error

# Load trained model
model = joblib.load("models/sklearn_model.joblib")
weights = model.coef_
bias = model.intercept_

# Save unquantized params
joblib.dump({"weights": weights, "bias": bias}, "models/unquant_params.joblib")

# Quantization
min_val = weights.min()
max_val = weights.max()
scale = (max_val - min_val) / 255.0

quantized = np.clip(np.round((weights - min_val) / scale), 0, 255).astype(np.uint8)
dequantized = quantized.astype(np.float32) * scale + min_val

# Save quantized weights
joblib.dump({
    "weights": quantized,
    "bias": bias,
    "scale": scale,
    "min": min_val
}, "models/quant_params.joblib")

# Reconstruct and evaluate
class ReconstructedModel:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

X, y = fetch_california_housing(return_X_y=True)
reconstructed = ReconstructedModel(dequantized, bias)
y_pred = reconstructed.predict(X)

r2 = r2_score(y, y_pred)
mae = mean_absolute_error(weights, dequantized)

# Print results
print("==== Quantization Report ====")
print(f"Weight range: [{min_val:.6f}, {max_val:.6f}]")
print(f"Scale: {scale:.8f}")
print(f"Quantization Error (MAE): {mae:.8f}")
print(f"Preserved Bias: {bias:.6f}")
print(f"[QUANTIZED] R² Score: {r2:.6f}")
print("Quantization complete.")
