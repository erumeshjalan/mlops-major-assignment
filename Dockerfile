# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and model
COPY src/ ./src/
COPY models/ ./models/

# Default command
CMD ["python", "src/predict.py"]
