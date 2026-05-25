FROM python:3.12-slim

WORKDIR /app

# Keep Python output unbuffered for logs
ENV PYTHONUNBUFFERED=1

# Install system deps required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and artifacts
COPY . /app

EXPOSE 8000

# Default model and feature paths for container runtime
ENV MODEL_PATH=models/muscle_growth_model.pkl
ENV FEATURES_PATH=feature_names.json

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "asyncio"]
