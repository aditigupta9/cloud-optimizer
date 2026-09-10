# Production-Grade Lightweight Python Base
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable immediate stdout flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

# Expose Flask telemetry dashboard port
EXPOSE 5000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/api/fleet_status || exit 1

# Execute Cloud AI Auto-Scaler Telemetry Server
CMD ["python", "app_flask.py"]