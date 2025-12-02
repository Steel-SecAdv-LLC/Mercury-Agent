FROM python:3.12-slim

LABEL maintainer="Steel Security Advisors LLC <support@steelsecurityadvisors.com>"
LABEL description="OMNI ♱ AVA: ML-Centric anomaly detection framework"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY omni_anomaly_engine/ ./omni_anomaly_engine/
COPY setup.py README.md LICENSE ./

# Install package
RUN pip install -e .

# Create directory for models and data
RUN mkdir -p /app/models /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import omni_anomaly_engine; print('OK')" || exit 1

# Entrypoint
ENTRYPOINT ["omni-anomaly"]
CMD ["--help"]
