# OMNI ♱ AVA Dockerfile
# Multi-stage build for security and optimization
# Copyright (C) 2025 Steel Security Advisory LLC

# =============================================================================
# Stage 1: Builder - Install dependencies and build the package
# =============================================================================
FROM python:3.12-slim AS builder

# Build arguments for version tracking and cache busting
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0
ARG CACHEBUST=1

LABEL maintainer="Steel Security Advisors LLC <support@steelsecurityadvisors.com>"
LABEL description="OMNI ♱ AVA: ML-Centric anomaly detection framework - Builder Stage"

WORKDIR /build

# Install build dependencies and upgrade OS packages for security
# CACHEBUST arg forces rebuild when changed to ensure latest security patches
RUN echo "Cache bust: ${CACHEBUST}" && apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only requirements first for better layer caching
COPY requirements-core.txt ./

# Install core dependencies (lightweight, no ML frameworks)
# For ML capabilities, use the ml-enabled stage or install torch separately
RUN pip install --no-cache-dir -r requirements-core.txt && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip list --outdated --format=json | python -c "import sys, json; pkgs = json.load(sys.stdin); [print(p['name']) for p in pkgs]" | xargs -r pip install --no-cache-dir --upgrade || true

# Copy application code
COPY src/ ./src/
COPY setup.py pyproject.toml README.md LICENSE ./

# Install package in production mode
RUN pip install --no-cache-dir -e .

# =============================================================================
# Stage 2: Security Scanner - Scan for vulnerabilities
# =============================================================================
FROM builder AS security-scan

# Install security scanning tools
RUN pip install --no-cache-dir safety bandit pip-audit

# Run security scans (non-blocking, log results)
RUN safety check --output text > /tmp/safety-report.txt 2>&1 || true
RUN pip-audit --output json > /tmp/pip-audit-report.json 2>&1 || true
RUN bandit -r src/ -f json -o /tmp/bandit-report.json 2>&1 || true

# Copy reports for potential extraction
RUN mkdir -p /security-reports && \
    cp /tmp/*.txt /tmp/*.json /security-reports/ 2>/dev/null || true

# =============================================================================
# Stage 3: Production - Minimal runtime image
# =============================================================================
FROM python:3.12-slim AS production

# Cache busting argument to force fresh package installations
ARG CACHEBUST=1

# Labels following OCI Image Spec
LABEL org.opencontainers.image.title="OMNI ♱ AVA"
LABEL org.opencontainers.image.description="ML-Centric multi-domain anomaly detection framework"
LABEL org.opencontainers.image.vendor="Steel Security Advisors LLC"
LABEL org.opencontainers.image.licenses="GPL-3.0+"
LABEL org.opencontainers.image.source="https://github.com/Steel-SecAdv-LLC/OMNI-AVA"
LABEL org.opencontainers.image.documentation="https://github.com/Steel-SecAdv-LLC/OMNI-AVA/docs"

# Security: Create non-root user
RUN groupadd --gid 1000 omniava && \
    useradd --uid 1000 --gid omniava --shell /bin/bash --create-home omniava

WORKDIR /app

# Install only runtime dependencies (no build tools) and upgrade OS packages for security
# CACHEBUST arg forces rebuild when changed to ensure latest security patches
RUN echo "Cache bust: ${CACHEBUST}" && apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libopencv-core410 \
    libopencv-imgproc410 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code with correct ownership
COPY --chown=omniava:omniava src/omni_anomaly_engine/ ./omni_anomaly_engine/
COPY --chown=omniava:omniava setup.py pyproject.toml README.md LICENSE ./

# Create necessary directories with correct permissions
RUN mkdir -p /app/models /app/data /app/logs /app/cache && \
    chown -R omniava:omniava /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    TORCH_HOME=/app/models \
    OMP_NUM_THREADS=4 \
    OMNI_LOG_LEVEL=INFO \
    OMNI_CACHE_DIR=/app/cache

# Security: Run as non-root user
USER omniava

# Health check with proper timeout and intervals
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import omni_anomaly_engine; from omni_anomaly_engine.engine import OmniAnomalyEngine; print('OK')" || exit 1

# Expose API port
EXPOSE 8000

# Default entrypoint (omni-ava as defined in setup.py console_scripts)
ENTRYPOINT ["omni-ava"]
CMD ["--help"]

# =============================================================================
# Stage 4: Development - Include dev tools
# =============================================================================
FROM production AS development

# Switch to root for installing dev dependencies
USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    bandit \
    ipython

# Copy test files
COPY --chown=omniava:omniava tests/ ./tests/

# Switch back to non-root user
USER omniava

# Override entrypoint for development
ENTRYPOINT ["/bin/bash"]
CMD ["-c", "echo 'Development container ready. Use pytest, black, etc.'"]

# =============================================================================
# Stage 5: API Server - FastAPI deployment
# =============================================================================
FROM production AS api-server

# Install uvicorn for API serving
USER root
RUN pip install --no-cache-dir uvicorn[standard]
USER omniava

# Environment for API server
ENV OMNI_API_HOST=0.0.0.0 \
    OMNI_API_PORT=8000 \
    OMNI_API_WORKERS=4

# Health check for API
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${OMNI_API_PORT}/health || exit 1

# Expose API port
EXPOSE 8000

# Start API server
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["omni_anomaly_engine.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
