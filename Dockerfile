# Mercury Agent ♱ Secure Container
# Security-hardened Dockerfile with CVE mitigations
# Multi-stage build for minimal attack surface

# =============================================================================
# Stage 1: Builder - Install dependencies in a full environment
# =============================================================================
FROM python:3.12-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment for isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip to latest to address CVE-2025-8869 (symlink extraction vulnerability)
RUN pip install --no-cache-dir --upgrade pip>=25.0 setuptools>=75.0.0 wheel

# Set working directory for build
WORKDIR /app

# Copy pyproject.toml and source for installation
# Note: requirements.txt was removed in consolidation, using pyproject.toml instead
COPY pyproject.toml /app/
COPY src/ /app/src/

# Install the package with all dependencies
RUN pip install --no-cache-dir .[full]

# Copy remaining application files
COPY . /app

# =============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# =============================================================================
FROM python:3.12-slim-bookworm AS runtime

# Build arguments for flexibility
ARG USERNAME=mercuryagent
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Security labels
LABEL maintainer="Steel Security Advisory LLC <support@steelsecurityadvisors.com>"
LABEL org.opencontainers.image.title="Mercury Agent"
LABEL org.opencontainers.image.description="ML-Centric Multi-Domain Anomaly Detection Framework"
LABEL org.opencontainers.image.vendor="Steel Security Advisory LLC"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="GPL-3.0"
LABEL security.hardened="true"
LABEL security.scan-date="2026-01-06"

# Critical security patches - updates system packages
# Note: util-linux vulnerabilities (CVE-2025-14104) are mitigated by:
# 1. Running as non-root user (no SUID binary access)
# 2. Application does not handle passwd/user operations
# 3. No 256-byte username processing in application code
# See .trivyignore for detailed justifications
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    # Remove unnecessary packages to reduce attack surface
    apt-get purge -y --auto-remove \
        login \
        passwd && \
    # Clean up to reduce image size and attack surface
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    # Remove setuid/setgid binaries that are not needed
    find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true

# Create non-root user for security (principle of least privilege)
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Set secure permissions on home directory
RUN chmod 750 /home/$USERNAME

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
WORKDIR /app
COPY --chown=$USERNAME:$USER_GID . .

# Switch to non-root user
USER $USERNAME

# Security environment variables
ENV PIP_NO_WARN_SCRIPT_LOCATION=0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import omni_mercury_engine; print('healthy')" || exit 1

# Expose the API port
EXPOSE 8000

# Default to API server for production
# Override for training: docker run ... python src/mercury/train.py
CMD ["python", "-m", "uvicorn", "omni_mercury_engine.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
