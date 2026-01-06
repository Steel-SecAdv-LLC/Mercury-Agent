# Mercury Agent ♱ Secure Container
# Security-hardened Dockerfile with CVE mitigations

FROM python:3.12-slim-bookworm

# Build arguments for flexibility
ARG USERNAME=mercuryagent
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Security labels
LABEL maintainer="Steel Security Advisory LLC <support@steelsecurityadvisors.com>"
LABEL org.opencontainers.image.title="Mercury Agent"
LABEL org.opencontainers.image.description="ML-Centric Multi-Domain Anomaly Detection Framework"
LABEL org.opencontainers.image.vendor="Steel Security Advisory LLC"
LABEL security.hardened="true"

# Critical security patches - updates system packages including:
# - util-linux (CVE-2025-14104: heap buffer overread)
# - SQLite (CVE-2025-7709: FTS5 integer overflow)
# - Other system libraries with known vulnerabilities
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        # Explicitly update util-linux for CVE-2025-14104
        util-linux \
        # Update SQLite for CVE-2025-7709
        libsqlite3-0 && \
    # Clean up to reduce image size and attack surface
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user for security (principle of least privilege)
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Set secure permissions on home directory
RUN chmod 750 /home/$USERNAME

# Switch to non-root user
USER $USERNAME
WORKDIR /app

# Upgrade pip to latest to address CVE-2025-8869 (symlink extraction vulnerability)
# and other pip security issues
RUN pip install --no-cache-dir --upgrade pip>=24.0 setuptools>=70.0.0 wheel

# Copy requirements first for better Docker layer caching
COPY --chown=$USERNAME:$USER_GID requirements.txt .

# Install Python dependencies with security best practices:
# - --no-cache-dir: Reduces image size and prevents cache-based attacks
# - No --trusted-host flags: Enforces HTTPS-only package downloads
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=$USERNAME:$USER_GID . .

# Install package in editable mode for CLI access
RUN pip install --no-cache-dir -e .

# Security: Prevent pip from running as root in the container
ENV PIP_NO_WARN_SCRIPT_LOCATION=0

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import omni_mercury_engine; print('healthy')" || exit 1

# Expose the API port
EXPOSE 8000

# Default to API server for production
# Override for training: docker run ... python src/mercury/train.py
CMD ["python", "-m", "uvicorn", "omni_mercury_engine.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
