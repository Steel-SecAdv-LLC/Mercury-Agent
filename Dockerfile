# Mercury Agent Secure Container
# Security-hardened Dockerfile with CVE mitigations
# Multi-stage build for minimal attack surface

# =============================================================================
# Stage 1: Builder - Install dependencies in a full environment
# =============================================================================
FROM python:3.13-slim-bookworm AS builder

# Install build dependencies
# gfortran + libopenblas-dev + pkg-config: required when pip falls back to
# building scipy from source (no pre-built wheel for the target ABI).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        libffi-dev \
        libssl-dev \
        libopenblas-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment for isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip to latest to address:
#   CVE-2025-8869 (symlink extraction in sdist archives)
#   CVE-2026-1703  (path traversal in wheel archives, GHSA-6vgw-5pg2-w6jp)
#   CVE-2026-6357  (arbitrary code execution via malicious wheel, fixed in 26.1)
# Python 3.13 implements PEP 706, so the vulnerable tar fallback is never used,
# but we pin to >=26.1 as defense-in-depth and to fully resolve all three CVEs.
RUN pip install --no-cache-dir --upgrade "pip>=26.1" "setuptools>=78.1.1" wheel

# Set working directory for build
WORKDIR /app

# Copy pyproject.toml and source for installation
# Note: requirements.txt was removed in consolidation, using pyproject.toml instead
COPY pyproject.toml /app/
COPY src/ /app/src/

# Install the package with all dependencies
RUN pip install --no-cache-dir ".[all]"

# Copy remaining application files
COPY . /app

# =============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# =============================================================================
FROM python:3.13-slim-bookworm AS runtime

# Build arguments for flexibility
ARG USERNAME=mercuryagent
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Security labels
LABEL maintainer="Steel Security Advisors LLC <steel.sa.llc@gmail.com>"
LABEL org.opencontainers.image.title="Mercury Agent"
LABEL org.opencontainers.image.description="ML-Centric Multi-Domain Anomaly Detection Framework"
LABEL org.opencontainers.image.vendor="Steel Security Advisors LLC"
LABEL org.opencontainers.image.version="1.6.0"
LABEL org.opencontainers.image.licenses="GPL-3.0"
LABEL security.hardened="true"
LABEL security.scan-date="2026-01-09"

# Critical security patches - updates system packages
# Note: util-linux vulnerabilities (CVE-2025-14104) are mitigated by:
# 1. Running as non-root user (no SUID binary access)
# 2. Application does not handle passwd/user operations
# 3. No 256-byte username processing in application code
# See .trivyignore for detailed justifications
RUN apt-get update && \
    apt-get upgrade -y && \
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
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Remove setuid/setgid binaries that are not needed (separate layer so
# an apt failure above is never silently masked by the trailing || true).
RUN find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true

# Create non-root user for security (principle of least privilege)
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Set secure permissions on home directory
RUN chmod 750 /home/$USERNAME

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade the base image's system Python pip to fix:
#   CVE-2025-8869 (symlink extraction)
#   CVE-2026-1703  (path traversal in wheel archives, GHSA-6vgw-5pg2-w6jp)
#   CVE-2026-6357  (arbitrary code execution via malicious wheel)
# The builder's venv already has pip>=26.1 via the copy above, but the base
# python:3.13-slim-bookworm image ships its own pip that Trivy detects.
RUN python -m pip install --upgrade --no-cache-dir "pip>=26.1" "setuptools>=78.1.1" && \
    pip cache purge

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
