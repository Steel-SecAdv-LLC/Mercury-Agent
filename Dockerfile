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
# Python 3.13+ implements PEP 706, so the vulnerable tar fallback is never used,
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

# Remove unused sample dataset fetchers before the virtualenv is copied into the
# runtime image; deleting them only after ``COPY --from=builder`` still leaves
# the registry-bearing files visible to layer-aware image scanners.
RUN find /opt/venv -path '*/site-packages/scipy/datasets' -type d -prune -exec rm -rf {} + && \
    find /opt/venv -path '*/site-packages/skimage/data' -type d -prune -exec rm -rf {} + && \
    test -z "$(find /opt/venv -path '*/site-packages/scipy/datasets/_fetchers.py' -print -quit)" && \
    test -z "$(find /opt/venv -path '*/site-packages/skimage/data/_fetchers.py' -print -quit)"

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
LABEL org.opencontainers.image.version="1.7.0"
LABEL org.opencontainers.image.licenses="GPL-3.0"
LABEL security.hardened="true"
LABEL security.scan-date="2026-01-09"

# Critical security patches - updates system packages.
# ``apt-get upgrade`` here is the canonical fix path for every OS-level
# CVE that ships with a Debian bookworm patch -- under the CI Trivy
# policy (``--severity CRITICAL,HIGH --ignore-unfixed``) the upgraded
# image reports zero findings, which is why Mercury no longer ships a
# ``.trivyignore`` waiver list.  The residual ``affected`` /
# ``will_not_fix`` Debian CVEs (e.g. util-linux CVE-2025-14104,
# zlib CVE-2023-45853) are scoped out by ``ignore-unfixed`` and are
# additionally mitigated below by: running as a non-root user, stripping
# SUID/SGID bits from every binary, and not invoking the vulnerable
# code paths (no passwd / setpwnam usage, no minizip usage).
RUN apt-get update && \
    # adduser: the upgraded apt in python:3.13-slim-bookworm depends on it,
    # but the slim base omits it.  Install before upgrade to unblock the
    # dependency resolver.
    apt-get install -y --no-install-recommends adduser && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    # Clean up to reduce image size and attack surface
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Security hardening: strip setuid/setgid bits from all binaries.
# This replaces the previous "apt-get purge login passwd" approach which
# broke on python:3.13-slim-bookworm due to the apt→adduser→passwd
# dependency chain — purging passwd cascades into removing adduser,
# which breaks the apt package manager.  Stripping SUID/SGID bits
# achieves the same privilege-escalation mitigation without breaking
# package dependencies.
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
# The builder's venv (copied above) already ships pip>=26.1, but the base
# python:3.13-slim-bookworm image carries its OWN pip under /usr/local that
# Trivy detects (pip-26.0.1.dist-info).  Because ``ENV PATH`` puts
# /opt/venv/bin first, a bare ``python -m pip`` would upgrade the *venv* pip
# (already patched) and leave the vulnerable *system* pip in place — so target
# the system interpreter explicitly via its absolute path, and drop the
# bundled ensurepip wheels that would otherwise re-seed a stale pip dist-info.
RUN /usr/local/bin/python -m pip install --upgrade --no-cache-dir "pip>=26.1" "setuptools>=78.1.1" && \
    find /opt/venv /usr/local/lib/python3.13 -name 'pip-26.0.1.dist-info' -type d -prune -exec rm -rf {} + && \
    rm -rf /usr/local/lib/python3.13/ensurepip/_bundled && \
    test -z "$(find /opt/venv /usr/local/lib/python3.13 -name 'pip-26.0.1.dist-info' -print -quit)" && \
    /usr/local/bin/python -c "import pip; v = tuple(map(int, pip.__version__.split('.')[:2])); assert v >= (26, 1), pip.__version__" && \
    pip cache purge

# Mercury never calls SciPy/scikit-image sample datasets in production.  Drop
# those bundled fetcher packages from the runtime image so the container does
# not ship unused network-fetching demo code or upstream registry strings that
# secret scanners classify as JWT-shaped material.
RUN find /opt/venv -path '*/site-packages/scipy/datasets' -type d -prune -exec rm -rf {} + && \
    find /opt/venv -path '*/site-packages/skimage/data' -type d -prune -exec rm -rf {} + && \
    test -z "$(find /opt/venv -path '*/site-packages/scipy/datasets/_fetchers.py' -print -quit)" && \
    test -z "$(find /opt/venv -path '*/site-packages/skimage/data/_fetchers.py' -print -quit)"

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
