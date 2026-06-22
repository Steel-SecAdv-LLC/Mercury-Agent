# Mercury Agent Secure Container
# Security-hardened Dockerfile with CVE mitigations
# Multi-stage build for minimal attack surface

# =============================================================================
# Stage 1: Builder - Install dependencies in a full environment
# =============================================================================
FROM python:3.14-slim-trixie AS builder

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
FROM python:3.14-slim-trixie AS runtime

# Build arguments for flexibility
ARG USERNAME=mercuryagent
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# OCI version label (build-time metadata; bump with pyproject [project].version
# or override via --build-arg MERCURY_VERSION=...). The runtime __version__ is
# SSOT-resolved at import; this image-metadata surface can't read it at build.
ARG MERCURY_VERSION=2.0.0

# Security labels
LABEL maintainer="Steel Security Advisors LLC <steel.sa.llc@gmail.com>"
LABEL org.opencontainers.image.title="Mercury Agent"
LABEL org.opencontainers.image.description="ML-Centric Multi-Domain Anomaly Detection Framework"
LABEL org.opencontainers.image.vendor="Steel Security Advisors LLC"
LABEL org.opencontainers.image.version="${MERCURY_VERSION}"
LABEL org.opencontainers.image.licenses="GPL-3.0"
LABEL security.hardened="true"
LABEL security.scan-date="2026-06-18"

# Critical security patches - updates system packages.
# ``apt-get upgrade`` here is the canonical fix path for every OS-level
# CVE that ships with a Debian trixie patch -- the blocking CI Trivy
# gates (``severity: CRITICAL,HIGH``, ``ignore-unfixed: false``,
# ``exit-code: 1``) fail on any fixable CRITICAL/HIGH finding, so this
# upgrade is what keeps them green.  The residual ``affected`` /
# ``will_not_fix`` Debian CVEs (no upstream fix available) are accepted
# only via the enumerated, 90-day-expiring ledger in the repo-root
# ``.trivyignore`` (rationale per entry; contract in SECURITY.md) and
# are additionally mitigated below by: running as a non-root user,
# stripping SUID/SGID bits from every binary, and not invoking the
# vulnerable code paths.
RUN apt-get update && \
    # adduser: the upgraded apt in the slim Debian base depends on it for the
    # dependency resolver during ``apt-get upgrade``.  Install it before the
    # upgrade, then purge it (and the perl-base it pulls) immediately after --
    # see the attack-surface-reduction step below.
    apt-get install -y --no-install-recommends adduser && \
    apt-get upgrade -y && \
    # NOTE: no libgl1-mesa-glx. Mercury depends on opencv-python-headless,
    # whose cv2 extension links no libGL/libGLX (verified: the wheel's
    # cv2.*.so has zero GL linkage), and the API container makes no cv2 GUI
    # calls. Installing the mesa GL stack only added an unused, unfixed-CVE
    # surface (CVE-2026-40393); dropping it removes the package and the CVE
    # rather than accepting it.
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libglib2.0-0 && \
    # Attack-surface reduction: ELIMINATE perl-base rather than accept its CVEs.
    # perl-base is a Debian "essential" package present in the base image, but
    # nothing in the runtime needs it -- Mercury is Python/Rust/C, and the
    # non-root user is created with ``useradd`` (from passwd) below, not
    # ``adduser``. ``adduser`` (a Perl script) is perl-base's only consumer here
    # and was installed solely to satisfy the upgrade dependency resolver above,
    # so both are purged now that all apt operations are complete. This removes
    # the perl-base CVE family -- CVE-2026-8376 / CVE-2026-42496 (Critical) and
    # CVE-2026-42497 / CVE-2026-48962 / CVE-2026-9538 (High), none with a Debian
    # trixie fix -- from the image, the same eliminate-don't-accept posture used
    # for the mesa GL stack. Verified: interpreter, sqlite3, pip and useradd all
    # work without perl; the blocking Trivy gate re-proves the package stays gone.
    apt-get purge -y --allow-remove-essential perl-base adduser && \
    # Clean up to reduce image size and attack surface
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Security hardening: strip setuid/setgid bits from all binaries.
# This replaces the previous "apt-get purge login passwd" approach which
# broke on the slim Debian base due to the apt→adduser→passwd
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
# python:3.13-slim-trixie image carries its OWN pip under /usr/local that
# Trivy detects.  Because ``ENV PATH`` puts /opt/venv/bin first, a bare
# ``python -m pip`` would upgrade the *venv* pip (already patched) and leave
# the vulnerable *system* pip in place — so target the system interpreter
# explicitly via its absolute path, and drop the bundled ensurepip wheels that
# would otherwise let ``python -m ensurepip`` re-seed a stale, pre-floor pip.
#
# The system stdlib path is derived from ``sysconfig`` rather than hardcoded to
# ``/usr/local/lib/python3.NN``.  A hardcoded minor-version path silently turns
# into a no-op the moment the base image's Python is bumped (e.g. 3.13 -> 3.14):
# the cleanup would "pass" vacuously while leaving the vulnerable bundled pip in
# the image.  Deriving the path keeps this hardening correct across base-image
# Python bumps, and the version assertions below verify *both* the system and
# venv interpreters resolve a patched pip rather than trusting a filename glob.
RUN /usr/local/bin/python -m pip install --upgrade --no-cache-dir "pip>=26.1" "setuptools>=78.1.1" && \
    SYS_STDLIB="$(/usr/local/bin/python -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')" && \
    rm -rf "${SYS_STDLIB}/ensurepip/_bundled" && \
    test ! -d "${SYS_STDLIB}/ensurepip/_bundled" && \
    /usr/local/bin/python -c "import pip; assert tuple(map(int, pip.__version__.split('.')[:2])) >= (26, 1), pip.__version__" && \
    /opt/venv/bin/python -c "import pip; assert tuple(map(int, pip.__version__.split('.')[:2])) >= (26, 1), pip.__version__" && \
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
