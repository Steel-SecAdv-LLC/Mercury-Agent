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
# git + cmake + ninja-build: required to clone and build the AMA Cryptography
# native PQC library (Mercury's mandatory crypto core; see
# scripts/build_ama_native.sh).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        libffi-dev \
        libssl-dev \
        libopenblas-dev \
        pkg-config \
        git \
        cmake \
        ninja-build && \
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
# setuptools floor is 83.0.0, not 78.1.1: 78.1.1 fixes only CVE-2025-47273, and
# CVE-2026-59890 has a published fix in 83.0.0. This repo's policy is that a
# fixable CVE is remediated, never accepted into .trivyignore. 83.0.0 is also
# what AMA v4.0.0's own preflight demands — the same floor
# scripts/build_ama_native.sh and .github/actions/build-ama-cryptography already
# declare — so all three build paths now state one number instead of three.
RUN pip install --no-cache-dir --upgrade "pip>=26.1" "setuptools>=83.0.0" "wheel>=0.47.0"

# Set working directory for build
WORKDIR /app

# Copy pyproject.toml and source for installation
# Note: requirements.txt was removed in consolidation, using pyproject.toml instead
COPY pyproject.toml /app/
COPY src/ /app/src/

# AMA_NO_CYTHON short-circuits AMA Cryptography's optional Cython/numpy build
# floor (the native C library is loaded via ctypes). It is consumed only by the
# AMA build in scripts/build_ama_native.sh below — `.[all]` does NOT pull the
# [pqc] extra, so it has no effect on the install on the next line. Exported here
# (builder-stage only; the runtime image is a fresh FROM) as belt-and-suspenders.
ENV AMA_NO_CYTHON=1

# Install the package with all features. AMA (the [pqc] extra) is intentionally
# NOT installed here; the native build below is its sole installer.
RUN pip install --no-cache-dir ".[all]"

# Re-apply the supply-chain floors AFTER the dependency resolver has run.
#
# Ordering is the whole point. The floors above are installed before
# ``pip install ".[all]"``, so any transitive requirement that resolves an older
# pin silently wins and the image ships the vulnerable version -- which is what
# the Container Security Scan was reporting: setuptools 70.3.0 present in the
# venv even though the earlier floor had already installed a newer one,
# alongside a clean build in the runtime prefix.
#
#   setuptools >= 83.0.0  CVE-2025-47273 (PackageIndex path traversal, fixed in
#                         78.1.1) AND CVE-2026-59890 (fixed in 83.0.0). The
#                         floor is the HIGHER of the two: ``only-if-needed``
#                         below will not move a package that already satisfies
#                         the constraint, so a floor of 78.1.1 would leave a
#                         resolved 78.1.1-82.x in place with CVE-2026-59890
#                         unfixed. A fixable CVE is remediated here, never
#                         accepted into .trivyignore.
#   msgpack    >= 1.2.1   GHSA-6v7p-g79w-8964 (out-of-bounds read on Unpacker
#                         reuse); pulled in transitively, so it has no direct
#                         declaration in pyproject to carry the floor.
#
# 83.0.0 is also exactly what AMA v4.0.0's preflight requires, so this floor and
# the one scripts/build_ama_native.sh applies a few lines below are now the same
# number rather than two that happen to converge.
#
# --upgrade-strategy only-if-needed keeps this from disturbing the resolved set
# beyond these two packages.
RUN pip install --no-cache-dir --upgrade --upgrade-strategy only-if-needed \
        "setuptools>=83.0.0" "msgpack>=1.2.1" && \
    python -c "import msgpack, setuptools; \
from packaging.version import Version; \
assert Version(setuptools.__version__) >= Version('83.0.0'), setuptools.__version__; \
assert tuple(msgpack.version) >= (1, 2, 1), msgpack.version; \
print('supply-chain floors held:', setuptools.__version__, msgpack.version)"

# Build and install the AMA Cryptography native PQC backend so the runtime image
# can import omni_mercury_engine (the import-time PQC gate requires ML-DSA-65 +
# Kyber-1024 + SPHINCS+ native availability — see omni_mercury_engine._pqc_gate).
# The shared object is co-located inside the installed ama_cryptography package
# so it travels with the venv into the runtime stage and loads without
# LD_LIBRARY_PATH. Pin matches pyproject's ama-cryptography git ref.
ARG AMA_REF=v4.0.0
COPY scripts/build_ama_native.sh /app/scripts/build_ama_native.sh
RUN AMA_REF="${AMA_REF}" bash /app/scripts/build_ama_native.sh

# Remove unused sample dataset fetchers before the virtualenv is copied into the
# runtime image; deleting them only after ``COPY --from=builder`` still leaves
# the registry-bearing files visible to layer-aware image scanners.
RUN find /opt/venv -path '*/site-packages/scipy/datasets' -type d -prune -exec rm -rf {} + && \
    find /opt/venv -path '*/site-packages/skimage/data' -type d -prune -exec rm -rf {} + && \
    test -z "$(find /opt/venv -path '*/site-packages/scipy/datasets/_fetchers.py' -print -quit)" && \
    test -z "$(find /opt/venv -path '*/site-packages/skimage/data/_fetchers.py' -print -quit)"

# The shipped image carries NO pip. pip 26.x vendors msgpack 1.1.2 and a
# setuptools 70.3.0 subset under ``pip/_vendor`` and documents them in a
# PEP 770 CycloneDX SBOM (``pip/_vendor/bom.cdx.json``) that the blocking
# Trivy gate parses — reporting GHSA-6v7p-g79w-8964 (msgpack) and
# CVE-2025-47273 (setuptools) against the image even when the venv's real
# msgpack/setuptools sit above their floors. No pip release with fixed
# vendored copies exists, so the eliminate-don't-accept posture applies (as
# with perl-base, gzip and the mesa GL stack in the runtime stage): the
# runtime container never installs packages, which makes pip build tooling,
# not a runtime dependency. First sweep any stale below-floor dist-info
# metadata (this is pip's last job here), then remove pip itself — before
# the venv is copied out, so no runtime layer ever carries it — and prove
# the package, its dist-info and its vendored SBOM are gone.
COPY scripts/enforce_wheel_floors.py /tmp/enforce_wheel_floors.py
RUN /opt/venv/bin/python /tmp/enforce_wheel_floors.py /opt/venv && \
    /opt/venv/bin/python /tmp/enforce_wheel_floors.py --check /opt/venv && \
    rm -f /tmp/enforce_wheel_floors.py && \
    /opt/venv/bin/python -m pip uninstall -y pip && \
    ! /opt/venv/bin/python -c "import pip" 2>/dev/null && \
    test -z "$(find /opt/venv -name 'bom.cdx.json' -print -quit)" && \
    test -z "$(find /opt/venv -type d -name 'pip' -path '*/site-packages/*' -print -quit)" && \
    test -z "$(find /opt/venv -type d -name 'pip-*.dist-info' -print -quit)"

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
ARG MERCURY_VERSION=2.1.0

# Security labels
LABEL maintainer="Steel Security Advisors LLC <steel.sa.llc@gmail.com>"
LABEL org.opencontainers.image.title="Mercury Agent"
LABEL org.opencontainers.image.description="ML-Centric Multi-Domain Anomaly Detection Framework"
LABEL org.opencontainers.image.vendor="Steel Security Advisors LLC"
LABEL org.opencontainers.image.version="${MERCURY_VERSION}"
LABEL org.opencontainers.image.licenses="GPL-3.0-or-later"
LABEL security.hardened="true"
LABEL security.scan-date="2026-07-02"

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
    # NOTE: no libglib2.0-0 either (2026-07-02). It was carried as cv2's
    # historical libgthread-2.0 import dependency, but the shipped
    # opencv-python-headless wheel (>= 4.13) vendors its full media stack
    # and links NO glib library (verified: readelf on the wheel's
    # cv2.abi3.so NEEDED list shows no libglib/libgthread, and cv2 import +
    # cvtColor/Canny run clean on a glib-less trixie base). Dropping it
    # eliminates the unfixed CVE-2026-58016 (Critical) and
    # CVE-2026-58014 / CVE-2026-58015 (High) rather than accepting them --
    # the same eliminate-don't-accept posture as the mesa note above.
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 && \
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
    #
    # gzip (2026-07-02): ELIMINATED on the same grounds. It is another Debian
    # "essential" package, but nothing in the runtime invokes gzip(1):
    # CPython's gzip/zlib stdlib modules use the linked libz (never the
    # binary), dpkg/apt decompress with their own internal code, and the
    # image never runs `tar -z`. Purging it removes the unfixed
    # CVE-2026-41992 (High -- LZH-decompression buffer overflow, Debian
    # trixie status "open", no fixed version) instead of accepting it.
    # Verified post-purge on the trixie base: apt-get update / install /
    # upgrade, dpkg, and `python -c "import gzip"` round-trip all work.
    apt-get purge -y --allow-remove-essential perl-base adduser gzip && \
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

# Remove the base image's system Python pip — the runtime container never
# installs packages, so pip here is pure attack surface. Elimination closes
# the pip CVE families harder than any upgrade could:
#   CVE-2025-8869 (symlink extraction)
#   CVE-2026-1703  (path traversal in wheel archives, GHSA-6vgw-5pg2-w6jp)
#   CVE-2026-6357  (arbitrary code execution via malicious wheel)
# all require an installer to exist — no pip, no install-time code path. It
# also removes pip 26.x's PEP 770 SBOM (``pip/_vendor/bom.cdx.json``), which
# names pip's vendored msgpack 1.1.2 / setuptools 70.3.0 copies and which the
# blocking Trivy gate otherwise reads as two HIGH findings with no fixed pip
# release available (see the builder-stage note — the venv's pip is removed
# there for the same reason). Same eliminate-don't-accept posture as
# perl-base, gzip and the mesa GL stack above.
#
# The bundled ensurepip wheels are dropped too, so ``python -m ensurepip``
# cannot re-seed an installer. The system stdlib path is derived from
# ``sysconfig`` rather than hardcoded to ``/usr/local/lib/python3.NN``: a
# hardcoded minor-version path silently turns into a no-op the moment the
# base image's Python is bumped (e.g. 3.13 -> 3.14) — the cleanup would
# "pass" vacuously while leaving the bundled wheels in the image. The
# negative-import probe proves absence rather than trusting a filename glob.
RUN /usr/local/bin/python -m pip uninstall -y pip && \
    SYS_STDLIB="$(/usr/local/bin/python -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')" && \
    rm -rf "${SYS_STDLIB}/ensurepip/_bundled" && \
    test ! -d "${SYS_STDLIB}/ensurepip/_bundled" && \
    ! /usr/local/bin/python -c "import pip" 2>/dev/null

# Verify — never mutate — the assembled runtime image, against what the
# scanner reads, not only what Python imports.
#
# Both of these were once true at the same time on the shipped image:
#
#   * the import assert below passed (``setuptools.__version__`` >= 83.0.0);
#   * the blocking container scan reported ``setuptools 70.3.0``
#     (CVE-2025-47273) and ``msgpack 1.1.2`` (GHSA-6v7p-g79w-8964).
#
# They measure different things. The assert reports the version Python
# *resolves* — the winner of the ``sys.path`` search. Trivy's ``python-pkg``
# analyzer reads ``*.dist-info/METADATA`` **files on disk** and any SBOM
# document it finds; it imports nothing and does not care which copy would
# win. The floors are therefore enforced and swept in the builder (where pip
# still exists to do it), and this stage re-proves everything on the final
# assembly — "verified in the stage that built it" is not "verified in the
# artifact that ships". ``--check`` mode never deletes anything: a violation
# here fails the build rather than being papered over, because deleting
# metadata at this point would blind the scanner instead of fixing the image.
# The pip-absence probes cover both interpreters and the two on-disk traces
# (dist-info, vendored SBOM) so a future base-image or builder change that
# re-introduces an installer fails loudly here, before the Trivy gate.
COPY scripts/enforce_wheel_floors.py /tmp/enforce_wheel_floors.py
RUN /opt/venv/bin/python /tmp/enforce_wheel_floors.py --check /opt/venv /usr/local /usr/lib && \
    /opt/venv/bin/python -c "import msgpack, setuptools; \
from packaging.version import Version; \
assert Version(setuptools.__version__) >= Version('83.0.0'), setuptools.__version__; \
assert tuple(msgpack.version) >= (1, 2, 1), msgpack.version; \
print('runtime venv floors held (import):', setuptools.__version__, msgpack.version)" && \
    ! /opt/venv/bin/python -c "import pip" 2>/dev/null && \
    ! /usr/local/bin/python -c "import pip" 2>/dev/null && \
    test -z "$(find /opt/venv /usr/local /usr/lib -name 'bom.cdx.json' -print -quit 2>/dev/null)" && \
    test -z "$(find /opt/venv /usr/local /usr/lib -type d -name 'pip' -path '*/site-packages/*' -print -quit 2>/dev/null)" && \
    test -z "$(find /opt/venv /usr/local /usr/lib -type d -name 'pip-*.dist-info' -print -quit 2>/dev/null)" && \
    rm -f /tmp/enforce_wheel_floors.py

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
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Fail the build loudly if the engine cannot import against the native AMA PQC
# backend baked into the venv above — the same contract the HEALTHCHECK and the
# k8s liveness/readiness probes enforce at runtime. This guarantees the shipped
# image is importable as the final non-root user rather than crash-looping on
# first start.
RUN python -c "import omni_mercury_engine; print('engine import OK — native AMA PQC backend verified')"

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import omni_mercury_engine; print('healthy')" || exit 1

# Expose the API port
EXPOSE 8000

# Default to API server for production
# Override for training: docker run ... python src/mercury/train.py
CMD ["python", "-m", "uvicorn", "omni_mercury_engine.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
