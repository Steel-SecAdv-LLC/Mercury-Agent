#!/bin/bash
# Mercury Agent — remote development environment bootstrap.
#
# Builds and installs AMA Cryptography (Mercury's PRIMARY post-quantum crypto
# backend) plus the project's ML/dev dependencies, so a fresh disposable
# container comes up with real native PQC and a runnable test/lint stack — no
# per-session manual install.
#
# AMA is not on PyPI and ships a native C library that must be compiled, so a
# plain dependency list cannot install it; this mirrors the verified procedure
# in .github/actions/build-ama-cryptography and src/omni_mercury_engine/_pqc_gate.py.
#
# Invocation — two supported paths:
#
#   1. As a session-start hook configured in the HOST environment itself
#      (kept outside this repository by design — the tree carries no
#      hosting-vendor configuration): point the host's hook at this script.
#   2. Manually, as the FIRST command in any fresh disposable container
#      (CI-like environments):
#
#          MERCURY_SETUP_FORCE=1 bash scripts/mercury_session_setup.sh
#
# The engine's PQC gate is fail-closed, so run this to completion before using
# Mercury in the session. Installs into the active Python environment: meant
# for disposable containers and CI-like environments, not developer
# workstations.
set -euo pipefail

# Safety: only modify disposable environments. Hosted remote sessions mark
# themselves via CLAUDE_CODE_REMOTE; anything else (e.g. a developer
# workstation where this fires as a hook) must opt in explicitly.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ] && [ "${MERCURY_SETUP_FORCE:-}" != "1" ]; then
  echo "mercury_session_setup: not a hosted remote session and MERCURY_SETUP_FORCE != 1 — leaving this environment untouched."
  exit 0
fi

AMA_REF="v3.2.0"                 # keep in lockstep with pyproject.toml [pqc] pin
AMA_SRC="/tmp/ama-cryptography"
AMA_LIB="${AMA_SRC}/build/lib"
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

ama_pqc_live() {
  LD_LIBRARY_PATH="${AMA_LIB}:${AMA_SRC}/build:${LD_LIBRARY_PATH:-}" python - <<'PY' 2>/dev/null
from ama_cryptography.pqc_backends import (
    DILITHIUM_AVAILABLE,
    KYBER_AVAILABLE,
    SPHINCS_AVAILABLE,
)
raise SystemExit(0 if all([DILITHIUM_AVAILABLE, KYBER_AVAILABLE, SPHINCS_AVAILABLE]) else 1)
PY
}

if ama_pqc_live; then
  echo "AMA Cryptography native PQC already present — skipping build."
else
  echo "Building AMA Cryptography ${AMA_REF} (primary PQC backend)..."
  # AMA's setup.py preflight enforces these build-system floors. --ignore-installed
  # because the base image's Debian-managed setuptools/wheel have no RECORD file
  # and so cannot be uninstalled/upgraded in place.
  python -m pip install --quiet --ignore-installed \
    "setuptools>=78.1.1" "wheel>=0.47.0" "cmake>=4.3.2"
  rm -rf "${AMA_SRC}"
  git clone --depth 1 --branch "${AMA_REF}" \
    https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git "${AMA_SRC}"
  cmake -S "${AMA_SRC}" -B "${AMA_SRC}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMA_USE_NATIVE_PQC=ON \
    -DAMA_BUILD_SHARED=ON \
    -DAMA_BUILD_STATIC=ON \
    -DAMA_BUILD_TESTS=OFF \
    -DAMA_BUILD_EXAMPLES=OFF
  cmake --build "${AMA_SRC}/build" -j "$(nproc)"
  ( cd "${AMA_SRC}" && AMA_NO_CYTHON=1 pip install --no-build-isolation --ignore-installed . )
fi

# Persist the native library path for the rest of the session so AMA's .so
# loads at import time (matches the LD_LIBRARY_PATH the PQC gate documents).
# Hosted agent harnesses export a session env file for cross-command
# persistence; honor the generic override first, then the harness's own.
ENV_FILE="${MERCURY_SETUP_ENV_FILE:-${CLAUDE_ENV_FILE:-}}"
if [ -n "${ENV_FILE}" ]; then
  echo "export LD_LIBRARY_PATH=\"${AMA_LIB}:${AMA_SRC}/build:\${LD_LIBRARY_PATH:-}\"" >> "${ENV_FILE}"
fi
export LD_LIBRARY_PATH="${AMA_LIB}:${AMA_SRC}/build:${LD_LIBRARY_PATH:-}"

# Mercury + ML stack (torch, lightning, etc.) and the CI lint/test tools, so
# tests and linters run out of the box. The [dev] extra is the single source
# of truth for the test/lint toolchain: it carries the pytest plugin set the
# suite's own configuration requires (pyproject sets ``asyncio_mode``, and the
# CI invocation uses ``-n``/``--timeout``/``--cov`` from
# pytest-xdist/-timeout/-cov) plus the black/mypy/types-requests pins that
# keep formatting and the type-ignore set byte-identical with CI. A bare
# ``pytest`` install provably cannot run the suite in CI configuration.
# flake8 and pydocstyle are not in [dev]; install them alongside, pydocstyle
# at the CI pin (mirrors the code-quality lane). AMA was installed above, so
# its cryptography>=48 is already present and this step won't try to
# uninstall the Debian-pinned one.
echo "Installing Mercury [ml,dev] extras + test/lint tooling..."
python -m pip install --quiet --ignore-installed -e "${PROJECT_DIR}[ml,dev]" \
  flake8 "pydocstyle==6.3.0"

echo "Mercury environment ready: AMA primary PQC backend live; [ml,dev] + tooling installed."
