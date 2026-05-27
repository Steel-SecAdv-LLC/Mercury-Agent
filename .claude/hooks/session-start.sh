#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# Builds and installs AMA Cryptography (Mercury's PRIMARY post-quantum crypto
# backend) plus the project's ML/dev dependencies, so every fresh remote
# container comes up with real native PQC and a runnable test/lint stack — no
# manual install per session.
#
# AMA is not on PyPI and ships a native C library that must be compiled, so a
# plain dependency list cannot install it; this mirrors the verified procedure
# in .github/actions/build-ama-cryptography and src/omni_mercury_engine/_pqc_gate.py.
set -euo pipefail

# Web-only: on a local machine the developer manages their own environment.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Async: install in the background so the session starts immediately. The build
# (AMA native lib + torch) is a few minutes on a cold container; the persisted
# LD_LIBRARY_PATH and packages become available shortly after the prompt opens.
# NOTE: this is the first line of stdout so Claude Code parses the async signal.
echo '{"async": true, "asyncTimeout": 600000}'

AMA_REF="v3.2.0"                 # keep in lockstep with pyproject.toml [pqc] pin
AMA_SRC="/tmp/ama-cryptography"
AMA_LIB="${AMA_SRC}/build/lib"
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"

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

# Persist the native library path for the whole session so AMA's .so loads at
# import time (matches the LD_LIBRARY_PATH the PQC gate documents).
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export LD_LIBRARY_PATH=\"${AMA_LIB}:${AMA_SRC}/build:\${LD_LIBRARY_PATH:-}\"" >> "${CLAUDE_ENV_FILE}"
fi
export LD_LIBRARY_PATH="${AMA_LIB}:${AMA_SRC}/build:${LD_LIBRARY_PATH:-}"

# Mercury + ML stack (torch, lightning, etc.) and the CI lint/test tools, so
# tests and linters run out of the box. AMA was installed above, so its
# cryptography>=48 is already present and this step won't try to uninstall the
# Debian-pinned one.
echo "Installing Mercury [ml] extras + test/lint tooling..."
python -m pip install --quiet --ignore-installed -e "${PROJECT_DIR}[ml]" \
  pytest "black>=26.3.1,<27.0.0" ruff flake8 mypy types-requests

echo "Session ready: AMA primary PQC backend live; Mercury [ml] + tooling installed."
