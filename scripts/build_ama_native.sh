#!/usr/bin/env bash
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Build and install the AMA Cryptography native PQC backend.
#
# AMA is Mercury's mandatory, sole crypto core: the agent surfaces
# (``omni_mercury_engine.engine`` — which the CLI and API serve — and
# ``omni_mercury_engine.crypto``) refuse to import unless ML-DSA-65 (Dilithium),
# Kyber-1024, and SPHINCS+ are loadable from the native C library (see
# ``omni_mercury_engine._pqc_gate``; the standalone analytic leaves import
# AMA-free for the quickstart).
# Mercury-Agent ships no ``CMakeLists.txt`` of its own; the native library lives
# in the upstream AMA-Cryptography repository, which this script clones at a
# pinned, immutable tag and builds.
#
# This is the canonical, single source of the build logic. It is invoked by:
#   * the Dockerfile builder stage (so the runtime image imports cleanly), and
#   * local/CI installs (docs/INSTALLATION.md, pqc-production-check.yml).
# It mirrors ``.github/actions/build-ama-cryptography``; keep the pinned
# ``AMA_REF`` in lockstep with that action and the pyproject ``ama-cryptography``
# git pin.
#
# After installing the Python package the freshly built ``libama_cryptography.so``
# is co-located inside the installed ``ama_cryptography`` package directory, so
# the backend loads from the module's own directory and needs no LD_LIBRARY_PATH
# at runtime (``ama_cryptography._find_native_library`` searches there first).
#
# Environment overrides:
#   AMA_REF        git tag/ref to build (default: v5.0.0)
#   AMA_REPO       repository URL (default: upstream Steel-SecAdv-LLC/AMA-Cryptography)
#   AMA_BUILD_DIR  scratch checkout/build directory (default: /tmp/ama-cryptography)
set -euo pipefail

# Preflight: name every missing prerequisite at once instead of failing midway
# with cmake's or git's own error. cmake itself is pip-installed below.
missing=""
command -v git >/dev/null 2>&1 || missing="${missing} git"
command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 \
  || command -v clang >/dev/null 2>&1 || missing="${missing} a-C-compiler(gcc/clang)"
command -v python >/dev/null 2>&1 || missing="${missing} python"
if [ -n "${missing}" ]; then
  echo "ERROR: cannot build the AMA native PQC backend; install:${missing}" >&2
  echo "  Debian/Ubuntu: apt-get install git build-essential" >&2
  echo "  RHEL/Fedora:   dnf install git gcc gcc-c++ make" >&2
  echo "  macOS:         xcode-select --install" >&2
  exit 1
fi

# Core count without Linux-only nproc (macOS has sysctl; POSIX has getconf).
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"

AMA_REF="${AMA_REF:-v5.0.0}"
AMA_REPO="${AMA_REPO:-https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git}"
AMA_BUILD_DIR="${AMA_BUILD_DIR:-/tmp/ama-cryptography}"

echo "==> Building AMA Cryptography native PQC backend (ref ${AMA_REF})"

# PEP 517 build-system floors AMA's setup.py enforces when installing with
# --no-build-isolation: setuptools>=84.0.0 (PYSEC-2025-49 + GHSA-cx63-2mw6-8hw5,
# raised to 84.0.0 by AMA v5.0.0 — anything older fails its preflight with
# ``AttributeError(install_layout) on bdist_wheel``), wheel>=0.48.0
# (GHSA-8rrh-rw8j-w5fx), and the PyPI cmake>=4.4.4 shim whose __version__ AMA's
# _check_cmake_version reads. AMA_NO_CYTHON short-circuits the Cython/numpy
# floor, so those are intentionally absent.
#
# These floors must stay in lockstep with the same three in
# .github/actions/build-ama-cryptography/action.yml: that composite action is
# what CI builds with, this script is what the Dockerfile and the session setup
# build with, and a floor raised in only one of them yields an image that fails
# AMA's preflight on exactly the lanes CI does not cover.
python -m pip install --upgrade "setuptools>=84.0.0" "wheel>=0.48.0" "cmake>=4.4.4"

rm -rf "${AMA_BUILD_DIR}"
git clone --branch "${AMA_REF}" --depth 1 "${AMA_REPO}" "${AMA_BUILD_DIR}"

cmake -S "${AMA_BUILD_DIR}" -B "${AMA_BUILD_DIR}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMA_USE_NATIVE_PQC=ON \
  -DAMA_BUILD_SHARED=ON \
  -DAMA_BUILD_STATIC=OFF \
  -DAMA_BUILD_TESTS=OFF \
  -DAMA_BUILD_EXAMPLES=OFF
cmake --build "${AMA_BUILD_DIR}/build" -j "${JOBS}"

# --no-build-isolation links the wheel against the .so just produced rather than
# re-running cmake into a transient build env. AMA_NO_CYTHON skips the optional
# Cython bindings (the Python API loads the .so via ctypes at runtime).
# This script is the sole installer of ama-cryptography (it lives in the [pqc]
# extra, which `pip install '.[all]'` does NOT pull). --no-deps keeps the install
# isolated (AMA has no runtime deps); --force-reinstall guarantees the installed
# Python package is the one built from this exact checkout (ABI-matched to the
# .so co-located below) even if a future change pre-installs it.
( cd "${AMA_BUILD_DIR}" && AMA_NO_CYTHON=1 python -m pip install \
    --no-build-isolation --force-reinstall --no-deps . )

# Co-locate the native shared object with the installed package so it loads
# without LD_LIBRARY_PATH (survives `COPY --from=builder /opt/venv` into the
# runtime image, and read-only-rootfs containers).
AMA_PKG_DIR="$(python -c 'import ama_cryptography, os; print(os.path.dirname(ama_cryptography.__file__))')"
# .so on Linux, .dylib on macOS; plain loop instead of mapfile so the script
# also runs under macOS's stock bash 3.2.
found=0
for so in $(find "${AMA_BUILD_DIR}/build" \( -name 'libama_cryptography.so*' -o -name 'libama_cryptography*.dylib' \) -print); do
  cp -P -p "${so}" "${AMA_PKG_DIR}/"
  echo "    co-located $(basename "${so}") -> ${AMA_PKG_DIR}/"
  found=1
done
if [ "${found}" -eq 0 ]; then
  echo "ERROR: AMA build produced no libama_cryptography shared library under ${AMA_BUILD_DIR}/build" >&2
  exit 1
fi

# Fail loud unless all three mandatory PQC algorithms load via the native backend
# (matches the engine's import-time gate and the build-ama-cryptography action).
python - <<'PY'
from ama_cryptography.pqc_backends import get_pqc_backend_info

info = get_pqc_backend_info()
required = ("dilithium_available", "kyber_available", "sphincs_available")
missing = [name for name in required if not info.get(name)]
if missing:
    raise SystemExit(f"AMA native PQC backend incomplete after build; missing {missing}: {info!r}")
print("==> AMA native PQC backend verified (ML-DSA-65 + Kyber-1024 + SPHINCS+)")
PY
