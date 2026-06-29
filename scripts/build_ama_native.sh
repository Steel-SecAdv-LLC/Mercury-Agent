#!/usr/bin/env bash
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Build and install the AMA Cryptography native PQC backend.
#
# AMA is Mercury's mandatory, sole crypto core: ``import omni_mercury_engine``
# refuses to proceed unless ML-DSA-65 (Dilithium), Kyber-1024, and SPHINCS+ are
# loadable from the native C library (see ``omni_mercury_engine._pqc_gate``).
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
#   AMA_REF        git tag/ref to build (default: v3.2.0)
#   AMA_REPO       repository URL (default: upstream Steel-SecAdv-LLC/AMA-Cryptography)
#   AMA_BUILD_DIR  scratch checkout/build directory (default: /tmp/ama-cryptography)
set -euo pipefail

AMA_REF="${AMA_REF:-v3.2.0}"
AMA_REPO="${AMA_REPO:-https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git}"
AMA_BUILD_DIR="${AMA_BUILD_DIR:-/tmp/ama-cryptography}"

echo "==> Building AMA Cryptography native PQC backend (ref ${AMA_REF})"

# PEP 517 build-system floors AMA's setup.py enforces when installing with
# --no-build-isolation: setuptools>=78.1.1 (PYSEC-2025-49 + GHSA-cx63-2mw6-8hw5),
# wheel>=0.47.0 (GHSA-8rrh-rw8j-w5fx), and the PyPI cmake>=4.3.2 shim whose
# __version__ AMA's _check_cmake_version reads. AMA_NO_CYTHON short-circuits the
# Cython/numpy floor, so those are intentionally absent.
python -m pip install --upgrade "setuptools>=78.1.1" "wheel>=0.47.0" "cmake>=4.3.2"

rm -rf "${AMA_BUILD_DIR}"
git clone --branch "${AMA_REF}" --depth 1 "${AMA_REPO}" "${AMA_BUILD_DIR}"

cmake -S "${AMA_BUILD_DIR}" -B "${AMA_BUILD_DIR}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMA_USE_NATIVE_PQC=ON \
  -DAMA_BUILD_SHARED=ON \
  -DAMA_BUILD_STATIC=OFF \
  -DAMA_BUILD_TESTS=OFF \
  -DAMA_BUILD_EXAMPLES=OFF
cmake --build "${AMA_BUILD_DIR}/build" -j "$(nproc)"

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
mapfile -t SO_FILES < <(find "${AMA_BUILD_DIR}/build" -name 'libama_cryptography.so*' -print)
if [ "${#SO_FILES[@]}" -eq 0 ]; then
  echo "ERROR: AMA build produced no libama_cryptography.so* under ${AMA_BUILD_DIR}/build" >&2
  exit 1
fi
for so in "${SO_FILES[@]}"; do
  cp -a "${so}" "${AMA_PKG_DIR}/"
  echo "    co-located $(basename "${so}") -> ${AMA_PKG_DIR}/"
done

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
