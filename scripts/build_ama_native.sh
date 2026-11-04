#!/usr/bin/env bash
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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
# Prebuilt first, source second. The cmake build is the slow phase of every
# install, so the script first asks ``scripts/fetch_ama_prebuilt.sh`` for the
# release archive built for this OS and CPU (the ``ama_cryptography`` wheels
# for every supported Python, each bundling the native library whose digest
# AMA signs into it, verified against the release's SHA256SUMS). When the
# archive has a wheel for this interpreter and its native backend loads here,
# the install is a download; when no archive or no matching wheel exists, the
# clone-and-cmake path below runs unchanged. A verification failure of a
# downloaded archive stops the script: it is never a reason to build silently.
#
# After installing the Python package the freshly built ``libama_cryptography.so``
# is co-located inside the installed ``ama_cryptography`` package directory, so
# the backend loads from the module's own directory and needs no LD_LIBRARY_PATH
# at runtime (``ama_cryptography._find_native_library`` searches there first).
#
# Environment overrides:
#   AMA_REF                git tag/ref to build (default: v5.0.0)
#   AMA_REPO               repository URL (default: upstream Steel-SecAdv-LLC/AMA-Cryptography)
#   AMA_BUILD_DIR          scratch checkout/build directory (default: /tmp/ama-cryptography)
#   AMA_PREBUILT           0 skips the prebuilt download and builds from source
#   AMA_PREBUILT_BASE_URL  where the prebuilt archives are (see fetch_ama_prebuilt.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v python >/dev/null 2>&1 || {
  echo "ERROR: cannot install the AMA native PQC backend; python is not on PATH" >&2
  exit 1
}

# This script drives a POSIX toolchain (gcc/clang, /tmp, flock-era paths).
# Under Git Bash / MSYS / Cygwin on Windows none of that holds, and the gcc
# probe below would fail with a misleading message. Refuse up front with the
# real options instead. AMA's own CMake supports MSVC: clone the repo, run
# `cmake -S . -B build && cmake --build build --config Release`, then
# `pip install --no-build-isolation .` from the clone — or build inside WSL,
# where this script works as written.
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*)
    echo "ERROR: this script targets POSIX hosts (Linux, macOS, WSL)." >&2
    echo "On native Windows build AMA with MSVC instead:" >&2
    echo "  git clone --branch \"${AMA_REF:-v5.0.0}\" --depth 1 <AMA repo> ama && cd ama" >&2
    echo "  cmake -S . -B build && cmake --build build --config Release" >&2
    echo "  pip install --no-build-isolation --force-reinstall --no-deps ." >&2
    echo "or run this script inside WSL." >&2
    exit 1
    ;;
esac

# Core count without Linux-only nproc (macOS has sysctl; POSIX has getconf).
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"

AMA_REF="${AMA_REF:-v5.0.0}"
AMA_REPO="${AMA_REPO:-https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git}"
AMA_BUILD_DIR="${AMA_BUILD_DIR:-/tmp/ama-cryptography}"

# Install into the active interpreter: pip when it has one, otherwise uv's pip
# front end (an environment ``uv sync`` created ships no pip).
py_install() {
  if python -m pip --version >/dev/null 2>&1; then
    python -m pip install "$@"
  elif command -v uv >/dev/null 2>&1; then
    uv pip install --python "$(command -v python)" "$@"
  else
    echo "ERROR: neither pip nor uv can install into $(command -v python)" >&2
    exit 1
  fi
}

# Install the ama_cryptography wheel in $1 that matches this interpreter
# (``--no-index --find-links`` lets the installer pick by tag; AMA tags each
# wheel for the Python that built it). Fails when no wheel fits.
install_prebuilt_wheel() {
  local version="${AMA_REF#v}"
  if python -m pip --version >/dev/null 2>&1; then
    python -m pip install --no-index --find-links "$1" --force-reinstall --no-deps \
      "ama-cryptography==${version}"
  else
    uv pip install --python "$(command -v python)" --no-index --find-links "$1" \
      --reinstall --no-deps "ama-cryptography==${version}"
  fi
}

# Co-locate the native shared object with the installed package so it loads
# without LD_LIBRARY_PATH (survives `COPY --from=builder /opt/venv` into the
# runtime image, and read-only-rootfs containers). Searches $1 recursively.
# .so on Linux, .dylib on macOS, .dll on Windows; plain loop instead of
# mapfile so the script also runs under macOS's stock bash 3.2.
colocate_native_library() {
  local pkg_dir so found=0
  pkg_dir="$(python -c 'import ama_cryptography, os; print(os.path.dirname(ama_cryptography.__file__))')"
  for so in $(find "$1" \( -name 'libama_cryptography.so*' -o -name 'libama_cryptography*.dylib' -o -name '*ama_cryptography*.dll' \) -print); do
    cp -P -p "${so}" "${pkg_dir}/"
    echo "    co-located $(basename "${so}") -> ${pkg_dir}/"
    found=1
  done
  [ "${found}" -eq 1 ]
}

# Fail loud unless all three mandatory PQC algorithms load via the native backend
# (matches the engine's import-time gate and the build-ama-cryptography action).
# Importing ``ama_cryptography`` also runs its own integrity check of the native
# library against the digest signed into the installed package.
verify_native_backend() {
  python - <<'PY'
from ama_cryptography.pqc_backends import get_pqc_backend_info

info = get_pqc_backend_info()
required = ("dilithium_available", "kyber_available", "sphincs_available")
missing = [name for name in required if not info.get(name)]
if missing:
    raise SystemExit(f"AMA native PQC backend incomplete after install; missing {missing}: {info!r}")
print("==> AMA native PQC backend verified (ML-DSA-65 + Kyber-1024 + SPHINCS+)")
PY
}

# ---------------------------------------------------------------------------
# Prebuilt release archive: the download replaces the clone and the cmake build.
# ---------------------------------------------------------------------------
if prebuilt_dir="$(bash "${SCRIPT_DIR}/fetch_ama_prebuilt.sh" "${AMA_BUILD_DIR}-prebuilt")"; then
  echo "==> Installing AMA Cryptography ${AMA_REF} from the prebuilt archive"
  # The wheel bundles the native library, so no co-location is needed here.
  if install_prebuilt_wheel "${prebuilt_dir}" && verify_native_backend; then
    exit 0
  fi
  echo "==> the prebuilt archive has no wheel for $(python -c 'import sys; print(sys.version.split()[0])') or its library did not load here; building from source" >&2
else
  rc=$?
  # 2 = no archive for this platform (fall through to the source build);
  # anything else is a failed verification and must stop here.
  [ "${rc}" -eq 2 ] || exit "${rc}"
fi

# ---------------------------------------------------------------------------
# Source build.
# ---------------------------------------------------------------------------
# Preflight: name every missing prerequisite at once instead of failing midway
# with cmake's or git's own error. cmake itself is pip-installed below.
missing=""
command -v git >/dev/null 2>&1 || missing="${missing} git"
command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 \
  || command -v clang >/dev/null 2>&1 || missing="${missing} a-C-compiler(gcc/clang)"
if [ -n "${missing}" ]; then
  echo "ERROR: cannot build the AMA native PQC backend; install:${missing}" >&2
  echo "  Debian/Ubuntu: apt-get install git build-essential" >&2
  echo "  RHEL/Fedora:   dnf install git gcc gcc-c++ make" >&2
  echo "  macOS:         xcode-select --install" >&2
  exit 1
fi

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
py_install --upgrade \
  "setuptools>=84.0.0" "wheel>=0.48.0" "cmake>=4.4.4"

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
export AMA_NO_CYTHON=1
( cd "${AMA_BUILD_DIR}" && py_install --no-build-isolation --force-reinstall --no-deps . )

if ! colocate_native_library "${AMA_BUILD_DIR}/build"; then
  echo "ERROR: AMA build produced no libama_cryptography shared library under ${AMA_BUILD_DIR}/build" >&2
  exit 1
fi

verify_native_backend
