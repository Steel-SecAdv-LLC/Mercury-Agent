#!/usr/bin/env bash
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Fetch a prebuilt AMA Cryptography native backend for this OS and CPU.
#
# The cmake build of AMA's native PQC library is the slow phase of every
# install. A release can carry that build as one archive per OS and
# architecture, holding the ``ama_cryptography`` wheels built against it for
# every supported Python (AMA bundles the shared library inside the wheel,
# signs the library's SHA3-256 into it at build time and verifies that at
# import, and tags the wheel for the interpreter that built it, so one wheel
# per Python version is needed). This script downloads the archive for the
# running platform, checks it against the release's ``SHA256SUMS``, and
# unpacks it. The callers (``scripts/build_ama_native.sh`` and the
# ``build-ama-cryptography`` composite action) install the wheel that matches
# their interpreter from the unpacked directory and run the same PQC
# capability probe the source build runs; the source build stays the fallback
# whenever no archive is available or no wheel matches.
#
# Usage:
#   fetch_ama_prebuilt.sh <dest-dir>
#
# Prints the destination directory on success.
#
# Exit status:
#   0  archive downloaded, verified and unpacked into <dest-dir>
#   2  no prebuilt is available: disabled, unsupported platform, the release
#      has no archive for this platform, or the download failed; the caller
#      builds from source
#   1  a downloaded archive does not match SHA256SUMS; the caller must stop
#      (a corrupt or tampered release is not a reason to build silently)
#
# Environment:
#   AMA_REF                 release tag (default: v5.0.0; must match the
#                           pyproject [pqc] pin)
#   AMA_PREBUILT            set to 0 to skip the download and build from source
#   AMA_PREBUILT_BASE_URL   directory URL holding SHA256SUMS and the archives
#                           (default: the ``ama-native-${AMA_REF}`` release of
#                           Steel-SecAdv-LLC/Mercury-Agent, published by
#                           .github/workflows/ama-prebuilt.yml; point it at the
#                           AMA repository's own release once that carries the
#                           same assets, or at a file:// mirror)
#
# Archive name: ama-cryptography-${AMA_REF}-${os}-${arch}.tar.gz with
# os in {linux, macos, windows} and arch in {x86_64, aarch64}.
set -euo pipefail

AMA_REF="${AMA_REF:-v5.0.0}"
AMA_PREBUILT="${AMA_PREBUILT:-1}"
AMA_PREBUILT_BASE_URL="${AMA_PREBUILT_BASE_URL:-https://github.com/Steel-SecAdv-LLC/Mercury-Agent/releases/download/ama-native-${AMA_REF}}"

dest="${1:?usage: fetch_ama_prebuilt.sh <dest-dir>}"

unavailable() {
  echo "==> prebuilt AMA ${AMA_REF}: $1; building from source" >&2
  exit 2
}

if [ "${AMA_PREBUILT}" = "0" ]; then
  unavailable "AMA_PREBUILT=0"
fi

case "$(uname -s)" in
  Linux) os=linux ;;
  Darwin) os=macos ;;
  MINGW*|MSYS*|CYGWIN*) os=windows ;;
  *) unavailable "unsupported operating system $(uname -s)" ;;
esac
case "$(uname -m)" in
  x86_64|amd64|AMD64) arch=x86_64 ;;
  aarch64|arm64) arch=aarch64 ;;
  *) unavailable "unsupported architecture $(uname -m)" ;;
esac

archive="ama-cryptography-${AMA_REF}-${os}-${arch}.tar.gz"
command -v curl >/dev/null 2>&1 || unavailable "curl is not installed"

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

fetch() {
  # -f turns an HTTP error into a non-zero exit; a missing asset is therefore
  # "unavailable", not a broken download that the caller must act on.
  curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 20 --max-time 600 \
    -o "$2" "${AMA_PREBUILT_BASE_URL%/}/$1"
}

fetch SHA256SUMS "${work}/SHA256SUMS" || unavailable "no SHA256SUMS at ${AMA_PREBUILT_BASE_URL}"
expected="$(awk -v name="${archive}" '$2 == name || $2 == "*" name {print $1}' "${work}/SHA256SUMS" | head -n 1)"
[ -n "${expected}" ] || unavailable "release has no ${archive}"
fetch "${archive}" "${work}/${archive}" || unavailable "could not download ${archive}"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    python - "$1" <<'PY'
import hashlib
import sys

with open(sys.argv[1], "rb") as handle:
    print(hashlib.sha256(handle.read()).hexdigest())
PY
  fi
}

actual="$(sha256_of "${work}/${archive}")"
if [ "${actual}" != "${expected}" ]; then
  echo "ERROR: ${archive} does not match SHA256SUMS (expected ${expected}, got ${actual})." >&2
  echo "       Refusing to install it. Set AMA_PREBUILT=0 to build from source instead." >&2
  exit 1
fi

rm -rf "${dest}"
mkdir -p "${dest}"
tar -xzf "${work}/${archive}" -C "${dest}"

wheels="$(find "${dest}" -maxdepth 2 -name 'ama_cryptography-*.whl' -print | wc -l | tr -d ' ')"
[ "${wheels}" -gt 0 ] || { echo "ERROR: ${archive} carries no ama_cryptography wheel" >&2; exit 1; }

echo "==> prebuilt AMA ${AMA_REF} for ${os}-${arch} verified (sha256 ${actual}; ${wheels} wheel(s))" >&2
echo "${dest}"
