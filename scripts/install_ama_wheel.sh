#!/usr/bin/env bash
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Build AMA Cryptography from a local, already-verified checkout into a wheel,
# then install exactly that wheel, hash-locked.
#
# Usage:  install_ama_wheel.sh SRC_DIR VERSION
#
# A directory install (``pip install .``) cannot be hash-checked, so OpenSSF
# Scorecard's Pinned-Dependencies check counts it as unpinned. Building the
# wheel first and installing it with ``--require-hashes`` against a requirement
# written from that wheel's own digest (scripts/ama_wheel_requirement.py) keeps
# the build identical -- ``--no-build-isolation`` against the native library
# the caller just compiled, ``--no-deps`` because AMA has no base runtime
# dependencies -- and makes the install refuse any file but the one just built.
#
# Installs into the interpreter ``python`` resolves to: with pip when it has
# one, otherwise with uv (an environment ``uv sync`` created ships no pip).
# The caller exports AMA_NO_CYTHON and the build floors beforehand.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SRC_DIR VERSION" >&2
  exit 2
fi
src="$1"
version="$2"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="$(command -v python)"
out="$(mktemp -d)"
trap 'rm -rf "${out}"' EXIT
req="${out}/ama-wheel.txt"

if python -m pip --version >/dev/null 2>&1; then
  ( cd "${src}" && python -m pip wheel --no-build-isolation --no-deps -w "${out}" . )
  python "${here}/ama_wheel_requirement.py" "${version}" "${out}" "${req}"
  python -m pip install --require-hashes --no-index --find-links "${out}" \
    --force-reinstall --no-deps -r "${req}"
elif command -v uv >/dev/null 2>&1; then
  uv build --no-config --wheel --no-build-isolation --python "${py}" --out-dir "${out}" "${src}"
  python "${here}/ama_wheel_requirement.py" "${version}" "${out}" "${req}"
  uv pip install --require-hashes --no-config --python "${py}" --no-index --find-links "${out}" \
    --reinstall --no-deps -r "${req}"
else
  echo "ERROR: neither pip nor uv can install into ${py}" >&2
  exit 1
fi
