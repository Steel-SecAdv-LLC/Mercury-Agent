#!/usr/bin/env bash
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Run the exact PR quality-gate matrix locally, before pushing.
#
# This is the single-command mirror of the blocking lanes in
# ``.github/workflows/ci.yml`` (Code Quality + Type Checking jobs). It exists
# because the pre-commit config is NOT that mirror: pre-commit's mypy hook runs
# ``--follow-imports=silent --ignore-missing-imports`` on changed files only
# (and is skipped in pre-commit CI entirely), so type errors that CI's three
# mypy lanes catch — cross-module [no-redef], [arg-type] against real
# signatures, [union-attr] — sail through a green pre-commit run and fail only
# after the push. Every gate below runs the same command, flags, and scope as
# its ci.yml counterpart.
#
# KEEP IN LOCKSTEP with .github/workflows/ci.yml:
#   * the Code Quality job ("Run Black/Ruff/Flake8", header + docstring gates)
#   * the Type Checking job (mypy src lane, lenient tests lane, and the
#     graduated strict lane's directory list)
# Tool pins (black==26.5.1, mypy==2.1.0, pydocstyle==6.3.0) are mirrored in
# pyproject and enforced across surfaces by scripts/check_pinned_tool_versions.py.
#
# Behaviour: every gate runs (no fail-fast), failures are summarised at the
# end, and the exit code is non-zero if any gate failed — one local run shows
# everything the CI matrix would show, instead of one failure per push.
#
# Usage:
#   bash scripts/run_ci_gates.sh            # all gates (mypy lanes dominate; ~5-10 minutes)
#   bash scripts/run_ci_gates.sh --fast     # skip the three mypy lanes (finishes in ~1 minute)
set -uo pipefail

FAST=0
[[ "${1:-}" == "--fast" ]] && FAST=1

declare -a FAILED=()
run_gate() {
  local name="$1"
  shift
  echo ""
  echo "==> ${name}"
  if "$@"; then
    echo "    PASS: ${name}"
  else
    echo "    FAIL: ${name}"
    FAILED+=("${name}")
  fi
}

# --- Code Quality job ------------------------------------------------------
run_gate "black --check (ci.yml: Run Black)" \
  black --check src/ tests/

run_gate "ruff (ci.yml: Run Ruff)" \
  ruff check src/ tests/ scripts/ tools/

run_gate "flake8 (ci.yml: Run Flake8)" \
  flake8 src/ tests/ scripts/ tools/ \
  --max-line-length=100 --extend-ignore=E203,W503,E402,E501,F841

run_gate "canonical headers (ci.yml: normalize_headers --check)" \
  python scripts/normalize_headers.py --check

run_gate "pydocstyle google convention (ci.yml: Run pydocstyle)" \
  pydocstyle src/omni_mercury_engine/ --convention=google

# --- Type Checking job (three lanes) ---------------------------------------
if [[ "${FAST}" -eq 0 ]]; then
  # Guard the pin before running any lane: a full [all,dev] install can leave
  # a user-site mypy (observed: 1.19.1 in ~/.local shadowing the pinned
  # 2.1.0), and an off-pin mypy flips the type-ignore set, producing false
  # unused-ignore/import errors that look like code regressions.
  MYPY_PIN="$(grep -oE 'mypy==[0-9.]+' pyproject.toml | head -1 | cut -d= -f3)"
  run_gate "mypy version matches the pyproject pin (${MYPY_PIN})" \
    bash -c "mypy --version | grep -qF 'mypy ${MYPY_PIN}' || { echo \"resolved \$(mypy --version) at \$(command -v mypy); expected ${MYPY_PIN} — a shadowing install (e.g. ~/.local/bin/mypy) must be removed\"; exit 1; }"

  run_gate "mypy src lane (ci.yml: Run MyPy)" \
    mypy src/omni_mercury_engine/ --show-error-codes

  run_gate "mypy lenient tests lane (ci.yml: Run MyPy on tests)" \
    mypy tests/ \
    --allow-untyped-defs \
    --allow-untyped-calls \
    --allow-subclassing-any \
    --disable-error-code untyped-decorator \
    --show-error-codes \
    --no-warn-unused-configs

  # Graduated strict directories — KEEP IN LOCKSTEP with the ci.yml list
  # ("Run MyPy strict on graduated test directories").
  run_gate "mypy graduated strict lane (ci.yml: strict test dirs)" \
    mypy tests/datasets/ tests/ethical/ tests/safeguards/ tests/tools/ \
    tests/loaders/ tests/narrative/ \
    --disallow-subclassing-any \
    --show-error-codes \
    --no-warn-unused-configs
else
  echo ""
  echo "==> (--fast: skipping the three mypy lanes)"
fi

# --- Summary ----------------------------------------------------------------
echo ""
echo "============================================================"
if [[ "${#FAILED[@]}" -eq 0 ]]; then
  echo "All CI quality gates passed locally."
  exit 0
fi
echo "FAILED gates (${#FAILED[@]}):"
for gate in "${FAILED[@]}"; do
  echo "  - ${gate}"
done
exit 1
