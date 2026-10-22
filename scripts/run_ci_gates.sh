#!/usr/bin/env bash
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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
# Tool pins (black==26.5.1, mypy==2.3.1, ruff==0.16.4, pydocstyle==6.3.0) are mirrored in
# pyproject and enforced across surfaces by scripts/check_pinned_tool_versions.py.
#
# Behaviour: every gate runs (no fail-fast), failures are summarised at the
# end, and the exit code is non-zero if any gate failed — one local run shows
# everything the CI matrix would show, instead of one failure per push.
#
# Usage:
#   bash scripts/run_ci_gates.sh            # all gates (mypy lanes dominate; ~5-10 minutes)
#   bash scripts/run_ci_gates.sh --fast     # skip the four mypy lanes (finishes in ~1 minute)
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
# format.yml checks a superset of ci.yml's black scope; mirror the superset.
run_gate "black --check (ci.yml: Run Black; format.yml scope)" \
  black --check src/ tests/ benchmarks/ scripts/ examples/

run_gate "ruff (ci.yml: Run Ruff)" \
  ruff check src/ tests/ scripts/ tools/

run_gate "flake8 (ci.yml: Run Flake8)" \
  flake8 src/ tests/ scripts/ tools/ \
  --max-line-length=100 --extend-ignore=E203,W503,E402,E501,F841

run_gate "canonical headers (ci.yml: normalize_headers --check)" \
  python scripts/normalize_headers.py --check

run_gate "pydocstyle google convention (ci.yml: Run pydocstyle)" \
  pydocstyle src/omni_mercury_engine/ --convention=google

run_gate "benchmark-integrity gate (ci.yml: ROADMAP row 17 offline fallback)" \
  python scripts/check_benchmark_integrity.py

run_gate "torch.load safety gate (ci.yml: no raw torch.load outside safe wrapper)" \
  python scripts/check_torch_load_safety.py

# --- Workflow Hardening job (repository script gates) ----------------------
# actionlint and zizmor need their own toolchains and are not mirrored here.
run_gate "workflow hardening checks (ci.yml: check_workflow_hardening)" \
  python scripts/check_workflow_hardening.py

run_gate "suppression hygiene (ci.yml: check_suppression_hygiene)" \
  python scripts/check_suppression_hygiene.py

run_gate "pinned dev-tool version parity (ci.yml: check_pinned_tool_versions)" \
  python scripts/check_pinned_tool_versions.py

run_gate "Lyapunov λ documentation drift (ci.yml: check_readme_lyapunov)" \
  python scripts/check_readme_lyapunov.py

run_gate "codebase-scale documentation drift (ci.yml: measure_codebase_scale --check)" \
  python scripts/measure_codebase_scale.py --check README.md

# --- Other repository drift gates CI runs --------------------------------
# These ran only in CI (or only inside the pytest lane), so a stale generated
# file reached a pushed head before anything local flagged it. Each takes a
# few seconds at most.
run_gate "capability inventory drift (pytest: test_committed_inventory_is_fresh)" \
  python scripts/generate_capability_inventory.py --check

run_gate "detector catalog drift (docs/DETECTORS.md vs DETECTOR_MANIFEST)" \
  python scripts/generate_detector_catalog.py --check

run_gate "documentation lint (ci.yml: doc_lint)" \
  python scripts/doc_lint.py

run_gate "neural coverage drift (ci.yml: neural_coverage --check)" \
  python scripts/neural_coverage.py --check

run_gate "weapons-gate corpus drift (ci.yml: ingest_weapons_gate_corpus --check)" \
  python scripts/ingest_weapons_gate_corpus.py --check

run_gate "Prometheus rule render drift (render_prometheus_rules --check)" \
  python scripts/render_prometheus_rules.py --check

# --- pqc-production-check job: tests/security without torch ----------------
# That lane installs only [dev,api]. scripts/no_torch hides torch the way an
# absent package does, so torch-dependent imports fail here as they do there.
run_gate "torch-free security tests (pqc-production-check.yml: pytest tests/security/)" \
  env PYTHONPATH=scripts/no_torch python -m pytest tests/security/ -q -p no:cacheprovider

# --- Type Checking job (three lanes) ---------------------------------------
if [[ "${FAST}" -eq 0 ]]; then
  # Guard the pin before running any lane: a full [all,dev] install can leave
  # a user-site mypy (observed: 1.19.1 in ~/.local shadowing the pinned
  # 2.1.0), and an off-pin mypy flips the type-ignore set, producing false
  # unused-ignore/import errors that look like code regressions.
  MYPY_PIN="$(grep -oE 'mypy==[0-9.]+' pyproject.toml | head -1 | cut -d= -f3)"
  run_gate "mypy version matches the pyproject pin (${MYPY_PIN})" \
    bash -c "mypy --version | grep -qF 'mypy ${MYPY_PIN}' || { echo \"resolved \$(mypy --version) at \$(command -v mypy); expected ${MYPY_PIN} — a shadowing install (e.g. ~/.local/bin/mypy) must be removed\"; exit 1; }"

  run_gate "dependency license gate (ci.yml: Type Checking, [ml] closure)" \
    python scripts/check_dependency_licenses.py --extras ml

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
    tests/loaders/ tests/narrative/ tests/fairness/ tests/scripts/ \
    tests/cyber/ tests/decision/ tests/distributed/ tests/emergent/ \
    tests/evaluation/ tests/federated/ tests/medical/ tests/metrics/ \
    tests/proofs/ tests/reasoning/ tests/research/ \
    tests/utils/ \
    --disallow-subclassing-any \
    --show-error-codes \
    --no-warn-unused-configs

  # Shipped-artifact scripts — KEEP IN LOCKSTEP with the ci.yml list
  # ("Run MyPy on shipped-artifact scripts").  2026-07-21: the whole
  # directory is on the gate — the historical pre-existing-errors
  # carve-out was retired when the last nine files were cleaned; a new
  # script must be added here AND in ci.yml when created.
  run_gate "mypy shipped-artifact scripts lane (ci.yml: scripts)" \
    mypy scripts/harvest_sigma_baseline.py \
    scripts/train_sigma_immutable.py \
    scripts/collect_sw_eng_metrics.py \
    scripts/fit_weapons_gate_calibration.py \
    scripts/train_gosnn_fusion.py \
    scripts/sweep_fusion_capacity.py \
    scripts/generate_docs_images.py \
    scripts/generate_calibration_visuals.py \
    scripts/measure_codebase_scale.py \
    scripts/update_readme_benchmarks.py \
    scripts/run_api.py \
    scripts/normalize_headers.py \
    scripts/generate_anomaly_panel.py \
    scripts/run_sigma_mutation_gate.py \
    scripts/check_benchmark_integrity.py \
    scripts/render_prometheus_rules.py \
    scripts/check_torch_load_safety.py \
    scripts/ci_gate_watchdog.py \
    scripts/check_dependency_licenses.py \
    scripts/record_gdacs_fixtures.py \
    --show-error-codes
else
  echo ""
  echo "==> (--fast: skipping the four mypy lanes)"
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
