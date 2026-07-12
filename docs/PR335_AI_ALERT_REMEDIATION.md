<!--
Copyright (C) 2025 Steel Security Advisors LLC
SPDX-License-Identifier: GPL-3.0-or-later
-->

# PR #335 — AI/Bot Alert Remediation & Sandbox Validation

This note is the audit trail for the AI/Bot (GitHub Copilot) review-alert
remediation delivered on top of the Detection-Science competitive-benchmark
work (PR #335). It records, for every alert surfaced by the automated reviewer:
the one-line alert, its root cause, the concrete root-cause fix (file + symbol),
the test that proves the fix, and the status. It also documents the exact
sandbox deployment used to exercise Mercury end-to-end and the validation
artifacts produced.

**No unresolved AI/Bot alerts remain; no debt-for-debt trades were accepted; all
pre-existing issues discovered during this work were resolved at the root.** No
check was suppressed, disabled, or narrowed to force CI green.

---

## 1. AI/Bot alert triage table

Ten Copilot review threads were open (unresolved) against the branch head. Each
was verified against the *current* tree, since several were marked `outdated`
(the code had already moved under them). They fall into two groups.

### 1a. Alerts fixed at the root in **this** remediation pass

| # | File · symbol | Alert (one line) | Root cause | Fix | Test |
|---|---|---|---|---|---|
| A | `scripts/verify_data_credentials.py` · `check_*` | Transport failure reported as the opaque `"HTTP 0"` (`r3566583308`) | `_get()` returns `(0, <exc text>)` on DNS/TLS/timeout; callers rendered only `f"HTTP {status}"`, discarding `body` (the exception) | New `_fail_detail(status, body)` surfaces the transport error text when `status==0` and a short provider-body preview on a real HTTP error; all five keyed checkers route their non-200 path through it | `tests/test_data_credentials_offline.py` (15 cases) |
| B | `tests/test_secret_wiring.py` · `test_wildfire_resolves_firms_secret_name` | Broad `except RuntimeError` masks a genuine regression as an "offline skip" (`r3566583301`, superseding `r3565646271`/`r3565720180`) | Any `RuntimeError` on engine import → `pytest.skip`, so a non-PQC `RuntimeError`/`NameError` from a loader refactor would go green | New `_import_skip_reason()` predicate: skip **only** an optional/transitive `ImportError` or the native-crypto PQC gate (`AMA/PQC is mandatory …`, mirroring `tests/test_calibration_brief.py`); re-raise everything else | `TestImportSkipClassification` (6 cases) + the behavioural test now runs green with the ML stack present |
| C | `tests/test_secret_wiring.py` · module docstring | Docstring claimed the behavioural check is "guarded with `importorskip`" — stale after the earlier try/except rewrite (`r3565742625`) | A prior fix replaced `importorskip` with a try/except but left the docstring describing the old mechanism | Docstring rewritten to describe the actual policy and point at `_import_skip_reason` | Covered by the predicate tests above |
| D | `scripts/verify_data_credentials.py` · `check_fred/nasa/alpha_vantage/openweathermap/run` | Pre-existing: functions missing Google-style docstrings | Docstrings never added when the checkers were written | Added concise one-line docstrings; module is now fully `pydocstyle --convention=google` clean | `pydocstyle` gate (green) |

### 1b. Alerts already fixed at the root in earlier branch commits (verified, threads resolved)

| # | File | Alert (one line) | Verified-current state |
|---|---|---|---|
| E | `.github/workflows/competitive-benchmark.yml` | Cache missed `data/adbench_embeddings` (`r3564977692`) | The `Cache ADBench datasets` step already keys on both `data/adbench` **and** `data/adbench_embeddings` |
| F | `benchmarks/competitive_benchmark.py` | Forked child returned the full score vector over IPC (`r3565720162`) | `_run_pyod_cell` reduces to `_metrics(...)` **inside** the child; only the small metrics dict crosses the `Queue` |
| G | `.github/workflows/network-tests.yml` | Comment said `WildfireDataLoader` (`r3565720168`) | Comment reads `WildfireLoader` (the real class) |
| H | `tests/test_secret_wiring.py` | `"fail-softed"` typo (`r3565646275`) | String is not present in the current file |
| I | `benchmarks/COMPETITIVE_BENCHMARK.md` | `git_commit` mislabeled as an ADBench dataset revision (`r3565742637`) | Provenance reads *"Benchmark code commit (Mercury repo `git rev-parse HEAD`)"* and notes dataset content is pinned by per-dataset sha256 |

> Group **1b** required no code change — the fixes already shipped in commits on
> this branch. This pass confirmed each against the current tree so the resolved
> threads reflect reality rather than being closed on trust.

---

## 2. Sandbox deployment (reproducible)

Mercury's engine import is gated on the mandatory AMA post-quantum-crypto native
backend; the competitive-benchmark tier/fusion detectors therefore need that
backend built. The comparison layer, guard-logic tier, and the secret-wiring /
credential tests do **not**.

```bash
# 1. Clean virtualenv (avoids the Debian-managed cryptography 41.x pin conflict)
python3 -m venv .venv && . .venv/bin/activate
python -m pip install --upgrade "pip>=26.1" wheel

# 2. Core + PyOD benchmark extra + dev tooling (no torch yet)
pip install -e ".[benchmark,dev]" flake8 "pydocstyle==6.3.0"

# 3. (For the tier/fusion lanes) the mandatory AMA native PQC backend + torch
bash scripts/build_ama_native.sh          # clones AMA-Cryptography @ v3.3.0, cmake build
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Verify the engine imports (PQC gate satisfied)
python -c "import omni_mercury_engine; print('engine OK')"
```

### Exercising the ensembles end-to-end, fully offline (no external services)

`scripts/../<sandbox>/e2e_offline.py` (reproduced in §4) drives a synthetic
anomaly set through the **same** `benchmarks.competitive_benchmark.evaluate_dataset`
path the real ADBench run uses, so the Mercury tier, the Mercury fusion lane, all
six PyOD baselines (each in its own forked, wall-clock-guarded cell), and
`summarize()` are all exercised with zero network calls.

### Test lanes

```bash
# The remediated surface + adjacent wiring, offline & deterministic
python -m pytest tests/test_secret_wiring.py tests/test_data_credentials_offline.py \
                 tests/test_pyod_comparison.py tests/benchmarks/ -q -m "not network"

# Exact CI quality-gate mirror (black, ruff, flake8, headers, pydocstyle [, mypy])
bash scripts/run_ci_gates.sh

# Live-data validation (opt-in): real ADBench download + competitive guard
MERCURY_NETWORK_TESTS=1 python benchmarks/competitive_regression_guard.py --check
MERCURY_NETWORK_TESTS=1 python -m pytest tests/test_pyod_comparison.py -q
```

---

## 3. Design notes / rationale

- **Surface the cause, don't hide it.** Both behavioural fixes (A, B) follow the
  same principle the branch already adopted elsewhere: an *environment* problem
  (uncached dataset, missing optional dep, unprovisioned PQC backend) skips
  cleanly, but a *code* problem (a real transport error worth diagnosing, a
  non-PQC `RuntimeError`, a `NameError`) must be loud. `_fail_detail` and
  `_import_skip_reason` are small, pure, independently unit-tested predicates —
  the classification logic is testable without the network or the ML stack.
- **Match existing house patterns.** The PQC skip marker is the exact string and
  policy used by `tests/test_calibration_brief.py`, kept in a shared module
  constant so the two sites can't drift.
- **No new dependencies, no runtime cost.** All changes are in a CI helper script
  and test code; Mercury's detection path is untouched.
- **`pyod_available()` deliberately stays narrow.** Consistent with the earlier
  review rounds, a *broken* install must surface loudly at the gate rather than be
  silently reported as "not installed" — the remediation preserves that.

---

## 4. Validation artifacts (captured in the sandbox)

**End-to-end orchestration, offline synthetic data** (`evaluate_dataset` → tier +
fusion + 6 PyOD baselines in forked wall-clock-guarded cells → `summarize`):

```
=== Per-method ROC-AUC on the synthetic set ===
  mercury_tier           AUC=1.0000  AP=1.0000  wall=0.1s
  mercury_fusion         AUC=0.9644  AP=0.5944  wall=4.8s
  isolation_forest       AUC=1.0000  AP=1.0000  wall=0.8s
  ecod                   AUC=1.0000  AP=1.0000  wall=1.5s
  copod                  AUC=1.0000  AP=1.0000  wall=1.6s
  local_outlier_factor   AUC=1.0000  AP=1.0000  wall=0.7s
  knn                    AUC=1.0000  AP=1.0000  wall=0.7s
  hbos                   AUC=1.0000  AP=1.0000  wall=3.3s
OK: tier + fusion + all 6 PyOD baselines produced finite AUC end-to-end (offline).
```

**Live-data competitive regression guard** (real 8-dataset ADBench subset,
Mercury tier vs PyOD, downloaded from the allowlisted transport):

```
=== network reachability to raw.githubusercontent.com === reachable: 200
COMPETITIVE REGRESSION GUARD: PASS (absolute floors + competitive gap held)
=== GUARD EXIT 0 ===
```

**Test results**

| Lane | Result |
|---|---|
| `tests/test_secret_wiring.py` + `tests/test_data_credentials_offline.py` | 32 passed |
| Adjacent regression sweep (wiring + comparison + benchmarks, offline) | 95 passed, 7 skipped (network fixtures) |
| `tests/test_pyod_comparison.py` (`MERCURY_NETWORK_TESTS=1`, real ADBench wine) | 14 passed, 1 skipped |
| `bash scripts/run_ci_gates.sh --fast` (black/ruff/flake8/headers/pydocstyle) | all gates pass (1373 files) |
| `mypy tests/` lenient lane (changed files) | Success, no issues |

Before → after, the exact regression this remediation closes:

```
# before:  check_eia(...) on a DNS failure ->  (False, "HTTP 0")
# after:   check_eia(...) on a DNS failure ->  (False, "transport error: ConnectTimeoutError: connection timed out")
```
