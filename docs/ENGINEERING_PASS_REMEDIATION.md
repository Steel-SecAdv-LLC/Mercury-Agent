# Engineering-pass remediation log (continuation of PR #348)

Audit trail for the engineering-pass round layered on top of the
`steel/maint1/2-coverage` maintenance round (PR #348). Every claim below
is backed by a measured local run; GitHub runners are the CI authority
and validated the same changes on the PR.

Environment note (reproducible, to prevent wasted effort): the package
refuses to import without the **native AMA-Cryptography v3.3.0 backend**
(`_pqc_gate`), which has no PyPI wheel — it is built from the upstream
repo via `scripts/build_ama_native.sh` (cmake + g++). All local runs
below use a venv with that backend built, CPU torch, and
`MERCURY_REQUIRES_ML=1 AMA_REQUIRE_REAL_PQC=true`.

## 1. Tier-1: PR #348 CI was red on Core Tests (Python 3.14)

Sole failing lane. 10 tests in `tests/tools/test_migrate_pkl.py` returned
exit 5 (`_EXIT_RESTRICTED_UNPICKLER_REFUSAL`) on 3.14 while 3.13 was
green. **Root cause (reproduced, not guessed):** Python 3.14 raises
`pickle.DEFAULT_PROTOCOL` 4→5; under protocol 5 numpy serialises arrays
via the PEP-574 zero-copy `numpy._core.numeric._frombuffer` path instead
of `multiarray._reconstruct`, and `_frombuffer` was not in migrate_pkl's
restricted-unpickler allow-list, so every array-bearing payload refused
before schema validation. Extracted the exact refused global from the
failing CI job log, then reproduced on local 3.11 with `protocol=5`.

**Fix:** allow-list `_frombuffer` (numpy 2.x `_core` + legacy 1.x `core`,
mirroring the existing dual `_reconstruct` entries). It is an inert
buffer→ndarray builder — same security posture as `_reconstruct`;
RCE-adjacent globals stay refused (17 malicious-payload refusal tests
still pass). Regression: the `legacy_pkl` fixture now parametrises over
protocol {4,5} and a new `TestProtocol5Reconstruction` pins the path
directly. Red-baseline verified: without the entry, protocol-5 → exit 5;
with it → exit 0.

## 2. verify-real-pqc lane (pre-existing #348 failure)

`pqc-production-check.yml` installs only `.[dev,api]` (no torch) then runs
`tests/security/`, but the round's new
`TestTrainedGateEndToEnd::test_benevolent_vector_passes` asserts
`backend == 'torch'` — impossible without torch, so the gate reported
`backend=='unavailable'` (fail-closed) and the assertion failed. Guarded
the trained-network class with `skipif(not HAS_TORCH)` (matching the
sibling `importorskip('torch')` tests in that dir); the no-torch
fail-closed path stays covered by `TestFailClosedContracts` /
`TestPQCUnavailableClassifier`. This is an env-guard, not a mask — the
class runs and enforces `backend=='torch'` in every torch-equipped lane.

## 3. Strict-mypy test-lane graduation (14 dirs)

Measured every `tests/` subdir under the strict flags and graduated the
14 already-clean ones (0 strict errors each; 131 files clean combined):
scripts, cyber, decision, distributed, emergent, evaluation, federated,
medical, metrics, proofs, reasoning, research, truth_decipher, utils.
A pure gate tightening (no source churn). `docs/MYPY_TEST_STRICT_MIGRATION.md`
is the dir-by-dir plan for the remaining ~331 errors across 19 dirs
(ordered easiest-first). `run_ci_gates.sh` brought back into lockstep
(it had silently drifted, missing even tests/fairness/).

## 4. ROADMAP rows 17 & 18 (plans + CI-gated mitigations)

- **Row 17 (benchmark refresh, external blocker):**
  `scripts/check_benchmark_integrity.py` — stdlib-only, no-network
  offline gate that fails closed unless the committed benchmark JSON is
  internally consistent AND the headline Mean/Median AUC + oracle-F1
  **recompute** from the genuine-labelled per-dataset rows (a
  restated-but-fabricated headline fails) AND the README block matches.
  Wired blocking in the workflow-hardening lane + `run_ci_gates.sh`;
  13-test unit suite. Row 17 now carries the exact 9-corpora dependency
  list (7 network/access-blocked, 2 data-quality single-class).
- **Row 18 (multilingual NL, future epic):**
  `tests/narrative/test_language_scope.py` promotes the README
  English-only honesty lock from prose to an enforced gate. Row 18 gains
  a concrete P1–P4 build-out plan + external dependency list.

## 5. Coverage uplift — six heavy under-covered modules (~0 → 98.6%)

621 deterministic, no-network behavioral tests. The pre-existing
`tests/test_*.py` of similar names targeted DIFFERENT modules; these are
the first real coverage of the named surface. Measured line+branch
(directory-form `--cov`; see §8 for why the dotted form is unusable):

| Module | Cover | Tests |
|---|---|---|
| `biometric/__init__.py` | 100.00% | 51 |
| `biometric/voice_recognition.py` | 99.16% | 52 |
| `biometric/fingerprint_recognition.py` | 100.00% | 69 |
| `biometric/iris_recognition.py` | 100.00% | 55 |
| `quantum_computing/circuits.py` | 96.45% | 72 |
| `quantum_computing/detector.py` | 99.14% | 52 |
| `quantum_computing/executor.py` | 100.00% | 69 |
| `quantum_computing/hybrid.py` | 99.22% | 60 |
| `core/adaptive_fusion.py` | 93.33% | 37 |
| `ml/compression.py` | 99.38% | 61 |
| `datasets/ocean.py` | 98.28% | 43 |
| **aggregate** | **98.62%** | **622 pass** |

3,307 statements + 968 branches over the target surface. `tests/biometric/`
and `tests/quantum_computing/` are new packages, now wired into the core
lane so the gain counts toward its non-regression floor.

### 5.1 Two root-cause defects surfaced by the coverage pass (fixed, pinned)

1. **`fingerprint_recognition.py` `_find_minutiae` (uint8 underflow).**
   `cn += abs(neighbors[k] - neighbors[(k+1)%8])` ran on raw uint8
   skeleton values, so a 0→1 ridge transition underflowed (0−1 → 255)
   instead of contributing 1. The crossing number was corrupted (a true
   ridge ending computed to 128, a bifurcation to 384), so RIDGE_ENDING /
   BIFURCATION minutiae were essentially never detected via the public
   `extract()` path — fingerprint matching silently ran on empty
   minutiae. **Fix:** cast each neighbour to signed `int` before the
   difference. Pinned by `TestCrossingNumberOverflow` (uint8 and int32
   paths now agree, no overflow) and a
   `test_fingerprint_random_suspicious` / low-contrast `poor_quality`
   split in the detector suite.
2. **`quantum_computing/detector.py` `detect()` (threshold=0.0).**
   `threshold = threshold or self._threshold` coalesced on truthiness, so
   an explicit `threshold=0.0` (a legitimate "flag anything above zero")
   was silently replaced by the 0.5 default. **Fix:**
   `self._threshold if threshold is None else threshold`. Pinned by
   `test_explicit_zero_threshold_is_respected`.

### 5.2 Non-blocking observations (documented, not fixed)

- `voice_recognition.py:643` computes `low_energy / total_energy` and
  discards it (the intended `low_ratio` is never assigned) — dead
  expression, no behavioral effect.
- Iris single-image extraction emits a benign numpy divide RuntimeWarning
  (stat over one sample); the presentation-attack verdict is asserted
  regardless.

## 6. Core-lane coverage re-measurement (task: fresh measured floor)

CI-identical invocation of the (now biometric+quantum-expanded) core
lane, native AMA, Python 3.11:

<!-- CORE_LANE_RESULT -->
- **Pre-expansion CI-identical run** (exact ci.yml selection, native AMA,
  Python 3.11): **4865 passed, 49 skipped, 0 failed; combined stmt+branch
  coverage 38.30%** (floor 30; matches the 38.26% the 3.14 CI lane reported).
  This is the artifact for "one CI-identical core-lane run" and confirms the
  round's changes keep the core lane green.
- **Post-expansion run** (this round adds `tests/biometric/` +
  `tests/quantum_computing/` to the core lane, so the six modules' 93–100%
  coverage now counts toward the core floor): rerun of the same selection
  with the two dirs added; measured coverage rises above the 38.30% baseline.
  The `COVERAGE_THRESHOLD_CORE` floor is **kept at 30** (the round only ADDS
  core-lane coverage, never removes it, so the existing floor stays a valid
  non-regression guarantee with an even larger cushion). Raising it is
  deferred to a dedicated re-calibration commit rather than bundled here, per
  the "do not raise a floor without a fresh measured justification" policy.
<!-- /CORE_LANE_RESULT -->

## 7. Mutation testing (task: cheap sample + survivor disposition)

<!-- MUTATION_RESULT -->
Cheap validation sample (bounds the harness before the full ~1-CPU-hour
PR run): `run_sigma_mutation_gate.py --max-mutants 12` over the 120
enumerated σ hot-path sites — baseline green in 5.0s, **12/12 killed,
0 survived, kill rate 100.0%** (int/float/bool/compare/arith/not mutants
across both `sigma_immutable_gate.py` and `sigma_immutable_corpus.py`),
byte-exact restoration verified. This confirms the harness and the
semantic pinning suites are effective; the full-matrix measurement stays
the authoritative 119/120 (99.2%) from PR #348, enforced blocking at
`--fail-under 90` in `mutation-testing.yml` (which also completed on the
PR runners this round).
<!-- /MUTATION_RESULT -->

The one accepted survivor from #348 (an `exc_info=True→False` flip inside
a diagnostic `logger.warning` in `sigma_immutable_gate.py`, log-verbosity
only, zero behavioral effect) remains accepted; compensating control is
the fail-closed contract pins in the two `*_semantics` suites.

## 8. Residual risks / environment notes

- **Dotted `--cov=<module>` is unusable in this env** (coverage 7.15.x's
  `find_spec` source resolution purges numpy from `sys.modules`; numpy
  2.4 refuses re-import). Directory-form `--cov=src/omni_mercury_engine`
  + `coverage report --include=<glob>` measures identically. Reconfirmed
  this round on an unrelated existing test, so it is not a test defect.
- The overall-85% coverage target is a directional goal; this round drove
  the six named heavy modules to 93–100% (from ~0). The long-tail modules
  (utils, verifiers, validation/data_loaders, etc.) remain the next
  targets and are tracked by the full-suite floor.

## Verification checklist

- [x] migrate_pkl 3.14 fix: reproduced + red-baseline verified; 47/47 file tests pass
- [x] verify-real-pqc: fixed; 29 file tests pass with torch, class skips without
- [x] 14 test dirs graduated to strict mypy; combined invocation 0 errors (131 files)
- [x] benchmark-integrity gate: PASS on committed JSON+README; headline recomputes to 1e-16
- [x] language-scope guard: 4 tests pass
- [x] coverage: 622 tests pass; six target modules 93–100% (98.62% aggregate)
- [x] two root-cause defects fixed + pinned; src stays strict-mypy/pydocstyle clean
- [x] black / ruff / flake8 / canonical-header clean across all changes
- [x] freshness gates green: codebase-scale, neural-coverage, λ-drift, benchmark-integrity
- [x] core-lane CI-identical run captured (38.30%, green); floor kept at 30 with a larger cushion after the biometric+quantum expansion (§6)
- [x] mutation cheap-sample: 12/12 killed, harness validated; documented survivor carried forward (§7)
