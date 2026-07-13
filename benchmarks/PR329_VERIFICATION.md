# PR #329 — independent verification report

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

This report records an **independent, end-to-end re-run** of the detector-tier
hardening deliverables and the six wired-subsystem commits carried by PR #329,
performed from a clean checkout with a freshly built real AMA Cryptography
v3.3.0 PQC backend (so the fail-closed import gate is satisfied, not stubbed).

It is a *verification* artifact: every claim below is backed by a command a
reviewer can re-run. Where a genuine failure surfaced, it was fixed at the root
and is called out explicitly (see §6).

## 1. Environment

| Component | Version |
| --- | --- |
| Branch / HEAD | `claude/release-hardening-blockers-1oyxiw` @ `81b5f82` (+ the fix in §6) |
| Python | 3.11.15 |
| numpy / scipy | 2.4.6 / 1.17.1 |
| torch (CPU) | 2.12.1+cpu |
| AMA Cryptography | **3.3.0** (native backend built from source; ML-DSA-65 + Kyber-1024 + SPHINCS+ all loadable) |
| prometheus-client | 0.25.0 / hypothesis 6.156.2 |

The AMA backend was built with `scripts/build_ama_native.sh` (the repo's
canonical builder) inside a clean virtualenv; `import omni_mercury_engine` then
passes `_pqc_gate._enforce_pqc_production_gate()` — confirming the PQC startup
gate against a real v3.3.0 build, as the acceptance criteria require.

## 2. Scope

The spec's five code deliverables (ensemble calibration, explicit NaN policy,
guard observability, digital-twin conditioning, SPOT purity) plus metadata
guarding, stronger testing, and the NAB empirical validation were **already
implemented** on this branch (they landed in the merged detector-tier hardening
commit `77b74f5`, the "35 confirmed defects" fix, which PR #329 is stacked on).
This report therefore *verifies* them rather than re-implementing them, and
separately validates the six subsystem-wiring commits unique to PR #329.

## 3. Detector-tier deliverables — test evidence

Command:

```bash
python -m pytest \
  tests/detectors/test_calibration_helpers.py \
  tests/detectors/test_ensemble_calibration.py \
  tests/detectors/test_digital_twin.py tests/detectors/test_digital_twin_conditioning.py \
  tests/detectors/test_spot_evt.py tests/detectors/test_spot_concurrency.py \
  tests/detectors/test_detection_observability.py \
  tests/detectors/test_detection_tier_property.py \
  tests/detectors/test_bocpd.py tests/detectors/test_bocpd_invariants.py \
  tests/detectors/test_rca.py tests/detectors/test_rca_walk_coverage.py
# -> 142 passed in 11.71s
python -m pytest tests/detectors/test_detection_config.py       # -> 32 passed
python -m pytest tests/detectors/test_detection_tier_integration.py  # (in the 97-test verbose run)

# whole directory, for completeness:
python -m pytest tests/detectors/     # -> 1050 passed, 3 skipped (CUDA-only skips) in 215s
```

| # | Deliverable | Verifying tests | Result |
| --- | --- | --- | --- |
| 1 | Ensemble calibration (ECDF/rank + isotonic/Platt + warm-up + config knobs; consensus combiner) | `test_ensemble_calibration.py` — `TestPava`, `TestCalibrators`, `TestEnsembleCalibrationConfig`, `TestConsensusCombiner::test_consensus_beats_best_member_on_complementary_series` (the **> 0.003** synthetic acceptance check) | ✅ pass |
| 2 | Explicit NaN policy `neutral/impute/flag/raise` on `bound_finite`/`finite_scores` | `test_detection_config.py::TestApplyNaNPolicy` (all four modes, array + `TestGuardFiniteScalar` scalar), `TestConfigResolution` (defaults < env < file < override) | ✅ pass |
| 3 | Guard observability — `omni_detector_nonfinite_corrected{detector,policy,field}` counter + structured logs | `test_detection_observability.py::TestGuardMetrics`, `TestGuardStructuredLogs`, `TestDetectorsMeterNonFiniteInput` (digital_twin/bocpd/rca meter live input) | ✅ pass |
| 4 | Scale-relative Tikhonov ridge `ridge_factor·trace(G)/d` (replaces the lstsq fallback) | `test_digital_twin_conditioning.py` — `TestRidgeParam`, `TestNearSingular`, `TestMagnitudeScaling`, `TestConditioningProperty` (bounded solution for any series) | ✅ pass |
| 5 | SPOT purity + thread-safety (pure `_tail_probability`/`_threshold_from_tail`; mutation-free `detect`) | `test_spot_concurrency.py::TestPurity` (pure, no-mutate, idempotent), `TestConcurrency` (16-thread parallel `detect`, distinct inputs) | ✅ pass |
| 7 | Metadata guarding (`z_q`, `gamma`) + unified magnitude cap | `test_detection_observability.py::test_spot_metadata_guard_metered`, `test_detection_tier_property.py::TestMetadataInvariants` | ✅ pass |
| 8 | Hypothesis property tests; BOCPD mass conservation; RCA `_walk`/adjacency coverage | `test_detection_tier_property.py` (Hypothesis), `test_bocpd_invariants.py::TestMassConservation*` (posterior sums to 1 each step), `test_rca_walk_coverage.py` (degenerate `_walk` fallback + adjacency inference) | ✅ pass |

## 4. Empirical validation — real NAB before/after

Command (downloads the canonical NAB series, seed 0, 30 series):

```bash
MERCURY_DATA_DIR=./.nab_cache \
  python -m benchmarks.reproduce_detection_tier_nab --seed 0 \
    --out benchmarks/detection_tier_nab_results.json \
    --analysis benchmarks/detection_tier_nab_analysis.md
```

Reproduced result (this run — data source: the repo's real
`omni_mercury_engine.datasets.timeseries.NABLoader`, **30 series**, seed 0):

| pipeline | combiner | calibration | best single | best-single AUC | ensemble AUC | ensemble − best | ensemble F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| before (pre-PR) | `average` | `none` | echo_state | 0.6230 | 0.6242 | 0.0012 | 0.2883 |
| after (this PR) | `consensus` | `rank` | echo_state | 0.6230 | **0.6350** | **0.0119** | 0.2895 |

> series: 30  best single: echo_state AUC=0.6230  consensus ensemble AUC=0.6350
> ensemble − best single = **0.0119** (threshold > 0.003): **PASS**

This is a byte-for-byte match of the committed `detection_tier_nab_analysis.md`:
the calibrated consensus ensemble beats the best single detector by **0.0119
ROC-AUC** (> 0.003 required). The `before` row (raw averaging, no per-detector
calibration) clears the bar by only 0.0012 — i.e. the **calibration + consensus
combiner is what earns the acceptance margin**, not the ensemble alone.

## 5. PR #329 subsystem-wiring commits — live-path validation

Each wired subsystem is proven **invoked on the live path**, not merely
importable:

```bash
python -m pytest tests/automl/ tests/explainability/ \
  tests/ethical/test_benevolence_cache_wiring.py tests/ethical/test_benevolence_cache.py
# -> 40 passed  (automl/explainability/benevolence)   [torch present]
python -m pytest tests/cognitive/ tests/ethical/ tests/test_drift_recalibration.py
# -> 781 passed, 1 skipped   (no meaning-level model serving — environmental)
```

| Commit | Subsystem | Live-path test(s) | Result |
| --- | --- | --- | --- |
| `097d2e6` | AutoML optimizer fixes + Tree SHAP + interaction matrix | `test_optimizer.py` (f1 no-recursion, continuous-score AUC, seeded), `test_shap_tree.py` (SHAP additivity, global interaction) | ✅ |
| `f587764` | `CachedBenevolenceScorer` on the engine ethics boundary | `test_benevolence_cache_wiring.py` (cached by default, repeat detection hits cache) | ✅ |
| `72510ce` | GDPR Art. 22 report via `detect_with_fusion(gdpr_report=True)` | `tests/explainability/test_gdpr_seam.py` (report attached when requested, absent by default, SHAP background captured by `fit_fusion`) | ✅ |
| `1c4a64e` | `engine.tune_fusion` + `mercury-agent tune` CLI | `tests/automl/test_tune_fusion.py` (end-to-end small, CLI smoke, requires both classes) | ✅ |
| `66eda45` | `CuriosityEngine` + `EnhancedAnomalyDetector` in `CognitiveOrchestrator.analyze()` | `test_orchestrator_novelty_wiring.py` (curiosity + enhanced detector invoked in `analyze`, no network I/O) | ✅ |
| `81b5f82` | AutoML samplers honour `seed=` | `test_optimizer.py::test_seeding_is_reproducible` | ✅ |

No regressions in the cognitive, ethical, or drift-recalibration suites.

## 6. Genuine failure found and fixed

Re-running the full ethical suite surfaced **one real, deterministic failure**
that PR #329 introduced through its own change:

```
tests/ethical/test_hard_enforcement.py::TestEngineFusionBoundary::
  test_detect_with_fusion_raises_on_benevolence_violation
AttributeError: property 'benevolence_threshold' of
  'CachedBenevolenceScorer' object has no setter
```

**Root cause.** Commit `f587764` wraps the engine's boundary scorer in
`CachedBenevolenceScorer` **by default**. That wrapper exposed
`benevolence_threshold` as a read-only pass-through property, so the previously
valid `engine._boundary_scorer.benevolence_threshold = 1.01` (runtime gate
tuning, exercised by the enforcement test) began raising `AttributeError` — a
compatibility regression from wrapping.

**Fix** (`src/omni_mercury_engine/cognitive/benevolence_cache.py`). Added a
`benevolence_threshold` setter that (a) delegates the assignment to the wrapped
scorer — preserving its `MINIMUM_BENEVOLENCE_FLOOR` clamp — and (b) **clears the
cache**. (b) is a correctness requirement, not just test-appeasement: cache keys
are `(ruleset_version, action, context)` and do **not** encode the threshold in
force at compute time, so without invalidation a decision ruled permissible
under the old threshold could be served as a hit after the bar is raised. This
restores a faithful drop-in for `BenevolenceScorer` (whose `enforce` always
re-evaluates against the current threshold).

Regression tests added to `tests/ethical/test_benevolence_cache.py`:
`test_threshold_setter_delegates_to_wrapped_scorer`,
`test_threshold_setter_preserves_floor_clamp`,
`test_threshold_setter_invalidates_cache`.

After the fix: `tests/ethical/` + `tests/cognitive/` + drift → **781 passed, 1
skipped**.

## 7. One-command reproduction

```bash
# 1. Build the real AMA v3.3.0 PQC backend (satisfies the import gate)
python -m venv .venv && . .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
bash scripts/build_ama_native.sh
python -m pip install -e ".[dev]" prometheus_client torch --index-url https://download.pytorch.org/whl/cpu

# 2. Detector-tier deliverables
python -m pytest tests/detectors/ -q

# 3. Real-NAB before/after
MERCURY_DATA_DIR=./.nab_cache python -m benchmarks.reproduce_detection_tier_nab --seed 0

# 4. Subsystem-wiring commits + no-regression sweep
python -m pytest tests/automl/ tests/explainability/ tests/ethical/ tests/cognitive/ \
  tests/test_drift_recalibration.py -q
```

## 8. Conclusion

All five code deliverables, metadata guarding, the stronger-testing suite, and
the NAB empirical criterion verify green against a real AMA v3.3.0 build; the six
PR #329 wiring commits are each proven on the live path. One genuine
wrapper-compatibility regression was found and fixed at the root with tests. The
detector-tier deliverables were pre-implemented on this branch, so this pass
confirms them rather than re-deriving them.
