<!--
Copyright (C) 2025 Steel Security Advisors LLC
SPDX-License-Identifier: GPL-3.0-or-later
-->

# steel/maint1/2-coverage — Maintenance Round Remediation Log

This note is the audit trail for maintenance round 1/2 (coverage focus).
It records, for every workstream: the gap or defect, its root cause, the
concrete fix (file + symbol), the test or gate that proves it, and the
measured before/after evidence. Decisions and residual risks are recorded
inline so nothing is carried as undocumented debt.

**No check was suppressed, disabled, or narrowed to force CI green. No
blanket `type: ignore` or lint suppressions were added (two narrow,
justified `noqa` markers follow the repo's established patterns). Every
gate floor in this round is set from a fresh local measurement, not
aspiration.**

Environment for all measurements: Python 3.11.15, `pip install -e
".[all,dev]"`, AMA-Cryptography v3.3.0 native backend built from the
pinned tag (all three PQC algorithms verified loadable), 4-core Linux
container, 2026-07-21.

---

## 1. Baseline

Full suite exactly as CI's ml-tests lane runs it (`pytest tests/ -n 4
--timeout=300 --cov=src/omni_mercury_engine --dist=loadgroup`,
`MERCURY_REQUIRES_ML=1`, `AMA_REQUIRE_REAL_PQC=true`):

- **12,328 passed, 99 skipped, 1 failed** in 13m32s.
- The single failure was `tests/test_measure_codebase_scale.py::
  test_readme_scale_block_is_in_sync` — the drift gate correctly flagging
  this branch's own new test files against the committed README scale
  block (regenerated in §7; not a product defect).
- **Total statement+branch coverage: 68.05%** (140,284 statements,
  39,028 branches) against the CI floor of 55.
- CI on `main` was fully green before this round (verified via the
  Actions API) — this round raises the bar rather than repairing a red
  build.

## 2. ROADMAP row 6 — intersectional fairness (closed)

| Gap / defect | Root cause | Fix (file · symbol) | Proof |
|---|---|---|---|
| Bias audits measured marginal parity only; a model fair on every marginal can still disadvantage a joint `(race, gender)` cell | No joint-subgroup metrics existed | `ml/fairness.py` · `build_intersectional_groups`, `FairnessAuditor.compute_intersectional_parity`, `compute_intersectional_equalized_odds`; sparse cells below `intersectional_min_group_size` are excluded from the disparity max and *reported*; all-small audits flag `insufficient_data` instead of fabricating a verdict | `tests/fairness/test_intersectional_metrics.py` — Simpson's-paradox regression: four joint cells at rates 0.8/0.2/0.2/0.8 give exactly-fair marginals (disparity 0.000) and joint disparity 0.300, flagged naming the worst-off cell |
| Engine's `_audit_fairness` passes `dict[str, ndarray]` into an ndarray-typed `audit()` under `# type: ignore[arg-type]`; the dict hit `np.unique()`, raised, and the broad `except` returned `None` — the fairness path was dead at runtime for its documented input shape | Input type never widened when the engine call site was written | `audit()` natively accepts `Mapping[str, ndarray]` and 2-D arrays with `feature_names`; single-feature report shape byte-compatible; `engine.py` ignore removed | `tests/fairness/test_intersectional_metrics.py::TestAuditIntersectional` (mapping input, 2-D input, single-feature compatibility, violation text) |
| `BiasmitigationProcessor._apply_threshold_optimization` zero-filled every prediction for integer-typed groups (raw array compared against stringified fit keys); unseen groups were dropped to 0.0 | dtype-blind key comparison + zeros initialisation | Match on string form; unseen groups fall back to the 0.5 default with a warning | `tests/fairness/test_bias_mitigation.py::TestThresholdOptimization` (integer-group regression, unseen-group default) |
| `_fit_reweighting` was a placeholder (every threshold hard-coded 0.5, no weights) | Never implemented | Real Kamiran–Calders reweighing `P(g)·P(y)/P(g,y)` + `get_sample_weights` (unseen combos → neutral 1.0; fit-before-use enforced) | Closed-form weight assertions on hand-counted data + hypothesis property: weighted joint frequency equals product of weighted marginals (the defining independence property) |

`tests/fairness/` (41 tests) is graduated into ci.yml's strict mypy lane
and added to the core-tests coverage lane.

## 3. ROADMAP row 8 — σ_Immutable mutation-testing gate (closed)

- **Harness**: `scripts/run_sigma_mutation_gate.py` — deterministic
  AST-based mutation over the σ hot path
  (`security/sigma_immutable_gate.py` + `sigma_immutable_corpus.py`, 124
  enumerated sites), in-place mutants with byte-exact restoration in a
  `finally`, red-baseline abort (exit 2 — a failing baseline would count
  every mutant as killed), timeout-as-killed, deterministic stride
  sampling, JSON report. `mutmut`/`cosmic-ray` were evaluated and
  rejected: their import/trampoline models fight the package's
  import-time native-PQC gate and provide no deterministic bounded-runtime
  mode. The harness itself is pinned by 11 unit tests
  (`tests/scripts/test_run_sigma_mutation_gate.py`) including
  restoration-on-failure and the red-baseline abort.
- **First full measurement (the honest number)**: with the pre-existing
  interface-level σ subset, **12/124 killed — 9.7%**. The subset pinned
  interfaces, not arithmetic: constant tweaks and operator swaps in the
  benevolence→σ-band projection, the 256-D vector builders, and the
  corpus-generation arithmetic all survived.
- **Response**: two semantic pinning suites written specifically to kill
  the survivor classes, each assertion annotated with the mutation class
  it kills — `tests/security/test_sigma_immutable_gate_semantics.py`
  (23 tests: closed-form projection values including the `>=` floor
  boundary, exact 256/27/180/33 layout, the
  `1.0 + 0.4·clip(0.5s+0.5a)` overlay at non-equivalent signal
  combinations, frozen evaluation verdicts, threshold clamping,
  verify-corpus-by-default, end-to-end trained-gate separation) and
  `tests/security/test_sigma_immutable_corpus_semantics.py` (corpus
  layout, labelling, signing arithmetic).
- **Re-measured kill rate with the strengthened subset**: **99.2%**
  (119/120; the site count dropped from 124 because two cleanups
  eliminated equivalent-mutant sites outright — the redundant
  `parse_corpus` re-copy and the inline `[:16]` fingerprint literals
  hoisted to `crypto_api.key_fingerprint`). Intermediate steps, all
  measured: corpus-only after its semantics suite 69/73 → 94.5%; the
  four corpus survivors were then eliminated (2 by cleanup, 2 pinned via
  the fingerprint contract); the ten gate-side survivors of the combined
  run (91.7%) were triaged — nine killed with fail-closed contract pins
  (unavailable gate must report `passes=False, score=0.0`; an
  `EthicalGate` without `_trained` is refused; an anchor exactly at the
  critical floor is compliant while just-below violates; refusal
  exceptions report the honest zero score; the `gate_load_error`
  fallback message survives the `or`→`and` swap; dual-gate signal
  defaults are exactly 0.0). **The single accepted survivor** is the
  `exc_info=True` flip inside a diagnostic `logger.warning`
  (`sigma_immutable_gate.py:192`) — log verbosity, zero behavioral
  effect. CI floor set at **90** (measured 99.2% minus a margin for
  future site additions) in `.github/workflows/mutation-testing.yml` —
  a blocking, path-filtered PR lane plus weekly cron; the workflow passes
  `check_workflow_hardening.py` and zizmor at CI's blocking threshold.

## 4. `scripts/` mypy-debt clearance (42 → 0)

All nine files carrying the documented pre-existing debt (counted
2026-07-20 in ci.yml's own lane comment) were cleaned with real typing
fixes — TypedDicts for record shapes, documented assert-narrowing at
consumer sites, justified `cast`s on `json.load` following the repo's
established pattern, precise matplotlib `Figure`/`Axes` types — zero
blanket ignores, zero config weakening, and behavior verified unchanged
(byte-identical README regeneration; numerically identical training
expressions). The ci.yml scripts lane now lists **every** `scripts/`
file, retiring the "pre-existing errors stay off the lane" carve-out.

## 5. Coverage uplift (the round's namesake)

| Module | Before | After | Suite |
|---|---|---|---|
| `core/calibration_pipeline.py` | 47.21% | 99.54% | `tests/core/test_calibration_pipeline.py` (67 tests, closed-form Youden-J/F1/cost-sensitive/KL/KS/fingerprint anchors + permutation-calibrated drift regressions; the unreachable scalar-to-list fallback in `compute_dataset_fingerprint` was removed — X is always reshaped 2-D so `.tolist()` always returns a list — leaving only the documented-unreachable ethical-verification `except` uncovered) |
| `detectors/advanced/point_adjustment.py` | 71.86% | 100.00% | `tests/detectors/test_point_adjustment.py` (35 tests, hand-analyzed two-segment scenario) |
| `validation/api_validators.py` | 61.35% | 100.00% | `tests/validation/test_api_validators.py` (119 tests, adversarial input incl. injection-shaped names, entity-expansion, NaN/Inf boundaries) |
| `metrics/benchmark_evaluator.py` | 55.15% | 100.00% | `tests/metrics/test_benchmark_evaluator.py` (38 tests, pen-and-paper AUROC/AUPRC/F1) |
| `core/di.py` | 0.00% | 94.34% | `tests/core/test_di.py` (56 tests; remaining misses are Protocol `...` bodies) |
| `detectors/cross_domain_frequency.py` | 0.00% | 98.44% | `tests/detectors/test_cross_domain_frequency.py` (31 tests; only an unreachable `except ImportError` uncovered) |
| `tools/migrate_pkl.py` | 11.33% | 100.00% | `tests/tools/test_migrate_pkl.py` (35 tests, strict lane; adversarial `os.system`/`builtins.eval`/`builtins.__import__` payloads refused with sentinel-file proof nothing executed; hardened-relaunch env contract poison-tested) |
| compat shims (`spectral_domain_oracle`, `core/self_healing`, `anomaly/__init__`) | 0.00% | 100.00% | `tests/test_compat_shims.py` (re-export identity vs canonical modules, deprecation-warning behavior) |
| σ hot path (`sigma_immutable_gate` + `corpus`) | (indirect) | 99.2% mutation kill (§3) | the two semantic suites (§3) |

Full-suite total: 68.05% → **[final re-measurement in flight at this
commit; the closing commit of this branch records the measured figure
and the corresponding `COVERAGE_THRESHOLD_FULL` raise from 55, with the
cushion noted in ci.yml]**.

## 6. Defects found and fixed at the root (this round)

| # | Defect | Root cause | Fix | Regression test |
|---|---|---|---|---|
| 1 | Engine fairness audit dead at runtime for its documented dict input (§2) | type-ignore masking + broad except | `audit()` accepts the dict shape natively | `TestAuditIntersectional::test_mapping_input_*` |
| 2 | Integer-group threshold mitigation silently zeroed all predictions (§2) | stringified-key comparison | string-form matching + unseen-group default | `TestThresholdOptimization::test_integer_groups_regression` |
| 3 | Reweighting mitigation was a placebo (§2) | placeholder never implemented | real Kamiran–Calders weights | `TestReweighing` (closed form + independence property) |
| 4 | `calibrate_all_thresholds()` served guardrail-violating registry values: returned result was clamped to `ANOMALY.MIN_THRESHOLD_FLOOR`/`MAX_THRESHOLD_CAP` but the registry kept the raw optimum, so `get_threshold()` exceeded the cap | clamp applied to the return path only | registry record rewritten to the clamped value with a `guardrail_clamped_from` provenance breadcrumb; registry == returned result asserted | `TestCalibrateAllThresholds::test_registry_value_respects_guardrail_cap` |
| 5 | `PointAdjustmentEvaluator(search_best_threshold=False)` ignored — evaluate() always searched | flag stored, never consulted | honored fail-loud: scores-only evaluation with search opted out raises (anomaly scores have arbitrary scale; a fabricated 0.5 default would invent an operating point) | `test_search_disabled_without_threshold_fails_loud` |
| 6 | `ServiceContainer.resolve` re-created falsy singletons on every resolve and ignored pre-registered falsy instances | cached-instance presence tested by truthiness instead of `is not None` | `is not None` check with rationale comment | `TestFalsySingletonDefect` (2 regressions) |
| 7 | Every entry in `ComponentFactory.create_detector`'s `detector_map` named a class that does not exist post-rename — all five built-in detector types raised `AttributeError` | bit rot: class renames never propagated to the string map | entries renamed to the real exports (`MercuryAnomalyDetector`, `TemporalAnomalyDetector`, `SpatialAnomalyDetector`, `DimensionalAnalyzer`, `SigmaDirectiveDetector`) | every entry instantiated for real, parametrized |
| 8 | `model_map`'s `neural`/`consciousness` entries likewise named nonexistent classes | same bit rot | renamed to `NeuralCognitiveModel` / `ConsciousnessPreservationModel` | both entries instantiated for real |
| 9 | `detect_drift`'s default `kl_threshold=0.1` false-flagged same-distribution data below n≈2000 — two independent N(0,1) samples at n=400 measure histogram symmetric-KL ~0.32–0.52 under 50-bin binning bias | fixed threshold applied to a biased finite-sample estimator | default KL decision is now permutation-calibrated: pool both samples, take 200 seeded same-size splits, score each with the same histogram estimator, and flag only when the observed KL exceeds the null's (1−α) quantile (`kl_threshold=None` sentinel; an explicit float keeps the historical fixed-threshold path unchanged; `DriftResult.kl_null_quantile` exposes the calibration; fixed default `permutation_seed=0` makes results reproducible) | `TestPermutationCalibratedDrift` (n=400 same-distribution not flagged, n=400 mean-shift flagged incl. KL-alone, legacy explicit-threshold pin, seeded determinism) |

Every sharp edge the suites initially documented was subsequently
**fixed at the root in this same round** (defect rows 10–14 below);
none is carried as accepted behavior:

| # | Sharp edge | Fix | Regression |
|---|---|---|---|
| 10 | `sanitize_string` truncated before escaping — entity expansion could exceed `max_length` | escape → strip → truncate, walking the cut back to the entity start (sound: every raw `&` is escaped, so any `&` in the escaped string opens a complete entity); output can no longer exceed the cap nor split an entity; the now-provably-dead length check in `validate_feature_names` removed | `TestSanitizeString` (4 new tests incl. parametrized never-exceeds/never-splits) |
| 11 | Out-of-range (incl. NaN) sensitivity flowed into `sanitized_data` | fail-closed: invalid values no longer propagate (`sanitized_data=None` for the failed parameter) | `TestValidateSensitivity` (2 regressions + request-level aggregation assert) |
| 12 | `ValidationErrorType.MISSING_REQUIRED` unconstructable | request validation now emits it for a missing required `data` key on both univariate and multivariate paths | `test_none_data_is_missing_required` (both request classes) |
| 13 | Multivariate path silent for sub-threshold Inf the univariate path warns about | identical warning semantics on both paths | `test_inf_ratio_under_limit_warns_like_univariate` |
| 14 | `BenchmarkEvaluator.evaluate()` raised a bare numpy `ValueError` on zero usable samples | early domain `ValueError` naming dataset and detector | `test_all_samples_skipped_raises_no_usable_samples_error` |

The single remaining in-suite documentation-only item is
`calibrate_all_thresholds`'s ethical-verification `except`, which is
unreachable without corrupting pipeline internals — kept as defensive
depth, documented at the suite.

## 7. Interconnectedness and docs

- All relative links across README/ARCHITECTURE/CONTRIBUTING/SECURITY
  and every `docs/*.md` resolve (0 broken; scripted sweep).
- README codebase-scale block regenerated to match the branch's final
  tree (`scripts/measure_codebase_scale.py`, the drift gate from §1).
- ROADMAP rows 6 and 8 closed with locking artifacts; CHANGELOG entry
  added under `[Unreleased]`.

## 8. Residual risks / open threads

- Mutation-gate survivors after strengthening: **exactly one**
  (`sigma_immutable_gate.py:192`, `exc_info=True→False` inside a
  diagnostic `logger.warning` — log verbosity, zero behavioral effect),
  classified in §3, the row-8 closure text, and the workflow comment.
- The dotted `--cov=<module>` form is unusable in this environment
  (coverage 7.15.2's `find_spec` source resolution purges numpy from
  `sys.modules`; numpy 2.4 refuses re-import). Directory-form `--cov`
  measures identically; noted so nobody burns time on it again.
- **Orphaned-module findings — RESOLVED in this round (ROADMAP row 19
  closed).** `core/di.py` had zero importers anywhere in
  src/tests/scripts/tools — earlier claims of importers were regex
  false positives (`core.di` with an unescaped dot) — its
  detector/model maps had bit-rotted to 100%-nonexistent class names
  (only possible in dead code; fixed), and its constructor injection is
  inert for every class defined under `from __future__ import
  annotations` (all of src/). Decision executed: **deprecated** with
  `DIDeprecationWarning` on import + DEPRECATION.md §1.3 entry, kept
  functional per the preservation policy, contract-pinned.
  `detectors/cross_domain_frequency.py` was fully implemented and
  documented but unreachable from any public surface. Decision
  executed: **wired** into `omni_mercury_engine.detectors` (lazy export
  + `__all__`), with the wiring pinned by `TestPublicWiring`.
