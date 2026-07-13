<!-- Copyright (C) 2025 Steel Security Advisors LLC -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Detector-tier hardening — process note & reproduction checklist

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

This note records the *decision boundary* for the **second** detector-tier
hardening pass in this PR — the pivot that shaped it, what was audited, why the
hardening approach was chosen, and a reproducible checklist a reviewer can run
locally to validate every claim. It builds on the first pass's 35-defect audit
and its `detectors/_calibration.py` primitives (see §5).

## 1. The pivot

The original ask was framed around detector *scores* — nudge the tier's benchmark
numbers up. Mining the repository, the transparent finding was that the tier's
individual detectors are already reasonable for what they are (classical,
pure-NumPy streaming detectors on the notoriously hard NAB point-labels), and
that chasing per-detector AUC would be low-leverage and easy to overfit.

**The pivot: stop chasing individual scores; harden the *seams* — the places where
the tier is actually brittle — and make every guard observable.** A monitoring
system fails in production not because one detector is 2% weaker, but because a
NaN slips through untracked, an ill-conditioned solve explodes on a large-magnitude
signal, two threads corrupt a shared detector's state, or a raw-average ensemble
is quietly worse than just running the single best detector. Those are the seams
this PR hardens.

This reframing is *also* what unlocked the benchmark target the original ask
wanted: once the ensemble calibrates each detector onto a common scale and
combines them robustly (rather than a raw mean), the unsupervised ensemble
finally beats the best single detector on real NAB by a clear margin.

## 2. What was audited (the seams)

A full read of the tier (`detectors/{detection_tier,spot_evt,digital_twin,bocpd,rca}.py`,
`core/{base,metrics,score_calibration}.py`, `benchmarks/detection_tier_benchmark.py`)
surfaced six concrete seams:

| # | Seam | Evidence in the pre-PR code | Failure mode |
|---|---|---|---|
| 1 | **Raw-score ensemble** | `_combine` did `score_matrix.mean(axis=1)` / `@ weights` on *uncalibrated* scores | The widest-score-range detector dominates; the ensemble can be worse than its best member |
| 2 | **Implicit NaN handling** | `np.nan_to_num(...)` buried in each detector's `_to_1d_f64`; `+inf → 1.8e308` | Undocumented, unconfigurable, silent; inf mapped to a value that itself breaks downstream math |
| 3 | **Unguarded metadata** | `spot_evt` emitted `{"z_q", "gamma"}` with no finite check | A NaN/Inf tail parameter leaks into alerts/fusion |
| 4 | **Fixed absolute ridge** | `digital_twin` used `gram + self.ridge*I` with `ridge=1e-6` | Negligible on large-magnitude Gram matrices → near-singular solve stays ill-conditioned |
| 5 | **SPOT state mutation in `detect()`** | `_stream_scores` mutated `self._n/_nt/_sigma/_zq` | `detect()` not idempotent; two concurrent calls on one instance corrupt each other |
| 6 | **Crop drops anomalies** | `_crop_to_anomaly` centred on the midpoint of first/last label | Midpoint can land in a normal gap and drop *all* anomalies → series silently discarded |

## 3. Why this approach

- **Calibration + robust consensus, not a smarter single detector.** The
  outlier-ensemble literature (Aggarwal & Sathe, *Outlier Ensembles*, 2017) is
  explicit that (a) heterogeneous detector scores must be normalised before
  combining and (b) a plain average is dominated by robust rank/quantile
  aggregation. The `rank`/ECDF calibration + high-quantile `consensus` combiner is
  the textbook-correct fix, and it is what clears the > 0.003 target on real NAB —
  not a bespoke, overfit weighting.
- **Explicit policy over implicit default.** NaN handling became a *named,
  configurable* policy with a conservative default (`neutral`) that preserves the
  historical behaviour, so nothing regresses silently while operators gain
  `impute` / `flag` / `raise` and a single documented magnitude regime.
- **Purity for concurrency.** Making the SPOT kernels pure functions of explicit
  local state is a smaller, safer change than a lock or a snapshot/restore hack,
  and it makes `detect()` idempotent as a bonus.
- **Observe every rescue.** A guard that silently fixes bad data hides a
  data-quality problem. Metering every correction turns each seam from a silent
  liability into a monitored signal.

## 4. Reproduction checklist

Every claim below is backed by a test or artefact in this PR. The tests are
written to **default to failing** — each asserts the hardened property and only
passes when it genuinely holds (there is no "accept unless proven wrong" path).

> Environment note: the detector-tier tests import only `omni_mercury_engine.detectors.*`
> and `core.metrics` / `core.centralized_constants`; they do not require the ML
> extra. `prometheus-client` and `hypothesis` are in the `[dev]` extra.

| Claim | Reproduce | Validates |
|---|---|---|
| Ensemble calibration implemented (rank/ecdf/isotonic/platt/none) + config | `pytest tests/detectors/test_ensemble_calibration.py` | Deliverable 1 |
| Calibrated consensus beats best single by > 0.003 (synthetic) | `pytest tests/detectors/test_ensemble_calibration.py -k consensus_beats` | Deliverable 1, acceptance |
| Calibrated consensus beats best single by > 0.003 (**real NAB**) | `MERCURY_DATA_DIR=./.nab_cache python -m benchmarks.reproduce_detection_tier_nab` → `benchmarks/detection_tier_nab_analysis.md` (+0.0119 on 30 series) | Empirical validation, acceptance |
| NaN policy explicit + configurable (neutral/impute/flag/raise) + magnitude regime | `pytest tests/detectors/test_detection_config.py` | Deliverable 2 |
| Guards metered (`omni_detector_nonfinite_corrected`) + structured logs | `pytest tests/detectors/test_detection_observability.py` | Deliverable 3 |
| Scale-relative Tikhonov ridge; near-singular / large-magnitude stable | `pytest tests/detectors/test_digital_twin_conditioning.py` | Deliverable 4 |
| SPOT purity + thread-safety (concurrent `detect()` no corruption) | `pytest tests/detectors/test_spot_concurrency.py` | Deliverable 5 |
| Metadata (`z_q`/`gamma`) guarded like scores | `pytest tests/detectors/test_detection_observability.py -k metadata` and `.../test_detection_tier_property.py -k metadata` | Deliverable 6 |
| BOCPD run-length mass conservation (posterior sums to 1 each step) | `pytest tests/detectors/test_bocpd_invariants.py` | Stronger testing |
| Property tests: score/metadata finiteness + bounds under arbitrary input | `pytest tests/detectors/test_detection_tier_property.py` | Stronger testing |
| RCA `_walk` degenerate + adjacency-inference paths covered | `pytest tests/detectors/test_rca_walk_coverage.py` | RCA coverage |
| `_crop_to_anomaly` fix (retains most anomalies; rescued 1 NAB series, 30 vs 29) | `pytest tests/benchmarks/test_detection_tier_realdata.py -k crop` and the NAB run's `n_datasets` | Deliverable, empirical |

Run the whole tier suite:

```bash
pytest tests/detectors/test_detection_config.py \
       tests/detectors/test_detection_observability.py \
       tests/detectors/test_spot_concurrency.py \
       tests/detectors/test_bocpd_invariants.py \
       tests/detectors/test_digital_twin_conditioning.py \
       tests/detectors/test_rca_walk_coverage.py \
       tests/detectors/test_ensemble_calibration.py \
       tests/detectors/test_detection_tier_property.py \
       tests/detectors/test_{spot_evt,digital_twin,bocpd,rca,detection_tier_integration}.py
```

Reproduce the real-NAB before/after benchmark (downloads the canonical NAB files
on first run, then reuses the cache; deterministic under `--seed`):

```bash
MERCURY_DATA_DIR=./.nab_cache python -m benchmarks.reproduce_detection_tier_nab \
    --out benchmarks/detection_tier_nab_results.json \
    --analysis benchmarks/detection_tier_nab_analysis.md
```

## 5. Relationship to the first-pass "35 confirmed defects"

This PR is a **two-pass** hardening of the same tier. The *first pass*
(`steel/detection-validation`, the earlier commits on this branch) was the
adversarial audit that surfaced and fixed **35 confirmed non-finite / crash /
correctness defects**, introduced the shared `detectors/_calibration.py`
primitives (`bound_finite`, `squash_scale`, `finite_scores`, `finite_features`),
and proved detector scores byte-identical on finite data — its findings-to-fix
map is the table in `docs/DETECTION_MECHANISMS.md` § *Robustness & hardening* and
its CHANGELOG entry. The *second pass* (this note) builds directly on those
primitives: it makes the guards a **configurable, observable NaN policy**, adds
**ensemble calibration + a robust consensus combiner**, and replaces two
first-pass stopgaps (the digital-twin `lstsq` fallback → scale-relative ridge; the
SPOT snapshot/restore → true purity) with stronger fixes.

Each hardened property in §4 is a *failing-by-default* assertion tied to a concrete
test or artefact — the auditable, "defaults-to-reject" verification a reviewer can
run: an item is accepted only when its test passes, not by assertion. The
first-pass 35 defects remain covered by
`tests/detectors/test_detector_robustness.py`; the second-pass properties by the
suites in §4.

## 6. Commit boundary

The work is grouped into logically-scoped commits so the decision boundary is
legible: (1) config + NaN policy + metrics, (2) numerical conditioning, (3) SPOT
purity, (4) ensemble calibration + consensus, (5) benchmark crop + reproduction
harness, (6) tests, (7) docs/changelog. This note is committed alongside them as
the single record of the pivot rationale.
