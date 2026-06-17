# Mercury Agent Benchmark Results

Applies to Mercury Agent **v1.7.x**. Last updated: 2026-05-20.

> **v1.7 update.** The current public headline is the committed
> `benchmarks/mercury_benchmark_results.json` run — **65 successful /
> 75 attempted**, Mean ROC-AUC **0.8466**, Median **0.9100**, Mean
> Oracle F1 **0.6428** (2026-05-19, commit 79e8335) — surfaced in the
> README "Latest Benchmark Results" block and regenerated on every
> push to `main` by `.github/workflows/benchmark.yml`. The FEMA
> Disaster label-polarity fix (v1.7.0) and the 11-loader reachability
> harness are reflected in that run (disaster AUC 0.9999; 10 loaders
> failed, NOAA StormEvents recovered) — see `docs/ROADMAP.md`
> cross-cutting entries "FEMA Disaster loader label polarity" and
> "Dataset reachability harness (unreachable-11)".

> **Reproducibility note.** The aggregate / per-dataset tables further
> down *this* document are a **legacy 51-success / 55-attempt run** of
> `benchmarks/mercury_benchmark.py` over the ensemble in isolation
> (**Mean ROC-AUC 0.8030 / Mean Oracle F1 0.5886**). That is the
> historical **CI regression-gate floor** (the gate trips 15% below
> it: ROC-AUC 0.68 / F1 0.50), preserved here for the auditable
> trajectory — it is **not** the current committed run. For the
> authoritative current figures see the committed
> `mercury_benchmark_results.json` and the README "Latest Benchmark
> Results" block (65/75, Mean ROC-AUC 0.8466). The externally-
> comparable subset is ADBench Mean AUC 0.8180.

## What This Measures

This document reports the empirical performance of `MercuryAnomalyDetector` — Mercury's
unsupervised anomaly detection ensemble — on labeled real-world datasets.

**Ensemble composition** (no external anomaly-detection dependencies):

| Component | Default Weight | Method |
|-----------|---------------|--------|
| ResonanceScore | 40% | FFT harmonic spectral anomaly (FFT at fit, O(n*d) inference) |
| KinematicScore | 30% | Physics-based jerk/curvature detection (O(n*d)) |
| InfoGeometryScore | 30% | Fisher Information Mahalanobis OOD (O(n*d^2) inference) |

> **Adaptive weighting:** After `fit()`, weights are recomputed proportional to each
> component's AUC separation from random. Components with AUC < 0.5 are zeroed out.
> The 40/30/30 defaults above are the fallback when all components produce
> near-random scores. See `_compute_adaptive_weights()` in `statistical.py`.

**Protocol:**
- Normal-only training (unsupervised) with `StandardScaler`
- ROC-AUC from continuous scores
- Oracle F1: best F1 over 101 thresholds from 0.0 to 1.0 (upper bound, not operational)
- No hyperparameter tuning was performed
- Datasets capped at 10,000 samples with stratified sampling

## How to Reproduce

```bash
python benchmarks/mercury_benchmark.py
```

Results are saved to `benchmarks/mercury_benchmark_results.json`.
The legacy aggregate / per-dataset tables below come from an earlier run of this
command (the CI regression-gate baseline); the current committed run is
summarized in the v1.7 update note above and the README "Latest Benchmark
Results" block.

## Aggregate Results

| Metric | Value |
|--------|-------|
| Datasets tested | 55 |
| Datasets successful | 51 |
| Datasets failed | 4 |
| Mean AUC | 0.8030 |
| Median AUC | 0.8852 |
| Std AUC | 0.1916 |
| Mean Oracle F1 | 0.5886 |
| Median Oracle F1 | 0.6250 |

### Transparent fitness substrate (Phase 1) — governed-fusion live suite

Phase 1 of the governed recursive self-improvement work (see
`docs/SELF_IMPROVEMENT_LOOP.md`) extends the same label-provenance
discipline that already gates the `datasets/` side to the live-API
`loaders/` path that feeds `research/governed_fusion/`. Of the 23 events
in the live headline manifest, only **2** have audited ground-truth labels
(independent of any scored feature) — both from `network_security`
(`batadal`, `nsl_kdd`). The other 21 threshold a scored feature
(statistical / circular) and are reported separately as leakage-flagged;
7 reconstructed-series events (`tsunami/*`, `energy/*`,
`pandemic/ebola_2014`) are reported in their own block by existing design.

| Bucket | n_events | Mean AUROC | Mean F1 | Notes |
|---|---:|---:|---:|---|
| `external_label` (FITNESS) | 2 | 0.7704 | 0.1863 | The transparent fitness signal Phase 2's promotion gate reads from. |
| `self_label` (leakage-flagged) | 21 | 0.8282 | 0.2854 | Reported for pipeline transparency; never folded into the headline that grades self-improvement. |
| `reconstructed` | 7 | n/a | n/a | Reported separately by existing suite design. |
| Mixed (historical) | 23 | 0.8231 | 0.2768 | The pre-Phase-1 headline; superseded by the provenance split above. |

The audit is committed in
`src/omni_mercury_engine/loaders/label_provenance.py`
(`LABEL_PROVENANCE_REGISTRY`) and gated by
`tests/loaders/test_label_provenance_gate.py` and
`tests/research/test_governed_fusion_label_provenance.py`. The
fusion-marginal ablation ledger
(`research/governed_fusion/ablation_ledger.json`, written by
`research/governed_fusion/measure_marginal_ablation.py`, gated by
`.github/workflows/ablation-ledger.yml`) tracks per-component leave-one-out
lift on the `external_label` subset only.

The external-label mean is *below* the historical mixed mean. Label leakage
does not only inflate; it can also degrade in either direction depending
on the geometry of the threshold rule. The discipline is unchanged
either way: do not grade self-improvement against labels that are a
function of the scored signal.

## Per-Component AUC

| Component | Mean AUC | Median AUC | n_datasets |
|-----------|----------|------------|------------|
| InfoGeometry | 0.8256 | 0.8760 | 51 |
| Resonance | 0.7623 | 0.8294 | 51 |
| Kinematic | 0.6013 | 0.6116 | 51 |
| **Ensemble** | **0.8030** | **0.8852** | **51** |

## Per-Dataset Results

| Dataset | n_samples | Anomaly Ratio | AUC | Oracle F1 | Threshold | Fit (ms) |
|---------|-----------|---------------|-----|-----------|-----------|----------|
| ADBench-31 | 5803 | 0.012 | 0.9990 | 0.8971 | 0.59 | 15 |
| ADBench-43 | 367 | 0.027 | 0.9989 | 0.9091 | 0.63 | 2 |
| ADBench-25 | 3062 | 0.032 | 0.9979 | 0.9029 | 0.53 | 22 |
| ADBench-32 | 49097 | 0.072 | 0.9968 | 0.9593 | 0.55 | 10 |
| ADBench-21 | 148 | 0.041 | 0.9930 | 0.8000 | 0.63 | 1 |
| ADBench-04 | 683 | 0.350 | 0.9924 | 0.9687 | 0.61 | 1 |
| ADBench-42 | 223 | 0.045 | 0.9897 | 0.9000 | 0.65 | 1 |
| ADBench-16 | 567498 | 0.004 | 0.9888 | 0.5135 | 0.50 | 7 |
| ADBench-38 | 3772 | 0.025 | 0.9855 | 0.7302 | 0.61 | 2 |
| ADBench-13 | 284807 | 0.002 | 0.9767 | 0.4545 | 0.70 | 24 |
| ADBench-40 | 1456 | 0.034 | 0.9756 | 0.7800 | 0.44 | 1 |
| NSL-KDD | 148517 | 0.481 | 0.9721 | 0.9388 | 0.43 | 31 |
| ADBench-27 | 5393 | 0.095 | 0.9576 | 0.7564 | 0.41 | 4 |
| ADBench-34 | 95156 | 0.000 | 0.9576 | 0.5000 | 0.73 | 4 |
| ADBench-18 | 351 | 0.359 | 0.9506 | 0.9084 | 0.56 | 2 |
| ADBench-03 | 95329 | 0.024 | 0.9474 | 0.8872 | 0.64 | 290 |
| ADBench-24 | 7603 | 0.092 | 0.9362 | 0.7270 | 0.44 | 23 |
| ADBench-06 | 1831 | 0.096 | 0.9303 | 0.7500 | 0.45 | 3 |
| ADBench-23 | 11183 | 0.023 | 0.9186 | 0.4513 | 0.61 | 4 |
| ADBench-37 | 340 | 0.091 | 0.9143 | 0.7209 | 0.39 | 1 |
| SMD | 75876 | 0.053 | 0.9066 | 0.5881 | 0.46 | 70 |
| ADBench-10 | 286048 | 0.010 | 0.9061 | 0.2093 | 0.48 | 17 |
| ADBench-17 | 1966 | 0.187 | 0.9032 | 0.7728 | 0.49 | 327 |
| ADBench-45 | 129 | 0.078 | 0.8883 | 0.6250 | 0.40 | 1 |
| ADBench-11 | 619326 | 0.059 | 0.8854 | 0.4950 | 0.40 | 14 |
| ADBench-02 | 7200 | 0.074 | 0.8852 | 0.5380 | 0.40 | 3 |
| BATADAL | 12938 | 0.017 | 0.8711 | 0.5358 | 0.47 | 42 |
| ADBench-20 | 1600 | 0.062 | 0.8502 | 0.5188 | 0.44 | 4 |
| ADBench-22 | 19020 | 0.352 | 0.8453 | 0.7886 | 0.35 | 11 |
| ADBench-35 | 4207 | 0.399 | 0.8077 | 0.7790 | 0.34 | 7 |
| ADBench-05 | 41188 | 0.113 | 0.8043 | 0.5468 | 0.42 | 85 |
| ADBench-08 | 202599 | 0.022 | 0.7954 | 0.2474 | 0.47 | 27 |
| ADBench-07 | 2114 | 0.220 | 0.7757 | 0.6439 | 0.35 | 2 |
| ADBench-30 | 6435 | 0.316 | 0.7725 | 0.7343 | 0.36 | 16 |
| ADBench-33 | 245057 | 0.208 | 0.7621 | 0.7002 | 0.31 | 4 |
| ADBench-28 | 6870 | 0.023 | 0.7604 | 0.2065 | 0.44 | 11 |
| ADBench-29 | 768 | 0.349 | 0.7333 | 0.7363 | 0.34 | 1 |
| ADBench-09 | 299285 | 0.062 | 0.7076 | 0.2959 | 0.32 | 827 |
| ADBench-14 | 214 | 0.042 | 0.6936 | 0.3030 | 0.44 | 1 |
| ADBench-12 | 1941 | 0.347 | 0.6918 | 0.7033 | 0.28 | 3 |
| ADBench-15 | 80 | 0.163 | 0.6878 | 0.5455 | 0.52 | 1 |
| ADBench-01 | 49534 | 0.030 | 0.5667 | 0.1273 | 0.28 | 28 |
| ADBench-41 | 3443 | 0.029 | 0.5372 | 0.1198 | 0.43 | 6 |
| ADBench-44 | 4819 | 0.053 | 0.5253 | 0.2072 | 0.25 | 2 |
| NAB | 69561 | 0.095 | 0.4697 | 0.2951 | 0.00 | 5 |
| ADBench-36 | 3686 | 0.017 | 0.4741 | 0.0805 | 0.49 | 118 |
| ADBench-46 | 198 | 0.237 | 0.4625 | 0.5562 | 0.30 | 2 |
| ADBench-39 | 240 | 0.125 | 0.4416 | 0.3704 | 0.17 | 1 |
| ADBench-19 | 6435 | 0.207 | 0.4101 | 0.5110 | 0.00 | 16 |
| ADBench-26 | 5216 | 0.029 | 0.3761 | 0.1078 | 0.30 | 16 |
| ADBench-47 | 1484 | 0.342 | 0.3752 | 0.6747 | 0.00 | 1 |

### Failed Datasets (4)

| Dataset | Reason |
|---------|--------|
| SMAP | Data source unavailable (OmniAnomaly mirror) |
| MSL | Data source unavailable (OmniAnomaly mirror) |
| CICIDS-2017 | All download sources failed |
| MIT-BIH | Requires wfdb library |

## Known Weaknesses

1. **KinematicScore underperforms on unordered tabular data.**
   KinematicScore computes derivatives (velocity, acceleration, jerk) via `np.diff`.
   This assumes adjacent rows are temporally ordered. On shuffled tabular data
   (e.g., ADBench datasets), derivatives are meaningless noise. The kinematic
   component achieved mean AUC 0.6013 across all datasets — near-random on
   unordered tabular data, more useful on time-series.

2. **Ensemble inversion on high-dimensional data.**
   On high-dimensional image-like features (optdigits, landsat, WPBC), the ensemble
   score can invert (anomalies score lower than normal). This manifests as
   ROC-AUC < 0.5 on 6 datasets.

3. **Oracle F1 is an upper bound, not operational performance.**
   The oracle threshold sweeps 101 values and picks the best F1. A deployed
   system would use a fixed threshold (e.g., 0.5) and would achieve lower F1.

4. **No hyperparameter tuning was performed.**
   All results use default parameters. Tuning per-dataset could improve
   performance but would also risk overfitting.

5. **InfoGeometryScore requires d < n.**
   The Fisher Information component inverts a d x d covariance matrix. When
   the number of features exceeds the number of training samples, the matrix
   is singular and a pseudo-inverse is used, degrading accuracy.

## Data Sources

- **ADBench** (47 datasets): Downloaded from the ADBench GitHub repository.
  Tabular anomaly detection benchmarks across diverse domains.
- **NSL-KDD**: Network intrusion detection from Canadian Institute for Cybersecurity.
- **SMD**: Server Machine Dataset from Tsinghua University (28 machines).
- **SMAP/MSL**: NASA spacecraft telemetry — OmniAnomaly mirror unavailable.
- **NAB**: Numenta Anomaly Benchmark (realKnownCause category).
- **BATADAL**: Water infrastructure attack detection (train + test with ATT_FLAG labels).
- **CICIDS-2017**: Modern network attack flows — all download sources failed.
- **MIT-BIH**: ECG arrhythmia dataset — requires wfdb library.

Some datasets require network access or credentials. Failed downloads are
recorded as errors in the results JSON, not replaced with synthetic data.

## Calibration Validation (MD-011, MD-003, MD-005)

A separate calibration validation harness measures the effect of supervised threshold
calibration, conformal coverage, and adaptive ensemble weights on the same datasets.
Unlike the mercury benchmark (unsupervised, normal-only training), the calibration
harness uses a labeled 60/20/20 train/calibration/test split with `fit_with_labels()`.

### How to Reproduce

```bash
python benchmarks/calibration_validation.py
# Skip conformal coverage (faster):
python benchmarks/calibration_validation.py --skip-conformal
```

Results are saved to `benchmarks/calibration_validation_results.json`.

### MD-011: Threshold Calibration Pipeline

`fit_with_labels()` triggers `ThresholdCalibrationPipeline` to select the best
threshold via Youden's J or F1-optimal strategy. Compared against the default 0.5
threshold on the same test scores:

| Metric | Value |
|--------|-------|
| Datasets tested | 40 |
| Calibration improved F1 | 32 (80%) |
| Calibration same | 2 (5%) |
| Calibration degraded | 6 (15%) |
| Mean Calibrated F1 | 0.4192 |
| Mean Uncalibrated F1 | 0.2763 |
| Mean Delta F1 | +0.1430 |

**Status: RESOLVED.** Calibration improves or matches F1 on 85% of datasets with
mean improvement of +0.143. The 6 degraded datasets have delta < 0.18 (small regressions
where the default 0.5 happened to be near-optimal for a high-AUC detector).

![Calibration Improvement](images/calibration_improvement.png)

### MD-005: Conformal Coverage

**Corrected metric:** The prior measurement used `evaluate_coverage()` which computes
overall prediction accuracy (predictions == labels), NOT the conformal coverage guarantee.
For anomaly detection with heavily imbalanced data, accuracy is dominated by the majority
class and is not the conformal guarantee.

The actual conformal guarantee (Vovk et al., 2005) is: the fraction of ALL test
nonconformity scores at or below the calibration quantile threshold should be >= coverage.

**Score-based coverage (corrected)** via `measure_score_based_coverage()`:

| Target | SplitConformal | CrossConformal (k=5) | Normal-class |
|--------|---------------|---------------------|-------------|
| 90% | 18/40 (45.0%) | 31/40 (77.5%) | 27/40 (67.5%) |
| 95% | 19/40 (47.5%) | 32/40 (80.0%) | 29/40 (72.5%) |
| 99% | 24/40 (60.0%) | 23/40 (57.5%) | 32/40 (80.0%) |

CrossConformalPredictor outperforms SplitConformalPredictor significantly (77.5% vs 45.0%
at 90% target) because it uses all calibration data across k=5 folds, producing a more
conservative (higher) threshold. Normal-class coverage (fraction of normal test points
with score <= threshold) is the practically meaningful guarantee for anomaly detection.

**Legacy accuracy-based metric** (for reference, not the conformal guarantee):

| Target | Meets Guarantee | Percentage |
|--------|----------------|------------|
| 90% | 12/40 | 30.0% |
| 95% | 8/40 | 20.0% |
| 99% | 1/40 | 2.5% |

**Status: PARTIALLY RESOLVED.** The conformal predictor implementation is correct. The prior
"low coverage" diagnosis was based on the wrong metric (prediction accuracy vs.
score-based coverage). CrossConformal achieves 77.5-80% empirical coverage across targets.
Does not meet the >90% dataset-level threshold for full resolution. The implementation
is correct; coverage gaps are inherent to split/cross conformal on small, heavily
imbalanced datasets.

The empirical-vs-target coverage scatter is included as a panel of the
consolidated calibration visualization shown above
([`images/calibration_improvement.png`](images/calibration_improvement.png));
the standalone `conformal_coverage.png` was retired in PR #139 when the
two views were merged into a single six-panel image.

### MD-003: Fusion Weight Cross-Validation

**L-BFGS-B cross-validation** via `run_fusion_weight_cv()` replicates the exact
optimization mechanism from `NeuroSymbolicHub._learn_fusion_weights()` (BCE loss,
L-BFGS-B optimizer) on the statistical detector's 3-component scores with
StratifiedKFold(n_splits=3):

| Weight Scheme | Mean Test F1 | Delta vs Optimal | Validated (< 0.02) |
|---------------|-------------|-----------------|-------------------|
| CV-Optimal | baseline | 0.000 | -- |
| Default (0.4/0.3/0.3) | -0.0099 | +0.010 | 29/40 (72.5%) |
| Adaptive (AUC-proportional) | -0.0031 | +0.003 | 33/40 (82.5%) |

Mean CV-optimal weights: R=0.516, K=0.087, I=0.397. This confirms that Resonance and
InfoGeometry carry most of the signal, while Kinematic contributes minimally on tabular
data (consistent with its near-random AUC on shuffled data).

**Adaptive weight distribution** (`_compute_adaptive_weights()`):

| Component | Default | Mean Adaptive | Std | Range |
|-----------|---------|--------------|-----|-------|
| Resonance | 0.40 | 0.360 | 0.172 | [0.000, 0.706] |
| Kinematic | 0.30 | 0.191 | 0.158 | [0.000, 1.000] |
| InfoGeometry | 0.30 | 0.448 | 0.184 | [0.000, 1.000] |

**Status: RESOLVED.** Both default (72.5%) and adaptive (82.5%) weights are validated
as near-optimal. The adaptive AUC-proportional weighting is closer to optimal than the
fixed defaults, with mean delta of only +0.003 F1.

![Adaptive Weight Distribution](images/adaptive_weight_distribution.png)

## CI Integration

The CI pipeline (`.github/workflows/benchmark.yml`) gates on `mercury_benchmark.py`
(Mercury detector in isolation) with regression thresholds set at 15% margin below
measured performance:

- **MIN_ROC_AUC: 0.68** — fail if mean AUC drops below this (measured: 0.803)
- **MIN_F1: 0.50** — fail if mean F1 drops below this (measured: 0.589)
- **MERCURY_ALLOW_SYNTHETIC: false** — no synthetic data fallbacks in CI

`empirical_benchmark.py` runs as a non-gating comparison step on scheduled/manual runs.

## References

1. Tavallaee et al., "A Detailed Analysis of the KDD CUP 99 Data Set", IEEE CISDA 2009
2. Taormina et al., "Battle of the Attack Detection Algorithms", J. Water Resources Planning 2018
3. Su et al., "Robust Anomaly Detection for Multivariate Time Series", KDD 2019
4. Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs", KDD 2018
5. Han et al., "ADBench: Anomaly Detection Benchmark", NeurIPS 2022

## Seven-Axis Evaluation Matrix

_Generated by `python -m benchmarks.seven_axis_runner` (seed=20260504). Do not hand-edit — regenerate with `python -m benchmarks.seven_axis_runner --regenerate-docs`._

| Axis | Score (higher is better) | Notes |
| --- | --- | --- |
| Generalization | 0.934 | OOD AUROC under bias / noise shift (ID=0.997 → OOD=0.934, Δ=+0.063). |
| Scalability | 0.904 | Empirical complexity slope d log(t) / d log(n) = 1.096 over N ∈ [200, 800, 3200]. |
| Data Efficiency | 1.000 | AUROC(N=50) / AUROC(N=1000) — measures how quickly the composer reaches asymptotic ranking quality (got 1.000 / 0.995). |
| Reasoning | 0.940 | Fraction of correctly-flagged anomalies for which the symbolic channel independently crossed the decision threshold. |
| Robustness | 0.964 | AUROC retention under additive noise σ ∈ [0.05, 0.15] (clean=0.996, worst-noise=0.959). |
| Transferability | 0.971 | Cross-domain AUROC retention (in=0.997, cross=0.968). |
| Interpretability | 1.000 | Fraction of decisions accompanied by at least one non-trivial FibringComposer diagnostic (correlation, decorrelation, or domain-affinity bias). |

<!-- end seven-axis-section -->
