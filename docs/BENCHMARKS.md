# Mercury Agent Benchmark Results

## What This Measures

This document reports the empirical performance of `StatisticalAnomalyDetector` — Mercury's
unsupervised anomaly detection ensemble — on labeled real-world datasets.

**Ensemble composition** (no external anomaly-detection dependencies):

| Component | Weight | Method |
|-----------|--------|--------|
| ResonanceScore | 40% | FFT harmonic spectral anomaly (FFT at fit, O(n*d) inference) |
| KinematicScore | 30% | Physics-based jerk/curvature detection (O(n*d)) |
| InfoGeometryScore | 30% | Fisher Information Mahalanobis OOD (O(n*d^2) inference) |

**Protocol:**
- Normal-only training (unsupervised) with `StandardScaler`
- ROC-AUC from continuous scores
- Oracle F1: best F1 over 101 thresholds from 0.0 to 1.0 (upper bound, not operational)
- No hyperparameter tuning was performed
- Datasets capped at 10,000 samples with stratified sampling

## How to Reproduce

```bash
python benchmarks/honest_benchmark.py
```

Results are saved to `benchmarks/honest_benchmark_results.json`.
Every number in this document must exist in that file. If the file does not exist,
the benchmark has not been run and no claims can be made.

## Per-Dataset Results

**Status: Not yet measured — run `python benchmarks/honest_benchmark.py`**

Once the benchmark has been run, this section will contain a table with columns:

| Dataset | Domain | ROC-AUC | Oracle F1 | Fit Time (s) | Score Time (s) |
|---------|--------|---------|-----------|---------------|-----------------|
| *(populated from honest_benchmark_results.json)* | | | | | |

To generate this table from results:

```python
import json

with open("benchmarks/honest_benchmark_results.json") as f:
    data = json.load(f)

for r in sorted(data["results"], key=lambda x: x.get("roc_auc", 0), reverse=True):
    if "error" not in r:
        print(f"| {r['dataset']} | {r.get('domain','')} | {r['roc_auc']:.3f} | {r['oracle_f1']:.3f} | {r['fit_time']:.2f} | {r['score_time']:.2f} |")
```

## Per-Component Analysis

**Status: Not yet measured — run `python benchmarks/honest_benchmark.py`**

The honest benchmark records per-component ROC-AUC for each dataset:
- `resonance_auc`: ResonanceScore alone
- `kinematic_auc`: KinematicScore alone
- `info_geometry_auc`: InfoGeometryScore alone

These are saved in `honest_benchmark_results.json` per dataset entry.

## Known Weaknesses

1. **KinematicScore underperforms on unordered tabular data.**
   KinematicScore computes derivatives (velocity, acceleration, jerk) via `np.diff`.
   This assumes adjacent rows are temporally ordered. On shuffled tabular data
   (e.g., ADBench datasets), derivatives are meaningless noise. Expect the
   kinematic component to contribute near-random AUC (~0.5) on such datasets.

2. **Ensemble inversion on image-like data.**
   On high-dimensional image-like features, the ensemble score can invert
   (anomalies score lower than normal). This manifests as ROC-AUC < 0.5.

3. **Oracle F1 is an upper bound, not operational performance.**
   The oracle threshold sweeps 101 values and picks the best F1. A deployed
   system would use a fixed threshold (e.g., 0.5) and would achieve lower F1.

4. **No hyperparameter tuning was performed.**
   All results use default parameters. Tuning per-dataset could improve
   performance but would also risk overfitting.

5. **InfoGeometryScore requires d < n.**
   The Fisher Information component inverts a d×d covariance matrix. When
   the number of features exceeds the number of training samples, the matrix
   is singular and a pseudo-inverse is used, degrading accuracy.

## Data Sources

- **ADBench** (47 datasets): Downloaded from the ADBench GitHub repository.
  Tabular anomaly detection benchmarks across diverse domains.
- **NSL-KDD**: Network intrusion detection from Canadian Institute for Cybersecurity.
- **SMD**: Server Machine Dataset from Tsinghua University (28 machines).
- **SMAP/MSL**: NASA spacecraft telemetry from OmniAnomaly mirror.
- **NAB**: Numenta Anomaly Benchmark (58 univariate time series).
- **BATADAL**: Water infrastructure attack detection (train + test with ATT_FLAG labels).
- **CICIDS-2017**: Modern network attack flows from CIC.
- **MIT-BIH**: ECG arrhythmia dataset from PhysioNet.

Some datasets require network access or credentials. Failed downloads are
recorded as errors in the results JSON, not replaced with synthetic data.

## CI Integration

The CI pipeline (`.github/workflows/benchmark.yml`) runs `empirical_benchmark.py`
with regression gates:

- **MIN_ROC_AUC: 0.70** — fail if mean AUC drops below this
- **MIN_F1: 0.35** — fail if mean F1 drops below this
- **MERCURY_ALLOW_SYNTHETIC: false** — no synthetic data fallbacks in CI

## References

1. Tavallaee et al., "A Detailed Analysis of the KDD CUP 99 Data Set", IEEE CISDA 2009
2. Taormina et al., "Battle of the Attack Detection Algorithms", J. Water Resources Planning 2018
3. Su et al., "Robust Anomaly Detection for Multivariate Time Series", KDD 2019
4. Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs", KDD 2018
5. Han et al., "ADBench: Anomaly Detection Benchmark", NeurIPS 2022
