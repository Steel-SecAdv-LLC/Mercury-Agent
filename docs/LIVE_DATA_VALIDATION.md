# Live Data Validation Guide

## Overview

Mercury's anomaly detection is validated on real-world datasets — no
synthetic data, no tuning. Results are measured, not estimated. Two
overlapping cuts of the validation set are referenced across the
documentation:

- **Canonical reproducibility set: 64 / 75.** 47 ADBench tabular +
  28 domain loaders attempted; 11 unavailable / rate-limited external
  sources and 1 known-broken loader (FEMA Disaster) excluded. This is
  the headline figure in the README and `CHANGELOG.md`.
- **`mercury_benchmark.py` direct CI gate: 51 / 55.** The subset that
  the CI benchmark workflow runs against the
  `MercuryAnomalyDetector` ensemble in isolation (47 ADBench + 4
  domain-specific loaders).

Both views report the same measured baseline.

**Current measured performance** (from `benchmarks/mercury_benchmark_results.json`):
- Mean AUC: 0.8030
- Median AUC: 0.8852
- Mean Oracle F1: 0.5886
- Datasets: 51/55 successful (CI subset) / 64/75 canonical

## Running Validation Locally

```bash
# Install dependencies
pip install -e ".[ml]"

# Run the mercury benchmark
python benchmarks/mercury_benchmark.py

# Results saved to benchmarks/mercury_benchmark_results.json
```

The benchmark caches downloaded datasets in `~/.omni_mercury/datasets/`.
Subsequent runs are faster.

## Dataset Categories

| Category | Count | Source |
|----------|-------|--------|
| ADBench tabular | 47 | Alibaba ADBench (via scipy.io) |
| Security | 2 | NSL-KDD, CICIDS-2017 |
| Time series | 3 | SMD, NAB, SMAP/MSL |
| Industrial | 1 | BATADAL |
| Medical | 1 | MIT-BIH |

## Key Results by Domain

| Dataset | AUC | Oracle F1 | Notes |
|---------|-----|-----------|-------|
| NSL-KDD | 0.9721 | 0.9388 | Network intrusion detection |
| BATADAL | 0.8711 | 0.5358 | Water treatment plant |
| SMD | 0.9066 | 0.5881 | Server machine dataset |

## Calibration Validation

In addition to the unsupervised mercury benchmark, a calibration validation harness
tests supervised threshold calibration, conformal coverage, and adaptive ensemble
weights on the same real-world datasets.

```bash
# Full run (includes conformal coverage — slower)
python benchmarks/calibration_validation.py

# Skip conformal for faster iteration
python benchmarks/calibration_validation.py --skip-conformal

# Run specific datasets
python benchmarks/calibration_validation.py --datasets lympho,smtp
```

Results are saved to `benchmarks/calibration_validation_results.json`.
See `docs/BENCHMARKS.md` for detailed results and analysis.

Key findings (40 datasets):
- Threshold calibration improves F1 on 80% of datasets (mean +0.143)
- Adaptive weights shift: InfoGeometry dominates (0.448 mean), Kinematic lowest (0.191)
- Conformal coverage is systematically overconfident (split conformal limitation)

## Troubleshooting

### Dataset download failures

**Symptom:** `LoaderError: Failed to download dataset`

**Fix:** Check network connectivity and proxy settings. ADBench datasets are fetched
from GitHub. If behind a firewall, set `HTTPS_PROXY`:

```bash
export HTTPS_PROXY=http://proxy:8080
python benchmarks/mercury_benchmark.py
```

### Cache corruption

**Symptom:** Benchmark produces different results on re-run, or loader crashes mid-run.

**Fix:** Clear the dataset cache:

```bash
rm -rf ~/.omni_mercury/datasets/
python benchmarks/mercury_benchmark.py
```

### Out of memory

**Symptom:** `MemoryError` or killed process during large datasets.

**Fix:** The benchmark caps datasets at 10,000 samples by default (`MAX_SAMPLES`).
If you're still running out of memory, reduce the value in `mercury_benchmark.py`.

### CUDA errors

**Symptom:** `CUDA out of memory` or `RuntimeError: CUDA error`.

**Fix:** MercuryAnomalyDetector runs on CPU only (numpy/scipy). CUDA errors come
from other parts of the stack (PyTorch models, visual detectors). Set:

```bash
CUDA_VISIBLE_DEVICES="" python benchmarks/mercury_benchmark.py
```

### Timeouts in CI

The full benchmark takes 5-15 minutes depending on hardware. CI is configured with
appropriate timeouts in `.github/workflows/benchmark.yml`. If timeouts occur:

1. Check if a new dataset was added that's unusually large
2. Verify `MAX_SAMPLES` hasn't been increased
3. Check CI runner specs (2+ CPU cores recommended)

## CI Integration

The benchmark runs in CI via `.github/workflows/benchmark.yml` with regression gates:

- `MIN_ROC_AUC: 0.68` (15% margin from measured 0.803)
- `MIN_F1: 0.50` (15% margin from measured 0.589)
- `MERCURY_ALLOW_SYNTHETIC: false` (no synthetic data fallbacks)

A PR that degrades detection performance below these thresholds will fail CI.
