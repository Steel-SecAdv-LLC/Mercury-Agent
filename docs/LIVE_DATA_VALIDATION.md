# Live Data Validation Guide

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-05-20.

## Overview

Mercury's anomaly detection is validated on real-world datasets — no
synthetic data, no tuning. Results are measured, not estimated.

The committed `benchmarks/mercury_benchmark_results.json` run is the
single source of truth:

- **Committed benchmark run: 66 successful / 75 attempted.**
  47 ADBench tabular + 28 domain loaders attempted; 10 external
  sources unavailable / rate-limited. Measured **Mean AUC 0.8251 /
  Median 0.8747 / Mean Oracle F1 0.5998** (2026-06-21, commit
  a7a194b), surfaced in the README *Latest Benchmark Results* block
  and regenerated on every push to `main`.
- **CI regression-gate floor (historical): Mean AUC 0.803 / Mean
  Oracle F1 0.589.** The benchmark workflow fails if ROC-AUC drops
  below 0.75 or Mean Oracle F1 below 0.55 — ~7% margins below this
  historical measured baseline (see `.github/workflows/benchmark.yml`).
- **Externally-comparable subset: ADBench Mean AUC 0.8251** — the
  numbers comparable to published detectors (the self-labeled
  environmental loaders are threshold-derived; see the README
  "Label provenance and comparability" split).

For the authoritative live figures always consult the README
*Latest Benchmark Results* block and `mercury_benchmark_results.json`.
The per-domain tables further down this page are an older illustrative
snapshot, not the committed run.

## Running Validation Locally

```bash
# Install dependencies
pip install -e ".[ml]"

# Run the mercury benchmark
python benchmarks/mercury_benchmark.py

# Results saved to benchmarks/mercury_benchmark_results.json
```

The benchmark writes downloaded datasets to `./data` and caches under
`./cache` (overridable via `MERCURY_DATA_DIR` / `MERCURY_CACHE_DIR`).
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
rm -rf ./data ./cache
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

- `MIN_ROC_AUC: 0.75` (~7% margin from measured 0.803)
- `MIN_F1: 0.55` (~7% margin from measured 0.589)
- `MERCURY_ALLOW_SYNTHETIC: false` (no synthetic data fallbacks)

A PR that degrades detection performance below these thresholds will fail CI.
