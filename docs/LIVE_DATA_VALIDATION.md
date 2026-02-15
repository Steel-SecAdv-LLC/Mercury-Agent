# Live Data Validation Guide

## Overview

Mercury's anomaly detection is validated on 51 real-world datasets (47 ADBench tabular
datasets + 4 domain-specific loaders). Results are measured, not estimated.

**Current measured performance** (from `benchmarks/honest_benchmark_results.json`):
- Mean AUC: 0.8030
- Median AUC: 0.8852
- Mean Oracle F1: 0.5886
- Datasets: 51/55 successful

## Running Validation Locally

```bash
# Install dependencies
pip install -e ".[ml]"

# Run the honest benchmark
python benchmarks/honest_benchmark.py

# Results saved to benchmarks/honest_benchmark_results.json
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

## Troubleshooting

### Dataset download failures

**Symptom:** `LoaderError: Failed to download dataset`

**Fix:** Check network connectivity and proxy settings. ADBench datasets are fetched
from GitHub. If behind a firewall, set `HTTPS_PROXY`:

```bash
export HTTPS_PROXY=http://proxy:8080
python benchmarks/honest_benchmark.py
```

### Cache corruption

**Symptom:** Benchmark produces different results on re-run, or loader crashes mid-run.

**Fix:** Clear the dataset cache:

```bash
rm -rf ~/.omni_mercury/datasets/
python benchmarks/honest_benchmark.py
```

### Out of memory

**Symptom:** `MemoryError` or killed process during large datasets.

**Fix:** The benchmark caps datasets at 10,000 samples by default (`MAX_SAMPLES`).
If you're still running out of memory, reduce the value in `honest_benchmark.py`.

### CUDA errors

**Symptom:** `CUDA out of memory` or `RuntimeError: CUDA error`.

**Fix:** MercuryAnomalyDetector runs on CPU only (numpy/scipy). CUDA errors come
from other parts of the stack (PyTorch models, visual detectors). Set:

```bash
CUDA_VISIBLE_DEVICES="" python benchmarks/honest_benchmark.py
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
