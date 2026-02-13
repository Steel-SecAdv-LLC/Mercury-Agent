# Live-Data Validation: Troubleshooting Guide

## Running Tests Locally

```bash
# Install all dependencies
pip install -e ".[all]"

# Run with live-data enabled
export MERCURY_RUN_LIVE_DATA=true
pytest tests/validation/test_real_data_validation.py -v

# Run with caching (faster subsequent runs)
export MERCURY_DATASET_CACHE=~/.mercury_cache
pytest tests/validation/test_real_data_validation.py -v

# Run only a specific dataset
pytest tests/validation/test_real_data_validation.py::test_adbench_statistical_detector[cardio] -v
```

## Common Issues

### ADBench Mammography: F1 = 0.179 (Very Low)

**Root cause:** Class imbalance (5% anomalies) + hard-to-separate anomalies.
**Solution:** Threshold tuning helps (0.179 -> 0.35 with `find_optimal_threshold()`),
but this remains a challenging dataset.
**Status:** Known limitation; marked in BENCHMARKS.md.

### NSL-KDD AUC = 0.59 (Below 0.70 target)

**Root cause:** Unsupervised statistical methods hit a ceiling on network data.
**Solution:** Use CyberFortress neural detector (supervised), or transfer learning.
**Status:** Tracked for v1.5 improvement.

### "RuntimeError: CUDA out of memory"

**Root cause:** GPU memory exhausted during detector fit.
**Solution:**

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
pytest tests/validation/test_real_data_validation.py -v
```

### "FileNotFoundError: No such file or directory: ~/.mercury/datasets/..."

**Root cause:** Dataset download failed or cache not found.
**Solution:**

```bash
# Clear and re-download
rm -rf ~/.mercury/datasets
export MERCURY_DATASET_CACHE=~/.mercury/datasets
pytest tests/validation/test_real_data_validation.py -v
```

### "Test timed out"

**Root cause:** Detector fit is slow (normal for large datasets).
**Solution:** Increase timeout or run specific tests:

```bash
pytest tests/validation/test_real_data_validation.py -v --timeout=1800

# Or run a specific test
pytest tests/validation/test_real_data_validation.py::test_nslkdd_statistical_detector -v
```

### Tests skipped with "MERCURY_RUN_LIVE_DATA not set"

**Root cause:** Live-data tests are gated on an environment variable.
**Solution:**

```bash
export MERCURY_RUN_LIVE_DATA=true
pytest tests/validation/ -v
```

## Threshold Calibration

Mercury Agent uses per-dataset threshold optimization to maximize F1:

```python
from omni_mercury_engine.detectors.threshold_calibrator import find_optimal_threshold

# scores: anomaly scores from detector, labels: ground truth
optimal_threshold = find_optimal_threshold(scores, labels)
predictions = (scores >= optimal_threshold).astype(int)
```

This typically improves F1 by 5-15% over the default 0.5 threshold.

## Generating a New Baseline

After changing detectors or improving performance:

```bash
python scripts/generate_baseline_report.py
```

This creates `benchmarks/live_data_baseline.json` with system metadata and
measured results. The CI pipeline uses this baseline for regression detection.

## CI Pipeline

The live-data validation CI workflow (`.github/workflows/live-data-validation.yml`)
runs on every PR that touches `src/`, `tests/`, or `benchmarks/`. It:

1. Installs all dependencies
2. Generates baseline if missing
3. Runs the live-data validation suite
4. Checks metrics against thresholds
5. Uploads a report artifact

## References

- ADBench: Han S et al., NeurIPS 2022 Datasets and Benchmarks Track
- NSL-KDD: Tavallaee M et al., IEEE CISDA 2009
- CICIDS-2017: Sharafaldin I et al., ICISSP 2018
