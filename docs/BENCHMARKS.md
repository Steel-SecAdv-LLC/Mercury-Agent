# Mercury Agent Benchmark Results

> All numbers measured on 2026-02-14 using `benchmarks/honest_benchmark.py`.
> Detector: `StatisticalAnomalyDetector` (Resonance 40% + Kinematic 30% + InfoGeometry 30%).
> No synthetic data. No fabrication. Every metric is from `benchmarks/honest_benchmark_results.json`.

## Aggregate Performance

| Metric | Value |
|--------|-------|
| Datasets Attempted | 49 |
| Datasets Succeeded | 49 |
| Mean ROC-AUC | 0.814 |
| Median ROC-AUC | 0.850 |
| Std ROC-AUC | 0.166 |
| Mean Oracle F1 | 0.529 |
| Median Oracle F1 | 0.540 |
| Std Oracle F1 | 0.265 |

## Per-Dataset Results

Sorted by ROC-AUC descending.

| Dataset | Domain | Samples | Features | Anomaly % | ROC-AUC | Oracle F1 | F1@0.5 |
|---------|--------|---------|----------|-----------|---------|-----------|--------|
| adbench-musk | adbench | 3062 | 166 | 3.2% | 0.999 | 0.921 | 0.204 |
| adbench-http | adbench | 567498 | 3 | 0.4% | 0.999 | 0.846 | 0.643 |
| adbench-satimage-2 | adbench | 5803 | 36 | 1.2% | 0.998 | 0.797 | 0.440 |
| adbench-WBC | adbench | 223 | 9 | 4.5% | 0.997 | 0.870 | 0.750 |
| adbench-Lymphography | adbench | 148 | 18 | 4.1% | 0.994 | 0.833 | 0.667 |
| adbench-WDBC | adbench | 367 | 30 | 2.7% | 0.993 | 0.800 | 0.000 |
| adbench-shuttle | adbench | 49097 | 9 | 7.2% | 0.992 | 0.958 | 0.676 |
| adbench-breastw | adbench | 683 | 9 | 35.0% | 0.989 | 0.936 | 0.911 |
| adbench-vowels | adbench | 1456 | 12 | 3.4% | 0.989 | 0.776 | 0.182 |
| adbench-InternetAds | adbench | 1966 | 1555 | 18.7% | 0.974 | 0.922 | 0.843 |
| adbench-thyroid | adbench | 3772 | 6 | 2.5% | 0.970 | 0.593 | 0.573 |
| adbench-fraud | adbench | 284807 | 29 | 0.2% | 0.966 | 0.281 | 0.154 |
| adbench-wine | adbench | 129 | 13 | 7.8% | 0.961 | 0.667 | 0.000 |
| adbench-Ionosphere | adbench | 351 | 32 | 35.9% | 0.951 | 0.881 | 0.552 |
| adbench-mnist | adbench | 7603 | 100 | 9.2% | 0.930 | 0.657 | 0.208 |
| adbench-cardio | adbench | 1831 | 21 | 9.6% | 0.925 | 0.678 | 0.156 |
| adbench-cover | adbench | 286048 | 10 | 1.0% | 0.922 | 0.174 | 0.000 |
| adbench-Stamps | adbench | 340 | 9 | 9.1% | 0.919 | 0.543 | 0.000 |
| adbench-backdoor | adbench | 95329 | 196 | 2.4% | 0.915 | 0.863 | 0.844 |
| adbench-PageBlocks | adbench | 5393 | 10 | 9.5% | 0.914 | 0.580 | 0.410 |
| NSL-KDD | security | 148517 | 41 | 48.1% | 0.907 | 0.838 | 0.168 |
| adbench-Hepatitis | adbench | 80 | 19 | 16.2% | 0.898 | 0.727 | 0.000 |
| adbench-optdigits | adbench | 5216 | 64 | 2.9% | 0.897 | 0.263 | 0.000 |
| BATADAL | industrial | 12938 | 43 | 1.7% | 0.867 | 0.540 | 0.394 |
| adbench-donors | adbench | 619326 | 10 | 5.9% | 0.850 | 0.273 | 0.018 |
| adbench-SpamBase | adbench | 4207 | 57 | 39.9% | 0.849 | 0.766 | 0.109 |
| adbench-mammography | adbench | 11183 | 6 | 2.3% | 0.846 | 0.368 | 0.271 |
| adbench-letter | adbench | 1600 | 32 | 6.2% | 0.832 | 0.404 | 0.000 |
| adbench-pendigits | adbench | 6870 | 16 | 2.3% | 0.803 | 0.151 | 0.000 |
| adbench-satellite | adbench | 6435 | 36 | 31.6% | 0.789 | 0.666 | 0.054 |
| adbench-speech | adbench | 3686 | 400 | 1.7% | 0.779 | 0.289 | 0.000 |
| adbench-annthyroid | adbench | 7200 | 6 | 7.4% | 0.772 | 0.374 | 0.304 |
| adbench-celeba | adbench | 202599 | 39 | 2.2% | 0.763 | 0.161 | 0.000 |
| adbench-skin | adbench | 245057 | 3 | 20.8% | 0.754 | 0.529 | 0.000 |
| adbench-magic.gamma | adbench | 19020 | 10 | 35.2% | 0.730 | 0.585 | 0.238 |
| adbench-campaign | adbench | 41188 | 62 | 11.3% | 0.720 | 0.377 | 0.000 |
| adbench-Cardiotocography | adbench | 2114 | 21 | 22.0% | 0.712 | 0.483 | 0.066 |
| adbench-Pima | adbench | 768 | 8 | 34.9% | 0.683 | 0.567 | 0.000 |
| adbench-smtp | adbench | 95156 | 3 | 0.0% | 0.660 | 0.400 | 0.009 |
| adbench-census | adbench | 299285 | 500 | 6.2% | 0.645 | 0.183 | 0.003 |
| adbench-fault | adbench | 1941 | 27 | 34.7% | 0.640 | 0.521 | 0.000 |
| adbench-glass | adbench | 214 | 7 | 4.2% | 0.624 | 0.143 | 0.000 |
| adbench-Wilt | adbench | 4819 | 5 | 5.3% | 0.585 | 0.132 | 0.000 |
| adbench-Waveform | adbench | 3443 | 21 | 2.9% | 0.585 | 0.091 | 0.000 |
| adbench-WPBC | adbench | 198 | 33 | 23.7% | 0.533 | 0.395 | 0.000 |
| adbench-yeast | adbench | 1484 | 8 | 34.2% | 0.516 | 0.509 | 0.004 |
| adbench-ALOI | adbench | 49534 | 27 | 3.0% | 0.493 | 0.059 | 0.006 |
| adbench-landsat | adbench | 6435 | 36 | 20.7% | 0.476 | 0.343 | 0.000 |
| adbench-vertebral | adbench | 240 | 6 | 12.5% | 0.383 | 0.225 | 0.000 |

## Datasets Below AUC 0.50

Three datasets scored below random:

- **ALOI** (0.493): High-dimensional object images. Spectral features not well-suited to image embeddings.
- **landsat** (0.476): Multi-spectral satellite data with 20.7% anomaly rate. High base rate reduces separability.
- **vertebral** (0.383): Small dataset (240 samples) with 12.5% anomaly rate. Insufficient training data for covariance estimation.

These are known hard cases for unsupervised anomaly detectors. No detector achieves strong performance on all 47 ADBench datasets.

## Detector Architecture

**StatisticalAnomalyDetector** uses three Mercury-original mathematical frameworks:

1. **ResonanceScore** (weight 0.4): FFT spectral density profiling at fit time; inference scores by feature deviation weighted by signal quality (noise ratio).
2. **KinematicScore** (weight 0.3): Physics-based derivative analysis (velocity, acceleration, jerk via `np.diff`). Effective on temporally-ordered data.
3. **InfoGeometryScore** (weight 0.3): Fisher Information Mahalanobis OOD scoring via Cholesky-decomposed precision matrix.

No sklearn anomaly detectors are used in the detection pipeline. Dependencies: `numpy`, `torch` (for tensor conversion only).

## Reproducing Results

```bash
python benchmarks/honest_benchmark.py
# Results saved to benchmarks/honest_benchmark_results.json
```

## CI Thresholds

Set in `.github/workflows/benchmark.yml` with 15% margin below measured baseline:

| Threshold | Value | Measured |
|-----------|-------|----------|
| MIN_ROC_AUC | 0.69 | 0.814 |
| MIN_F1 | 0.45 | 0.529 |

## Data Sources

- **ADBench**: 47 datasets from [Minqi824/ADBench](https://github.com/Minqi824/ADBench) (MIT License)
- **NSL-KDD**: Network intrusion detection from [defcom17/NSL_KDD](https://github.com/defcom17/NSL_KDD)
- **BATADAL**: Water infrastructure attacks from [batadal.net](https://www.batadal.net/)

All data downloaded at benchmark runtime. No cached or pre-processed results.
