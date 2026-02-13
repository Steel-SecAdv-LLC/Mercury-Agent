# API Reference

## Detectors

### StatisticalAnomalyDetector

Z-score, IQR, and IsolationForest ensemble for tabular anomaly detection.

```python
from omni_mercury_engine.detectors import StatisticalAnomalyDetector

detector = StatisticalAnomalyDetector()
detector.fit(X_train)              # Fit on training data (numpy array)
result = detector.detect(X_test)   # Returns dict with scores and predictions

scores = result["scores"]          # Continuous anomaly scores [0, 1]
is_anomaly = result["is_anomaly"]  # Boolean array
threshold = result["threshold"]    # Effective threshold used
```

**Config options:**

```python
detector = StatisticalAnomalyDetector(config={
    "z_threshold": 3.0,         # Z-score threshold
    "iqr_multiplier": 1.5,      # IQR outlier multiplier
    "contamination": 0.1,       # IsolationForest contamination
    "threshold": 0.5,           # Anomaly classification threshold
    "auto_calibrate": True,     # Auto-calibrate threshold from scores
})
```

### TemporalAnomalyDetector

LSTM-based detector for time-series anomaly detection.

```python
from omni_mercury_engine.detectors import TemporalAnomalyDetector

detector = TemporalAnomalyDetector()
detector.fit(X_train)              # Fit on time-series training data
result = detector.detect(X_test)   # Returns dict with scores

scores = result["scores"]          # Continuous anomaly scores [0, 1]
```

**Config options:**

```python
detector = TemporalAnomalyDetector(config={
    "window_size": 10,           # Temporal window size
    "change_threshold": 2.0,     # Sudden change detection threshold
    "auto_calibrate": True,      # Auto-calibrate threshold
})
```

## Threshold Calibration

Per-dataset threshold optimization to maximize F1.

```python
from omni_mercury_engine.detectors.threshold_calibrator import (
    find_optimal_threshold,
    ThresholdOptimizer,
)

# Simple: find best threshold for one dataset
threshold = find_optimal_threshold(scores, labels)
predictions = (scores >= threshold).astype(int)

# Advanced: cache thresholds for multiple datasets
optimizer = ThresholdOptimizer()
optimizer.optimize("cardio", scores_cardio, labels_cardio)
optimizer.optimize("thyroid", scores_thyroid, labels_thyroid)
optimizer.save("thresholds.json")
```

## Dataset Loaders

### ADBenchLoader

47 tabular anomaly detection datasets from NeurIPS 2022.

```python
from omni_mercury_engine.datasets.adbench import ADBenchLoader
from omni_mercury_engine.datasets.base import DatasetConfig

config = DatasetConfig(name="adbench-cardio", preprocessing={"dataset": "cardio"})
loader = ADBenchLoader(config)
loader.download()
X, y = loader._load_raw()   # X: features, y: binary labels
X = loader.preprocess(X)     # Z-score normalization
```

Available datasets: cardio, thyroid, mammography, breastw, Ionosphere, Pima,
satellite, shuttle, wine, glass, musk, arrhythmia, optdigits, pendigits,
vertebral, WBC, and 31 more (47 total).

### NSLKDDLoader

Network intrusion detection dataset (148K records).

```python
from omni_mercury_engine.datasets.security import NSLKDDLoader
from omni_mercury_engine.datasets.base import DatasetConfig

config = DatasetConfig(name="nsl-kdd", preprocessing={"binary": True})
loader = NSLKDDLoader(config)
features, labels = loader.load_data()
features = loader.preprocess(features)
```

### CICIDSLoader

Modern network intrusion dataset (2.8M flows).

```python
from omni_mercury_engine.datasets.security import CICIDSLoader
from omni_mercury_engine.datasets.base import DatasetConfig

config = DatasetConfig(name="cicids", preprocessing={"binary": True})
loader = CICIDSLoader(config)
features, labels = loader.load_data()
```

## Dataset Cache

Filesystem cache for downloaded datasets.

```python
from omni_mercury_engine.datasets.cache import DatasetCache

cache = DatasetCache()  # Uses ~/.mercury/datasets or MERCURY_DATASET_CACHE env var
cached = cache.get("adbench_cardio")
if cached is None:
    # Download and cache
    cache.set("adbench_cardio", {"X": X, "y": y})
```

## See Also

- Source docstrings for full parameter documentation
- [BENCHMARKS.md](BENCHMARKS.md) for measured performance on real data
- [LIVE_DATA_VALIDATION.md](LIVE_DATA_VALIDATION.md) for running benchmarks
