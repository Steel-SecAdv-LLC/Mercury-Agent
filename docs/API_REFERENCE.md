# API Reference

## MercuryAnomalyDetector

Mercury's original anomaly detection ensemble combining three mathematical frameworks.
Mercury-native — only numpy and scipy required.

```python
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

detector = MercuryAnomalyDetector()
detector.fit(X_train)
result = detector.detect(X_test)

scores = result["scores"]                    # Continuous [0, 1]
is_anomaly = result["is_anomaly"]            # Boolean array
components = result["ensemble_components"]   # Per-component scores
```

### Ensemble Components

| Component | Default Weight | Method |
|-----------|---------------|--------|
| ResonanceScore | 40% | FFT spectral density profiling |
| KinematicScore | 30% | Derivative-based dynamics (velocity, acceleration, jerk) |
| InfoGeometryScore | 30% | Fisher information / Mahalanobis distance |

> **Note:** Weights are **adaptive** after `fit()`. The detector computes per-component
> AUC separation and assigns weights proportional to each component's discriminative
> power. Components with AUC < 0.5 (inverted signal) receive zero weight. The
> 40/30/30 split above is the **fallback default** used only when all components
> produce near-random scores. See `_compute_adaptive_weights()` in `statistical.py`.

### Config Options

```python
detector = MercuryAnomalyDetector(config={
    "z_threshold": 3.0,        # Z-score threshold for outlier detection
    "iqr_multiplier": 1.5,     # IQR fence multiplier
    "threshold": 0.5,          # Decision threshold for is_anomaly
    "auto_calibrate": False,   # Auto-calibrate threshold from score distribution
})
```

### `fit(data) -> MercuryAnomalyDetector`

Fit on training data. Computes baselines for all three components:
- Distributional statistics (mean, std, quartiles)
- Kinematic baselines (jerk/acceleration mean and std per feature)
- Information-geometric manifold (mean, regularized precision matrix)
- FFT spectral profiles per feature

**Args:** `data` — numpy array or torch tensor, shape `(n_samples,)` or `(n_samples, n_features)`.

**Returns:** Self (for method chaining).

### `detect(data) -> dict`

Run anomaly detection. Returns a dictionary with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `scores` | `ndarray` | Combined ensemble scores in [0, 1] |
| `is_anomaly` | `ndarray[bool]` | Boolean anomaly predictions |
| `z_scores` | `ndarray` | Raw z-scores per feature |
| `z_score_continuous` | `ndarray` | Normalized z-score intensity [0, 1] |
| `iqr_scores` | `ndarray` | Continuous IQR-based scores [0, 1] |
| `resonance_scores` | `ndarray` | FFT harmonic anomaly scores [0, 1] |
| `kinematic_scores` | `ndarray` | Physics dynamics scores [0, 1] |
| `info_geometry_scores` | `ndarray` | Fisher OOD scores [0, 1] |
| `ensemble_components` | `dict` | `{"resonance": ..., "kinematic": ..., "info_geometry": ...}` |
| `threshold` | `float` | Effective threshold (may be auto-calibrated) |
| `calibration_diagnostics` | `dict\|None` | Diagnostics when auto-calibrated |
| `detector_type` | `str` | Always `"statistical"` |
| `iqr_flags` | `ndarray[bool]` | Legacy boolean IQR anomalies |

### Auto-Calibration

```python
detector = MercuryAnomalyDetector()
detector.fit(X_train)
detector.enable_auto_calibration(contamination=0.05)
result = detector.detect(X_test)
# result["threshold"] is now auto-calibrated
```

### Backward Compatibility

`StatisticalAnomalyDetector` is retained as an alias:

```python
from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector
# StatisticalAnomalyDetector is MercuryAnomalyDetector
```
