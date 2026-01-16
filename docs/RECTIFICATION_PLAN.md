# Mercury-Agent Rectification Plan

## Executive Summary

This document provides a comprehensive rectification plan for Mercury-Agent's anomaly detection system. The analysis identifies 7 critical issues causing significant performance degradation, with ROC-AUC gaps of 0.3-0.86 compared to baseline IsolationForest.

**Current Performance vs Baseline:**

| Dataset | Anomaly % | Mercury-Agent | IsolationForest | Gap |
|---------|-----------|---------------|-----------------|-----|
| covtype | 0.48% | 0.063 | 0.916 | -0.853 |
| kddcup99 | 3.0% | 0.071 | 0.937 | -0.866 |
| digits_8 | 9.6% | 0.442 | 0.756 | -0.314 |
| breast_cancer | 37% | 0.228 | 0.819 | -0.591 |

**Target Performance:**
- Minimum: ≥0.80 mean ROC-AUC across datasets
- Target: Beat IsolationForest by 0.1
- Stretch: Top performance with <10% FPR

---

## Issue Analysis

### Issue #1: Untrained Fusion Neural Network [CRITICAL]

**Location:** `src/omni_mercury_engine/engine.py:522-534`

**Evidence:**
```python
def _init_fusion(self) -> None:
    """Initialize ML fusion components."""
    if self.mode == "fusion":
        self.fusion_model = OmniFusionModel()  # Random weights!
        self.fusion_model.to(self.device)
        self.fusion_inference = FusionInference(...)
```

**Root Cause:** `OmniFusionModel()` initializes with random weights. The `train_with_advanced_optimizers()` method exists in `fusion_network.py:408-585` but is never called during initialization or inference.

**Impact:** Random fusion weights produce meaningless anomaly probabilities, degrading ROC-AUC.

---

### Issue #2: Logistic Regression Fallback Fails Silently [HIGH]

**Location:** `benchmarks/empirical_benchmark.py:2017-2022`

**Evidence:**
```python
# Check if we have enough variance in scores
score_std = np.std(score_matrix, axis=0)
valid_detectors_mask = score_std > 0.01
if valid_detectors_mask.sum() < 1:
    logger.warning("No detectors with score variance, skipping weight learning")
    return  # Silent return without setting _fusion_trained = False!
```

**Root Cause:** When detector scores have low variance (due to Issue #4), fusion weight learning is silently skipped without proper fallback.

**Impact:** Cascading failure - untrained fusion produces random outputs.

---

### Issue #3: Discrete Score Destruction [HIGH]

**Location:** `src/omni_mercury_engine/detectors/statistical.py:96-104`

**Evidence:**
```python
z_score_flags = np.any(np.abs(z_scores) > self.z_threshold, axis=1)  # BOOLEAN

combined_scores = (
    z_score_flags.astype(float) * 0.4       # 0.0 or 0.4
    + iqr_anomalies.astype(float) * 0.3    # 0.0 or 0.3
    + (if_anomalies == -1).astype(float) * 0.3  # 0.0 or 0.3
)
# OUTPUT: Only {0.0, 0.3, 0.4, 0.7, 1.0} - 5 discrete values!
```

**Root Cause:** Boolean flags destroy continuous score information from z-scores, IQR, and IsolationForest.

**Impact:** Loss of ranking granularity; 5 discrete values cannot capture score distribution, harming ROC-AUC.

---

### Issue #4: Normal-Only Fitting Defeats Anomaly Detection [MEDIUM]

**Location:** `benchmarks/empirical_benchmark.py:1950-1954`

**Evidence:**
```python
# Step 1: Fit all detectors on normal training data
normal_mask = y == 0
X_normal = X[normal_mask] if normal_mask.sum() > 10 else X
logger.info(f"Fitting detectors on {len(X_normal)} normal samples...")
```

**Root Cause:** Detectors fitted on normal-only data produce uniform low scores on test data, causing zero variance.

**Impact:** Triggers Issue #2's silent skip, leading to untrained fusion.

---

### Issue #5: Contamination Mismatch [MEDIUM]

**Location:** `src/omni_mercury_engine/detectors/statistical.py:49-53`

**Evidence:**
```python
self.contamination = self.config.get("contamination", 0.1)  # Hardcoded default

self.isolation_forest = IsolationForest(
    contamination=self.contamination,  # Uses 0.1 regardless of dataset
    random_state=42,
)
```

**Root Cause:** Hardcoded 0.1 contamination doesn't adapt to dataset characteristics.

**Impact:** Over-flags in low-anomaly datasets (covtype: 0.48%), high false positives.

---

### Issue #6: Feature Dimension Mismatch in Fusion Model [HIGH]

**Location:** `src/omni_mercury_engine/ml/fusion_network.py:355-357`

**Evidence:**
```python
elif features.dim() == 2:
    # Creates NEW linear layer on every forward pass - memory leak!
    proj = nn.Linear(features.shape[1], self.hidden_dim).to(features.device)
    encoded_features[name] = proj(features)
```

**Root Cause:** When feature dimensions don't match expected, a new projection layer is created per forward pass. These layers aren't tracked in `model.parameters()`.

**Impact:** Memory leak, no gradient flow, inconsistent projections between batches.

---

### Issue #7: No Score Continuity in Pipeline [LOW]

**Location:**
- `src/omni_mercury_engine/detectors/temporal.py:131` - `np.minimum(z_score / 3.0, 1.0)`
- `src/omni_mercury_engine/detectors/directive.py:161` - `np.minimum(normalized_diffs / self.convergence_threshold, 1.0)`

**Evidence:** Scores clipped to [0, 1] with hard thresholds lose ranking information for extreme anomalies.

**Impact:** Reduced variance for fusion; extreme anomalies indistinguishable.

---

## Implementation Roadmap

### Phase 1: Restore Signal Integrity [P0 - Critical]

**Goal:** Preserve continuous scores from detectors.
**Expected Impact:** +0.4 ROC-AUC

#### Fix 1.1: Continuous Statistical Scores

**File:** `src/omni_mercury_engine/detectors/statistical.py`
**Lines:** 92-104

```python
# BEFORE (discrete):
z_score_flags = np.any(np.abs(z_scores) > self.z_threshold, axis=1)
combined_scores = (
    z_score_flags.astype(float) * 0.4
    + iqr_anomalies.astype(float) * 0.3
    + (if_anomalies == -1).astype(float) * 0.3
)

# AFTER (continuous):
def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
    """Detect anomalies with continuous scores for ML fusion."""
    if not self._is_fitted:
        raise DetectorException("Detector must be fitted before detection")

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Compute continuous z-score intensity (not boolean flags)
    z_scores = self._compute_z_scores(data)
    z_score_intensity = np.max(np.abs(z_scores), axis=1) / (self.z_threshold + 1e-8)
    z_score_continuous = np.clip(z_score_intensity, 0, 3.0) / 3.0  # Normalize to [0, 1]

    # Compute continuous IQR scores (distance from bounds)
    iqr_scores = self._compute_iqr_scores(data)

    # Use IsolationForest decision_function for continuous scores
    # decision_function returns negative for anomalies, so negate and normalize
    if_raw_scores = -self.isolation_forest.decision_function(data)
    if_normalized = (if_raw_scores - if_raw_scores.min()) / (if_raw_scores.max() - if_raw_scores.min() + 1e-8)

    # Combine continuous scores
    combined_scores = (
        z_score_continuous * 0.4
        + iqr_scores * 0.3
        + if_normalized * 0.3
    )

    is_anomaly = combined_scores > self.threshold

    return {
        "is_anomaly": is_anomaly,
        "scores": combined_scores,
        "z_scores": z_scores,
        "z_score_continuous": z_score_continuous,
        "iqr_scores": iqr_scores,
        "isolation_forest_scores": if_normalized,
        "detector_type": "statistical",
    }

def _compute_iqr_scores(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Compute continuous IQR-based anomaly scores."""
    iqr = self.q3 - self.q1 + 1e-8
    lower_bound = self.q1 - self.iqr_multiplier * iqr
    upper_bound = self.q3 + self.iqr_multiplier * iqr

    # Distance from bounds (0 = within bounds, >0 = outside)
    lower_dist = np.maximum(lower_bound - data, 0)
    upper_dist = np.maximum(data - upper_bound, 0)

    # Max distance across features, normalized by IQR
    dist_from_bounds = np.maximum(lower_dist, upper_dist)
    normalized_dist = dist_from_bounds / iqr

    # Aggregate across features and clip
    scores = np.mean(normalized_dist, axis=1)
    return np.clip(scores, 0, 1)
```

#### Fix 1.2: Adaptive Contamination Estimation

**File:** `src/omni_mercury_engine/detectors/statistical.py`
**Lines:** 45-55

```python
def __init__(self, config: dict[str, Any] | None = None) -> None:
    super().__init__(config)
    self.z_threshold = self.config.get("z_threshold", 3.0)
    self.iqr_multiplier = self.config.get("iqr_multiplier", 1.5)

    # Allow contamination from config, but will be adaptively estimated in fit()
    self._config_contamination = self.config.get("contamination", None)
    self.contamination = 0.1  # Default, will be updated

    self.scaler = StandardScaler()
    self.isolation_forest: IsolationForest | None = None  # Lazy init after contamination estimate

    self.mean: np.ndarray[Any, Any] | None = None
    self.std: np.ndarray[Any, Any] | None = None
    self.q1: np.ndarray[Any, Any] | None = None
    self.q3: np.ndarray[Any, Any] | None = None

def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> StatisticalAnomalyDetector:
    """Fit detector with adaptive contamination estimation."""
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Compute statistics
    self.mean = np.mean(data, axis=0)
    self.std = np.std(data, axis=0) + 1e-8
    self.q1 = np.percentile(data, 25, axis=0)
    self.q3 = np.percentile(data, 75, axis=0)

    # Adaptive contamination estimation using z-scores
    if self._config_contamination is not None:
        self.contamination = self._config_contamination
    else:
        z_scores = (data - self.mean) / self.std
        # Estimate based on statistical outliers (|z| > 3)
        outlier_fraction = np.mean(np.any(np.abs(z_scores) > 3.0, axis=1))
        # Clamp to reasonable range [0.001, 0.5]
        self.contamination = float(np.clip(outlier_fraction * 2, 0.001, 0.5))

    # Initialize IsolationForest with estimated contamination
    self.isolation_forest = IsolationForest(
        contamination=self.contamination,
        random_state=42,
        n_estimators=100,
    )

    self.scaler.fit(data)
    self.isolation_forest.fit(data)

    self._is_fitted = True
    return self
```

---

### Phase 2: Train OmniFusionModel [P0 - Critical]

**Goal:** Implement automatic training for the fusion neural network.
**Expected Impact:** +0.3 ROC-AUC

#### Fix 2.1: Auto-Training in Engine Initialization

**File:** `src/omni_mercury_engine/engine.py`

Add new method and modify `_init_fusion`:

```python
def _init_fusion(self) -> None:
    """Initialize ML fusion components with training support."""
    if self.mode == "fusion":
        self.fusion_model = OmniFusionModel()
        self.fusion_model.to(self.device)
        self.fusion_inference = FusionInference(
            model=self.fusion_model,
            device=str(self.device),
        )
        self._fusion_trained = False  # Track training state
        logger.info(
            "OmniFusionModel initialized (untrained). Call fit_fusion() or "
            "train_fusion_model() before detection for optimal performance."
        )

def fit_fusion(
    self,
    X: np.ndarray,
    y: np.ndarray | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    validation_split: float = 0.2,
    contamination: float | None = None,
) -> dict[str, Any]:
    """
    Fit the fusion model on training data with semi-supervised learning.

    This method extracts features from all detectors and trains the OmniFusionModel
    to produce calibrated anomaly scores. Supports both supervised (with labels)
    and semi-supervised (estimated pseudo-labels) training.

    Args:
        X: Training features (n_samples, n_features)
        y: Optional training labels (1=anomaly, 0=normal). If None, uses
           semi-supervised learning with pseudo-labels from detector consensus.
        epochs: Maximum training epochs (default: 50)
        batch_size: Training batch size (default: 32)
        learning_rate: Learning rate for optimizer (default: 0.001)
        early_stopping_patience: Epochs without improvement before stopping
        validation_split: Fraction of data for validation
        contamination: Expected anomaly fraction for pseudo-labeling. If None,
                      estimated from data using adaptive methods.

    Returns:
        Dictionary with training metrics including final_loss, best_loss,
        epochs_trained, and convergence information.

    Example:
        >>> engine = OmniMercuryEngine(mode="fusion")
        >>> metrics = engine.fit_fusion(X_train, y_train, epochs=100)
        >>> print(f"Training loss: {metrics['best_loss']:.4f}")
    """
    from torch.utils.data import DataLoader, TensorDataset

    if self.mode != "fusion":
        raise ValueError("fit_fusion() requires mode='fusion'")

    # GPU check with fallback
    device = self.device
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable, falling back to CPU")
        device = torch.device("cpu")

    # Fit all base detectors first
    logger.info(f"Fitting {len(self.detectors)} base detectors on {len(X)} samples...")
    for name, detector in self.detectors.items():
        try:
            if not detector.is_fitted():
                detector.fit(X)
                logger.debug(f"Fitted detector: {name}")
        except Exception as e:
            logger.warning(f"Failed to fit detector {name}: {e}")

    # Extract features from all detectors
    logger.info("Extracting detector features for fusion training...")
    detector_features = {}
    for name, detector in self.detectors.items():
        try:
            features = detector.extract_features(X)
            detector_features[name] = features
        except Exception as e:
            logger.warning(f"Failed to extract features from {name}: {e}")

    if not detector_features:
        raise RuntimeError("No detector features could be extracted")

    # Generate pseudo-labels if not provided (semi-supervised)
    if y is None:
        logger.info("No labels provided, using semi-supervised pseudo-labeling...")
        y = self._generate_pseudo_labels(X, detector_features, contamination)

    # Prepare training data
    labels_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Create dataset
    n_samples = len(X)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val

    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Training loop with early stopping
    self.fusion_model.train()
    self.fusion_model.to(device)

    optimizer = torch.optim.AdamW(
        self.fusion_model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0
    loss_history = []

    for epoch in range(epochs):
        # Training
        self.fusion_model.train()
        train_losses = []

        for start_idx in range(0, n_train, batch_size):
            end_idx = min(start_idx + batch_size, n_train)
            batch_indices = train_indices[start_idx:end_idx]

            # Get batch features
            batch_features = {
                name: feat[batch_indices].to(device)
                for name, feat in detector_features.items()
            }
            batch_labels = labels_tensor[batch_indices].to(device)

            optimizer.zero_grad()
            outputs = self.fusion_model(batch_features)
            loss = torch.nn.functional.binary_cross_entropy(
                outputs["anomaly_probs"], batch_labels
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        self.fusion_model.eval()
        val_losses = []

        with torch.no_grad():
            for start_idx in range(0, n_val, batch_size):
                end_idx = min(start_idx + batch_size, n_val)
                batch_indices = val_indices[start_idx:end_idx]

                batch_features = {
                    name: feat[batch_indices].to(device)
                    for name, feat in detector_features.items()
                }
                batch_labels = labels_tensor[batch_indices].to(device)

                outputs = self.fusion_model(batch_features)
                loss = torch.nn.functional.binary_cross_entropy(
                    outputs["anomaly_probs"], batch_labels
                )
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses) if val_losses else avg_train_loss
        loss_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in self.fusion_model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: train_loss={avg_train_loss:.4f}, "
                f"val_loss={avg_val_loss:.4f}"
            )

    # Restore best model
    if best_state is not None:
        self.fusion_model.load_state_dict(best_state)

    self.fusion_model.eval()
    self._fusion_trained = True

    logger.info(f"Fusion training complete. Best val_loss: {best_val_loss:.4f}")

    return {
        'final_loss': loss_history[-1]['val_loss'] if loss_history else 0.0,
        'best_loss': best_val_loss,
        'epochs_trained': len(loss_history),
        'loss_history': loss_history,
        'early_stopped': epochs_without_improvement >= early_stopping_patience,
    }

def _generate_pseudo_labels(
    self,
    X: np.ndarray,
    detector_features: dict[str, torch.Tensor],
    contamination: float | None = None,
) -> np.ndarray:
    """
    Generate pseudo-labels using detector consensus for semi-supervised learning.

    Uses adaptive contamination estimation and ensemble voting from detector
    scores to identify likely anomalies for training.
    """
    n_samples = len(X)

    # Collect scores from all detectors
    all_scores = []
    for name, detector in self.detectors.items():
        try:
            result = detector.detect(X)
            scores = result.get("scores", result.get("is_anomaly", np.zeros(n_samples)))
            if isinstance(scores, (list, np.ndarray)):
                scores = np.array(scores).flatten()
                if len(scores) == n_samples:
                    all_scores.append(scores)
        except Exception:
            continue

    if not all_scores:
        # Fallback: use distance from mean
        mean = np.mean(X, axis=0)
        distances = np.linalg.norm(X - mean, axis=1)
        all_scores = [distances / (distances.max() + 1e-8)]

    # Ensemble score (average)
    ensemble_score = np.mean(all_scores, axis=0)

    # Estimate contamination if not provided
    if contamination is None:
        # Use IQR-based estimation
        q1, q3 = np.percentile(ensemble_score, [25, 75])
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        contamination = float(np.mean(ensemble_score > upper_fence))
        contamination = max(0.001, min(contamination, 0.5))

    # Threshold at (1 - contamination) percentile
    threshold = np.percentile(ensemble_score, (1 - contamination) * 100)
    pseudo_labels = (ensemble_score > threshold).astype(float)

    logger.info(
        f"Generated pseudo-labels: contamination={contamination:.4f}, "
        f"n_anomalies={int(pseudo_labels.sum())}/{n_samples}"
    )

    return pseudo_labels
```

#### Fix 2.2: Dynamic Feature Dimensions in OmniFusionModel

**File:** `src/omni_mercury_engine/ml/fusion_network.py`
**Lines:** 288-296, 348-358

```python
def __init__(
    self,
    feature_dims: dict[str, int] | None = None,
    hidden_dim: int = 128,
    num_heads: int = 4,
    dropout: float = 0.1,
    num_classes: int = 10,
):
    super().__init__()

    if feature_dims is None:
        feature_dims = {
            "statistical": 10,
            "temporal": 32,
            "spatial": 32,
            "dimensional": 50,
            "directive": 20,
            "quantum": 16,
            "astrophysical": 24,
            "biometric": 128,
            "affective": 64,
            "neural": 48,
            "consciousness": 32,
            "security": 40,
            "resilience": 16,
        }

    self.feature_dims = feature_dims
    self.hidden_dim = hidden_dim

    # Track dynamically created projection layers
    self._dynamic_projections: nn.ModuleDict = nn.ModuleDict()

    # ... rest of initialization ...

def _get_or_create_projection(
    self,
    name: str,
    input_dim: int,
    device: torch.device,
) -> nn.Module:
    """
    Get or create a projection layer for dynamic feature dimensions.

    This addresses the memory leak issue where new layers were created
    on every forward pass. Layers are cached in _dynamic_projections
    and properly tracked in model.parameters().
    """
    key = f"{name}_{input_dim}"

    if key not in self._dynamic_projections:
        proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
        ).to(device)
        self._dynamic_projections[key] = proj
        logger.debug(f"Created dynamic projection for {name}: {input_dim} -> {self.hidden_dim}")

    return self._dynamic_projections[key]

def forward(
    self,
    detector_features: dict[str, torch.Tensor],
    detector_scores: dict[str, torch.Tensor] | None = None,
    return_attention: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Forward pass with dynamic feature dimension handling.
    """
    encoded_features = {}

    for name, features in detector_features.items():
        actual_dim = features.shape[1] if features.dim() == 2 else features.shape[-1]
        expected_dim = self.feature_dims.get(name)

        if name in self.encoders:
            # Use specialized encoder if dimension matches
            if expected_dim is None or actual_dim == expected_dim:
                try:
                    encoded_features[name] = self.encoders[name](features)
                    continue
                except RuntimeError as e:
                    logger.debug(f"Encoder {name} failed: {e}, using dynamic projection")

            # Dimension mismatch - use dynamic projection
            proj = self._get_or_create_projection(name, actual_dim, features.device)
            encoded_features[name] = proj(features)

        elif name in self.generic_encoders:
            if expected_dim is None or actual_dim == expected_dim:
                encoded_features[name] = self.generic_encoders[name](features)
            else:
                proj = self._get_or_create_projection(name, actual_dim, features.device)
                encoded_features[name] = proj(features)

        elif features.dim() == 2 and features.shape[1] == self.hidden_dim:
            encoded_features[name] = features

        elif features.dim() == 2:
            # Dynamic projection for unknown detectors
            proj = self._get_or_create_projection(name, actual_dim, features.device)
            encoded_features[name] = proj(features)

    # ... rest of forward pass ...
```

---

### Phase 3: Fix Detector Fitting Strategy [P1 - High]

**Goal:** Enable semi-supervised fitting on full training data.
**Expected Impact:** +0.1 ROC-AUC

#### Fix 3.1: Semi-Supervised Fitting in Benchmark

**File:** `benchmarks/empirical_benchmark.py`
**Lines:** 1950-1970

```python
def _train_fusion_on_features(self, X: np.ndarray, y: np.ndarray) -> None:
    """
    Learn optimal detector weights using semi-supervised approach.

    Changed from normal-only fitting to full dataset fitting with
    contamination-aware training.
    """
    if self.engine is None:
        logger.warning("Cannot train fusion: engine not initialized")
        return

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        logger.info(f"Learning detector weights from {len(X)} training samples...")

        # CHANGED: Fit on FULL training data, not normal-only
        # This ensures detectors learn the full data distribution
        # for better anomaly discrimination
        logger.info(f"Fitting detectors on full training set ({len(X)} samples)...")

        fitted_detectors = []
        for name, detector in self.engine.detectors.items():
            try:
                if hasattr(detector, "fit"):
                    # Pass contamination hint if detector supports it
                    if hasattr(detector, 'contamination'):
                        detector.contamination = self.contamination
                    detector.fit(X)
                    fitted_detectors.append(name)
                    logger.debug(f"Fitted detector: {name}")
            except Exception as e:
                logger.debug(f"Failed to fit detector {name}: {e}")

        if not fitted_detectors:
            logger.warning("No detectors could be fitted, using fallback")
            self._fusion_trained = False
            return

        # ... rest of method unchanged, but add explicit fallback flag

    except Exception as e:
        logger.warning(f"Fusion weight learning failed: {e}")
        self._fusion_trained = False  # ADDED: Explicit flag set
```

---

### Phase 4: Calibration Improvements [P2 - Medium]

**Goal:** Post-hoc score calibration for better probability estimates.
**Expected Impact:** Better calibrated probabilities

#### Fix 4.1: Isotonic Regression Post-Calibration

**File:** `src/omni_mercury_engine/engine.py` (new method)

```python
def calibrate_scores(
    self,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "isotonic",
) -> None:
    """
    Calibrate fusion model scores using held-out calibration data.

    Args:
        X_cal: Calibration features
        y_cal: Calibration labels (0=normal, 1=anomaly)
        method: Calibration method ("isotonic" or "platt")
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    # Get raw scores
    raw_scores = []
    for sample in X_cal:
        result = self.detect_with_fusion(sample.reshape(1, -1))
        raw_scores.append(result['anomaly_prob'])
    raw_scores = np.array(raw_scores)

    if method == "isotonic":
        self._calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        self._calibrator.fit(raw_scores, y_cal)
    elif method == "platt":
        self._calibrator = LogisticRegression(C=1.0, solver='lbfgs')
        self._calibrator.fit(raw_scores.reshape(-1, 1), y_cal)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    self._calibration_method = method
    logger.info(f"Score calibration fitted using {method} regression")
```

---

## Validation Test Suite

### Unit Tests for Phase 1

**File:** `tests/test_signal_integrity.py`

```python
"""Unit tests for signal integrity fixes."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector


class TestContinuousScores:
    """Test that scores are continuous, not discrete."""

    @pytest.fixture
    def toy_data(self):
        """Generate toy classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            weights=[0.9, 0.1],  # 10% anomalies
            random_state=42,
        )
        return X, y

    def test_scores_are_continuous(self, toy_data):
        """Verify scores have more than 5 unique values."""
        X, y = toy_data
        detector = StatisticalAnomalyDetector()
        detector.fit(X)
        result = detector.detect(X)

        scores = result["scores"]
        unique_scores = np.unique(scores)

        # Should have many unique values, not just 5
        assert len(unique_scores) > 10, (
            f"Expected >10 unique scores for continuous output, got {len(unique_scores)}"
        )

    def test_roc_auc_improvement(self, toy_data):
        """Verify ROC-AUC improves with continuous scores."""
        X, y = toy_data
        detector = StatisticalAnomalyDetector()
        detector.fit(X)
        result = detector.detect(X)

        scores = result["scores"]
        auc = roc_auc_score(y, scores)

        # Target: >0.6 ROC-AUC on toy data
        assert auc > 0.6, f"Expected ROC-AUC >0.6, got {auc:.3f}"

    def test_adaptive_contamination(self, toy_data):
        """Verify contamination is adaptively estimated."""
        X, y = toy_data
        detector = StatisticalAnomalyDetector()  # No contamination config
        detector.fit(X)

        # Should estimate contamination, not use default 0.1
        # With 10% anomalies, estimate should be in reasonable range
        assert 0.001 <= detector.contamination <= 0.5


class TestIsolationForestScores:
    """Test IsolationForest score extraction."""

    def test_decision_function_used(self):
        """Verify decision_function (continuous) is used, not predict."""
        from sklearn.datasets import make_blobs

        X, _ = make_blobs(n_samples=50, n_features=5, centers=2, random_state=42)
        detector = StatisticalAnomalyDetector()
        detector.fit(X)
        result = detector.detect(X)

        if_scores = result.get("isolation_forest_scores")
        assert if_scores is not None

        # Scores should be continuous [0, 1]
        assert if_scores.min() >= 0
        assert if_scores.max() <= 1
        assert len(np.unique(if_scores)) > 5
```

### Integration Tests for Phase 2

**File:** `tests/test_fusion_training.py`

```python
"""Integration tests for fusion model training."""
import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from omni_mercury_engine.engine import OmniMercuryEngine


class TestFusionTraining:
    """Test fusion model training functionality."""

    @pytest.fixture
    def engine(self):
        """Create engine in fusion mode."""
        return OmniMercuryEngine(mode="fusion", device="cpu")

    @pytest.fixture
    def training_data(self):
        """Generate training data."""
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=10,
            n_classes=2,
            weights=[0.85, 0.15],
            random_state=42,
        )
        return X.astype(np.float32), y

    def test_fit_fusion_supervised(self, engine, training_data):
        """Test supervised fusion training."""
        X, y = training_data

        metrics = engine.fit_fusion(
            X, y,
            epochs=20,
            batch_size=32,
            early_stopping_patience=5,
        )

        assert engine._fusion_trained
        assert "best_loss" in metrics
        assert metrics["epochs_trained"] > 0

    def test_fit_fusion_semi_supervised(self, engine, training_data):
        """Test semi-supervised fusion training without labels."""
        X, _ = training_data

        metrics = engine.fit_fusion(
            X, y=None,  # No labels
            epochs=20,
            contamination=0.15,
        )

        assert engine._fusion_trained
        assert metrics["epochs_trained"] > 0

    def test_trained_model_improves_auc(self, engine, training_data):
        """Verify trained model improves ROC-AUC over untrained."""
        X, y = training_data

        # Split data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        # Test untrained
        scores_untrained = []
        for sample in X_test:
            result = engine.detect_with_fusion(sample.reshape(1, -1))
            scores_untrained.append(result["anomaly_prob"])
        auc_untrained = roc_auc_score(y_test, scores_untrained)

        # Train
        engine.fit_fusion(X_train, y_train, epochs=30)

        # Test trained
        scores_trained = []
        for sample in X_test:
            result = engine.detect_with_fusion(sample.reshape(1, -1))
            scores_trained.append(result["anomaly_prob"])
        auc_trained = roc_auc_score(y_test, scores_trained)

        # Trained should be significantly better
        assert auc_trained > auc_untrained + 0.05, (
            f"Expected trained AUC > untrained+0.05, "
            f"got {auc_trained:.3f} vs {auc_untrained:.3f}"
        )


class TestDynamicDimensions:
    """Test dynamic feature dimension handling."""

    def test_dimension_mismatch_handled(self):
        """Verify dimension mismatches don't crash the model."""
        from omni_mercury_engine.ml.fusion_network import OmniFusionModel

        model = OmniFusionModel()

        # Create features with mismatched dimensions
        features = {
            "statistical": torch.randn(4, 15),  # Expected 10, got 15
            "temporal": torch.randn(4, 32),     # Matches expected
        }

        # Should not raise, should use dynamic projection
        output = model(features)
        assert "anomaly_probs" in output
        assert output["anomaly_probs"].shape == (4, 1)

    def test_no_memory_leak(self):
        """Verify dynamic projections are cached, not recreated."""
        from omni_mercury_engine.ml.fusion_network import OmniFusionModel

        model = OmniFusionModel()

        features = {
            "statistical": torch.randn(4, 15),
        }

        # First forward
        model(features)
        n_params_1 = sum(p.numel() for p in model.parameters())

        # Multiple forwards with same dimensions
        for _ in range(10):
            model(features)

        n_params_2 = sum(p.numel() for p in model.parameters())

        # Should have same number of parameters (projection cached)
        assert n_params_1 == n_params_2
```

---

## GitHub Issue Templates

### Issue #1: Untrained Fusion Neural Network

```markdown
---
name: Bug Fix - Untrained Fusion Neural Network
about: OmniFusionModel initializes with random weights causing poor performance
title: "[BUG] Fix #1: OmniFusionModel random weight initialization"
labels: bug, critical, fusion-model
assignees: ''
---

## Description
The OmniFusionModel in `engine.py:529` is initialized with random weights but
`train_with_advanced_optimizers()` is never called, causing fusion to produce
meaningless scores.

## Root Cause
- `_init_fusion()` creates OmniFusionModel with random PyTorch initialization
- No automatic training mechanism exists
- Users unaware they need to call training before inference

## Impact
- ROC-AUC degradation: 0.063-0.442 vs 0.756-0.937 baseline
- Affects all fusion-mode detection
- Life-critical applications (medical anomaly detection) receive unreliable scores

## Proposed Fix
1. Add `fit_fusion()` method to OmniMercuryEngine
2. Implement semi-supervised training with pseudo-labels
3. Add `_fusion_trained` flag to track state
4. Warn users if `detect_with_fusion()` called without training

## Acceptance Criteria
- [ ] `fit_fusion()` method trains OmniFusionModel
- [ ] Semi-supervised mode works without labels
- [ ] ROC-AUC improves by >0.3 on benchmark datasets
- [ ] Unit tests pass with 5-fold CV

## Files to Modify
- `src/omni_mercury_engine/engine.py`
- `src/omni_mercury_engine/ml/fusion_network.py`
- `tests/test_fusion_training.py` (new)

## Branch
`fix/issue-1-train-omnifusion-model`
```

### Issue #3: Discrete Score Destruction

```markdown
---
name: Bug Fix - Discrete Score Destruction
about: Statistical detector outputs only 5 discrete score values
title: "[BUG] Fix #3: Preserve continuous scores in StatisticalAnomalyDetector"
labels: bug, high-priority, detectors
assignees: ''
---

## Description
`StatisticalAnomalyDetector.detect()` converts continuous scores to boolean
flags before combining, producing only 5 discrete values: {0.0, 0.3, 0.4, 0.7, 1.0}.

## Evidence
```python
# Line 96-102 in statistical.py
z_score_flags = np.any(np.abs(z_scores) > self.z_threshold, axis=1)  # BOOLEAN
combined_scores = (
    z_score_flags.astype(float) * 0.4  # 0.0 or 0.4
    + iqr_anomalies.astype(float) * 0.3  # 0.0 or 0.3
    + (if_anomalies == -1).astype(float) * 0.3  # 0.0 or 0.3
)
```

## Impact
- Loss of ranking granularity
- ROC-AUC cannot distinguish between anomaly severities
- Upstream fusion receives coarse-grained inputs

## Proposed Fix
1. Replace boolean flags with continuous intensity scores
2. Use `decision_function()` for IsolationForest (returns continuous scores)
3. Compute IQR distance scores instead of boolean threshold
4. Normalize all scores to [0, 1] before combining

## Acceptance Criteria
- [ ] `detect()` returns scores with >10 unique values
- [ ] ROC-AUC improves by >0.1 on benchmark datasets
- [ ] Backward compatible (same dict keys returned)

## Files to Modify
- `src/omni_mercury_engine/detectors/statistical.py`
- `tests/test_signal_integrity.py` (new)

## Branch
`fix/issue-3-continuous-scores`
```

---

## Risk Mitigation

### Risk 1: Training Instability

**Risk:** Semi-supervised pseudo-labels may be noisy, causing unstable training.

**Mitigation:**
- Use ensemble voting from multiple detectors for pseudo-labels
- Apply label smoothing (0.1) to reduce overconfidence
- Implement early stopping with patience=10
- Use gradient clipping (max_norm=1.0)

### Risk 2: Memory Overhead

**Risk:** Dynamic projection layers increase memory usage.

**Mitigation:**
- Cache projections in `_dynamic_projections` ModuleDict
- Limit cache size with LRU eviction if needed
- Log warnings when many dynamic projections created

### Risk 3: Backward Compatibility

**Risk:** API changes may break existing user code.

**Mitigation:**
- Maintain same `detect()` return dict structure
- Add new keys (e.g., `z_score_continuous`) without removing old ones
- Deprecate old behavior with warnings, not errors

### Risk 4: Computational Cost

**Risk:** Training adds overhead for users who just want inference.

**Mitigation:**
- Make training optional (can use untrained model with warning)
- Provide pretrained checkpoint download option
- Limit default epochs to 50 with early stopping
- GPU check with automatic CPU fallback

---

## Benchmark Validation Protocol

### 5-Fold Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def validate_fixes(X, y, n_folds=5):
    """Run 5-fold CV to validate ROC-AUC improvements."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {
        'mercury_aucs': [],
        'baseline_aucs': [],
        'mercury_pr_aucs': [],
        'baseline_pr_aucs': [],
    }

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Mercury-Agent with fixes
        engine = OmniMercuryEngine(mode="fusion")
        engine.fit_fusion(X_train, y_train, epochs=50)

        mercury_scores = []
        for sample in X_test:
            result = engine.detect_with_fusion(sample.reshape(1, -1))
            mercury_scores.append(result['anomaly_prob'])

        results['mercury_aucs'].append(roc_auc_score(y_test, mercury_scores))
        results['mercury_pr_aucs'].append(average_precision_score(y_test, mercury_scores))

        # Baseline IsolationForest
        from sklearn.ensemble import IsolationForest
        baseline = IsolationForest(random_state=42)
        baseline.fit(X_train)
        baseline_scores = -baseline.decision_function(X_test)

        results['baseline_aucs'].append(roc_auc_score(y_test, baseline_scores))
        results['baseline_pr_aucs'].append(average_precision_score(y_test, baseline_scores))

    # Statistical significance test
    from scipy.stats import ttest_rel
    t_stat, p_value = ttest_rel(results['mercury_aucs'], results['baseline_aucs'])

    return {
        'mercury_mean_auc': np.mean(results['mercury_aucs']),
        'mercury_std_auc': np.std(results['mercury_aucs']),
        'baseline_mean_auc': np.mean(results['baseline_aucs']),
        'baseline_std_auc': np.std(results['baseline_aucs']),
        'improvement': np.mean(results['mercury_aucs']) - np.mean(results['baseline_aucs']),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }
```

---

## Summary

This rectification plan addresses 7 critical issues with prioritized fixes:

| Priority | Issue | Fix | Expected Impact |
|----------|-------|-----|-----------------|
| P0 | #3 Discrete Scores | Continuous preservation | +0.4 ROC-AUC |
| P0 | #1 Untrained NN | fit_fusion() method | +0.3 ROC-AUC |
| P1 | #5 Contamination | Adaptive estimation | +0.15 ROC-AUC |
| P1 | #4 Normal-Only Fit | Semi-supervised | +0.1 ROC-AUC |
| P2 | #6 Dim Mismatch | Dynamic projections | Stability |
| P2 | #7 No Continuity | Remove clipping | Better calibration |
| P2 | #2 LR Fallback | Explicit flag | Transparency |

**Total Expected Improvement:** +0.7-0.95 ROC-AUC

**Target Achievement:**
- Minimum (≥0.80 mean ROC-AUC): Achievable with Phase 1 + Phase 2
- Target (Beat IF by 0.1): Achievable with all phases
- Stretch (<10% FPR): Achievable with calibration phase
