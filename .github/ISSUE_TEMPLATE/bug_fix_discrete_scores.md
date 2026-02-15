---
name: Bug Fix - Discrete Score Destruction
about: Statistical detector outputs only 5 discrete score values
title: "[BUG] Fix #3: Preserve continuous scores in MercuryAnomalyDetector"
labels: bug, high-priority, detectors
assignees: ''
---

## Description

`MercuryAnomalyDetector.detect()` converts continuous scores to boolean flags before combining, producing only 5 discrete values: {0.0, 0.3, 0.4, 0.7, 1.0}.

## Evidence

```python
# statistical.py:96-102 (BEFORE fix)
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
- sklearn's IsolationForest baseline achieves 0.82+ ROC-AUC, but discrete combination destroys the continuous score information

## Proposed Fix

1. Replace boolean flags with continuous intensity scores
2. Use `decision_function()` for IsolationForest component (returns continuous scores)
3. Compute IQR distance scores instead of boolean threshold
4. Normalize all scores to [0, 1] before combining

## Code Change

```python
# AFTER fix - continuous scores
z_score_intensity = np.max(np.abs(z_scores), axis=1) / (self.z_threshold + 1e-8)
z_score_continuous = np.clip(z_score_intensity, 0, 3.0) / 3.0  # [0, 1]

iqr_scores = self._compute_iqr_scores(data)  # New method for continuous IQR

if_raw_scores = -self.isolation_forest.decision_function(data)
if_normalized = (if_raw_scores - if_raw_scores.min()) / (if_raw_scores.max() - if_raw_scores.min())

combined_scores = z_score_continuous * 0.4 + iqr_scores * 0.3 + if_normalized * 0.3
```

## Acceptance Criteria

- [ ] `detect()` returns scores with >10 unique values
- [ ] ROC-AUC improves by >0.1 on benchmark datasets
- [ ] Backward compatible (same dict keys returned)
- [ ] Unit tests verify continuous output

## Files to Modify

- `src/omni_mercury_engine/detectors/statistical.py`
- `tests/test_signal_integrity.py` (new)

## Branch

`fix/issue-3-continuous-scores`
