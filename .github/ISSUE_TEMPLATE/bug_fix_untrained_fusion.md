---
name: Bug Fix - Untrained Fusion Neural Network
about: OmniFusionModel initializes with random weights causing poor performance
title: "[BUG] Fix #1: OmniFusionModel random weight initialization"
labels: bug, critical, fusion-model
assignees: ''
---

## Description

The OmniFusionModel in `engine.py:529` is initialized with random weights but `train_with_advanced_optimizers()` is never called, causing fusion to produce meaningless scores.

## Root Cause

- `_init_fusion()` creates OmniFusionModel with random PyTorch initialization
- No automatic training mechanism exists
- Users unaware they need to call training before inference

## Impact

- ROC-AUC degradation: 0.063-0.442 vs 0.756-0.937 baseline
- Affects all fusion-mode detection
- Life-critical applications (medical anomaly detection) receive unreliable scores

## Evidence

```python
# engine.py:529
def _init_fusion(self) -> None:
    if self.mode == "fusion":
        self.fusion_model = OmniFusionModel()  # Random weights!
        # train_with_advanced_optimizers() never called
```

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
