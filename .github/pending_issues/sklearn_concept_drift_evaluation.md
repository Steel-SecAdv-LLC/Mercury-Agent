---
title: "Replace sklearn dependencies in ml/concept_drift_evaluation.py"
labels: ["tech-debt", "sklearn-removal", "non-blocking"]
assignees: []
---

## Context

PR #134 eliminated sklearn from `benchmarks/real_data_benchmarks.py` and
`benchmarks/mercury_benchmark.py` with native numpy implementations.
The `src/omni_mercury_engine/ml/concept_drift_evaluation.py` module still
uses sklearn and was identified as a separate work item.

## Current sklearn usage

```
Line 946: from sklearn.base import clone
Line 984: from sklearn.metrics import (...)
```

## Required replacement

- `sklearn.base.clone` — implement a generic `_clone_estimator()` that
  copies constructor parameters via `get_params()` / `__init__()`.
- `sklearn.metrics` — already have native AUC/precision/recall/F1
  implementations in benchmarks that can be extracted to shared utility.

## Acceptance criteria

- [ ] Zero `from sklearn` imports in `concept_drift_evaluation.py`
- [ ] `tests/test_advanced_ml_capabilities.py` concept drift tests pass
- [ ] No functional regression in drift detection pipeline

## Tracking

Opened per PR #134 Phase 2 deliverable.
