---
title: "Replace sklearn dependencies in ml/cross_domain_transfer.py"
labels: ["tech-debt", "sklearn-removal", "non-blocking"]
assignees: []
---

## Context

PR #134 eliminated sklearn from `benchmarks/real_data_benchmarks.py` and
`benchmarks/mercury_benchmark.py` with native numpy implementations.
The `src/omni_mercury_engine/ml/cross_domain_transfer.py` module still uses
sklearn heavily and was identified as a separate work item.

## Current sklearn usage

```
Line  63: import sklearn  # noqa: F401  (availability check)
Line 412: from sklearn.ensemble import GradientBoostingClassifier
Line 418: from sklearn.linear_model import LogisticRegression
Line 547: from sklearn.ensemble import GradientBoostingClassifier
Line 554: from sklearn.linear_model import LogisticRegression
Line 676: from sklearn.ensemble import GradientBoostingClassifier
Line 682: from sklearn.linear_model import LogisticRegression
Line 1082: from sklearn.ensemble import GradientBoostingClassifier
Line 1088: from sklearn.linear_model import LogisticRegression
Line 1226: from sklearn.ensemble import GradientBoostingClassifier
Line 1232: from sklearn.linear_model import LogisticRegression
Line 1374: from sklearn.ensemble import GradientBoostingClassifier
Line 1380: from sklearn.linear_model import LogisticRegression
Line 1585: from sklearn.preprocessing import StandardScaler
Line 1598: from sklearn.preprocessing import LabelEncoder
Line 1674: from sklearn.metrics import (...)
```

## Required replacement

This is the largest sklearn consumer. Replacements needed:
- `GradientBoostingClassifier` (12 sites) — significant implementation
  effort; consider a lightweight boosting stub or optional dependency guard.
- `LogisticRegression` (12 sites) — implement via iterative reweighted
  least squares or gradient descent with L2 regularization.
- `StandardScaler` / `LabelEncoder` — already have native implementations
  in `benchmarks/real_data_benchmarks.py` and `benchmarks/mercury_benchmark.py`
  that can be extracted to a shared utility.
- `sklearn.metrics` — already have native AUC/precision/recall/F1.

## Recommended approach

1. Extract shared native implementations to `src/omni_mercury_engine/ml/_native_utils.py`
2. Replace `StandardScaler`, `LabelEncoder`, metrics first (easy wins)
3. Replace `LogisticRegression` with native implementation
4. `GradientBoostingClassifier` — either native mini-GBM or make sklearn
   an optional soft dependency with graceful ImportError handling

## Acceptance criteria

- [ ] Zero `from sklearn` imports in `cross_domain_transfer.py`
- [ ] `tests/test_advanced_ml_capabilities.py` cross-domain tests pass
- [ ] `tests/test_error_handling_coverage.py` CORAL tests pass
- [ ] No functional regression in domain adaptation pipeline

## Tracking

Opened per PR #134 Phase 2 deliverable.
