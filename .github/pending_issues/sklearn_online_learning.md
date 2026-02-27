---
title: "Replace sklearn dependencies in ml/online_learning.py"
labels: ["tech-debt", "sklearn-removal", "non-blocking"]
assignees: []
---

## Context

PR #134 eliminated sklearn from `benchmarks/real_data_benchmarks.py` and
`benchmarks/mercury_benchmark.py` with native numpy implementations.
The `src/omni_mercury_engine/ml/online_learning.py` module still uses
sklearn and was identified as a separate work item.

## Current sklearn usage

```
Line 309: from sklearn.linear_model import SGDClassifier
Line 371: from sklearn.linear_model import PassiveAggressiveClassifier
```

## Required replacement

- `SGDClassifier` — implement native SGD with hinge/log loss, L2 penalty,
  and `partial_fit()` interface for streaming updates.
- `PassiveAggressiveClassifier` — implement the PA-I algorithm
  (Crammer et al. 2006) with configurable aggressiveness parameter C.

Both are used inside lazy imports (runtime, not top-level), so they only
affect users who explicitly invoke online learning features.

## Acceptance criteria

- [ ] Zero `from sklearn` imports in `online_learning.py`
- [ ] `tests/test_advanced_ml_capabilities.py` online learning tests pass
- [ ] No functional regression in streaming anomaly detection pipeline

## Tracking

Opened per PR #134 Phase 2 deliverable.
