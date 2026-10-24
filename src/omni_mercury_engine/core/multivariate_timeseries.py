# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multivariate time-series anomaly detection — statistical LTG-shaped baseline. (moved; shim).

The implementation now lives in :mod:`omni_mercury_engine.detectors.multivariate_timeseries`; this shim re-exports it
so released ``omni_mercury_engine.core.multivariate_timeseries`` imports keep
working unchanged. New code should import from the new location.

Motivated by the LTG method (Springer, 2025:
https://link.springer.com/article/10.1007/s44443-025-00024-3), which combines Long
short-term memory, Temporal convolution and Graph convolution.

**This module does not implement that architecture.** It implements a
deterministic statistical baseline with the same three-branch *shape*: the
"LSTM" branch is a per-window mean, the "temporal convolution" branch is a
per-window standard deviation, and the "graph" branch is a feature-correlation
summary. There are no learned parameters, no recurrent state and no convolution
kernels anywhere in this file. The names are kept because they map onto the
paper's branches and the reconstruction-error score they feed; the docstrings
say plainly what each one computes.

The previous docstring described this as an LTG implementation and the detector
returned a ``roc_auc_estimate`` computed as ``0.5 + 0.4 * tanh(separation)`` from
its *own* scores and its *own* thresholded predictions -- no labels were involved
at any point, so the number could not be an AUC of anything and rose whenever the
detector was merely self-consistent. It has been removed rather than renamed: a
fabricated metric is worse than no metric, because it invites comparison with
real ones. Callers that want a real AUC must supply ground-truth labels and use
``sklearn.metrics.roc_auc_score`` (or Mercury's evaluation harness) over the
returned ``anomaly_scores``.
"""

from __future__ import annotations

from omni_mercury_engine.detectors.multivariate_timeseries import (
    ChaosMultivariateFusion,
    MultivariateTSDetector,
)

__all__ = [
    "ChaosMultivariateFusion",
    "MultivariateTSDetector",
]
