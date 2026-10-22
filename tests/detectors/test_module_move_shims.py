# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Released import paths of moved detector modules keep working, identically.

Seven detector modules moved into ``detectors/`` (from ``core/``, ``ml/`` and
``models/``); each old module is a shim re-exporting the new location.  These
tests pin the promise: every public name the released module served is still
importable from the old path and is the *same object* as the new location's.
"""

from __future__ import annotations

import importlib
import types

import pytest

#: old subpath -> (new detectors module, released public names)
MOVES: dict[str, tuple[str, tuple[str, ...]]] = {
    "core.adaptive_detector": (
        "adaptive_detector",
        (
            "AdaptiveAnomalyDetector",
            "CovarianceAwareDetector",
            "TemporalPatternDetector",
        ),
    ),
    "core.info_geometry": ("info_geometry", ("InformationGeometryDetector",)),
    "core.multivariate_timeseries": ("multivariate_timeseries", ("MultivariateTSDetector",)),
    "core.topological_analysis": ("topological_analysis", ("TopologicalAnomalyDetector",)),
    "core.conformal_martingale": ("conformal_martingale", ("ConformalChangeDetector",)),
    "ml.drift": (
        "drift",
        (
            "ChiSquaredDriftDetector",
            "CovarianceDriftDetector",
            "PrincipalAxisDriftDetector",
            "OnlineDriftDetector",
            "EnsembleDriftDetector",
            "create_drift_detector",
        ),
    ),
    "models.lstm_ae": ("lstm_ae", ("LSTMAutoencoderDetector",)),
}


@pytest.mark.parametrize("old_subpath", sorted(MOVES))
def test_old_path_serves_the_new_module_objects(old_subpath: str) -> None:
    new_name, names = MOVES[old_subpath]
    old = importlib.import_module(f"omni_mercury_engine.{old_subpath}")
    new = importlib.import_module(f"omni_mercury_engine.detectors.{new_name}")
    for name in names:
        assert getattr(old, name) is getattr(
            new, name
        ), f"{old_subpath}.{name} is not the object served by detectors.{new_name}"


@pytest.mark.parametrize("old_subpath", sorted(MOVES))
def test_shim_reexports_every_public_name_of_the_new_module(old_subpath: str) -> None:
    """The shim may never silently drop a name the new module exports."""
    new_name, _ = MOVES[old_subpath]
    old = importlib.import_module(f"omni_mercury_engine.{old_subpath}")
    new = importlib.import_module(f"omni_mercury_engine.detectors.{new_name}")
    new_public = set(getattr(new, "__all__", ())) or {
        n
        for n in dir(new)
        if not n.startswith("_")
        and not isinstance(getattr(new, n), types.ModuleType)
        and getattr(getattr(new, n), "__module__", new.__name__) == new.__name__
    }
    missing = {n for n in new_public if not hasattr(old, n)}
    assert not missing, f"shim {old_subpath} is missing re-exports: {sorted(missing)}"
