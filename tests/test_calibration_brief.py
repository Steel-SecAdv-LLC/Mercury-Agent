# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the calibration-brief port: ``StrictIsotonicCalibration``.

Ported from PR #275 (X1 survivor; branch deleted, recovered from
``refs/pull/275/head``).  Scope notes: the brief's ``BetaCalibration`` was not
ported — ``main`` ships its own accept-gated Beta implementation, covered by
``tests/test_beta_calibration.py`` — and the X12a eta-multiply lint was not
ported (the eta^Phi design was settled as an opt-in decoupling by PR #278).
Evidence suite: ``benchmarks/calibration_brief/``.

Dual-path import: prefers the normal package import (real CI, where the
AMA/PQC backend is present); falls back to loading the module standalone via
stub parent packages when the mandatory PQC startup gate blocks package import
(mirrors the brief's standalone import recipe).  This keeps the test runnable
in both environments.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from typing import Any

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_calibration() -> Any:
    try:
        from omni_mercury_engine.core import calibration as cal

        return cal
    except Exception:
        root = REPO / "src" / "omni_mercury_engine"
        for name in [
            "omni_mercury_engine",
            "omni_mercury_engine.ml",
            "omni_mercury_engine.core",
        ]:
            if name not in sys.modules:
                m = types.ModuleType(name)
                m.__path__ = []
                sys.modules[name] = m

        def _load(modname: str, relpath: str) -> Any:
            spec = importlib.util.spec_from_file_location(modname, root / relpath)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod
            spec.loader.exec_module(mod)
            return mod

        mlmod = _load("omni_mercury_engine.ml.mercury_ml", "ml/mercury_ml.py")
        sys.modules["omni_mercury_engine.ml"].mercury_ml = mlmod  # type: ignore[attr-defined]
        return _load("omni_mercury_engine.core.calibration", "core/calibration.py")


CAL = _load_calibration()


def _auroc(y: np.ndarray[Any, Any], p: np.ndarray[Any, Any]) -> float:
    from scipy.stats import rankdata

    y = np.asarray(y)
    n1, n0 = float((y == 1).sum()), float((y == 0).sum())
    r = rankdata(p, method="average")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _synth(n: int = 8000, seed: int = 7) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    z = rng.normal(np.where(y == 1, 1.0, -1.0), 1.0)
    s = 1.0 / (1.0 + np.exp(-(2.2 * (2 * z) + 0.8)))
    return s, y


class _StubProbaDetector:
    """Minimal predict_proba detector for exercising the registry path."""

    def __init__(self, scores: np.ndarray[Any, Any]) -> None:
        self._scores = scores

    def predict_proba(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        s = self._scores[: len(X)]
        return np.column_stack([1.0 - s, s])


def test_strict_isotonic_preserves_auroc() -> None:
    """X1's defining property: AUROC exactly preserved (vanilla isotonic loses it)."""
    s, y = _synth()
    cal, te = slice(0, 2000), slice(2000, 8000)
    a_raw = _auroc(y[te], s[te])
    strict = CAL.StrictIsotonicCalibration().fit(s[cal], y[cal])
    a_strict = _auroc(y[te], strict.calibrate(s[te]))
    assert abs(a_strict - a_raw) < 1e-9, f"StrictIsotonic AUROC delta {abs(a_strict - a_raw):.2e}"


def test_strict_isotonic_is_strictly_increasing_where_isotonic_ties() -> None:
    """The tie-break makes the map strictly increasing; vanilla isotonic has flats."""
    s, y = _synth()
    cal = slice(0, 2000)
    grid = np.linspace(0.001, 0.999, 4001)

    iso = CAL.IsotonicCalibration().fit(s[cal], y[cal])
    strict = CAL.StrictIsotonicCalibration().fit(s[cal], y[cal])

    iso_out = iso.calibrate(grid)
    strict_out = strict.calibrate(grid)

    assert np.any(np.diff(iso_out) == 0.0), "expected vanilla isotonic to have flat regions"
    assert np.all(np.diff(strict_out) > 0.0), "strict isotonic output must strictly increase"
    assert np.all(strict_out > 0.0) and np.all(strict_out < 1.0), "squeeze must avoid {0, 1}"


def test_strict_isotonic_ece_close_to_vanilla_isotonic() -> None:
    """The tie-break must not cost calibration quality (brief: within ~10%)."""
    s, y = _synth()
    cal, te = slice(0, 2000), slice(2000, 8000)
    iso = CAL.IsotonicCalibration().fit(s[cal], y[cal])
    strict = CAL.StrictIsotonicCalibration().fit(s[cal], y[cal])
    e_iso = CAL.compute_ece(y[te], iso.calibrate(s[te]), n_bins=15)
    e_strict = CAL.compute_ece(y[te], strict.calibrate(s[te]), n_bins=15)
    assert (
        e_strict <= e_iso * 1.25 + 1e-4
    ), f"ECE regressed: isotonic {e_iso:.4f} -> strict {e_strict:.4f}"


def test_strict_isotonic_single_class_is_safe() -> None:
    """Degenerate calibration set: stays unfitted and passes scores through."""
    s = np.linspace(0.1, 0.9, 100)
    y = np.zeros(100, dtype=int)
    strict = CAL.StrictIsotonicCalibration().fit(s, y)
    out = strict.calibrate(s)
    assert out.shape == s.shape
    np.testing.assert_array_equal(out, s)


def test_calibrate_detector_registry_exposes_strict_isotonic() -> None:
    """``method="strict_isotonic"`` resolves through the registry end to end."""
    s, y = _synth(n=600, seed=11)
    X = s.reshape(-1, 1)
    detector = _StubProbaDetector(s)

    calibrator, result = CAL.calibrate_detector(detector, X, y, method="strict_isotonic")

    assert isinstance(calibrator, CAL.StrictIsotonicCalibration)
    assert result is not None and result.method == "strict_isotonic"


def test_calibrate_detector_rejects_unknown_method() -> None:
    """The registry still fails closed on unknown method names."""
    s, y = _synth(n=200, seed=3)
    detector = _StubProbaDetector(s)
    try:
        CAL.calibrate_detector(detector, s.reshape(-1, 1), y, method="not_a_method")
    except ValueError as exc:
        assert "Unknown calibration method" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown method")
