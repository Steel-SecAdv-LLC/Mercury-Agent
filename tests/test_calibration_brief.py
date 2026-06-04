"""Tests for the calibration-brief additions: BetaCalibration,
StrictIsotonicCalibration (src/omni_mercury_engine/core/calibration.py) and the
X12a eta-multiply lint (tools/lint_no_eta_score_multiply.py).

Dual-path import: prefers the normal package import (real CI, where the AMA/PQC
backend is present); falls back to loading the two modules standalone via stub
parent packages when the mandatory PQC startup gate blocks package import
(mirrors the Brief's standalone import recipe).  This keeps the test runnable in
both environments.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_calibration():
    try:
        from omni_mercury_engine.core import calibration as cal  # type: ignore
        return cal
    except Exception:
        root = REPO / "src" / "omni_mercury_engine"
        for name in ["omni_mercury_engine", "omni_mercury_engine.ml",
                     "omni_mercury_engine.core"]:
            if name not in sys.modules:
                m = types.ModuleType(name)
                m.__path__ = []  # type: ignore[attr-defined]
                sys.modules[name] = m

        def _load(modname: str, relpath: str):
            spec = importlib.util.spec_from_file_location(modname, root / relpath)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod
            spec.loader.exec_module(mod)
            return mod

        mlmod = _load("omni_mercury_engine.ml.mercury_ml", "ml/mercury_ml.py")
        sys.modules["omni_mercury_engine.ml"].mercury_ml = mlmod  # type: ignore[attr-defined]
        return _load("omni_mercury_engine.core.calibration", "core/calibration.py")


def _load_linter():
    spec = importlib.util.spec_from_file_location(
        "lint_eta", REPO / "tools" / "lint_no_eta_score_multiply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CAL = _load_calibration()


def _auroc(y, p):
    from scipy.stats import rankdata
    y = np.asarray(y)
    n1, n0 = float((y == 1).sum()), float((y == 0).sum())
    r = rankdata(p, method="average")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _synth(n=8000, seed=7):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    z = rng.normal(np.where(y == 1, 1.0, -1.0), 1.0)
    s = 1.0 / (1.0 + np.exp(-(2.2 * (2 * z) + 0.8)))
    return s, y


def test_beta_preserves_auroc_exactly():
    s, y = _synth()
    cal, te = slice(0, 2000), slice(2000, 8000)
    a_raw = _auroc(y[te], s[te])
    beta = CAL.BetaCalibration().fit(s[cal], y[cal])
    a_cal = _auroc(y[te], beta.calibrate(s[te]))
    assert abs(a_cal - a_raw) < 1e-9, f"Beta changed AUROC by {abs(a_cal - a_raw):.2e}"


def test_beta_reduces_ece():
    s, y = _synth()
    cal, te = slice(0, 2000), slice(2000, 8000)
    beta = CAL.BetaCalibration().fit(s[cal], y[cal])
    e_raw = CAL.compute_ece(y[te], s[te], n_bins=15)
    e_cal = CAL.compute_ece(y[te], beta.calibrate(s[te]), n_bins=15)
    assert e_cal < e_raw / 2, f"Beta did not halve ECE: {e_raw:.4f} -> {e_cal:.4f}"


def test_strict_isotonic_preserves_auroc():
    s, y = _synth()
    cal, te = slice(0, 2000), slice(2000, 8000)
    a_raw = _auroc(y[te], s[te])
    strict = CAL.StrictIsotonicCalibration().fit(s[cal], y[cal])
    a_strict = _auroc(y[te], strict.calibrate(s[te]))
    assert abs(a_strict - a_raw) < 1e-9, f"StrictIsotonic AUROC delta {abs(a_strict - a_raw):.2e}"


def test_beta_single_class_is_safe():
    s = np.linspace(0.1, 0.9, 100)
    y = np.zeros(100, dtype=int)
    beta = CAL.BetaCalibration().fit(s, y)
    out = beta.calibrate(s)
    assert out.shape == s.shape


def test_lint_flags_eta_score_multiply():
    lint = _load_linter()
    assert lint.lint_source("fusion_score = weighted_sum * (eta ** 1.618)\n")
    assert lint.lint_source(
        "ethical_scaling = eta ** k\nfusion_score = weighted_sum * ethical_scaling\n")
    assert not lint.lint_source("fusion_score = w_R * r + w_H * h + w_O * o\n")


def test_lint_flags_real_fusion_module():
    """The known V12d/V6 site (fusion.py) must be detected by the gate-hardening lint."""
    lint = _load_linter()
    path = REPO / "src" / "omni_mercury_engine" / "core" / "three_r" / "fusion.py"
    violations = lint.lint_file(str(path))
    assert violations, "lint failed to flag the known eta-multiply in fusion.py"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL TESTS PASS")
