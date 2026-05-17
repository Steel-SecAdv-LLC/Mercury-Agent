"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

End-to-end audit of the 21-probe Anomaly Math Arrest ensemble.

Phase 2 of the May 2026 audit cure asserts the ensemble is the
*dominant* anomaly-detection path and that no IsolationForest fallback
remains.  This file is the regression that pins both contracts:

1. ``AnomalyMathArrest`` is the live detection surface (importable,
   fits, detects, predicts) across representative domain hints —
   security-adjacent (earthquake), medical (pandemic), ocean (marine),
   and climate-equivalent (default).
2. The ensemble runs *every* probe whose ``fit_quality`` was good
   enough to register; no probe-call site silently degrades to a
   single-method fallback.
3. ``grep IsolationForest src/`` returns only documentation strings
   that explain what the ensemble replaced — not a live import or
   call site (i.e., the retirement is by deletion, not flag-gating).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from omni_mercury_engine.detectors.math_arrest.arrest import (
    PROBE_PRESETS,
    AnomalyMathArrest,
)
from omni_mercury_engine.detectors.math_arrest.base_probe import (
    BaseEquationProbe,
)

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture
def deterministic_corpus() -> tuple[np.ndarray, np.ndarray]:
    """Reproducible normal/anomalous corpus exercising the ensemble.

    The training corpus is Gaussian noise; the anomalous evaluation
    corpus is the same Gaussian baseline with three large step-shifts
    and two extreme spikes injected at known indices.  This pattern
    activates probes that key on local-to-global deviation,
    distributional shift, and high-frequency bursts — a single-probe
    fallback would not separate the two on every domain.
    """
    rng = np.random.default_rng(42)
    n = 1024
    train = rng.standard_normal(n)
    eval_clean = rng.standard_normal(n)
    eval_anomalous = eval_clean.copy()
    eval_anomalous[100:120] += 8.0
    eval_anomalous[400:420] -= 8.0
    eval_anomalous[700] = 50.0
    eval_anomalous[850] = -50.0
    return train, eval_anomalous


# ---------------------------------------------------------------------------
# Audit 1: AnomalyMathArrest is callable end-to-end across domain hints.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "domain",
    ["earthquake", "tsunami", "pandemic", "marine", "geomagnetic", "default"],
)
def test_ensemble_runs_end_to_end_per_domain(
    domain: str, deterministic_corpus: tuple[np.ndarray, np.ndarray]
) -> None:
    train, eval_anomalous = deterministic_corpus
    arrest = AnomalyMathArrest(domain=domain)
    arrest.fit(train)

    # standard_normal(int) returns ndarray; mypy's overload picks the scalar
    # branch, so cast through ``np.asarray`` to recover the array view.
    eval_clean: np.ndarray = np.asarray(
        np.random.default_rng(7).standard_normal(train.shape[0])
    )
    clean_scores = arrest.detect(eval_clean)
    anomalous_scores = arrest.detect(eval_anomalous)

    # Shape and bound contracts.
    assert clean_scores.shape == eval_clean.shape
    assert anomalous_scores.shape == eval_anomalous.shape
    assert np.all((clean_scores >= 0.0) & (clean_scores <= 1.0))
    assert np.all((anomalous_scores >= 0.0) & (anomalous_scores <= 1.0))

    # Discrimination contract: scores at the *injected anomaly indices*
    # must, on average, exceed scores at the same indices on a clean
    # eval corpus.  This is the core "ensemble is the dominant path"
    # check — if a single probe were the silent fallback the
    # discrimination collapses on at least one domain.
    anomaly_indices = list(range(100, 120)) + list(range(400, 420)) + [700, 850]
    assert anomalous_scores[anomaly_indices].mean() > clean_scores[anomaly_indices].mean(), (
        f"Domain {domain!r}: ensemble did not discriminate at injected "
        "anomaly indices — a fallback may be silently active."
    )


# ---------------------------------------------------------------------------
# Audit 2: 21 probes are wired, registered, and fit successfully.
# ---------------------------------------------------------------------------


def test_all_21_probes_registered() -> None:
    arrest = AnomalyMathArrest()
    assert len(arrest._probes) == 21
    for probe in arrest._probes:
        assert isinstance(probe, BaseEquationProbe)


def test_all_probes_participate_after_fit(
    deterministic_corpus: tuple[np.ndarray, np.ndarray],
) -> None:
    normal, _ = deterministic_corpus
    arrest = AnomalyMathArrest()
    arrest.fit(normal)

    fitted = [p for p in arrest._probes if p.is_fitted]
    # The contract is that the *vast majority* of probes register; we
    # accept that a small number may decline on a particular corpus
    # (e.g., quantum probes that need stricter conditioning) but at
    # least 18/21 must contribute or the ensemble has degraded.
    assert len(fitted) >= 18, (
        f"Only {len(fitted)}/21 probes fit on the deterministic corpus — "
        "ensemble is silently degrading toward a single-probe fallback."
    )


def test_presets_align_with_ensemble_contract() -> None:
    # ``all`` must be the full 21; ``forensic`` must include every probe
    # because forensic mode is the strictest auditing surface.
    assert len(PROBE_PRESETS["all"]) == 21
    assert len(PROBE_PRESETS["forensic"]) == 21


# ---------------------------------------------------------------------------
# Audit 3: No live IsolationForest fallback in src/.
# ---------------------------------------------------------------------------


def test_no_live_isolationforest_in_src() -> None:
    """Every ``IsolationForest`` reference in src/ must be a comment or
    docstring explaining what the ensemble replaced — never a live
    import or call.  A regression that re-imports ``IsolationForest``
    or instantiates ``IsolationForest(...)`` will fail this test.
    """
    live_pattern = re.compile(
        r"""
        (
            ^\s*from\s+sklearn[^\n]*\bIsolationForest\b   # live import
        |   ^\s*import\s+IsolationForest                  # live import
        |   IsolationForest\s*\(                          # instantiation
        )
        """,
        re.VERBOSE | re.MULTILINE,
    )

    offenders: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "IsolationForest" not in text:
            continue
        if live_pattern.search(text):
            offenders.append(str(path.relative_to(SRC_ROOT)))

    assert offenders == [], (
        "Live IsolationForest reference detected in src/ — the ensemble "
        "must be the sole anomaly path. Offending files: " + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Audit 4: predict() is binary, deterministic, and not a soft fallback.
# ---------------------------------------------------------------------------


def test_predict_is_binary_and_repeatable(
    deterministic_corpus: tuple[np.ndarray, np.ndarray],
) -> None:
    normal, anomalous = deterministic_corpus
    arrest = AnomalyMathArrest(domain="default")
    arrest.fit(normal)

    pred_a = arrest.predict(anomalous)
    pred_b = arrest.predict(anomalous)

    assert pred_a.shape == anomalous.shape
    assert set(np.unique(pred_a).tolist()).issubset({0, 1})
    np.testing.assert_array_equal(pred_a, pred_b)
