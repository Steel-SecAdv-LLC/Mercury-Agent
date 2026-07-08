# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The run_tier_ensemble one-call entrypoint for the streaming detector tier."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.detection_tier import run_tier_ensemble


def test_runner_scores_and_flags_a_burst() -> None:
    rng = np.random.default_rng(0)
    series = rng.normal(0, 1, 400)
    series[200:210] += 8.0
    r = run_tier_ensemble(series, subset=("spectral_residual", "spot_evt", "bocpd"))

    assert r["n_points"] == 400
    scores = np.asarray(r["scores"])
    assert scores.shape == (400,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert len(r["uncertainty"]) == 400
    assert r["method"] == "average"  # default without labels


def test_runner_conformal_fp_control_concentrates_on_burst() -> None:
    rng = np.random.default_rng(1)
    series = rng.normal(0, 1, 400)
    series[200:210] += 8.0
    r = run_tier_ensemble(
        series, subset=("spectral_residual", "spot_evt", "bocpd"), conformal_alpha=0.05
    )
    flagged = {i for i, f in enumerate(r["conformal_flags"]) if f}
    assert flagged & set(range(195, 215))
    assert "conformal_threshold" in r


def test_runner_stacking_requires_labels() -> None:
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        run_tier_ensemble(rng.normal(0, 1, 100), method="stacking", subset=("spectral_residual",))
