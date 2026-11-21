# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The OAE / 3R sensitivity surface, pinned.

``research/omni_equation/harness_oae_sensitivity.py`` varied the golden-ratio
weights (each axis scaled 0.8–1.2, renormalised), the served 3R recursion
depth (3–7 around the shipped 5) and the Lyapunov λ (0.15–0.35 around the
shipped 0.25) on the four recorded NAB files. Measured surface:

* weights: mean AUROC spans 0.7864–0.8044 over the whole ±20% band, golden
  (0.7962) within 0.001 of uniform — the equation does not depend brittlely
  on the golden split, and no perturbation clears noise;
* depth: 0.7962–0.7992 across 3–7 — flat;
* λ: the fusion score is *exactly* flat; the parameter sets only the decay
  envelope ``ε·e^(−λt)`` the stability monitor reads.

The shipped constants therefore stand, now on a measured surface rather than
convention. These tests hold the artifact's shape and re-verify the λ
flatness live.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "oae_sensitivity.json"

#: Stability ceilings over the measured spreads (0.018 weights, 0.003 depth),
#: with room for numerical drift only. A spread above these means a dial has
#: become decision-relevant and the surface must be re-published.
MAX_WEIGHT_SPREAD = 0.05
MAX_DEPTH_SPREAD = 0.02


def _artifact() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(ARTIFACT.read_text())
    return result


def test_the_surface_is_complete() -> None:
    artifact = _artifact()
    assert artifact["schema"] == "mercury.research.oae_sensitivity/v1"
    weights = artifact["weights"]
    assert {"golden", "uniform"} <= set(weights)
    # 3 axes x 4 non-unit scales + golden + uniform.
    assert len(weights) == 14
    assert len(artifact["recursion_depth"]) == 5
    assert sum(1 for k in artifact["lambda"] if k.startswith("lambda_")) == 5
    files = set(artifact["protocol"]["files"])
    for cell in weights.values():
        assert set(cell["per_file_auc"]) == files


def test_the_golden_weights_are_not_a_brittle_dependence() -> None:
    weights = _artifact()["weights"]
    values = [cell["mean_auc"] for cell in weights.values()]
    spread = max(values) - min(values)
    assert spread <= MAX_WEIGHT_SPREAD, (
        f"the ±20% weight band now spans {spread:.4f} of mean AUROC; a dial "
        "has become decision-relevant — re-run the sensitivity harness"
    )
    assert abs(weights["golden"]["mean_auc"] - weights["uniform"]["mean_auc"]) <= 0.02


def test_the_recursion_depth_is_flat_around_the_shipped_value() -> None:
    depth = _artifact()["recursion_depth"]
    values = [cell["mean_auc"] for cell in depth.values()]
    assert max(values) - min(values) <= MAX_DEPTH_SPREAD
    assert "depth_5" in depth


def test_lambda_never_moves_the_fusion_score() -> None:
    """Live: identical (R, H, O) triples through compute() at two λ values."""
    from omni_mercury_engine.core.three_r.fusion import OmniAvaEquation

    rng = np.random.default_rng(0)
    triples = rng.random((32, 3))
    low = OmniAvaEquation(convergence_rate=0.15)
    high = OmniAvaEquation(convergence_rate=0.35)
    for r, h, o in triples:
        a = low.compute(float(r), float(h), float(o))
        b = high.compute(float(r), float(h), float(o))
        assert a.fusion_score == b.fusion_score
        assert a.lyapunov_bound != b.lyapunov_bound  # the envelope does move

    recorded = _artifact()["lambda"]
    for key, cell in recorded.items():
        if key.startswith("lambda_"):
            assert cell["max_abs_score_delta_vs_0.15"] == 0.0


def test_no_dial_change_was_adopted() -> None:
    """The shipped constants stand: golden weights, depth 5, λ 0.25."""
    from omni_mercury_engine.core.centralized_constants import LYAPUNOV
    from omni_mercury_engine.core.three_r.fusion import OmniAvaEquation

    assert LYAPUNOV.LAMBDA_CONVERGENCE == 0.25  # exactly representable
    equation = OmniAvaEquation()
    assert equation.weights["w_R"] == pytest.approx(0.4472, abs=0.001)
    from research.omni_equation.harness_oae_weights import RECURSION_DEPTH

    assert RECURSION_DEPTH == 5
