# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The run_tier_ensemble one-call entrypoint for the streaming detector tier."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.detection_tier import (
    localize_root_cause,
    run_tier_ensemble,
)


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


def test_runner_attribution_exposes_per_detector_scores() -> None:
    rng = np.random.default_rng(3)
    series = rng.normal(0, 1, 300)
    series[150:158] += 8.0
    subset = ("spectral_residual", "spot_evt", "bocpd")
    r = run_tier_ensemble(series, subset=subset, include_attribution=True)

    assert r["detector_names"] == list(subset)
    matrix = np.asarray(r["per_detector_scores"])
    assert matrix.shape == (300, len(subset))  # n_points x n_detectors
    assert np.all((matrix >= 0.0) & (matrix <= 1.0))
    # Attribution is not returned unless requested.
    assert "per_detector_scores" not in run_tier_ensemble(series, subset=subset)


def _causal_chain_fault(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A 4-node system where a fault originates at node 0 and propagates 0->1->2.

    Node 3 is independent, so root-cause attribution must rank the causal-chain
    nodes far above node 3.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (300, 4))
    base[:, 1] += 0.8 * base[:, 0]
    base[:, 2] += 0.8 * base[:, 1]
    obs = base.copy()
    obs[-1, 0] += 8.0
    obs[-1, 1] += 6.0
    obs[-1, 2] += 4.0
    return obs, base[:-1]


def test_localize_root_cause_attributes_the_causal_chain() -> None:
    obs, train = _causal_chain_fault()
    result = localize_root_cause(obs, train=train)

    assert result["n_nodes"] == 4
    assert result["n_rows"] == obs.shape[0]
    # Ranked descending by attribution, one entry per node.
    attributions = [e["attribution"] for e in result["ranked"]]
    assert attributions == sorted(attributions, reverse=True)
    # The independent node 3 must be the least-attributed, well below the chain.
    by_node = {e["node"]: e["attribution"] for e in result["ranked"]}
    assert by_node[3] == min(by_node.values())
    assert by_node[3] < max(by_node.values()) / 3


def test_localize_root_cause_top_k_and_names() -> None:
    obs, train = _causal_chain_fault(1)
    result = localize_root_cause(
        obs, train=train, top_k=2, node_names=["pump", "valve", "tank", "aux"]
    )
    assert len(result["ranked"]) == 2
    # Names are attached and index-consistent.
    for entry in result["ranked"]:
        assert entry["name"] == ["pump", "valve", "tank", "aux"][entry["node"]]
    assert result["top_root_cause"] == result["ranked"][0]


def test_localize_root_cause_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match=r"2-D"):
        localize_root_cause(np.zeros(5))
    with pytest.raises(ValueError, match=r"node_names"):
        localize_root_cause(np.zeros((3, 4)), node_names=["a", "b"])
