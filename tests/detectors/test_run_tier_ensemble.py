# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The run_tier_ensemble one-call entrypoint for the streaming detector tier."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.detection_tier import (  # type: ignore[attr-defined,unused-ignore]
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


@pytest.mark.parametrize("contamination", [0.05, 0.1])
@pytest.mark.parametrize("subset", [None, ("spectral_residual", "spot_evt", "bocpd")])
def test_label_free_flags_honour_contamination(
    contamination: float, subset: tuple[str, ...] | None
) -> None:
    """Without labels the threshold is the (1 - contamination) quantile of the
    ensemble score. The automatic method selector mistook the averaged rank
    scores for a bimodal distribution and split them with Otsu, flagging 35-53%
    of a series with one short burst."""
    rng = np.random.default_rng(0)
    series = rng.normal(0, 1, 200)
    series[100:108] += 7.0
    result = run_tier_ensemble(series, subset=subset, contamination=contamination)
    assert result["n_flagged"] <= round(contamination * 200) + 1
    assert result["threshold"] == pytest.approx(
        float(np.quantile(result["scores"], 1 - contamination)), abs=1e-9
    )


class TestReferenceCalibration:
    """``reference=`` anchors the fit, threshold and conformal cut to known-normal data.

    Without it (and without labels) the tier calibrates on the series it
    scores, so the flag rate is set by the parameters, not by evidence; the
    result says so in ``calibration_basis``. Measured on the four recorded NAB
    files (probation as the reference, alpha 0.05) the reference-calibrated
    conformal flags mark 0.011-0.051 of the NAB-normal points.
    """

    SUBSET = ("spectral_residual", "spot_evt", "bocpd")

    def test_reference_fit_scores_only_the_input(self) -> None:
        rng = np.random.default_rng(10)
        reference = rng.normal(0, 1, 300)
        series = rng.normal(0, 1, 400)
        series[200] += 8.0
        r = run_tier_ensemble(series, reference=reference, subset=self.SUBSET, conformal_alpha=0.05)
        assert r["fit_on"] == "reference"
        assert r["n_reference"] == 300
        assert r["calibration_basis"] == "reference"
        assert r["n_points"] == 400
        assert r["conformal_flags"][200] == 1
        normal = [f for i, f in enumerate(r["conformal_flags"]) if i != 200]
        assert float(np.mean(normal)) <= 0.08

    def test_without_a_reference_the_basis_is_the_scored_series(self) -> None:
        rng = np.random.default_rng(11)
        r = run_tier_ensemble(rng.normal(0, 1, 200), subset=self.SUBSET, conformal_alpha=0.05)
        assert r["calibration_basis"] == "scored_series"
        assert "fit_on" not in r

    def test_labels_and_reference_are_mutually_exclusive(self) -> None:
        rng = np.random.default_rng(12)
        with pytest.raises(ValueError, match="mutually exclusive"):
            run_tier_ensemble(
                rng.normal(0, 1, 100),
                labels=np.zeros(100, dtype=int),
                reference=rng.normal(0, 1, 100),
                subset=self.SUBSET,
            )

    def test_a_short_reference_is_refused(self) -> None:
        rng = np.random.default_rng(13)
        with pytest.raises(ValueError, match="at least 8 points"):
            run_tier_ensemble(rng.normal(0, 1, 100), reference=np.zeros(4), subset=self.SUBSET)

    def test_rca_is_excluded_and_recorded_under_a_reference(self) -> None:
        """rca fits a node graph over its training points, so it cannot score a
        series of another length; it is dropped visibly, not silently kept or lost."""
        rng = np.random.default_rng(14)
        r = run_tier_ensemble(rng.normal(0, 1, 400), reference=rng.normal(0, 1, 300))
        assert "rca" not in r["members"]
        assert "rca" in r["excluded_members"]
        with pytest.raises(ValueError, match="rca cannot be used with a reference"):
            run_tier_ensemble(
                rng.normal(0, 1, 400),
                reference=rng.normal(0, 1, 300),
                subset=("spectral_residual", "rca"),
            )

    def test_an_equal_length_reference_keeps_the_full_tier(self) -> None:
        rng = np.random.default_rng(15)
        r = run_tier_ensemble(
            rng.normal(0, 1, 60),
            reference=rng.normal(0, 1, 60),
            subset=("spectral_residual", "rca"),
        )
        assert set(r["members"]) == {"spectral_residual", "rca"}
        assert "excluded_members" not in r
