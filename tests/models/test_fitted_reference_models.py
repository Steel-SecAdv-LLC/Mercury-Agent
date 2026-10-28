# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Competency pins for the fitted-reference retrofit of the exotic models.

Measured on 2026-10-27 (full-manifest evidence sweep): ``astrophysical``
scored a planted 12-sigma burst at AUROC 0.000 (inverted by construction:
its score was proximity-to-horizon = 1/norm), ``quantum`` at 0.33 (below
chance), and ``consciousness`` at 0.644 with clean/burst score means of
0.228 vs 0.224 (no separation). All three scored with fixed, unfitted
directions. The retrofit (:mod:`omni_mercury_engine.models.fitted_reference`)
keeps each model's feature identity and replaces the direction-guess with
calibrated deviation from a reference fitted on normal data.

These are competency tests, not liveness tests: each pin demands real
separation of a planted anomaly or a floor on real labeled data (ADBench
clinical fixtures, sweep protocol: fit on the full contaminated matrix).
Floors sit just under the recorded run - they assert the capability with
headroom for numeric noise, never a tuned number. The declared blind spot
(scale-invariant feature maps cannot see pure amplitude collapse) is itself
pinned, so the ontology's ``cannot_detect`` claim stays measured truth.

Recorded run (2026-10-27, sweep protocol over thyroid/wbc/cardio/
pima_real/breastw): astrophysical mean 0.805, quantum 0.825,
consciousness 0.719; planted high-energy burst AUROC 1.000 for all three.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omni_mercury_engine.models.astrophysical import AstrophysicalAnomalyModel
from omni_mercury_engine.models.consciousness import ConsciousnessPreservationModel
from omni_mercury_engine.models.fitted_reference import (
    MIN_REFERENCE_ROWS,
    FittedReferenceScorer,
)
from omni_mercury_engine.models.quantum import QuantumAnomalyModel

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "adbench"
_ADBENCH = ("thyroid", "wbc", "cardio", "pima_real", "breastw")

MODELS = {
    "astrophysical": AstrophysicalAnomalyModel,
    "quantum": QuantumAnomalyModel,
    "consciousness": ConsciousnessPreservationModel,
}

#: Real-data floors, just under the recorded run (0.805 / 0.825 / 0.719).
_ADBENCH_MEAN_FLOOR = {"astrophysical": 0.75, "quantum": 0.77, "consciousness": 0.66}


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with midranks (no sklearn dependency)."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _scores(model: object, rows: np.ndarray) -> np.ndarray:
    return np.ravel(model.predict(rows)["anomaly_scores"]).astype(float)  # type: ignore[attr-defined]


class _FeatureStub(FittedReferenceScorer):
    """Minimal host: features are the rows themselves."""

    def extract_features(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=np.float64)


class TestMixinContract:
    def test_fit_refuses_a_reference_too_small_to_mean_anything(self) -> None:
        with pytest.raises(ValueError, match=str(MIN_REFERENCE_ROWS)):
            _FeatureStub().fit(np.random.default_rng(0).normal(size=(MIN_REFERENCE_ROWS - 1, 4)))

    def test_fit_refuses_an_all_constant_reference(self) -> None:
        with pytest.raises(ValueError, match="constant"):
            _FeatureStub().fit(np.ones((32, 4)))

    def test_unfitted_scoring_is_refused_not_served(self) -> None:
        with pytest.raises(RuntimeError, match="unfitted"):
            _FeatureStub().score_features(np.zeros((2, 4)))

    def test_scores_are_bounded_and_monotone_beyond_the_reference(self) -> None:
        """A 24-sigma row must outrank a 12-sigma row - no tie plateau."""
        rng = np.random.default_rng(1)
        stub = _FeatureStub().fit(rng.normal(size=(128, 6)))
        mild = np.full((1, 6), 12.0)
        wild = np.full((1, 6), 24.0)
        out = stub.reference_scores(np.vstack([mild, wild]))
        s = out["anomaly_scores"]
        assert np.all((s >= 0.0) & (s < 1.0))
        assert s[1] > s[0], "tail must stay strictly monotone in deviation"
        assert 0.0 < out["confidence"] <= 1.0


class TestPlantedBurstCompetency:
    """The exact failure the old heuristics measured: a 12-sigma burst."""

    @pytest.mark.parametrize("name", sorted(MODELS))
    def test_high_energy_burst_is_separated(self, name: str) -> None:
        rng = np.random.default_rng(3)
        model = MODELS[name]().fit(rng.normal(size=(200, 300)))
        clean = rng.normal(size=(30, 300))
        burst = rng.normal(size=(30, 300))
        for row in burst:
            i = rng.integers(20, 280)
            row[i : i + 3] += 12.0
        y = np.r_[np.zeros(30), np.ones(30)].astype(int)
        auc = _auroc(y, np.r_[_scores(model, clean), _scores(model, burst)])
        assert auc >= 0.95, f"{name}: burst AUROC {auc:.3f} (recorded run: 1.000)"

    def test_astrophysical_catches_energy_collapse_too(self) -> None:
        """Raw-energy features make astrophysical direction-free in practice."""
        rng = np.random.default_rng(3)
        model = AstrophysicalAnomalyModel().fit(rng.normal(size=(200, 300)))
        clean = rng.normal(size=(30, 300))
        collapsed = rng.normal(size=(30, 300)) * 0.05
        y = np.r_[np.zeros(30), np.ones(30)].astype(int)
        auc = _auroc(y, np.r_[_scores(model, clean), _scores(model, collapsed)])
        assert auc >= 0.95, f"collapse AUROC {auc:.3f} (recorded run: 1.000)"

    @pytest.mark.parametrize("name", ["quantum", "consciousness"])
    def test_declared_blind_spot_scale_invariance_is_real(self, name: str) -> None:
        """The ontology's ``cannot_detect: amplitude-scale anomalies`` is measured.

        Both feature maps normalise amplitude away (quantum amplitude
        encoding; consciousness FFT amplitude shares), so a pure energy
        collapse is invisible BY REPRESENTATION, not by scoring bug. Pinning
        it keeps the declaration honest; if a future feature change makes
        collapse visible, this pin should be flipped into a competency floor.
        """
        rng = np.random.default_rng(3)
        model = MODELS[name]().fit(rng.normal(size=(200, 300)))
        clean = rng.normal(size=(30, 300))
        collapsed = rng.normal(size=(30, 300)) * 0.05
        gap = abs(float(np.mean(_scores(model, collapsed))) - float(np.mean(_scores(model, clean))))
        assert gap < 0.25, (
            f"{name} now separates amplitude collapse (gap {gap:.2f}); "
            "update the ontology's cannot_detect and replace this pin with a floor"
        )


class TestRealDataFloors:
    """Sweep protocol on committed ADBench clinical fixtures: fit on the
    full contaminated matrix, score the same rows (the hardest honest
    unsupervised condition; robust median/MAD must absorb the anomalies)."""

    @pytest.mark.parametrize("name", sorted(MODELS))
    def test_adbench_mean_auroc_floor(self, name: str) -> None:
        aucs = []
        for ds in _ADBENCH:
            d = np.load(_FIXTURES / f"{ds}.npz")
            x, y = d["X"].astype(float), d["y"].astype(int)
            if len(x) > 2000:
                idx = np.sort(np.random.default_rng(0).permutation(len(x))[:2000])
                x, y = x[idx], y[idx]
            model = MODELS[name]().fit(x)
            aucs.append(_auroc(y, _scores(model, x)))
        mean = float(np.mean(aucs))
        assert mean >= _ADBENCH_MEAN_FLOOR[name], (
            f"{name}: ADBench mean AUROC {mean:.3f} fell below the floor "
            f"{_ADBENCH_MEAN_FLOOR[name]} (recorded run: see module docstring)"
        )


class TestQuantumRegressions:
    def test_narrow_input_no_longer_crashes_the_phasor_product(self) -> None:
        """Latent width bug: inputs with <8 columns padded amplitudes but not
        states, crashing coherence; surfaced by tabular data on 2026-10-27."""
        feats = QuantumAnomalyModel().extract_features(
            np.random.default_rng(0).normal(size=(16, 6))
        )
        assert feats.shape == (16, 16)
        assert np.all(np.isfinite(feats))

    def test_noise_path_scores_the_noise_modified_features(self) -> None:
        rng = np.random.default_rng(5)
        model = QuantumAnomalyModel({"seed": 5})
        model.fit(rng.normal(size=(64, 12)))
        result = model.predict_with_noise(rng.normal(size=(6, 12)))
        assert len(np.ravel(result["anomaly_scores"])) == 6
        assert "anomaly_score_basis" in result


class TestFalseAlarmByConstruction:
    """What the scores do to held-out NORMAL data - the modality gate's
    question, answered at score level since these models emit no verdict.

    The ECDF calibration makes this a designed property: on held-out rows
    drawn from the fitted reference's distribution, scores are ~uniform, so
    the share above any threshold q is ~(1 - q). Pinned at q = 0.95 over the
    same five seeds the modality gate uses.
    """

    @pytest.mark.parametrize("name", sorted(MODELS))
    def test_heldout_normal_share_above_q95_is_about_5_percent(self, name: str) -> None:
        rates = []
        for seed in (7, 11, 23, 101, 999):
            rng = np.random.default_rng(seed)
            model = MODELS[name]().fit(rng.normal(size=(120, 8)))
            held_out = rng.normal(size=(120, 8))
            rates.append(float(np.mean(_scores(model, held_out) > 0.95)))
        rate = float(np.mean(rates))
        assert rate == pytest.approx(0.05, abs=0.04), (
            f"{name}: {rate:.4f} of held-out normal rows score above 0.95; "
            "the ECDF construction promises ~0.05 - investigate before re-pinning"
        )
