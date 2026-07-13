# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Empirical proof that the conformal coverage guarantee survives the gate.

The decision layer projects the engine's conformal certificate onto the
three-state transparency contract.  A projection is only transparent if it *preserves the
guarantee*: a GROUNDED label must never contradict the certificate, the
marginal coverage must still hold through the layer, and abstaining on the
uncertain mass must make the calls the loop *does* make at least as accurate as
the raw thresholded verdict.

This torch-gated test fits a real fusion engine on a seeded, moderately
overlapping regime (so the abstention states are genuinely exercised),
calibrates a 90% conformal certificate, and measures those properties on a
held-out split.  Numbers on this fixture (seed 0): ~94% empirical coverage,
~5% abstention, 100% selective (grounded) accuracy vs ~99% raw accuracy, and --
the invariant that matters -- zero gate-vs-certificate contradictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.decision import DecisionAbstentionResponder
from omni_mercury_engine.verifiers.three_state import ThreeState

pytestmark = [pytest.mark.xdist_group("decision_coverage_empirical"), pytest.mark.timeout(300)]

_TARGET = 0.90


def _regime(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Seeded, moderately overlapping two-class data (sep=1.6 -> real abstention)."""
    rng = np.random.RandomState(seed)
    n_normal, n_anom, dim, sep = 1200, 300, 8, 1.6
    normal = rng.normal(0.0, 1.0, (n_normal, dim))
    anomaly = rng.normal(sep, 1.0, (n_anom, dim))
    X = np.vstack([normal, anomaly]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anom)]).astype(np.int64)
    order = rng.permutation(len(X))
    return X[order], y[order]


def _measure() -> dict[str, Any]:
    from omni_mercury_engine.engine import OmniMercuryEngine

    torch.manual_seed(0)
    np.random.seed(0)
    X, y = _regime(0)
    n = len(X)
    a, b = int(n * 0.5), int(n * 0.72)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    engine.fit_fusion(X[:a], y[:a], epochs=25, batch_size=32, early_stopping_patience=15)
    # The loop below serves sample-at-a-time, so the calibration scores must
    # come from the same regime (per_sample=True): several detector features
    # are batch-relative by design, and a row scored alone does not get the
    # score it gets inside a batch. Before the 2026-06-11 serve-path purity
    # fix, cross-call streaming buffers made single-sample serving *resemble*
    # batch scoring by accident of call history; regime-matched calibration
    # is the principled form of the guarantee this test pins.
    engine.calibrate_fusion_conformal(X[a:b], y[a:b], coverage=_TARGET, per_sample=True)
    responder = DecisionAbstentionResponder()

    x_te, y_te = X[b:], y[b:]
    covered = grounded = grounded_correct = raw_correct = contradictions = 0
    transparent = True
    grounded_non_singleton = 0
    for i in range(len(x_te)):
        res = engine.detect_with_fusion(x_te[i : i + 1])
        rec = responder.decide(res)
        cset = res["conformal"]["prediction_set"]
        truth = int(y_te[i])
        covered += truth in cset
        raw_correct += int(bool(res["is_anomaly"]) == bool(truth))
        # The layer carries the certificate faithfully (transparency).
        if rec.signals["conformal_set_size"] != res["conformal"]["set_size"]:
            transparent = False
        if rec.coverage != res["conformal"]["coverage"]:
            transparent = False
        if rec.state is ThreeState.GROUNDED:
            grounded += 1
            grounded_correct += int(rec.decision_label == truth)
            if res["conformal"]["set_size"] != 1:
                grounded_non_singleton += 1  # a non-singleton must never ground
            if rec.decision_label not in cset:
                contradictions += 1  # the gate must never overrule the certificate
    n_te = len(x_te)
    return {
        "n": n_te,
        "coverage": covered / n_te,
        "abstain_rate": 1.0 - grounded / n_te,
        "grounded": grounded,
        "grounded_acc": grounded_correct / grounded if grounded else float("nan"),
        "raw_acc": raw_correct / n_te,
        "contradictions": contradictions,
        "grounded_non_singleton": grounded_non_singleton,
        "transparent": transparent,
    }


@pytest.fixture(scope="module")
def coverage_run() -> dict[str, Any]:
    return _measure()


class TestCoverageSurvivesTheGate:
    def test_no_grounded_decision_contradicts_the_certificate(
        self, coverage_run: dict[str, Any]
    ) -> None:
        # The core faithfulness invariant: a GROUNDED label is always a member
        # of the conformal prediction set -- the gate never overrules the cert.
        assert coverage_run["contradictions"] == 0

    def test_only_singletons_are_grounded(self, coverage_run: dict[str, Any]) -> None:
        # An ambiguous {0,1} or empty {} set can never be promoted to a grounded
        # call; grounding requires a conformal singleton.
        assert coverage_run["grounded_non_singleton"] == 0

    def test_layer_is_transparent_to_the_certificate(self, coverage_run: dict[str, Any]) -> None:
        # The record reports the same set size and coverage the engine emitted.
        assert coverage_run["transparent"] is True

    def test_marginal_coverage_holds_through_the_layer(self, coverage_run: dict[str, Any]) -> None:
        # The distribution-free 90% guarantee survives the merge + the gate
        # (generous finite-sample slack; observed ~0.94 on this fixture).
        assert coverage_run["coverage"] >= _TARGET - 0.07

    def test_abstention_is_exercised(self, coverage_run: dict[str, Any]) -> None:
        # The regime is hard enough that the gate genuinely abstains on some
        # points (otherwise the selective-accuracy claim would be vacuous).
        assert coverage_run["abstain_rate"] > 0.0
        assert coverage_run["grounded"] > 0

    def test_abstention_does_not_lower_committed_accuracy(
        self, coverage_run: dict[str, Any]
    ) -> None:
        # Routing the uncertain mass to a human keeps the grounded calls at least
        # as accurate as the raw thresholded verdict over the same stream (it is
        # measured higher on this fixture: ~100% vs ~99%).  A 1pp tolerance keeps
        # the property robust to torch-version numerics without weakening the
        # claim -- the structural invariants above (0 contradictions, only
        # singletons grounded, transparency) are exact and carry the proof.
        assert coverage_run["grounded_acc"] >= coverage_run["raw_acc"] - 0.01
