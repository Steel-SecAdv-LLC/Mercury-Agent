# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Trained-gate contract for ``MultiHeadAttentionFusion``.

The torch attention path must be unreachable until genuine trained weights
are loaded (EthicalGate convention): historically the module ran its random
initialisation under ``no_grad`` and presented a fixed random projection as
learned fusion.  Untrained inference must be the deterministic phi-weighted
reference average — identical with and without torch installed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.core.global_omni_scalar_network import (
    PHI,
    MultiHeadAttentionFusion,
)


def _reference_average(states: list[np.ndarray]) -> np.ndarray:
    padded = []
    for state in states:
        p = np.zeros(37)
        p[: min(len(state), 37)] = state[:37]
        padded.append(p)
    stacked = np.stack(padded)
    phi_weights = np.tile(np.array([PHI, 1.0, 1.0 / PHI]), len(padded) // 3 + 1)[: len(padded)]
    phi_weights = phi_weights / phi_weights.sum()
    return np.average(stacked, axis=0, weights=phi_weights)


def _states() -> list[np.ndarray]:
    rng = np.random.default_rng(11)
    return [rng.normal(size=37) for _ in range(4)]


def test_untrained_fuse_is_the_phi_reference_even_with_torch() -> None:
    fusion = MultiHeadAttentionFusion(load_shipped_weights=False)
    assert fusion._trained is False
    states = _states()
    result = fusion.fuse(states)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, _reference_average(states))


def test_untrained_fuse_is_deterministic_across_instances() -> None:
    states = _states()
    a = MultiHeadAttentionFusion(load_shipped_weights=False).fuse(states)
    b = MultiHeadAttentionFusion(load_shipped_weights=False).fuse(states)
    np.testing.assert_array_equal(a, b)


def test_load_trained_weights_activates_learned_path() -> None:
    donor = MultiHeadAttentionFusion(load_shipped_weights=False)
    payload = {
        "attention": donor.attention.state_dict(),
        "projection": donor.projection.state_dict(),
        "output_projection": donor.output_projection.state_dict(),
    }
    fusion = MultiHeadAttentionFusion(load_shipped_weights=False)
    fusion.load_trained_weights(payload)
    assert fusion._trained is True
    states = _states()
    result = fusion.fuse(states)
    assert isinstance(result, np.ndarray)
    # The learned path is a different computation from the reference average.
    assert not np.allclose(result, _reference_average(states))


def test_load_trained_weights_missing_module_fails_loud() -> None:
    fusion = MultiHeadAttentionFusion(load_shipped_weights=False)
    with pytest.raises(KeyError):
        fusion.load_trained_weights({"attention": fusion.attention.state_dict()})


@pytest.mark.parametrize(
    "thresholds",
    [
        {"demote_act_below": -0.1, "demote_clear_above": 0.8},
        {"demote_act_below": 0.2, "demote_clear_above": 1.5},
        {"demote_act_below": 0.9, "demote_clear_above": 0.1},  # inverted
        {"demote_act_below": 0.5, "demote_clear_above": 0.5},  # degenerate
        {"demote_act_below": 0.2},  # missing key
        "not-a-dict",
        None,
    ],
)
def test_detection_head_with_invalid_thresholds_fails_loud(thresholds: object) -> None:
    """A consequential head must never serve with degenerate operating points.

    Out-of-range or inverted thresholds turn the disagreement overlay
    inert-or-unconditional (e.g. ``demote_clear_above=0.0`` demotes every
    grounded negative); a payload carrying a head with such thresholds is
    refused at load rather than silently serving a broken demotion rule.
    """
    from typing import Any

    from omni_mercury_engine.core.attention_fusion_stack import TrainableFusionStack

    stack = TrainableFusionStack()
    payload: dict[str, Any] = {
        "projection": stack.projection.state_dict(),
        "attention": stack.attention.state_dict(),
        "output_projection": stack.output_projection.state_dict(),
        "detection_head": stack.detection_head.state_dict(),
    }
    if thresholds is not None:
        payload["decision_thresholds"] = thresholds
    fusion = MultiHeadAttentionFusion(load_shipped_weights=False)
    with pytest.raises(RuntimeError):
        fusion.load_trained_weights(payload)
    assert fusion.detection_head is None
    assert fusion.decision_thresholds is None


def test_default_construction_serves_shipped_winner_when_present() -> None:
    """When the merit-gated checkpoint ships, default construction loads it."""
    from omni_mercury_engine.models.checkpoint_paths import load_shipped_checkpoint

    try:
        load_shipped_checkpoint("gosnn_attention_fusion")
    except FileNotFoundError:
        pytest.skip("no shipped gosnn_attention_fusion checkpoint in this build")
    fusion = MultiHeadAttentionFusion()
    assert fusion._trained is True
    result = fusion.fuse(_states())
    assert isinstance(result, np.ndarray)
    assert np.all(np.isfinite(result))


def test_eval_artifact_decision_matches_shipped_state() -> None:
    """The committed eval verdict and the shipped checkpoint must agree.

    The training program commits its verdict to
    ``artifacts/gosnn_fusion.eval.json`` whether it ships or refuses. A
    SHIPPED decision without the checkpoint (or a refusal decision alongside
    one) would mean the artifact narrates a different reality than the
    package serves -- exactly the drift this suite exists to prevent.
    """
    import json
    from pathlib import Path

    from omni_mercury_engine.models.checkpoint_paths import shipped_checkpoint_path

    artifact = Path(__file__).resolve().parents[2] / "artifacts" / "gosnn_fusion.eval.json"
    if not artifact.exists():
        pytest.skip("no committed gosnn fusion eval artifact (repo-layout test)")
    verdict = json.loads(artifact.read_text())
    decision = str(verdict["decision"])
    shipped = shipped_checkpoint_path("gosnn_attention_fusion").exists()
    assert decision.startswith("SHIPPED") == shipped, (
        f"eval artifact says {decision[:60]!r} but shipped checkpoint "
        f"exists={shipped}; the verdict and the package disagree"
    )


def test_consequential_verdict_matches_shipped_head() -> None:
    """The consequential claim and the shipped payload must agree.

    The detection-metric gate records ``consequential.shipped`` in the eval
    artifact; a detection head (plus its decision thresholds) may exist in
    the checkpoint payload if and only if that verdict says it shipped.
    Otherwise the artifact narrates an observability-only posture while the
    package serves a consequential head (or vice versa).
    """
    import json
    from pathlib import Path

    from omni_mercury_engine.models.checkpoint_paths import (
        load_shipped_checkpoint,
        shipped_checkpoint_path,
    )

    artifact = Path(__file__).resolve().parents[2] / "artifacts" / "gosnn_fusion.eval.json"
    if not artifact.exists():
        pytest.skip("no committed gosnn fusion eval artifact (repo-layout test)")
    verdict = json.loads(artifact.read_text())
    # Artifacts written before the detection-gate era carry no block; that
    # reads as "nothing consequential shipped".
    consequential = bool(verdict.get("consequential", {}).get("shipped", False))
    if not shipped_checkpoint_path("gosnn_attention_fusion").exists():
        assert not consequential, "consequential verdict without any checkpoint"
        return
    payload, _provenance = load_shipped_checkpoint("gosnn_attention_fusion")
    has_head = "detection_head" in payload
    has_thresholds = isinstance(payload.get("decision_thresholds"), dict)
    assert has_head == consequential, (
        f"eval artifact consequential.shipped={consequential} but the shipped "
        f"payload {'carries' if has_head else 'lacks'} a detection_head"
    )
    if has_head:
        assert has_thresholds, "a consequential head must ship its thresholds"


def test_gate_scores_the_head_against_the_engines_own_verdict() -> None:
    """The merit gate must baseline the head against the engine's anomaly_prob.

    A consequential disagreement head that separates anomalies *worse* than
    the engine's own calibrated verdict -- the very verdict the overlay would
    second-guess -- cannot help: when the two disagree the engine is the one
    more often right, so demoting there removes net-correct verdicts. The
    2026-07-17 measurement made this decisive (engine 0.961 vs best
    GOSNN-input head 0.904), and the gate now records the engine baseline so
    it can never ship a head that only beats the weaker phi/mean reference
    fusions. This pins that the recorded evidence carries that baseline.
    """
    import json
    from pathlib import Path

    artifact = Path(__file__).resolve().parents[2] / "artifacts" / "gosnn_fusion.eval.json"
    if not artifact.exists():
        pytest.skip("no committed gosnn fusion eval artifact (repo-layout test)")
    verdict = json.loads(artifact.read_text())
    # Degenerate-harvest refusals never reach the baseline stage; they carry
    # no baselines block and are out of scope for this contract.
    baselines = verdict.get("baselines")
    if baselines is None:
        pytest.skip("degenerate-harvest refusal artifact carries no baselines")
    assert "engine_anomaly_prob" in baselines, (
        "the merit gate must baseline the consequential head against the "
        "engine's own anomaly_prob (the verdict the overlay second-guesses); "
        f"recorded baselines were {sorted(baselines)}"
    )
    # The gate's stated constraint must name the engine baseline too, so the
    # artifact's narrated logic matches what the code enforces.
    constraint = str(verdict.get("gate", {}).get("constraint", ""))
    assert (
        "anomaly_prob" in constraint or "engine" in constraint.lower()
    ), "the gate constraint text must disclose the engine-verdict baseline"
