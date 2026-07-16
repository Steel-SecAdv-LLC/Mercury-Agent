# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""GOSNN attention fusion must receive genuine per-call input variation.

Regression: detector/model anomaly scores reach the engine as ``torch.Tensor``
(``_normalize_scores`` always returns a tensor), but the GOSNN base-scalar
builder filtered on ``(np.ndarray, float, int)`` and silently dropped every one.
The fusion's per-call base member was therefore always empty and its input
constant across calls -- the degenerate harvest recorded in
``artifacts/gosnn_fusion.eval.json`` (1 unique state list across 403 calls) and
a fused output with exactly zero per-call effect. This suite pins that the
fusion now sees genuine per-call variation, that its harmonic synergy reflects
that variation (real downstream influence on the observability metadata), and
that surfacing it does NOT perturb the σ_Immutable ethical gate or the emitted
anomaly probability.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "adbench" / "pima_real.npz"


@pytest.mark.skipif(not FIXTURE.exists(), reason="pima ADBench fixture not present")
def test_fusion_input_and_synergy_vary_per_call() -> None:
    from omni_mercury_engine.core.global_omni_scalar_network import get_global_scalar_network
    from omni_mercury_engine.engine import OmniMercuryEngine

    data = np.load(FIXTURE)
    x = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.int64)
    half = len(x) // 2
    x_train, y_train, x_test = x[:half], y[:half], x[half:]

    net = get_global_scalar_network()
    fusion = net.attention_fusion
    original_fuse = fusion.fuse
    recorded: list[list[np.ndarray[Any, Any]]] = []

    def _recording_fuse(states: list[np.ndarray[Any, Any]], *args: Any, **kwargs: Any) -> Any:
        recorded.append([np.asarray(s, dtype=np.float64).copy() for s in states])
        return original_fuse(states, *args, **kwargs)

    fusion.fuse = _recording_fuse  # type: ignore[method-assign]
    try:
        engine = OmniMercuryEngine()
        rng = np.random.default_rng(0)
        fit_idx = rng.choice(len(x_train), size=min(50, len(x_train)), replace=False)
        engine.fit_fusion(x_train[fit_idx], y_train[fit_idx])

        fusion_scores: list[float | None] = []
        contributions: list[float | None] = []
        anomaly_probs: list[float | None] = []
        gates: list[bool | None] = []
        for i in rng.choice(len(x_test), size=6, replace=False):
            result = engine.detect_with_fusion(x_test[i : i + 1])
            meta = result.get("gosnn_metadata", {})
            fusion_scores.append(meta.get("enhancement_fusion_score"))
            contributions.append(meta.get("intelligence_contribution"))
            anomaly_probs.append(result.get("anomaly_prob"))
            gates.append(meta.get("ethical_gate_passed"))
    finally:
        fusion.fuse = original_fuse  # type: ignore[method-assign]

    assert recorded, "fuse() was never called on the detect path"
    # The per-call base member (states[0]) carries this sample's detector scores.
    assert all(
        len(states[0]) > 0 for states in recorded
    ), "GOSNN fusion base member is empty; per-call detector scores were dropped"
    # The harvested input must not collapse to a single constant list.
    signatures = {
        hashlib.sha256(b"".join(np.round(s, 9).tobytes() for s in states)).hexdigest()
        for states in recorded
    }
    assert len(signatures) >= 2, "GOSNN fusion input is constant across calls (degenerate harvest)"
    # Genuine downstream influence: the fused output (its mean = fusion score, and
    # the derived intelligence contribution) reflects the per-call state. These
    # vary on both the trained-attention and the phi-reference fuse paths.
    assert len({round(s, 6) for s in fusion_scores if s is not None}) >= 2
    assert len({round(c, 6) for c in contributions if c is not None}) >= 2
    # Safety: surfacing per-call variation must not break the ethical gate ...
    assert all(g is True for g in gates)
    # ... nor destabilise the emitted anomaly probability.
    assert all(a is not None and np.isfinite(a) for a in anomaly_probs)
