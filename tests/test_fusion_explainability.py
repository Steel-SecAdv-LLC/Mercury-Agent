"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Serve-path explainability wiring tests (WS5).

``OmniMercuryEngine.detect_with_fusion(explain=True)`` must attach a real
Integrated-Gradients attribution of the *same* calibrated fusion probability the
result reports (via ``score_fusion``), plus its faithfulness scores. These tests
pin: (a) the explanation is absent by default (cost-gated), (b) it is present and
well-formed when requested, explaining the served decision.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import omni_mercury_engine.engine as engine_mod
from omni_mercury_engine.engine import OmniMercuryEngine, default_fusion_checkpoint_path


@pytest.fixture(scope="module")
def loaded_engine() -> OmniMercuryEngine:
    """A fusion engine with the shipped default checkpoint loaded."""
    eng = OmniMercuryEngine(mode="fusion", device="cpu")
    eng.load_model(default_fusion_checkpoint_path())
    return eng


def _benign_batch() -> np.ndarray:
    """A small benign batch (>=2 rows so base detectors can fit on the batch)."""
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 1.0, size=(8, 16)).astype(np.float32)


def test_explanation_absent_by_default(loaded_engine: OmniMercuryEngine) -> None:
    """Without ``explain=True`` the result carries no explanation (cost-gated)."""
    result = loaded_engine.detect_with_fusion(_benign_batch(), domain="general")
    assert "explanation" not in result


def test_explain_attaches_wellformed_attribution(
    loaded_engine: OmniMercuryEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``explain=True`` attaches an IG attribution + faithfulness for the sample."""
    # Turn IG steps down so the test is fast; the wiring is identical.
    monkeypatch.setattr(engine_mod, "_EXPLAIN_IG_STEPS", 6)
    result = loaded_engine.detect_with_fusion(_benign_batch(), domain="general", explain=True)

    assert "explanation" in result
    expl = result["explanation"]
    # One importance per input feature, all finite.
    importances = expl["feature_importances"]
    assert len(importances) == 16
    assert all(np.isfinite(fi["importance"]) for fi in importances)
    # Faithfulness was evaluated against the same predict path.
    assert "comprehensiveness" in expl["faithfulness_scores"]
    assert np.isfinite(expl["faithfulness_scores"]["comprehensiveness"])
    # The explanation explains the *served* probability, not a proxy.
    assert abs(expl["prediction"] - result["anomaly_prob"]) < 0.05


def test_explanation_is_json_serialisable(
    loaded_engine: OmniMercuryEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attached explanation round-trips through JSON (it ships in results)."""
    import json

    monkeypatch.setattr(engine_mod, "_EXPLAIN_IG_STEPS", 6)
    result = loaded_engine.detect_with_fusion(_benign_batch(), domain="general", explain=True)
    # Should not raise.
    json.dumps(result["explanation"])
