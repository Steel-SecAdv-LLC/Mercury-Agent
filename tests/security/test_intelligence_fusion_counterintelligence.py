# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Counterintelligence screening inside IntelligenceFusionEngine's evidence path.

Pins two things: on today's engine-format reports (no source, provenance,
time or location fields) the screen changes nothing; and when reports do
carry those fields, derivative copies and one source reporting under several
disciplines stop counting as independent corroboration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.exceptions import InputModalityError
from omni_mercury_engine.security.intelligence_fusion import (
    IntelligenceDiscipline,
    IntelligenceFusionEngine,
)

if TYPE_CHECKING:
    from pathlib import Path

DISCIPLINES = [d.value for d in IntelligenceDiscipline]


def _engine(**kwargs: Any) -> IntelligenceFusionEngine:
    return IntelligenceFusionEngine(
        enable_neurosymbolic=False, enable_cryptanalysis=False, **kwargs
    )


def _random_reports(rng: np.random.Generator) -> dict[str, Any]:
    chosen = rng.choice(DISCIPLINES, size=int(rng.integers(1, 7)), replace=False)
    reports: dict[str, Any] = {}
    for discipline in chosen:
        report: dict[str, Any] = {"confidence": float(rng.uniform(0, 1))}
        if rng.random() < 0.8:
            report["threat_score"] = float(rng.uniform(0, 1))
        reports[str(discipline)] = report
    return reports


def test_engine_format_reports_are_scored_exactly_as_without_the_screen() -> None:
    screened, plain = _engine(), _engine(counterintelligence=False)
    rng = np.random.default_rng(0)
    for _ in range(200):
        reports = _random_reports(rng)
        a, b = screened.fuse_intelligence(reports), plain.fuse_intelligence(reports)
        assert (a.threat_level, a.confidence, a.risk_score) == (
            b.threat_level,
            b.confidence,
            b.risk_score,
        )
        assert a.primary_intel_sources == b.primary_intel_sources
        assert a.corroborating_sources == b.corroborating_sources
        assert a.counterintelligence is not None and b.counterintelligence is None


def test_a_relayed_copy_is_not_independent_corroboration() -> None:
    # "signals" relays the open-source report: one origin, not two.
    reports = {
        "open_source": {
            "report_id": "osint-1",
            "source_id": "outlet-A",
            "reported_at": 1_000.0,
            "confidence": 0.9,
            "threat_score": 0.8,
        },
        "signals": {
            "report_id": "sig-1",
            "source_id": "station-B",
            "reported_at": 2_000.0,
            "derived_from": ["osint-1"],
            "confidence": 0.9,
            "threat_score": 0.8,
        },
    }
    screened = _engine().fuse_intelligence(reports)
    plain = _engine(counterintelligence=False).fuse_intelligence(reports)
    ci = screened.counterintelligence
    assert ci is not None
    assert ci["fusion_weight_multipliers"]["signals"] == 0.0
    # Corroboration 1/3 instead of 2/3: the confidence halves.
    assert screened.confidence == pytest.approx(plain.confidence / 2)
    assert "signals" not in screened.primary_intel_sources + screened.corroborating_sources


def test_one_source_under_two_disciplines_counts_once() -> None:
    reports = {
        "open_source": {"source_id": "S1", "confidence": 0.6, "threat_score": 0.5},
        "human": {"source_id": "S1", "confidence": 0.6, "threat_score": 0.5},
        "signals": {"source_id": "S2", "confidence": 0.6, "threat_score": 0.5},
    }
    screened = _engine().fuse_intelligence(reports)
    plain = _engine(counterintelligence=False).fuse_intelligence(reports)
    assert plain.confidence == pytest.approx(0.6)  # three disciplines: full corroboration
    assert screened.confidence < plain.confidence


def test_off_modality_input_still_raises_input_modality_error() -> None:
    with pytest.raises(InputModalityError):
        _engine().fuse_intelligence(np.zeros((3, 4)))  # type: ignore[arg-type]


def test_merit_gate_measures_the_screened_evidence_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    engine = _engine()
    rng = np.random.default_rng(1)
    cases = [_random_reports(rng) for _ in range(30)]
    # Labels equal to the evidence path's own classes: the network cannot win,
    # so the call must refuse whatever the random weights do.
    heldout = [(r, engine._evidence_assessment(r, engine._screen(r))[0]) for r in cases]
    seen: list[Any] = []
    original = engine._evidence_assessment

    def spy(reports: dict[str, Any], ci: Any = None) -> tuple[int, float, float]:
        seen.append(ci)
        return original(reports, ci)

    monkeypatch.setattr(engine, "_evidence_assessment", spy)
    path = tmp_path / "weights.pt"
    torch.save(engine.fusion_network.state_dict(), path)
    with pytest.raises(RuntimeError):
        engine.load_trained_weights(path, heldout)
    assert len(seen) == 30 and all(ci is not None for ci in seen)
