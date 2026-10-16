# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Counterintelligence screening inside IntelligenceFusionEngine's evidence path.

Pins two things: on today's engine-format reports (no source, provenance,
time or location fields) the screen changes nothing; and when reports do
carry those fields, derivative copies and one source reporting under several
disciplines stop counting as independent corroboration.

Also pins the evidence contract (scores and confidences are probabilities;
only scored reports count), the trained path (screened, finite inputs only),
and the merit gate (deduplicated cases, recorded and alpha-spent retries,
no unmeasured temporal branch).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.exceptions import InputModalityError
from omni_mercury_engine.security.intelligence_fusion import (
    MERIT_ALPHA,
    MIN_HELDOUT_CASES,
    AllSourceFusionNetwork,
    IntelligenceDiscipline,
    IntelligenceFusionEngine,
    merit_alpha_for_attempt,
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
    # S1 is one origin: human (prior 0.85) represents it over open_source
    # (0.75), so two independent origins remain -> corroboration 2/3.
    assert screened.confidence == pytest.approx(0.6 * 2 / 3)
    assert screened.risk_score == pytest.approx(0.5)
    assert screened.primary_intel_sources == ["human", "signals"]
    ci = screened.counterintelligence
    assert ci is not None
    assert ci["fusion_weight_multipliers"] == {"human": 1.0, "open_source": 0.0, "signals": 1.0}


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


# --------------------------------------------------------------------------- #
# Evidence contract: scores and confidences are probabilities.
# --------------------------------------------------------------------------- #
BASE = {d: {"confidence": 0.9, "threat_score": 0.9} for d in ("signals", "imagery", "human")}


@pytest.mark.parametrize("bad", [-1e6, -0.01, 1.01, 50.0])
def test_one_out_of_range_score_cannot_decide_the_fused_level(bad: float) -> None:
    # Regression: only the weighted mean was clipped, so one report at -1e6
    # forced LOW and one at 50 forced CRITICAL whatever its weight.
    engine = _engine()
    reports = {**BASE, "open_source": {"confidence": 0.1, "threat_score": bad}}
    result, base = engine.fuse_intelligence(reports), engine.fuse_intelligence(BASE)
    assert (result.threat_level, result.risk_score, result.confidence) == (
        base.threat_level,
        base.risk_score,
        base.confidence,
    )
    assert "open_source" not in result.primary_intel_sources + result.corroborating_sources
    assert any(
        "open_source: threat_score" in n and "outside [0, 1]" in n for n in result.evidence_notes
    )


def test_out_of_range_score_alone_is_not_evidence() -> None:
    result = _engine().fuse_intelligence({"open_source": {"confidence": 0.9, "threat_score": 50}})
    assert (result.threat_level, result.risk_score, result.confidence) == ("LOW", 0.0, 0.0)


def test_out_of_range_confidence_is_dropped_not_clipped() -> None:
    reports = {
        "signals": {"confidence": 0.3, "threat_score": 0.9},
        "imagery": {"confidence": 1e9, "threat_score": 0.9},
    }
    result = _engine().fuse_intelligence(reports)
    # The score counts (corroboration 2/3); only the valid confidence is averaged.
    assert result.confidence == pytest.approx(0.3 * 2 / 3)
    assert any("imagery: confidence" in n for n in result.evidence_notes)


def test_empty_reports_are_not_corroboration_or_sources() -> None:
    # Regression: report.get("confidence", 0.5) let {} count as a corroborating
    # discipline with a fabricated confidence, and list it as a primary source.
    one = {"signals": {"confidence": 0.9, "threat_score": 0.9}}
    padded = {**one, "open_source": {}, "cyber": {}, "human": {"confidence": 1.0}}
    a, b = _engine().fuse_intelligence(one), _engine().fuse_intelligence(padded)
    assert (b.threat_level, b.risk_score, b.confidence) == (
        a.threat_level,
        a.risk_score,
        a.confidence,
    )
    assert b.confidence == pytest.approx(0.9 / 3)
    assert b.primary_intel_sources == ["signals"] and b.corroborating_sources == []
    assert any("cyber: no threat_score" in n for n in b.evidence_notes)


def test_scored_report_without_confidence_gets_no_assumed_confidence() -> None:
    reports = {
        "signals": {"confidence": 0.6, "threat_score": 0.9},
        "imagery": {"threat_score": 0.9},
    }
    result = _engine().fuse_intelligence(reports)
    assert result.confidence == pytest.approx(0.6 * 2 / 3)  # not mean(0.6, 0.5) * 2/3


@pytest.mark.parametrize("counterintelligence", [True, False])
@pytest.mark.parametrize("value", [[{"confidence": 0.9, "threat_score": 0.9}], ({},), "text", 3])
def test_non_mapping_discipline_values_are_refused_before_screening(
    counterintelligence: bool, value: object
) -> None:
    # Regression: a list was accepted by the screen, skipped by evidence
    # scoring and crashed source ranking with AttributeError.
    engine = _engine(counterintelligence=counterintelligence)
    with pytest.raises(InputModalityError, match="one report mapping per discipline"):
        engine.fuse_intelligence({"signals": value, "imagery": {"threat_score": 0.5}})


def test_engine_level_hijack_and_spoof_keep_the_genuine_report() -> None:
    genuine = {"confidence": 0.95, "threat_score": 0.95, "reported_at": 5.0}
    imagery = {"confidence": 0.95, "threat_score": 0.95}
    hijack = {
        "communications": {
            "report_id": "signals",
            "confidence": 0.1,
            "threat_score": 0.0,
            "reported_at": 1.0,
        },
        "signals": genuine,
        "imagery": imagery,
    }
    screened = _engine().fuse_intelligence(hijack)
    plain = _engine(counterintelligence=False).fuse_intelligence(hijack)
    assert (screened.threat_level, screened.risk_score) == (plain.threat_level, plain.risk_score)
    assert "signals" in screened.primary_intel_sources
    spoof = {
        "open_source": {
            "source_id": "S1",
            "reported_at": 1.0,
            "confidence": 0.1,
            "threat_score": 0.0,
        },
        "signals": {"source_id": "S1", **genuine},
        "imagery": imagery,
    }
    result = _engine().fuse_intelligence(spoof)
    ci = result.counterintelligence
    assert ci is not None
    assert ci["fusion_weight_multipliers"]["signals"] == 1.0
    assert ci["fusion_weight_multipliers"]["open_source"] == 0.0
    assert result.risk_score == pytest.approx(0.95)
    assert [f["kind"] for f in ci["flags"]] == ["conflicting_claims"]


# --------------------------------------------------------------------------- #
# Trained path: screened, finite inputs only.
# --------------------------------------------------------------------------- #
def _trained_engine(seed: int = 0) -> IntelligenceFusionEngine:
    import torch

    torch.manual_seed(seed)
    engine = _engine()
    engine.fusion_network = AllSourceFusionNetwork(input_dim=128)
    engine._trained = True  # white-box: the decision path, not the merit gate
    return engine


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1e6, 50.0, "high", None])
def test_trained_path_never_decides_from_unusable_features(bad: object) -> None:
    # Regression: NaN reached the network, every logit was NaN, argmax read
    # LOW, and confidence/risk were NaN with threat_detected False.
    engine = _trained_engine()
    reports = {**BASE, "cyber": {"confidence": 0.9, "threat_score": bad}}
    result = engine.fuse_intelligence(reports)
    assert math.isfinite(result.confidence) and math.isfinite(result.risk_score)
    expected = engine._evidence_assessment(reports, engine._screen(reports))
    assert (result.threat_level, result.confidence, result.risk_score) == (
        ["LOW", "MODERATE", "SUBSTANTIAL", "SEVERE", "CRITICAL"][expected[0] - 1],
        expected[1],
        expected[2],
    )
    assert any("fusion network did not score" in n for n in result.evidence_notes)
    with pytest.raises(ValueError, match=r"cyber\.threat_score"):
        engine.extract_features(reports)


def test_non_finite_network_output_fails_closed() -> None:
    import torch

    engine = _trained_engine()
    final = engine.fusion_network.threat_classifier[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.bias.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        engine.fuse_intelligence(BASE)


RELAY = {
    "open_source": {
        "report_id": "o",
        "source_id": "A",
        "reported_at": 1.0,
        "confidence": 0.9,
        "threat_score": 0.1,
    },
    "signals": {
        "report_id": "s",
        "source_id": "B",
        "reported_at": 2.0,
        "derived_from": ["o"],
        "confidence": 0.9,
        "threat_score": 0.99,
    },
}


def test_trained_path_does_not_show_zero_weight_copies_to_the_network() -> None:
    # Regression: the network got the raw reports, so a relayed copy the
    # screen weighted 0 still drove the trained decision.
    engine = _trained_engine()
    result = engine.fuse_intelligence(RELAY)
    alone = engine._network_assessment(engine.fusion_network, {"open_source": RELAY["open_source"]})
    assert (result.threat_level, result.confidence, result.risk_score) == (
        ["LOW", "MODERATE", "SUBSTANTIAL", "SEVERE", "CRITICAL"][alone[0] - 1],
        alone[1],
        alone[2],
    )
    assert any(n.startswith("signals: weight 0") for n in result.evidence_notes)


def test_trained_path_defers_fractional_weights_to_evidence_scoring() -> None:
    # One entity 5,000+ km apart within a minute: both reports get x0.5.
    reports = {
        "open_source": {
            "source_id": "S1",
            "entity_id": "E",
            "reported_at": 0.0,
            "location": {"lat": 0.0, "lon": 0.0},
            "confidence": 0.9,
            "threat_score": 0.9,
        },
        "signals": {
            "source_id": "S2",
            "entity_id": "E",
            "reported_at": 60.0,
            "location": {"lat": 50.0, "lon": 50.0},
            "confidence": 0.9,
            "threat_score": 0.2,
        },
    }
    engine = _trained_engine()
    ci = engine._screen(reports)
    assert ci is not None and ci.fusion_weight_multipliers == {"open_source": 0.5, "signals": 0.5}
    result = engine.fuse_intelligence(reports)
    assert (result.confidence, result.risk_score) == engine._evidence_assessment(reports, ci)[1:]
    assert any("fractional screening weights" in n for n in result.evidence_notes)


def test_trained_path_does_not_use_the_unmeasured_temporal_branch() -> None:
    engine = _trained_engine()
    timeline = [{"threat_level": 5, "confidence": 0.9, "num_sources": 4}] * 6
    with_context = engine.fuse_intelligence(BASE, temporal_context=timeline)
    without = engine.fuse_intelligence(BASE)
    assert (with_context.threat_level, with_context.confidence, with_context.risk_score) == (
        without.threat_level,
        without.confidence,
        without.risk_score,
    )
    assert with_context.temporal_patterns  # still summarised for the analyst
    assert any("temporal_context not fed" in n for n in with_context.evidence_notes)


# --------------------------------------------------------------------------- #
# Merit gate: distinct cases, recorded and alpha-spent retries.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def gate_pool(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A seeded candidate, cases where it agrees with evidence, and discordant ones."""
    import torch

    torch.manual_seed(0)
    network = AllSourceFusionNetwork(input_dim=128)
    folder = tmp_path_factory.mktemp("gate")
    checkpoint = folder / "candidate.pt"
    torch.save(network.state_dict(), checkpoint)
    zero = AllSourceFusionNetwork(input_dim=128)
    with torch.no_grad():
        for parameter in zero.parameters():
            parameter.zero_()
    zero_checkpoint = folder / "zero.pt"  # always LOW: never better than evidence
    torch.save(zero.state_dict(), zero_checkpoint)

    engine = _engine()
    rng = np.random.default_rng(0)
    agree: list[tuple[dict[str, Any], int]] = []
    wins: list[tuple[dict[str, Any], int]] = []
    for _ in range(300):
        reports = {
            str(name): {
                "confidence": float(rng.uniform(0.2, 1.0)),
                "relevance": float(rng.uniform(0.0, 1.0)),
                "indicators": ["x"] * int(rng.integers(0, 12)),
                "threat_score": float(rng.uniform(0.02, 0.98)),
            }
            for name in rng.choice(["open_source", "cyber", "financial", "signals"], 2, False)
        }
        ci = engine._screen(reports)
        by_network = engine._trained_assessment(network, reports, ci)[0]
        if by_network == engine._evidence_assessment(reports, ci)[0]:
            agree.append((reports, by_network))
        else:
            wins.append((reports, by_network))  # labelled for the network
    assert len(agree) >= MIN_HELDOUT_CASES and len(wins) >= 6
    return {"checkpoint": checkpoint, "zero": zero_checkpoint, "agree": agree, "wins": wins}


def test_duplicated_cases_cannot_reach_the_case_minimum(gate_pool: dict[str, Any]) -> None:
    # Regression: 25 copies of one agreeing case plus 5 copies of one
    # discordant win passed as n=30 with p=1/32.
    heldout = [gate_pool["agree"][0]] * 25 + [gate_pool["wins"][0]] * 5
    engine = _engine()
    with pytest.raises(ValueError, match="2 distinct held-out cases"):
        engine.load_trained_weights(gate_pool["checkpoint"], heldout)
    assert not engine._trained and engine.merit_attempts == ()


def test_duplicated_wins_count_once_in_the_mcnemar_test(gate_pool: dict[str, Any]) -> None:
    agree = gate_pool["agree"][:MIN_HELDOUT_CASES]
    duplicated = agree + [gate_pool["wins"][0]] * 5
    engine = _engine()
    with pytest.raises(RuntimeError, match="refused"):
        engine.load_trained_weights(gate_pool["checkpoint"], duplicated)
    (attempt,) = engine.merit_attempts
    assert (attempt.merit.n_cases, attempt.merit.n_supplied) == (MIN_HELDOUT_CASES + 1, 35)
    assert (attempt.merit.network_only, attempt.merit.p_value) == (1, 0.5)
    # Five distinct wins on the same agreeing cases do pass.
    merit = _engine().load_trained_weights(gate_pool["checkpoint"], agree + gate_pool["wins"][:5])
    assert merit.admitted and merit.p_value == pytest.approx(1 / 32)


def test_retries_on_one_heldout_set_are_recorded_and_alpha_spent(
    gate_pool: dict[str, Any],
) -> None:
    heldout = gate_pool["agree"][:MIN_HELDOUT_CASES] + gate_pool["wins"][:5]  # p = 1/32
    assert merit_alpha_for_attempt(1) == MERIT_ALPHA
    assert merit_alpha_for_attempt(2) == MERIT_ALPHA / 2 < 1 / 32 < MERIT_ALPHA
    engine = _engine()
    with pytest.raises(RuntimeError, match="attempt 1"):
        engine.load_trained_weights(gate_pool["zero"], heldout)
    # Shuffled order is the same held-out set: this is attempt 2, at alpha/2.
    with pytest.raises(RuntimeError, match="attempt 2"):
        engine.load_trained_weights(gate_pool["checkpoint"], heldout[::-1])
    assert not engine._trained
    first, second = engine.merit_attempts
    assert first.fingerprint == second.fingerprint
    assert [a.merit.attempt for a in engine.merit_attempts] == [1, 2]
    assert second.merit.alpha == MERIT_ALPHA / 2 and not second.merit.admitted
    # A different held-out set starts its own count.
    other = gate_pool["agree"][1 : MIN_HELDOUT_CASES + 1] + gate_pool["wins"][:5]
    assert engine.load_trained_weights(gate_pool["checkpoint"], other).attempt == 1
    assert engine._trained


def test_trained_path_defers_when_source_information_moves_reliability() -> None:
    # The network's features use the fixed discipline table; a graded source
    # moves the adjusted reliability, which only the evidence path can apply.
    reports = {
        **BASE,
        "signals": {"source_id": "S1", "source_grade": "E", "confidence": 0.9, "threat_score": 0.9},
    }
    engine = _trained_engine()
    ci = engine._screen(reports)
    assert ci is not None
    adjusted = ci.adjusted_reliability["signals"]
    assert adjusted is not None and adjusted < engine.int_reliability_scores["signals"]
    result = engine.fuse_intelligence(reports)
    assert (result.confidence, result.risk_score) == engine._evidence_assessment(reports, ci)[1:]
    assert any("source information adjusted them" in n for n in result.evidence_notes)
    # Engine-format reports leave reliability untouched: the network decides.
    plain = engine.fuse_intelligence(BASE)
    assert not any("did not score" in n for n in plain.evidence_notes)
