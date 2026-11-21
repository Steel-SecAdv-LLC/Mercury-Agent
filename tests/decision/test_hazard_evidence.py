# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A decision about a hazard is read beside the record of decisions about that hazard.

"Anomaly, 90% coverage" reads the same for every hazard, and has not held the
same for every hazard. ``decision/hazard_evidence.py`` serves what was
measured on real events held out (``research/hazard_calibration``); the
decision layer adds it to a record as a caveat, and the catalog serves it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omni_mercury_engine.decision import DecisionAbstentionResponder, hazard_evidence

EVIDENCE: dict[str, Any] = json.loads(
    Path(hazard_evidence.__file__).with_name("hazard_evidence.json").read_text(encoding="utf-8")
)


def _certified(**more: Any) -> dict[str, Any]:
    return {
        "anomaly_prob": 0.97,
        "is_anomaly": True,
        "threshold_used": 0.5,
        "conformal": {"prediction_set": [1], "set_size": 1, "coverage": 0.9},
        "gosnn_metadata": {"ethical_gate_passed": True},
        **more,
    }


@pytest.fixture(autouse=True)
def _read_afresh() -> Any:
    hazard_evidence.published.cache_clear()
    yield
    hazard_evidence.published.cache_clear()


def test_a_measured_hazard_gives_its_coverage_and_abstention() -> None:
    told = EVIDENCE["hazards"]["tsunami"]

    caveat = hazard_evidence.caveat("tsunami")

    assert caveat is not None
    assert f"held the true label {told['certificate']['coverage']:.1%} of the time" in caveat
    assert f"{told['decision']['abstained']:.1%} of rows were abstained on" in caveat
    assert f"on {len(told['events'])} real events" in caveat
    assert "not this result's own certificate" in caveat
    assert EVIDENCE["measured_on"] in caveat and "research/hazard_calibration" in caveat


def test_a_hazard_that_could_not_be_measured_says_there_is_no_record() -> None:
    one_event = hazard_evidence.caveat("landslide")
    unfetched = hazard_evidence.caveat("Wildfire ")

    assert one_event is not None and "no held-out record exists for landslide" in one_event
    assert "one event" in one_event
    assert unfetched is not None and "could not be read" in unfetched
    assert hazard_evidence.for_hazard("wildfire") == {
        "hazard": "wildfire",
        "measured_on": EVIDENCE["measured_on"],
        "coverage_target": EVIDENCE["coverage_target"],
        "not_fetched": True,
    }


@pytest.mark.parametrize("name", [None, "", "security", "no-such-hazard"])
def test_of_a_name_it_knows_nothing_of_it_says_nothing(name: str | None) -> None:
    assert hazard_evidence.for_hazard(name) is None
    assert hazard_evidence.caveat(name) is None


def test_the_decision_carries_the_record_and_is_not_changed_by_it() -> None:
    responder = DecisionAbstentionResponder()

    about_a_hazard = responder.decide(_certified(), domain="tsunami")
    about_nothing_measured = responder.decide(_certified(), domain="security")

    assert any("held-out record for tsunami" in caveat for caveat in about_a_hazard.caveats)
    assert not any("held-out" in caveat for caveat in about_nothing_measured.caveats)
    for field in ("state", "disposition", "decision_label", "abstained", "decision_confidence"):
        assert getattr(about_a_hazard, field) == getattr(about_nothing_measured, field), field
    assert about_a_hazard.abstained is False and about_a_hazard.decision_label == 1


def test_evidence_that_cannot_be_read_is_no_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    for content in (None, "not json", '{"hazards": []}'):
        path = tmp_path / "evidence.json"
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(content)
        monkeypatch.setattr(hazard_evidence, "_PATH", path)
        hazard_evidence.published.cache_clear()

        assert hazard_evidence.published() == {}
        assert hazard_evidence.caveat("tsunami") is None
        record = DecisionAbstentionResponder().decide(_certified(), domain="tsunami")
        assert not any("held-out" in caveat for caveat in record.caveats)


def test_the_catalog_serves_it_whole_and_with_each_domain() -> None:
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from omni_mercury_engine.api.routes import catalog

    app = FastAPI()
    app.include_router(catalog.router)
    client = TestClient(app)

    assert client.get("/api/v1/catalog/hazard-evidence").json() == EVIDENCE
    tsunami = client.get("/api/v1/catalog/domains/tsunami").json()["held_out_evidence"]
    assert tsunami["certificate"] == EVIDENCE["hazards"]["tsunami"]["certificate"]
    assert tsunami["measured_on"] == EVIDENCE["measured_on"]
    unmeasured = [
        record
        for record in client.get("/api/v1/catalog/domains").json()["domains"]
        if record["held_out_evidence"] is None
    ]
    assert unmeasured  # most domains have no held-out record, and say so with a null
