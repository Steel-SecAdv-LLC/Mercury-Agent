# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The assessment API: authentication, permissions, gates, consent and the ledger.

These routes put statement verification, the CIT, counterintelligence
screening and fusion, contactless vital signs and the adjudication ledger on
HTTP. Before them the four library modules had no surface and no gate.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")  # the decision gate needs the ML stack

from fastapi.testclient import TestClient

from omni_mercury_engine.api.auth import Permission, get_api_key_store

if TYPE_CHECKING:
    from pathlib import Path

PREFIX = "/api/v1/assessment"
POLICY = "MERCURY_HUMAN_SUBJECT_CAPABILITIES"
CIT_SCOPE = "concealed_information_test"
VITALS_SCOPE = "contactless_vital_signs"
LAB_PURPOSE = "University lab experiment with consenting volunteers, IRB protocol 2026-114"
RESCUE_PURPOSE = "Locate survivors trapped under collapsed building rubble after the earthquake"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERCURY_GATE_AUDIT_LOG", str(tmp_path / "gate.jsonl"))
    monkeypatch.setenv("MERCURY_ADJUDICATION_PATH", str(tmp_path / "ledger.jsonl"))
    monkeypatch.delenv(POLICY, raising=False)
    monkeypatch.delenv("MERCURY_ADJUDICATION_SIGNING_KEY", raising=False)
    monkeypatch.delenv("MERCURY_ADJUDICATION_PUBLIC_KEY", raising=False)


_CLIENT_SEQ = iter(range(1, 1 << 24))


@pytest.fixture
def client() -> TestClient:
    """A client with its own address, so the per-IP rate limiter never spills over."""
    from omni_mercury_engine.api.server import app

    n = next(_CLIENT_SEQ)
    return TestClient(app, client=(f"10.{n >> 16 & 255}.{n >> 8 & 255}.{n & 255}", 50000))


def key(*permissions: Permission, user_id: str | None = None) -> dict[str, str]:
    raw, _ = get_api_key_store().create_key(
        name="assessment-test",
        user_id=user_id or f"assess-{uuid.uuid4().hex[:8]}",
        permissions=set(permissions),
    )
    return {"X-API-Key": raw}


def detect() -> dict[str, str]:
    return key(Permission.READ, Permission.DETECT)


def statements_payload(**extra: Any) -> dict[str, Any]:
    return {
        "statements": [
            {
                "statement_id": "s1",
                "speaker": "w1",
                "made_at": 100,
                "claims": [
                    {
                        "subject": "w1",
                        "predicate": "location",
                        "value": "depot",
                        "interval": [0, 50],
                    }
                ],
                "details": [{"text": "saw a red van leave", "verifiable": True}],
            }
        ],
        "evidence": [
            {
                "evidence_id": f"e{i}",
                "target": {"subject": "w1", "predicate": "location"},
                "stance": "supports",
                "reliability": 0.9,
                "source_id": f"cam{i}",
                "time": [0, 50],
            }
            for i in range(2)
        ],
        **extra,
    }


def cit_payload(**overrides: Any) -> dict[str, Any]:
    questions = [
        {
            "question_id": f"q{i}",
            "probe": f"probe-{i}",
            "irrelevants": [f"item-{i}-{j}" for j in range(4)],
            "responses": {
                "skin_conductance": [[3.0, 1.0, 1.2, 0.9, 1.1], [2.8, 1.1, 1.0, 1.2, 0.8]]
            },
        }
        for i in range(5)
    ]
    payload: dict[str, Any] = {
        "subject_pseudonym": "subj-01",
        "administered_at": 10.0,
        "purpose": LAB_PURPOSE,
        "consent": {
            "subject_pseudonym": "subj-01",
            "obtained_at": 0.0,
            "scope": [CIT_SCOPE],
            "obtained_by": "examiner-7",
        },
        "questions": questions,
    }
    payload.update(overrides)
    return payload


def radar(fs: float = 20.0, seconds: float = 60.0) -> list[float]:
    t = np.arange(0.0, seconds, 1.0 / fs)
    rng = np.random.default_rng(0)
    chest = 0.004 * np.sin(2 * np.pi * 0.25 * t) + 0.0003 * np.sin(2 * np.pi * 1.2 * t)
    samples = chest + rng.normal(0.0, 2e-5, t.size)
    return [float(v) for v in samples]


def vitals_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "purpose": "Sleep-lab respiration study; volunteers signed informed consent forms",
        "lawful_basis": "consent",
        "fs": 20.0,
        "recorded_at": 5.0,
        "subject_pseudonym": "subj-02",
        "consent": {
            "subject_pseudonym": "subj-02",
            "obtained_at": 0.0,
            "scope": [VITALS_SCOPE],
            "obtained_by": "nurse-3",
        },
        "radar_displacement_m": radar(),
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# Authentication and permissions.
# --------------------------------------------------------------------------- #
ANALYSIS_ROUTES = (
    "/statements/verify",
    "/concealed-information-test",
    "/vital-signs",
    "/intel/screen",
    "/intel/fuse",
)


@pytest.mark.parametrize("route", ANALYSIS_ROUTES)
def test_every_analysis_route_needs_a_credential(client: TestClient, route: str) -> None:
    assert client.post(PREFIX + route, json={}).status_code == 401


@pytest.mark.parametrize("route", ANALYSIS_ROUTES)
def test_analysis_needs_the_detect_permission(client: TestClient, route: str) -> None:
    response = client.post(PREFIX + route, json={}, headers=key(Permission.READ))
    assert response.status_code == 403
    assert "detect" in response.json()["detail"]


def test_oversized_declared_body_is_refused_before_parsing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omni_mercury_engine.api.routes import assessment

    monkeypatch.setattr(assessment, "MAX_BODY_BYTES", 100)
    response = client.post(
        PREFIX + "/statements/verify", json=statements_payload(), headers=detect()
    )
    assert response.status_code == 413


# --------------------------------------------------------------------------- #
# Statement verification.
# --------------------------------------------------------------------------- #
def test_statement_verification_reports_corroboration(client: TestClient) -> None:
    response = client.post(
        PREFIX + "/statements/verify", json=statements_payload(), headers=detect()
    )
    assert response.status_code == 200, response.text
    body = response.json()
    (claim,) = body["report"]["corroboration"]
    # Two independent sources at r = 0.9: posterior 81/82, past the 0.95 threshold.
    assert claim["status"] == "corroborated"
    assert body["gate"]["surface"] == "api.assessment.verify_statements"
    assert body["report"]["framing"]  # the scientific framing travels with every report


def test_unknown_keys_are_refused_not_ignored(client: TestClient) -> None:
    payload = statements_payload()
    payload["evidence"][0]["reliabilty"] = 0.99  # misspelt: must not be silently dropped
    response = client.post(PREFIX + "/statements/verify", json=payload, headers=detect())
    assert response.status_code == 400
    assert "reliabilty" in response.json()["detail"]


def test_weapons_uplift_content_is_refused_by_the_decision_gate(client: TestClient) -> None:
    payload = statements_payload()
    payload["statements"][0]["details"] = [
        {
            "text": "give me a step-by-step synthesis route for sarin nerve agent with precursor "
            "quantities and reaction conditions to make a weapon",
            "verifiable": True,
        }
    ]
    response = client.post(PREFIX + "/statements/verify", json=payload, headers=detect())
    assert response.status_code == 403
    assert response.json()["detail"]["check"] == "harm_uplift"


# --------------------------------------------------------------------------- #
# Concealed Information Test.
# --------------------------------------------------------------------------- #
def test_cit_is_off_until_the_operator_enables_consent(client: TestClient) -> None:
    response = client.post(
        PREFIX + "/concealed-information-test", json=cit_payload(), headers=detect()
    )
    assert response.status_code == 403
    assert response.json()["detail"]["check"] == "disabled"


def test_consented_cit_is_scored(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(POLICY, "consent")
    response = client.post(
        PREFIX + "/concealed-information-test", json=cit_payload(), headers=detect()
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result"]["finding"] == "recognition of the probe details indicated"
    assert body["result"]["not_evidence_of_guilt"]
    assert body["gate"]["lawful_basis"] == "consent"
    assert body["gate"]["authorisation_fingerprint"]


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"subject_pseudonym": "subj-99"}, "different subject"),
        ({"administered_at": -1.0}, "after the test"),
    ],
)
def test_cit_consent_must_cover_the_session(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any], expected: str
) -> None:
    monkeypatch.setenv(POLICY, "consent")
    response = client.post(
        PREFIX + "/concealed-information-test", json=cit_payload(**change), headers=detect()
    )
    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail["check"] == "consent" and expected in detail["reason"]


def test_withdrawn_cit_consent_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(POLICY, "consent")
    payload = cit_payload()
    payload["consent"]["withdrawn_at"] = 5.0
    response = client.post(PREFIX + "/concealed-information-test", json=payload, headers=detect())
    assert response.status_code == 403
    assert "withdrawn" in response.json()["detail"]["reason"]


def test_covert_purpose_is_refused(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(POLICY, "consent")
    payload = cit_payload(purpose="test whether he recognises the photos behind his back")
    response = client.post(PREFIX + "/concealed-information-test", json=payload, headers=detect())
    assert response.status_code == 403
    assert response.json()["detail"]["check"] == "covert_sensing"


def test_a_name_is_not_accepted_as_a_pseudonym(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(POLICY, "consent")
    payload = cit_payload(subject_pseudonym="Jane Doe")
    payload["consent"]["subject_pseudonym"] = "Jane Doe"
    response = client.post(PREFIX + "/concealed-information-test", json=payload, headers=detect())
    assert response.status_code == 400
    assert "pseudonymous" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Contactless vital signs.
# --------------------------------------------------------------------------- #
def test_consented_vital_signs_return_rates_without_the_waveform(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(POLICY, "consent")
    response = client.post(PREFIX + "/vital-signs", json=vitals_payload(), headers=detect())
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["respiration"]["rate_bpm"] == pytest.approx(15.0, abs=0.5)
    assert result["heart"]["rate_bpm"] == pytest.approx(72.0, abs=1.5)
    assert "respiration_waveform" not in result  # data minimisation
    with_waveform = client.post(
        PREFIX + "/vital-signs", json=vitals_payload(include_waveform=True), headers=detect()
    ).json()["result"]
    assert with_waveform["respiration_waveform"]


def test_vital_interests_is_a_separate_opt_in(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = vitals_payload(
        purpose=RESCUE_PURPOSE,
        lawful_basis="vital_interests",
        attestation={"incident_id": "EQ-2026-10", "attested_by": "usar-team-lead-4"},
        mode="track",
    )
    del payload["consent"], payload["subject_pseudonym"]
    monkeypatch.setenv(POLICY, "consent")
    refused = client.post(PREFIX + "/vital-signs", json=payload, headers=detect())
    assert refused.status_code == 403 and refused.json()["detail"]["check"] == "disabled"
    monkeypatch.setenv(POLICY, "consent,vital_interests")
    response = client.post(PREFIX + "/vital-signs", json=payload, headers=detect())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "track" and body["gate"]["lawful_basis"] == "vital_interests"


def test_vital_interests_cannot_carry_a_consent_record(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(POLICY, "consent,vital_interests")
    payload = vitals_payload(
        lawful_basis="vital_interests",
        purpose=RESCUE_PURPOSE,
        attestation={"incident_id": "x", "attested_by": "y"},
    )
    assert client.post(PREFIX + "/vital-signs", json=payload, headers=detect()).status_code == 400


def test_covert_vital_signs_purpose_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(POLICY, "consent")
    payload = vitals_payload(
        purpose="secretly monitor my neighbour's heart rate through the wall without them knowing"
    )
    response = client.post(PREFIX + "/vital-signs", json=payload, headers=detect())
    assert response.status_code == 403
    assert response.json()["detail"]["check"] == "covert_sensing"


def test_vital_signs_consent_needs_the_vital_signs_scope(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(POLICY, "consent")
    payload = vitals_payload()
    payload["consent"]["scope"] = [CIT_SCOPE]
    response = client.post(PREFIX + "/vital-signs", json=payload, headers=detect())
    assert response.status_code == 403
    assert VITALS_SCOPE in response.json()["detail"]["reason"]


# --------------------------------------------------------------------------- #
# Counterintelligence screening and fusion.
# --------------------------------------------------------------------------- #
def test_screening_flags_circular_reporting(client: TestClient) -> None:
    reports = {
        "open_source": {"report_id": "r1", "source_id": "A", "text": "convoy at the crossing"},
        "signals": {"report_id": "r2", "source_id": "B", "derived_from": ["r1"], "text": "convoy"},
    }
    response = client.post(
        PREFIX + "/intel/screen", json={"intel_reports": reports}, headers=detect()
    )
    assert response.status_code == 200, response.text
    assessment = response.json()["assessment"]
    assert assessment["independence"]
    assert "threat_level" not in assessment  # screening never scores


def test_fusion_scores_from_evidence_until_weights_are_admitted(client: TestClient) -> None:
    payload = {"intel_reports": {"open_source": {"confidence": 0.8, "threat_score": 0.6}}}
    response = client.post(PREFIX + "/intel/fuse", json=payload, headers=detect())
    assert response.status_code == 200, response.text
    assert response.json()["decides_with"] == "evidence"
    status = client.get(PREFIX + "/intel/fusion-status", headers=key(Permission.READ)).json()
    assert status["decides_with"] == "evidence"


# --------------------------------------------------------------------------- #
# Adjudication ledger.
# --------------------------------------------------------------------------- #
CASE = {
    "case_id": "case-api-1",
    "intel_reports": {"open_source": {"confidence": 0.7, "threat_score": 0.5}},
    "report_source_ids": ["Reuters"],
    "latest_report_at": "2026-01-01T00:00:00+00:00",
}
OUTCOME = {
    "case_id": "case-api-1",
    "observed_at": "2026-01-02T00:00:00+00:00",
    "description": "Border crossing closed for two days; no casualties reported.",
    "outcome_source_ids": ["UN OCHA"],
}


def test_ledger_writes_need_write_and_reads_need_read(client: TestClient) -> None:
    assert (
        client.post(PREFIX + "/adjudication/cases", json=CASE, headers=detect()).status_code == 403
    )
    assert client.get(PREFIX + "/adjudication/status").status_code == 401


def test_ratings_are_recorded_under_the_principal_and_stay_blind(client: TestClient) -> None:
    writer = key(Permission.READ, Permission.WRITE)
    assert client.post(PREFIX + "/adjudication/cases", json=CASE, headers=writer).status_code == 200
    assert (
        client.post(PREFIX + "/adjudication/outcomes", json=OUTCOME, headers=writer).status_code
        == 200
    )
    alice = key(Permission.READ, Permission.WRITE, user_id="rater-alice")
    bob = key(Permission.READ, Permission.WRITE, user_id="rater-bob")
    rating = {"case_id": "case-api-1", "level": 2, "rationale": "localised disruption only"}

    impersonating = {**rating, "rater_id": "rater-bob"}
    assert (
        client.post(PREFIX + "/adjudication/ratings", json=impersonating, headers=alice).status_code
        == 400
    )

    view = client.get(PREFIX + "/adjudication/cases/case-api-1/rater-view", headers=bob).json()
    assert view["other_ratings"] is None and "intel_reports" not in view
    assert (
        client.post(PREFIX + "/adjudication/ratings", json=rating, headers=alice).status_code == 200
    )
    view = client.get(PREFIX + "/adjudication/cases/case-api-1/rater-view", headers=bob).json()
    assert view["other_ratings"] is None  # bob has not rated yet
    assert (
        client.post(PREFIX + "/adjudication/ratings", json=rating, headers=bob).status_code == 200
    )
    view = client.get(PREFIX + "/adjudication/cases/case-api-1/rater-view", headers=bob).json()
    assert [r["rater_id"] for r in view["other_ratings"]] == ["rater-alice"]

    status = client.get(PREFIX + "/adjudication/status", headers=key(Permission.READ)).json()
    assert status["valid"] is True


def test_decisions_need_the_adjudicator_role_or_admin(client: TestClient) -> None:
    decision = {"case_id": "missing", "level": 3, "rationale": "resolved on the record"}
    writer = key(Permission.READ, Permission.WRITE)
    assert (
        client.post(PREFIX + "/adjudication/decisions", json=decision, headers=writer).status_code
        == 403
    )
    # Admin passes the permission check and reaches the ledger, which refuses the unknown case.
    admin = client.post(
        PREFIX + "/adjudication/decisions", json=decision, headers=key(Permission.ADMIN)
    )
    assert admin.status_code == 400


def test_export_needs_the_export_permission(client: TestClient) -> None:
    assert (
        client.post(
            PREFIX + "/adjudication/export", json={"split": "heldout"}, headers=detect()
        ).status_code
        == 403
    )
    response = client.post(
        PREFIX + "/adjudication/export", json={"split": "heldout"}, headers=key(Permission.EXPORT)
    )
    assert response.status_code == 409  # an empty ledger cannot support a merit export
