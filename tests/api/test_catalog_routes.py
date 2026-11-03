# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The catalog routes publish the engine's own declarations, whole and anonymous.

``/api/v1/catalog/domains`` and ``/api/v1/catalog/detectors`` serve the
detector manifest, the anomaly ontology, each loader's declared contract and
the knowledge profiles. These tests pin that every manifest detector and
every loader/hazard domain is served with its profile, that a record carries
the discipline, institutions, lineage and references the console and the
agent read, that two detectors are never served with the same mechanism,
that unknown names answer 404 without a stack trace, and that the routes
need no credentials and appear in the OpenAPI document.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST
from omni_mercury_engine.core.knowledge_profiles import DOMAIN_PROFILES
from omni_mercury_engine.core.loader_declarations import loader_domains
from omni_mercury_engine.detectors.hazard_diagnostics import KNOWN_HAZARDS


@pytest.fixture(scope="module")
def client() -> TestClient:
    from omni_mercury_engine.api.server import app

    return TestClient(app)


def test_every_manifest_detector_is_served_with_its_profile(client: TestClient) -> None:
    body = client.get("/api/v1/catalog/detectors").json()
    assert body["count"] == len(DETECTOR_MANIFEST)
    served = {d["name"]: d for d in body["detectors"]}
    assert set(served) == {e.name for e in DETECTOR_MANIFEST}
    for name, record in served.items():
        assert record["profile"] is not None, f"{name} served without a knowledge profile"
        assert record["ontology"] is not None, f"{name} served without its ontology row"
        for key in (
            "discipline",
            "lineage",
            "mechanism",
            "practitioners",
            "use_when",
            "not_for",
            "references",
        ):
            assert record["profile"][key], f"{name}: profile.{key} is empty"


def test_detector_records_are_distinguishable(client: TestClient) -> None:
    records = client.get("/api/v1/catalog/detectors").json()["detectors"]
    mechanisms = [r["profile"]["mechanism"] for r in records]
    assert len(set(mechanisms)) == len(mechanisms)
    lineages = [r["profile"]["lineage"] for r in records]
    assert len(set(lineages)) == len(lineages)


def test_one_detector_carries_manifest_ontology_and_profile(client: TestClient) -> None:
    record = client.get("/api/v1/catalog/detectors/bocpd").json()
    assert record["class_name"] == "BOCPDDetector"
    assert record["module"] == "omni_mercury_engine.detectors.bocpd"
    assert "change_point" in record["ontology"]["detects"]
    assert "point" in record["ontology"]["cannot_detect"]
    assert "Adams" in record["profile"]["lineage"]
    assert any("arxiv.org/abs/0710.3742" in ref["url"] for ref in record["profile"]["references"])


def test_an_adapter_record_names_what_it_wraps(client: TestClient) -> None:
    record = client.get("/api/v1/catalog/detectors/sepsis").json()
    assert record["wraps"].endswith(":SepsisDetector")
    assert "qSOFA" in record["profile"]["mechanism"]
    assert "Singer" in record["profile"]["lineage"]


def test_every_loader_and_hazard_domain_is_served_with_its_profile(client: TestClient) -> None:
    body = client.get("/api/v1/catalog/domains").json()
    served = {d["domain"]: d for d in body["domains"]}
    expected = set(loader_domains()) | set(KNOWN_HAZARDS)
    assert set(served) == expected
    assert body["count"] == len(expected)
    for domain, record in served.items():
        assert record["profile"] is not None, f"{domain} served without a knowledge profile"
        profile = record["profile"]
        assert profile["institutions"], f"{domain}: no institutions"
        assert any(i["url"].startswith("https://") for i in profile["institutions"]), domain
        assert profile["professions"] and profile["expertise"] and profile["references"], domain
        assert record["hazard_viz"] == (domain in KNOWN_HAZARDS)
        if domain in loader_domains():
            assert record["loader"] is not None, f"{domain}: loader declaration missing"
            assert record["loader"]["feature_columns"] is not None


def test_a_loader_domain_carries_its_declared_contract(client: TestClient) -> None:
    record = client.get("/api/v1/catalog/domains/sepsis").json()
    loader = record["loader"]
    assert loader["class_name"] == "SepsisLoader"
    assert loader["label_source"] == "ground_truth"
    assert "HR" in loader["feature_columns"]
    assert loader["source_url"].startswith("https://physionet.org/")
    assert record["profile"]["discipline"].startswith("Critical-care")
    assert any("PhysioNet" in i["name"] for i in record["profile"]["institutions"])
    assert "sepsis" in record["related_detectors"]


def test_a_hazard_only_domain_is_served_without_a_loader(client: TestClient) -> None:
    record = client.get("/api/v1/catalog/domains/schumann").json()
    assert record["hazard_viz"] is True
    assert record["loader"] is None
    assert record["profile"]["domain"] == "schumann"
    assert "schumann_resonance" in record["related_detectors"]


def test_a_keyed_domain_says_which_key(client: TestClient) -> None:
    record = client.get("/api/v1/catalog/domains/wildfire").json()
    assert record["loader"]["requires_api_key"] is True
    assert record["loader"]["api_key_env_var"] == "NASA_FIRMS_MAP_KEY"
    assert "FIRMS_MAP_KEY" in record["loader"]["api_key_env_fallbacks"]


def test_domain_records_are_distinguishable(client: TestClient) -> None:
    records = client.get("/api/v1/catalog/domains").json()["domains"]
    methods = [r["profile"]["method"] for r in records]
    assert len(set(methods)) == len(methods)
    disciplines = {r["profile"]["discipline"] for r in records}
    assert len(disciplines) >= len(records) - 2  # severe-storms meteorology legitimately recurs


def test_unknown_names_answer_404_cleanly(client: TestClient) -> None:
    for path in ("/api/v1/catalog/detectors/no-such", "/api/v1/catalog/domains/no-such"):
        response = client.get(path)
        assert response.status_code == 404
        assert "No such" in response.json()["detail"]


def test_catalog_routes_are_anonymous_and_documented(client: TestClient) -> None:
    openapi = client.get("/openapi.json").json()
    for path in ("/api/v1/catalog/domains", "/api/v1/catalog/detectors"):
        assert path in openapi["paths"], path
        assert not openapi["paths"][path]["get"].get(
            "security"
        ), f"{path} must not require credentials"
    assert client.get("/api/v1/catalog/detectors", headers={}).status_code == 200


def test_profiles_in_the_api_are_the_registry_profiles(client: TestClient) -> None:
    record = client.get("/api/v1/catalog/domains/earthquake").json()
    assert record["profile"]["title"] == DOMAIN_PROFILES["earthquake"].title
    assert [i["name"] for i in record["profile"]["institutions"]] == [
        i.name for i in DOMAIN_PROFILES["earthquake"].institutions
    ]
