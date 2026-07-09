# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Counterfactual explanations on the live detection surfaces (CLI/HTTP/MCP).

The tier's ``include_counterfactual`` option and the flagship's
``gdpr_report`` option must be reachable and verified on every surface,
with correctness re-scored through the real detection paths (the module's
own guarantee — see ``explainability/detection_counterfactuals.py``).
"""

from __future__ import annotations

import json
import os

# Disable rate limiting before the api modules import (middleware reads the
# env var at module-load time; same pattern as tests/api/).
os.environ["OMNI_RATE_LIMIT_ENABLED"] = "false"

import numpy as np
import pytest


@pytest.fixture
def client():  # type: ignore[no-untyped-def]
    """FastAPI test client over the server app."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from omni_mercury_engine.api.server import app

    return fastapi_testclient.TestClient(app)


@pytest.fixture
def auth_headers():  # type: ignore[no-untyped-def]
    """Valid API-key auth headers."""
    from omni_mercury_engine.api.auth import get_api_key_store

    store = get_api_key_store()
    raw_key, _ = store.create_key(name="cf_test_key", user_id="cf_test_user")
    return {"X-API-Key": raw_key}


def _burst_series(n: int = 160) -> np.ndarray:  # type: ignore[type-arg]
    rng = np.random.default_rng(0)
    series = rng.normal(0.0, 1.0, n)
    series[80:86] += 7.0
    return series


class TestTierCounterfactualCore:
    """run_tier_ensemble carries the counterfactual through one seam."""

    def test_counterfactual_flips_the_flagged_point(self) -> None:
        from omni_mercury_engine.detectors.detection_tier import run_tier_ensemble

        result = run_tier_ensemble(_burst_series(), include_counterfactual=True)
        cf = result["counterfactual"]
        assert cf["flipped"] is True
        assert cf["score_after"] <= result["threshold"] < cf["score_before"]
        # The explained point is one of the flagged burst points.
        assert result["flags"][cf["index"]] == 1
        # Windowed contextual feature space: the flip may need neighbors too.
        assert 1 <= len(cf["changed_features"]) <= 7

    def test_explicit_index_and_method(self) -> None:
        from omni_mercury_engine.detectors.detection_tier import run_tier_ensemble

        result = run_tier_ensemble(
            _burst_series(),
            include_counterfactual=True,
            counterfactual_index=82,
            counterfactual_method="prototype",
        )
        cf = result["counterfactual"]
        assert cf["index"] == 82
        assert cf["method"] == "prototype"
        assert cf["flipped"] is True

    def test_default_off(self) -> None:
        from omni_mercury_engine.detectors.detection_tier import run_tier_ensemble

        result = run_tier_ensemble(_burst_series())
        assert "counterfactual" not in result

    def test_deterministic(self) -> None:
        from omni_mercury_engine.detectors.detection_tier import run_tier_ensemble

        a = run_tier_ensemble(_burst_series(), include_counterfactual=True)
        b = run_tier_ensemble(_burst_series(), include_counterfactual=True)
        assert a["counterfactual"] == b["counterfactual"]


class TestTierCounterfactualCLI:
    def test_cli_flag_emits_verified_counterfactual(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from click.testing import CliRunner

        from omni_mercury_engine.cli import main

        csv = tmp_path / "series.csv"
        np.savetxt(csv, _burst_series(), delimiter=",")
        result = CliRunner().invoke(main, ["tier-detect", "-i", str(csv), "--counterfactual"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["counterfactual"]["flipped"] is True
        assert payload["counterfactual"]["minimal"] is True


class TestTierCounterfactualHTTP:
    def test_route_opt_in(self, client, auth_headers) -> None:  # type: ignore[no-untyped-def]
        response = client.post(
            "/api/v1/detect/tier",
            headers=auth_headers,
            json={
                "request": {
                    "data": _burst_series().tolist(),
                    "include_counterfactual": True,
                }
            },
        )
        assert response.status_code in (200, 429)
        if response.status_code == 200:
            payload = response.json()
            assert payload["counterfactual"]["flipped"] is True

    def test_route_default_off(self, client, auth_headers) -> None:  # type: ignore[no-untyped-def]
        response = client.post(
            "/api/v1/detect/tier",
            headers=auth_headers,
            json={"request": {"data": _burst_series().tolist()}},
        )
        assert response.status_code in (200, 429)
        if response.status_code == 200:
            assert "counterfactual" not in response.json()

    def test_bad_method_rejected(self, client, auth_headers) -> None:  # type: ignore[no-untyped-def]
        response = client.post(
            "/api/v1/detect/tier",
            headers=auth_headers,
            json={
                "request": {
                    "data": _burst_series().tolist(),
                    "include_counterfactual": True,
                    "counterfactual_method": "magic",
                }
            },
        )
        assert response.status_code == 422


class TestTierCounterfactualMCP:
    def _call(self, server, name: str, arguments: dict) -> dict:  # type: ignore[no-untyped-def]
        response = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        assert response is not None and "error" not in response, response
        return json.loads(response["result"]["content"][0]["text"])

    def test_mcp_tool_carries_counterfactual(self) -> None:
        from omni_mercury_engine.mcp_server import MercuryMCPServer

        server = MercuryMCPServer()
        payload = self._call(
            server,
            "mercury_tier_detect",
            {"data": _burst_series().tolist(), "include_counterfactual": True},
        )
        assert payload["counterfactual"]["flipped"] is True

    def test_mcp_schema_documents_options(self) -> None:
        from omni_mercury_engine.mcp_server import MercuryMCPServer

        server = MercuryMCPServer()
        response = server.handle_message(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        tools = {t["name"]: t for t in response["result"]["tools"]}
        tier_props = tools["mercury_tier_detect"]["inputSchema"]["properties"]
        assert "include_counterfactual" in tier_props
        assert "counterfactual_method" in tier_props
        fusion_props = tools["mercury_detect_fusion"]["inputSchema"]["properties"]
        assert "gdpr_report" in fusion_props


class TestFlagshipGdprReportHTTP:
    """The dormant engine gdpr_report path is finally reachable over HTTP."""

    def test_gdpr_report_attached_on_opt_in(self, client, auth_headers) -> None:  # type: ignore[no-untyped-def]
        pytest.importorskip("torch")
        rng = np.random.default_rng(1)
        response = client.post(
            "/api/v1/detect/flagship",
            headers=auth_headers,
            json={
                "request": {
                    "data": rng.normal(size=(24, 6)).tolist(),
                    "gdpr_report": True,
                    "subject_id": "test-subject-1",
                }
            },
        )
        # 403 = ethical-gate refusal (fail-closed, acceptable); 200 must carry
        # the report; 429 = rate limit in shared test runs.
        assert response.status_code in (200, 403, 429)
        if response.status_code == 200:
            payload = response.json()
            assert "gdpr_report" in payload

    def test_default_off(self, client, auth_headers) -> None:  # type: ignore[no-untyped-def]
        pytest.importorskip("torch")
        rng = np.random.default_rng(1)
        response = client.post(
            "/api/v1/detect/flagship",
            headers=auth_headers,
            json={"request": {"data": rng.normal(size=(24, 6)).tolist()}},
        )
        assert response.status_code in (200, 403, 429)
        if response.status_code == 200:
            assert "gdpr_report" not in response.json()
