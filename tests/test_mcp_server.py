# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the Mercury MCP server -- the universal interconnect.

Drives the server with crafted JSON-RPC 2.0 messages (the same wire an MCP client
speaks) and asserts the protocol handshake, tool discovery, tool execution, and
fail-closed/honest behaviour, all offline."""

from __future__ import annotations

import io
import json

import numpy as np

from omni_mercury_engine.agentic.capabilities.web_research import SearchResult, WebResearcher
from omni_mercury_engine.mcp_server import PROTOCOL_VERSION, MercuryMCPServer


def _offline_assistant() -> object:
    """A GeneralAssistant whose researcher can never reach the network."""
    from omni_mercury_engine.agentic.capabilities import GeneralAssistant

    def _dead(url: str, timeout: float) -> tuple[int, str, str]:
        raise OSError("network down")

    return GeneralAssistant(researcher=WebResearcher(transport=_dead, enable_ddg_fallback=True))


def _online_assistant() -> object:
    """A GeneralAssistant with a stubbed search provider + page transport."""
    from omni_mercury_engine.agentic.capabilities import GeneralAssistant

    page = "Conformal prediction gives finite-sample coverage guarantees. " * 5

    def _transport(url: str, timeout: float) -> tuple[int, str, str]:
        return 200, f"<html><body><p>{page}</p></body></html>", url

    def _provider(query: str, max_results: int) -> list[SearchResult]:
        return [SearchResult(title="Coverage", url="https://ex.test/c", snippet="cov")]

    researcher = WebResearcher(transport=_transport, search_provider=_provider)
    return GeneralAssistant(researcher=researcher)


class TestProtocol:
    def test_initialize_handshake(self) -> None:
        server = MercuryMCPServer()
        resp = server.handle_message(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        assert resp is not None
        assert resp["jsonrpc"] == "2.0" and resp["id"] == 1
        result = resp["result"]
        assert result["protocolVersion"] == PROTOCOL_VERSION
        assert result["serverInfo"]["name"] == "mercury-agent"
        assert "tools" in result["capabilities"]

    def test_initialized_notification_has_no_response(self) -> None:
        server = MercuryMCPServer()
        resp = server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert resp is None
        assert server._initialized is True

    def test_ping(self) -> None:
        server = MercuryMCPServer()
        resp = server.handle_message({"jsonrpc": "2.0", "id": 9, "method": "ping"})
        assert resp is not None and resp["result"] == {}

    def test_unknown_method_is_jsonrpc_error(self) -> None:
        server = MercuryMCPServer()
        resp = server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "does/not/exist"})
        assert resp is not None and resp["error"]["code"] == -32601

    def test_invalid_request_without_method(self) -> None:
        server = MercuryMCPServer()
        resp = server.handle_message({"jsonrpc": "2.0", "id": 3})
        assert resp is not None and resp["error"]["code"] == -32600


class TestToolDiscovery:
    def test_tools_list_is_self_describing_manifest(self) -> None:
        server = MercuryMCPServer()
        resp = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert resp is not None
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert {
            "mercury_detect_anomaly",
            "mercury_score_ethics",
            "mercury_research",
            "mercury_answer",
            "mercury_write_document",
            "mercury_calibrate_confidence",
        } <= names
        # Every tool advertises a JSON-Schema input contract.
        for t in tools:
            assert t["inputSchema"]["type"] == "object"
            assert t.get("description")

    def test_manifest_matches_tools_list(self) -> None:
        server = MercuryMCPServer()
        manifest = server.manifest()
        listed = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert listed is not None
        assert manifest == listed["result"]["tools"]


def _call(server: MercuryMCPServer, name: str, arguments: dict) -> dict:
    resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    assert resp is not None
    return resp["result"]


class TestToolCalls:
    def test_detect_anomaly_flags_outlier(self) -> None:
        server = MercuryMCPServer()
        rng = np.random.default_rng(0)
        data = rng.normal(size=(50, 3))
        data[0] += 15.0  # a clear outlier
        result = _call(server, "mercury_detect_anomaly", {"data": data.tolist()})
        assert result["isError"] is False
        payload = json.loads(result["content"][0]["text"])
        assert payload["n"] == 50
        assert len(payload["scores"]) == 50
        assert payload["is_anomaly"][0] is True  # the injected outlier is flagged

    def test_score_ethics_blocks_harmful_action(self) -> None:
        server = MercuryMCPServer()
        result = _call(
            server,
            "mercury_score_ethics",
            {"action": "destroy and kill with permanent irreversible violence"},
        )
        payload = json.loads(result["content"][0]["text"])
        assert payload["is_permissible"] is False
        assert payload["harm_score"] > 0.0

    def test_score_ethics_euphemism_blocked(self) -> None:
        server = MercuryMCPServer()
        result = _call(server, "mercury_score_ethics", {"action": "make them disappear quietly"})
        payload = json.loads(result["content"][0]["text"])
        assert payload["is_permissible"] is False

    def test_calibrate_confidence_reports_oof(self) -> None:
        server = MercuryMCPServer()
        rng = np.random.default_rng(0)
        s = rng.uniform(0, 1, 400)
        y = (rng.uniform(0, 1, 400) < s**2).astype(int)  # miscalibrated
        result = _call(
            server,
            "mercury_calibrate_confidence",
            {"scores": s.tolist(), "labels": y.tolist(), "method": "auto"},
        )
        payload = json.loads(result["content"][0]["text"])
        assert payload["eval_protocol"] == "cv_oof"
        assert "brier_delta_ci" in payload

    def test_research_is_failclosed_offline(self) -> None:
        server = MercuryMCPServer(assistant=_offline_assistant())
        result = _call(server, "mercury_research", {"query": "anything"})
        assert result["isError"] is False  # honest report, not a transport error
        payload = json.loads(result["content"][0]["text"])
        assert payload["available"] is False

    def test_research_cited_report_online(self) -> None:
        server = MercuryMCPServer(assistant=_online_assistant())
        result = _call(
            server, "mercury_research", {"query": "conformal prediction", "max_sources": 1}
        )
        payload = json.loads(result["content"][0]["text"])
        assert payload["available"] is True
        assert payload["document"]

    def test_unknown_tool_is_error_result(self) -> None:
        server = MercuryMCPServer()
        result = _call(server, "mercury_nonexistent", {})
        assert result["isError"] is True

    def test_bad_arguments_are_error_result_not_crash(self) -> None:
        server = MercuryMCPServer()
        result = _call(server, "mercury_score_ethics", {"action": ""})
        assert result["isError"] is True


class TestStdioLoop:
    def test_serve_stdio_roundtrip(self) -> None:
        server = MercuryMCPServer()
        requests = (
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            + "\n"
            + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
            + "\n"
            + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            + "\n"
        )
        stdin = io.StringIO(requests)
        stdout = io.StringIO()
        server.serve_stdio(stdin=stdin, stdout=stdout)
        lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        # initialize -> response; notification -> no response; tools/list -> response.
        assert len(lines) == 2
        assert lines[0]["id"] == 1 and "result" in lines[0]
        assert lines[1]["id"] == 2 and "tools" in lines[1]["result"]

    def test_parse_error_is_reported(self) -> None:
        server = MercuryMCPServer()
        stdin = io.StringIO("{not valid json}\n")
        stdout = io.StringIO()
        server.serve_stdio(stdin=stdin, stdout=stdout)
        line = json.loads(stdout.getvalue().strip())
        assert line["error"]["code"] == -32700
