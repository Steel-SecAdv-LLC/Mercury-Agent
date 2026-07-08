# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the Mercury MCP server -- the universal interconnect.

Drives the server with crafted JSON-RPC 2.0 messages (the same wire an MCP client
speaks) and asserts the protocol handshake, tool discovery, tool execution, and
fail-closed/honest behaviour, all offline.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from omni_mercury_engine.agentic.capabilities.web_research import SearchResult, WebResearcher

if TYPE_CHECKING:
    from typing import Any
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


def _call(server: MercuryMCPServer, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    assert resp is not None
    result = resp["result"]
    assert isinstance(result, dict)
    return result


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

    def test_intel_tools_are_selectable(self) -> None:
        """The intel streams are advertised as first-class, discoverable MCP tools."""
        server = MercuryMCPServer()
        names = {t["name"] for t in server.manifest()}
        assert {
            "mercury_verify_claims",
            "mercury_check_provenance",
            "mercury_self_consistency",
            "mercury_value_metrics",
        } <= names

    def test_verify_claims_refutes_false_symbolic_claim(self) -> None:
        server = MercuryMCPServer()
        result = _call(
            server,
            "mercury_verify_claims",
            {"text": "Note that 91 is prime, a well-known fact."},
        )
        payload = json.loads(result["content"][0]["text"])
        assert payload["allowed"] is False  # hard mode blocks a refuted claim
        assert payload["n_refuted"] >= 1
        assert any(
            v["kind"] == "primality" and v["status"] == "refuted" for v in payload["verdicts"]
        )

    def test_answer_emission_is_verifier_guarded(self, monkeypatch: Any) -> None:
        """A research/answer emission carrying an oracle-refuted claim is blocked live."""
        server = MercuryMCPServer(assistant=_online_assistant())
        # Force the assistant's answer to carry a refutable claim, then confirm
        # the MCP emission guard (hard mode) blocks it rather than emitting it.
        monkeypatch.setattr(
            server._research_assistant(),
            "answer",
            lambda question, **kw: "The number 91 is prime.",
        )
        result = _call(server, "mercury_answer", {"question": "is 91 prime?"})
        assert result["isError"] is True
        assert "verifier" in result["content"][0]["text"].lower()

    def test_answer_emission_soft_mode_flags_not_blocks(self, monkeypatch: Any) -> None:
        server = MercuryMCPServer(assistant=_online_assistant())
        monkeypatch.setenv("MERCURY_VERIFIER_MODE", "soft")
        monkeypatch.setattr(
            server._research_assistant(),
            "answer",
            lambda question, **kw: "The number 91 is prime.",
        )
        result = _call(server, "mercury_answer", {"question": "is 91 prime?"})
        assert result["isError"] is False  # soft mode annotates, does not block
        payload = json.loads(result["content"][0]["text"])
        assert payload.get("verifier_flags")

    def test_value_metrics_board_is_served(self) -> None:
        server = MercuryMCPServer()
        result = _call(server, "mercury_value_metrics", {})
        payload = json.loads(result["content"][0]["text"])
        assert "closed_feedback_loop" in payload["streams"]
        assert payload["streams"]["verifier_in_loop"]["target"] == 1.0

    def test_unknown_tool_is_error_result(self) -> None:
        server = MercuryMCPServer()
        result = _call(server, "mercury_nonexistent", {})
        assert result["isError"] is True

    def test_bad_arguments_are_error_result_not_crash(self) -> None:
        server = MercuryMCPServer()
        result = _call(server, "mercury_score_ethics", {"action": ""})
        assert result["isError"] is True


class _FakeFusionEngine:
    """A stand-in for OmniMercuryEngine to drive the fusion tool torch-free."""

    def __init__(self, result: dict[str, Any] | None = None, raises: Exception | None = None):
        self._result = result or {}
        self._raises = raises
        self.calls: list[tuple[tuple[int, ...], object]] = []

    def detect_with_fusion(self, X: Any, domain: object = None) -> dict[str, Any]:
        self.calls.append((tuple(X.shape), domain))
        if self._raises is not None:
            raise self._raises
        return self._result


class TestDetectionInterconnectTools:
    """The flagship-fusion and detector-tier tools are first-class MCP tools.

    Together with the CLI (`detect -d fusion`, `tier-detect`) and HTTP
    (`/detect/flagship`, `/detect/tier`) these unify Mercury's detection across
    every surface -- an MCP client reaches the SAME engines, not stand-ins.
    """

    def test_new_detection_tools_are_advertised(self) -> None:
        names = {t["name"] for t in MercuryMCPServer().manifest()}
        assert {"mercury_detect_fusion", "mercury_tier_detect"} <= names

    def test_detect_fusion_returns_calibrated_decision(self) -> None:
        engine = _FakeFusionEngine(
            result={
                "anomaly_prob": 0.87,
                "is_anomaly": True,
                "class_prediction": 3,
                "severity": 0.42,
                "threshold_used": 0.5,
                "detector_importance": {"statistical": 0.6, "temporal": 0.4},
                "gosnn_metadata": {
                    "ethical_gate_passed": True,
                    "sigma_immutable_score": 0.98,
                    "sigma_immutable_threshold": 0.96,
                    "sigma_immutable_backend": "torch",
                },
                "mode": "fusion",
            }
        )
        server = MercuryMCPServer(engine=engine)
        rng = np.random.default_rng(0)
        result = _call(server, "mercury_detect_fusion", {"data": rng.normal(size=(30, 5)).tolist()})
        assert result["isError"] is False
        payload = json.loads(result["content"][0]["text"])
        assert payload["anomaly_prob"] == 0.87
        assert payload["is_anomaly"] is True
        assert payload["ethical_gate"]["passed"] is True
        assert payload["ethical_gate"]["sigma_immutable_score"] == 0.98
        assert engine.calls and engine.calls[0][0] == (30, 5)

    def test_detect_fusion_blocked_by_ethical_gate_is_error(self) -> None:
        from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError

        engine = _FakeFusionEngine(
            raises=EthicalConstraintViolationError("blocked", 0.1, 0.96, check="sigma_immutable")
        )
        server = MercuryMCPServer(engine=engine)
        rng = np.random.default_rng(1)
        result = _call(server, "mercury_detect_fusion", {"data": rng.normal(size=(20, 4)).tolist()})
        # Fail closed: the refusal surfaces as a clean isError naming the gate.
        assert result["isError"] is True
        text = result["content"][0]["text"].lower()
        assert "ethical gate" in text and "sigma_immutable" in text

    def test_detect_fusion_bad_matrix_is_error(self) -> None:
        server = MercuryMCPServer(engine=_FakeFusionEngine())
        result = _call(server, "mercury_detect_fusion", {"data": [1.0, 2.0, 3.0]})  # 1-D
        assert result["isError"] is True
        assert "2-d" in result["content"][0]["text"].lower()

    def test_tier_detect_flags_injected_burst(self) -> None:
        server = MercuryMCPServer()
        rng = np.random.default_rng(0)
        series = rng.normal(0, 1, 200)
        series[100:108] += 7.0
        result = _call(
            server,
            "mercury_tier_detect",
            {
                "data": series.tolist(),
                "subset": ["spectral_residual", "spot_evt", "bocpd"],
                "conformal_alpha": 0.05,
            },
        )
        assert result["isError"] is False
        payload = json.loads(result["content"][0]["text"])
        assert payload["n_points"] == 200
        flagged = {i for i, f in enumerate(payload["conformal_flags"]) if f}
        assert flagged & set(range(95, 115))

    def test_tier_detect_stacking_without_labels_is_error(self) -> None:
        server = MercuryMCPServer()
        rng = np.random.default_rng(2)
        result = _call(
            server,
            "mercury_tier_detect",
            {"data": rng.normal(0, 1, 60).tolist(), "method": "stacking"},
        )
        assert result["isError"] is True

    def test_localize_root_cause_ranks_the_chain(self) -> None:
        server = MercuryMCPServer()
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, (300, 4))
        base[:, 1] += 0.8 * base[:, 0]
        base[:, 2] += 0.8 * base[:, 1]
        obs = base.copy()
        obs[-1, 0] += 8.0
        obs[-1, 1] += 6.0
        obs[-1, 2] += 4.0
        result = _call(
            server,
            "mercury_localize_root_cause",
            {"observations": obs.tolist(), "train": base[:-1].tolist(), "top_k": 3},
        )
        assert result["isError"] is False
        payload = json.loads(result["content"][0]["text"])
        assert payload["n_nodes"] == 4
        assert len(payload["ranked"]) == 3
        assert payload["top_root_cause"]["node"] in {0, 1, 2}  # a chain node, not the independent 3

    def test_localize_root_cause_bad_input_is_error(self) -> None:
        server = MercuryMCPServer()
        result = _call(server, "mercury_localize_root_cause", {"observations": [1.0, 2.0, 3.0]})
        assert result["isError"] is True

    def test_detect_fusion_real_engine_end_to_end(self) -> None:
        """No injection: the tool builds the real flagship engine and scores a matrix."""
        pytest.importorskip("torch")
        server = MercuryMCPServer()
        rng = np.random.default_rng(0)
        result = _call(server, "mercury_detect_fusion", {"data": rng.normal(size=(30, 5)).tolist()})
        assert result["isError"] is False, result["content"][0]["text"]
        payload = json.loads(result["content"][0]["text"])
        assert 0.0 <= float(payload["anomaly_prob"]) <= 1.0
        assert "ethical_gate" in payload


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


class TestMalformedInputHardening:
    """A single malformed message must never crash the server or wrongly reply."""

    def test_non_dict_params_is_invalid_params_not_crash(self) -> None:
        # A truthy non-object params (e.g. a JSON array) slips past `or {}`; the
        # server must return -32602, never raise AttributeError on params.get.
        server = MercuryMCPServer()
        resp = server.handle_message(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": [1, 2]}
        )
        assert resp is not None and resp["error"]["code"] == -32602

    def test_non_dict_params_does_not_kill_serve_loop(self) -> None:
        server = MercuryMCPServer()
        stdin = io.StringIO(
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": [1, 2]})
            + "\n"
            + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"})
            + "\n"
        )
        stdout = io.StringIO()
        server.serve_stdio(stdin=stdin, stdout=stdout)
        lines = [json.loads(x) for x in stdout.getvalue().splitlines() if x.strip()]
        # Both messages answered -> the bad params did not kill the loop.
        assert {ln["id"] for ln in lines} == {1, 2}
        assert lines[0]["error"]["code"] == -32602
        assert lines[1]["result"] == {}  # ping still served

    def test_malformed_notification_gets_no_response(self) -> None:
        server = MercuryMCPServer()
        # No id => notification; even malformed, JSON-RPC forbids a reply.
        assert server.handle_message({"jsonrpc": "2.0"}) is None
        assert server.handle_message({"jsonrpc": "1.0", "method": "x"}) is None
        # A malformed *request* (has an id) still gets an error.
        bad_req = server.handle_message({"jsonrpc": "1.0", "id": 7})
        assert bad_req is not None and bad_req["error"]["code"] == -32600


def _boom_guard(self: object, text: str, *, source: str = "generation") -> object:
    """A verifier that raises at runtime (a bug), not one that is unavailable."""
    raise RuntimeError("verifier exploded")


class TestGuardEmissionFailClosed:
    """A verifier *runtime* fault must fail closed, never silently disable gating."""

    def test_runtime_error_fails_closed_in_hard_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.intel import verifier_loop as vl

        monkeypatch.setenv("MERCURY_VERIFIER_MODE", "hard")
        monkeypatch.setattr(vl.VerifierLoop, "guard_emission", _boom_guard)
        guard = MercuryMCPServer._guard_emission("2 is prime.", source="unit")
        assert guard["allowed"] is False  # emission blocked, not emitted ungated
        assert guard["mode"] == "verifier_error"
        assert guard["blocked"]  # the fault surfaces as a block, not silence

    def test_runtime_error_soft_mode_flags_but_allows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.intel import verifier_loop as vl

        monkeypatch.setenv("MERCURY_VERIFIER_MODE", "soft")
        monkeypatch.setattr(vl.VerifierLoop, "guard_emission", _boom_guard)
        guard = MercuryMCPServer._guard_emission("2 is prime.", source="unit")
        assert guard["allowed"] is True  # soft mode never blocks...
        assert guard["mode"] == "verifier_error_soft"
        assert guard["flagged"]  # ...but the fault is flagged, not swallowed

    def test_clean_verifier_still_blocks_a_false_claim_in_hard_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Sanity: with no injected fault, a genuinely refuted claim still blocks
        # (default mode is hard) -- the fail-closed change did not weaken gating.
        monkeypatch.delenv("MERCURY_VERIFIER_MODE", raising=False)
        guard = MercuryMCPServer._guard_emission("91 is prime.", source="unit")
        assert guard["allowed"] is False  # 91 = 7*13 -> refuted -> blocked


class TestGuardEmissionImportHandling:
    """Only a *missing module* degrades to allow; a symbol-import bug fails closed."""

    def test_module_not_found_degrades_to_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _fake(name: str, *a: object, **k: object) -> object:
            if name == "omni_mercury_engine.intel.verifier_loop":
                raise ModuleNotFoundError(
                    "No module named 'omni_mercury_engine.intel.verifier_loop'"
                )
            return real_import(name, *a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _fake)
        guard = MercuryMCPServer._guard_emission("2 is prime.", source="unit")
        assert guard["allowed"] is True and guard["mode"] == "unavailable"

    def test_symbol_import_error_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def _fake(name: str, *a: object, **k: object) -> object:
            if name == "omni_mercury_engine.intel.verifier_loop":
                raise ImportError("cannot import name 'VerifierMode'")  # a bug, not unavailability
            return real_import(name, *a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _fake)
        # Not downgraded to "unavailable": it propagates (emission blocked at call_tool).
        with pytest.raises(ImportError):
            MercuryMCPServer._guard_emission("2 is prime.", source="unit")


class TestCheckProvenanceVerifiedType:
    """`verified` is validated as a real bool, not truthiness-coerced (a truthy
    string like "false" must not be read as True on a hazardous boundary)."""

    def test_string_false_is_rejected_not_treated_as_true(self) -> None:
        server = MercuryMCPServer()
        result = _call(
            server,
            "mercury_check_provenance",
            {"text": "a benign claim", "sources": ["s1"], "verified": "false"},
        )
        assert result["isError"] is True
        assert "boolean" in result["content"][0]["text"].lower()

    def test_bool_verified_is_accepted(self) -> None:
        server = MercuryMCPServer()
        result = _call(
            server,
            "mercury_check_provenance",
            {"text": "a benign claim", "sources": ["s1"], "verified": True},
        )
        assert result["isError"] is False
