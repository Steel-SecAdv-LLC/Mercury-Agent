# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury as an MCP server -- the universal interconnect.

Any AI system that speaks the Model Context Protocol (MCP-capable desktop
assistants, IDE agents, and orchestrators) can *link up to and run* Mercury
through this server: it advertises Mercury's capabilities as discoverable, self-
describing MCP tools and executes them on request. No Mercury-specific client,
SDK, or glue code is required on the other side -- MCP is the contract.

**Dependency-free.** The whole transport is JSON-RPC 2.0 over stdio implemented
with the standard library only (``json`` + ``sys``) -- no ``mcp`` package, no
web framework, no new dependency. Newline-delimited JSON messages are read from
stdin and written to stdout, exactly as the MCP stdio transport specifies.

**Honest and fail-closed**, like the rest of Mercury:

* Every outward tool (research / answer / write_document) passes Mercury's own
  fail-closed benevolence gate before acting.
* A tool whose backing stack is unavailable in the environment (e.g. the ML
  detection engine on a slim install) returns an ``isError`` result explaining
  why -- it never fabricates a capability it cannot deliver.
* The research tools degrade honestly to "no sources reachable" offline rather
  than inventing answers.

The tool surface (``tools/list``) doubles as a machine-readable **capability
manifest** (:meth:`MercuryMCPServer.manifest`), so the same self-description can
back other transports (HTTP/JSON-RPC) without duplication.

Run it::

    mercury-agent mcp            # serve on stdio
    python -m omni_mercury_engine.mcp_server

and point any MCP client at that command.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "mercury-agent"

# JSON-RPC 2.0 error codes (subset used here).
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603


class ToolError(RuntimeError):
    """A tool failed in a way the caller should see (returned as ``isError``)."""


@dataclass
class ToolSpec:
    """One MCP tool: its advertised schema and its handler.

    ``handler`` maps validated ``arguments`` to a result string (JSON-encoded for
    machine consumption). Raising :class:`ToolError` surfaces a clean, model-
    visible error; any other exception is caught and reported as an internal
    tool error so the server never crashes on a bad call.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]

    def advertise(self) -> dict[str, Any]:
        """The ``tools/list`` entry for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class MercuryMCPServer:
    """Expose Mercury's capabilities to any MCP client over stdio.

    Capabilities are built lazily (so importing this module pulls in nothing
    heavy) and may be injected for testing. Construct once and either drive it
    message-by-message via :meth:`handle_message` (pure, for tests/other
    transports) or run the stdio loop via :meth:`serve_stdio`.
    """

    def __init__(
        self,
        *,
        benevolence_scorer: Any | None = None,
        assistant: Any | None = None,
    ) -> None:
        """Initialize the server; capabilities are injectable for testing."""
        self._scorer = benevolence_scorer
        self._assistant = assistant
        self._initialized = False
        self._tools: dict[str, ToolSpec] = {}
        self._register_tools()

    # -- lazily-built capabilities ----------------------------------------

    def _benevolence(self) -> Any:
        if self._scorer is None:
            from omni_mercury_engine.cognitive.ethical_bounding import (
                MINIMUM_BENEVOLENCE_FLOOR,
                BenevolenceScorer,
            )

            self._scorer = BenevolenceScorer(benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR)
        return self._scorer

    def _research_assistant(self) -> Any:
        if self._assistant is None:
            from omni_mercury_engine.agentic.capabilities import GeneralAssistant

            self._assistant = GeneralAssistant(benevolence_scorer=self._benevolence())
        return self._assistant

    # -- tool registry -----------------------------------------------------

    def _register_tools(self) -> None:
        specs = [
            ToolSpec(
                name="mercury_detect_anomaly",
                description=(
                    "Unsupervised batch anomaly detection over a numeric matrix using "
                    "Mercury's statistical ensemble. Fits on the supplied batch and "
                    "returns a per-row anomaly score, a boolean flag, and the operating "
                    "threshold. (Batch scoring; not a pre-trained calibrated model.)"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "description": "Rows of equal-length numeric feature vectors.",
                            "items": {"type": "array", "items": {"type": "number"}},
                        }
                    },
                    "required": ["data"],
                },
                handler=self._tool_detect_anomaly,
            ),
            ToolSpec(
                name="mercury_score_ethics",
                description=(
                    "Score an action against Mercury's benevolence/harm gate. Returns "
                    "the benevolence score, harm, severity, permissibility, the two-axis "
                    "weapons/mass-casualty verdict (hazard domain, operational intent, "
                    "disposition), and the explanation -- the same fail-closed ethics "
                    "used across Mercury."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "description": "The action to evaluate."},
                        "context": {
                            "type": "object",
                            "description": "Optional structured context for the action.",
                        },
                    },
                    "required": ["action"],
                },
                handler=self._tool_score_ethics,
            ),
            ToolSpec(
                name="mercury_research",
                description=(
                    "Research a question on the open web and return a cited report "
                    "(extractive, never fabricated). Harm-gated and fail-closed: returns "
                    "an honest 'unavailable' when offline/blocked or 'refused' when the "
                    "ethics gate blocks the query."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_sources": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
                handler=self._tool_research,
            ),
            ToolSpec(
                name="mercury_answer",
                description=(
                    "Answer a question with sentences extracted verbatim from web "
                    "sources, with citations. Honest/fail-closed when offline."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "max_sources": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["question"],
                },
                handler=self._tool_answer,
            ),
            ToolSpec(
                name="mercury_write_document",
                description=(
                    "Generate a Markdown/HTML/text document from structured sections, "
                    "gated by Mercury's benevolence check (returns an error if refused)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "sections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "heading": {"type": "string"},
                                    "body": {"type": "string"},
                                },
                                "required": ["heading", "body"],
                            },
                        },
                        "format": {"type": "string", "enum": ["markdown", "html", "text"]},
                    },
                    "required": ["title", "sections"],
                },
                handler=self._tool_write_document,
            ),
            ToolSpec(
                name="mercury_calibrate_confidence",
                description=(
                    "Fit Mercury's cross-validated, accept-gated confidence calibrator "
                    "on (score, label) pairs and return the measured out-of-fold "
                    "ECE/Brier and the accept decision. Calibrated probabilities are "
                    "deployed only on a statistically significant improvement."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "scores": {"type": "array", "items": {"type": "number"}},
                        "labels": {"type": "array", "items": {"type": "integer"}},
                        "method": {
                            "type": "string",
                            "enum": ["auto", "platt", "isotonic", "strict_isotonic", "temperature"],
                        },
                    },
                    "required": ["scores", "labels"],
                },
                handler=self._tool_calibrate_confidence,
            ),
        ]
        self._tools = {s.name: s for s in specs}

    # -- tool handlers -----------------------------------------------------

    @staticmethod
    def _tool_detect_anomaly(args: dict[str, Any]) -> str:
        import numpy as np

        rows = args.get("data")
        if not isinstance(rows, list) or not rows:
            raise ToolError("'data' must be a non-empty array of numeric rows")
        try:
            X = np.asarray(rows, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ToolError(f"'data' is not a numeric matrix: {exc}") from exc
        if X.ndim != 2:
            raise ToolError("'data' must be 2-D (rows of equal-length feature vectors)")
        try:
            from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
        except Exception as exc:  # pragma: no cover - slim-install path
            raise ToolError(
                f"anomaly detection stack unavailable in this environment: {exc}"
            ) from exc
        detector = MercuryAnomalyDetector()
        detector.fit(X)
        result = detector.detect(X)
        scores = np.asarray(result["scores"], dtype=float).reshape(-1)
        flags = np.asarray(result["is_anomaly"]).reshape(-1).astype(bool)
        return json.dumps(
            {
                "n": int(X.shape[0]),
                "threshold": float(result.get("threshold", 0.0)),
                "scores": [round(float(s), 6) for s in scores],
                "is_anomaly": [bool(f) for f in flags],
                "n_anomalies": int(flags.sum()),
                "note": "unsupervised batch scoring (fit on the supplied batch)",
            }
        )

    def _tool_score_ethics(self, args: dict[str, Any]) -> str:
        action = args.get("action")
        if not isinstance(action, str) or not action.strip():
            raise ToolError("'action' must be a non-empty string")
        context = args.get("context") or {}
        if not isinstance(context, dict):
            raise ToolError("'context' must be an object")
        score = self._benevolence().score_action(action, context)
        return json.dumps(
            {
                "action": action,
                "benevolence_score": round(float(score.benevolence_score), 6),
                "harm_score": round(float(score.harm_score), 6),
                "severity_score": round(float(score.severity_score), 6),
                "is_permissible": bool(score.is_permissible),
                "threshold": round(float(self._benevolence().benevolence_threshold), 6),
                # Two-axis weapons/mass-casualty uplift verdict (see
                # docs/HARM_POLICY.md), surfaced so an MCP caller can see the
                # hazard-domain/operational-intent routing and disposition, not
                # just the scalar benevolence/harm.
                "hazard_domain": getattr(score, "hazard_domain", "none"),
                "operational_intent": getattr(score, "operational_intent", "mechanism"),
                "weapons_disposition": getattr(score, "weapons_disposition", "allow"),
                "explanation": score.explanation,
            }
        )

    def _tool_research(self, args: dict[str, Any]) -> str:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("'query' must be a non-empty string")
        max_sources = int(args.get("max_sources", 5))
        report = self._research_assistant().research_report(query, max_sources=max_sources)
        return json.dumps(report.to_dict())

    def _tool_answer(self, args: dict[str, Any]) -> str:
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ToolError("'question' must be a non-empty string")
        max_sources = int(args.get("max_sources", 3))
        answer = self._research_assistant().answer(question, max_sources=max_sources)
        return json.dumps({"question": question, "answer": answer})

    def _tool_write_document(self, args: dict[str, Any]) -> str:
        from omni_mercury_engine.agentic.capabilities.document_generator import Section

        title = args.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ToolError("'title' must be a non-empty string")
        raw_sections = args.get("sections")
        if not isinstance(raw_sections, list) or not raw_sections:
            raise ToolError("'sections' must be a non-empty array of {heading, body}")
        # Fail closed on a malformed section rather than silently dropping it:
        # the advertised inputSchema requires each item to be an object with a
        # non-empty 'heading' and 'body', so an MCP client that sends anything
        # else gets a clear, actionable error instead of a silently shorter doc.
        sections = []
        for i, s in enumerate(raw_sections):
            if not isinstance(s, dict):
                raise ToolError(f"section[{i}] must be an object with 'heading' and 'body'")
            heading = s.get("heading")
            body = s.get("body")
            if not isinstance(heading, str) or not isinstance(body, str):
                raise ToolError(f"section[{i}] requires string 'heading' and 'body'")
            sections.append(Section(heading, body))
        fmt = str(args.get("format", "markdown"))
        document = self._research_assistant().write_document(title, sections, fmt=fmt)
        if document is None:
            raise ToolError("document refused by the benevolence gate")
        return json.dumps({"format": document.fmt, "content": document.content})

    @staticmethod
    def _tool_calibrate_confidence(args: dict[str, Any]) -> str:
        import numpy as np

        from omni_mercury_engine.core.confidence import CalibratedConfidence

        scores = args.get("scores")
        labels = args.get("labels")
        if not isinstance(scores, list) or not isinstance(labels, list):
            raise ToolError("'scores' and 'labels' must be arrays")
        if len(scores) != len(labels):
            raise ToolError("'scores' and 'labels' must have equal length")
        method = str(args.get("method", "auto"))
        try:
            cc = CalibratedConfidence(method=method)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        report = cc.fit(np.asarray(scores, dtype=float), np.asarray(labels, dtype=float))
        out = report.to_dict()
        out["is_calibrated"] = bool(cc.is_calibrated)
        return json.dumps(out)

    # -- public surfaces ---------------------------------------------------

    def manifest(self) -> list[dict[str, Any]]:
        """Return the ``tools/list`` payload -- a machine-readable capability list."""
        return [spec.advertise() for spec in self._tools.values()]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a tool and return an MCP ``tools/call`` result (with ``isError``)."""
        spec = self._tools.get(name)
        if spec is None:
            return self._tool_error_result(f"unknown tool {name!r}")
        try:
            text = spec.handler(arguments or {})
        except ToolError as exc:
            return self._tool_error_result(str(exc))
        except Exception as exc:  # defensive: never crash the server on a tool bug
            logger.exception("tool %s failed", name)
            return self._tool_error_result(f"internal tool error: {type(exc).__name__}: {exc}")
        return {"content": [{"type": "text", "text": text}], "isError": False}

    @staticmethod
    def _tool_error_result(message: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": message}], "isError": True}

    # -- JSON-RPC dispatch -------------------------------------------------

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle one JSON-RPC message; return a response, or ``None`` for a notification."""
        if message.get("jsonrpc") != "2.0" or "method" not in message:
            # A message with no ``id`` is a notification; per JSON-RPC 2.0 the
            # server must NOT reply to one, even a malformed one. Only an
            # identifiable request (has an ``id``) gets an error envelope.
            if "id" not in message:
                return None
            return self._error(message.get("id"), _INVALID_REQUEST, "invalid JSON-RPC request")
        method = message["method"]
        msg_id = message.get("id")
        params = message.get("params") or {}
        is_notification = "id" not in message

        # Notifications (no id) never get a response.
        if is_notification:
            if method == "notifications/initialized":
                self._initialized = True
            return None

        if method == "initialize":
            return self._result(
                msg_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": _server_version()},
                },
            )
        if method == "ping":
            return self._result(msg_id, {})
        if method == "tools/list":
            return self._result(msg_id, {"tools": self.manifest()})
        if method == "tools/call":
            # ``params`` may be any JSON value; a truthy non-object (e.g. an
            # array) slips past the ``or {}`` guard above, so re-check before
            # dereferencing it -- otherwise ``params.get`` raises and (without
            # the serve_stdio guard) would kill the loop.
            if not isinstance(params, dict):
                return self._error(msg_id, _INVALID_PARAMS, "'params' must be an object")
            name = params.get("name")
            if not isinstance(name, str):
                return self._error(msg_id, _INVALID_PARAMS, "tools/call requires a string 'name'")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                return self._error(msg_id, _INVALID_PARAMS, "'arguments' must be an object")
            return self._result(msg_id, self.call_tool(name, arguments))
        return self._error(msg_id, _METHOD_NOT_FOUND, f"unknown method {method!r}")

    def serve_stdio(self, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> None:
        """Serve MCP over newline-delimited JSON-RPC on stdio until EOF."""
        import sys

        rstream = stdin if stdin is not None else sys.stdin
        wstream = stdout if stdout is not None else sys.stdout
        logger.info("Mercury MCP server starting on stdio (protocol %s)", PROTOCOL_VERSION)
        for line in rstream:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._write(wstream, self._error(None, _PARSE_ERROR, "parse error"))
                continue
            if not isinstance(message, dict):
                self._write(wstream, self._error(None, _INVALID_REQUEST, "invalid request"))
                continue
            # Defense in depth: no single malformed message may ever kill the
            # serve loop. Any unexpected error becomes a JSON-RPC internal error.
            try:
                response = self.handle_message(message)
            except Exception as exc:  # pragma: no cover - belt-and-suspenders
                logger.exception("handle_message crashed")
                response = self._error(
                    message.get("id"), _INTERNAL_ERROR, f"internal error: {type(exc).__name__}"
                )
            if response is not None:
                self._write(wstream, response)

    # -- JSON-RPC envelope helpers ----------------------------------------

    @staticmethod
    def _write(wstream: IO[str], payload: dict[str, Any]) -> None:
        wstream.write(json.dumps(payload) + "\n")
        wstream.flush()

    @staticmethod
    def _result(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _server_version() -> str:
    try:
        from omni_mercury_engine._version import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - version helper is always present
        return "0"


def serve() -> None:
    """Entry point: run the Mercury MCP server on stdio."""
    logging.basicConfig(level=logging.INFO)
    MercuryMCPServer().serve_stdio()


if __name__ == "__main__":  # pragma: no cover
    serve()


__all__ = ["PROTOCOL_VERSION", "MercuryMCPServer", "ToolError", "ToolSpec", "serve"]
