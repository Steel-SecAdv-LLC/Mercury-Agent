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
        engine: Any | None = None,
    ) -> None:
        """Initialize the server; capabilities are injectable for testing."""
        self._scorer = benevolence_scorer
        self._assistant = assistant
        self._engine = engine
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

    def _fusion_engine(self) -> Any:
        """The flagship OmniMercuryEngine (fusion mode), built once and cached.

        This is the SAME neuro-symbolic fusion path the ``mercury-agent detect
        -d fusion`` CLI runs -- trained fusion network + GOSNN scalar
        integration + the σ_Immutable second hard ethical gate -- so an MCP
        client reaches Mercury's flagship detector, not a reduced statistical
        stand-in. ``require_explicit_fit=False`` mirrors the CLI: there is no
        train/test split here, so the engine relies on the shipped default
        fusion checkpoint (loaded below) plus the auto-fit-on-first-batch path
        for the base detectors. The stdio server is single-threaded, so one
        cached engine is safe without locking.
        """
        if self._engine is None:
            from omni_mercury_engine.engine import OmniMercuryEngine

            engine = OmniMercuryEngine(mode="fusion", require_explicit_fit=False)
            engine.load_default_fusion_checkpoint()
            self._engine = engine
        return self._engine

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
                name="mercury_detect_fusion",
                description=(
                    "Flagship neuro-symbolic anomaly detection over a numeric feature "
                    "matrix using Mercury's full OmniMercuryEngine fusion path: the "
                    "trained fusion network, GOSNN scalar integration, and the "
                    "sigma_Immutable hard ethical gate (the same engine the "
                    "'mercury-agent detect -d fusion' CLI runs, loaded from the shipped "
                    "default checkpoint). Returns a calibrated anomaly probability, the "
                    "decision, severity, per-detector importances, and the ethical-gate "
                    "metadata. Requires the ML stack; returns a clear error on a slim "
                    "install, and is *blocked* (never silently allowed) if the ethical "
                    "gate refuses the input."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "description": "Rows of equal-length numeric feature vectors.",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                        "domain": {
                            "type": "string",
                            "description": (
                                "Optional domain for GOSNN threshold tuning " "(e.g. 'medical')."
                            ),
                        },
                    },
                    "required": ["data"],
                },
                handler=self._tool_detect_fusion,
            ),
            ToolSpec(
                name="mercury_tier_detect",
                description=(
                    "Streaming detector-tier ensemble over a 1-D numeric series "
                    "(torch-free, no ML extra required): the calibrated statistical / "
                    "state-space / streaming tier. Returns per-point calibrated anomaly "
                    "probabilities, flags at the calibrated threshold, cross-detector "
                    "uncertainty, and -- when 'conformal_alpha' is set -- flags with a "
                    "distribution-free false-positive guarantee (FPR <= alpha). The same "
                    "runner behind 'mercury-agent tier-detect' and 'POST /detect/tier'."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "description": "A 1-D numeric anomaly series.",
                            "items": {"type": "number"},
                            "minItems": 8,
                        },
                        "labels": {
                            "type": "array",
                            "description": "Optional per-point 0/1 labels (enables stacking/BMA).",
                            "items": {"type": "integer"},
                        },
                        "subset": {
                            "type": "array",
                            "description": "Detector names to include (default: the full tier).",
                            "items": {"type": "string"},
                        },
                        "method": {
                            "type": "string",
                            "enum": ["stacking", "bma", "average", "consensus"],
                        },
                        "contamination": {"type": "number", "minimum": 0.0, "maximum": 0.5},
                        "conformal_alpha": {
                            "type": "number",
                            "exclusiveMinimum": 0.0,
                            "exclusiveMaximum": 1.0,
                            "description": "Distribution-free FP rate; adds conformal flags.",
                        },
                    },
                    "required": ["data"],
                },
                handler=self._tool_tier_detect,
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
            ToolSpec(
                name="mercury_verify_claims",
                description=(
                    "Route the checkable claims in a text through Mercury's oracle-"
                    "validated verifiers (primality, Collatz, propositional "
                    "tautology/SAT via a Tseitin->DPLL transform, and physics "
                    "dimensional consistency) and return a per-claim verdict "
                    "(confirmed/refuted/unavailable). This is the verifier-in-the-"
                    "loop that gates Mercury's own research/answer emissions."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to scan for claims."}
                    },
                    "required": ["text"],
                },
                handler=self._tool_verify_claims,
            ),
            ToolSpec(
                name="mercury_check_provenance",
                description=(
                    "Enforce provenance at the output boundary: decide whether a "
                    "candidate emission on a (possibly hazardous) topic may be "
                    "emitted given its cited sources, using Mercury's own "
                    "weapons/mass-casualty gate to decide when attribution is "
                    "required. Returns whether it is emitted, whether the boundary "
                    "enforced (withheld/redacted), and the mode."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The emission text."},
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Source identifiers/citations carried with the text.",
                        },
                        "verified": {
                            "type": "boolean",
                            "description": "Whether the sources were independently checked.",
                        },
                    },
                    "required": ["text"],
                },
                handler=self._tool_check_provenance,
            ),
            ToolSpec(
                name="mercury_self_consistency",
                description=(
                    "Score N sampled reasoning-path answers for disagreement and "
                    "apply the calibrated decision rule (widen confidence toward "
                    "0.5 with disagreement, abstain when the paths are too split). "
                    "Returns the plurality answer, a disagreement in [0,1], and the "
                    "decision (positive/negative/abstain)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "answers": {
                            "type": "array",
                            "description": "The sampled reasoning-path answers to vote over.",
                        },
                        "prob": {
                            "type": "number",
                            "description": "Optional base calibrated probability for the decision.",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["answers"],
                },
                handler=self._tool_self_consistency,
            ),
            ToolSpec(
                name="mercury_value_metrics",
                description=(
                    "Return Mercury's intelligence-layer value board: each stream's "
                    "declared, measured value metric with its baseline, target, and "
                    "direction (the single source of truth the CI lanes enforce)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "stream": {
                            "type": "string",
                            "description": "Optional single stream to return; omit for all.",
                        }
                    },
                },
                handler=self._tool_value_metrics,
            ),
        ]
        self._tools = {s.name: s for s in specs}

    # -- tool handlers -----------------------------------------------------

    @staticmethod
    def _tool_detect_anomaly(args: dict[str, Any]) -> str:
        import numpy as np

        X = MercuryMCPServer._coerce_matrix(args.get("data"))
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

    @staticmethod
    def _coerce_matrix(rows: Any) -> Any:
        """Validate ``args['data']`` as a 2-D numeric matrix (shared by detect tools)."""
        import numpy as np

        if not isinstance(rows, list) or not rows:
            raise ToolError("'data' must be a non-empty array of numeric rows")
        try:
            X = np.asarray(rows, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ToolError(f"'data' is not a numeric matrix: {exc}") from exc
        if X.ndim != 2:
            raise ToolError("'data' must be 2-D (rows of equal-length feature vectors)")
        return X

    def _tool_detect_fusion(self, args: dict[str, Any]) -> str:
        X = self._coerce_matrix(args.get("data"))
        domain = args.get("domain")
        if domain is not None and not isinstance(domain, str):
            raise ToolError("'domain' must be a string")
        # Import the gate exception before the try so the fail-closed refusal
        # (a blocked detection) is surfaced distinctly from an internal fault.
        try:
            from omni_mercury_engine.engine import EthicalConstraintViolationError
        except Exception as exc:  # pragma: no cover - slim-install path
            raise ToolError(
                f"flagship fusion engine unavailable in this environment: {exc}"
            ) from exc
        try:
            engine = self._fusion_engine()
        except Exception as exc:  # pragma: no cover - slim-install path (torch/checkpoint)
            raise ToolError(
                f"flagship fusion engine unavailable in this environment: {exc}"
            ) from exc
        try:
            result = engine.detect_with_fusion(X, domain=domain)
        except EthicalConstraintViolationError as exc:
            # Fail closed and honestly: the detection was *refused* by a hard
            # ethical gate, not merely errored -- say which gate and why.
            raise ToolError(
                f"flagship detection blocked by the '{exc.check}' ethical gate: {exc}"
            ) from exc
        gosnn = result.get("gosnn_metadata", {})
        if not isinstance(gosnn, dict):
            gosnn = {}
        return json.dumps(
            {
                "n": int(X.shape[0]),
                "anomaly_prob": round(float(result.get("anomaly_prob", 0.0)), 6),
                "is_anomaly": bool(result.get("is_anomaly", False)),
                "class_prediction": result.get("class_prediction"),
                "severity": round(float(result.get("severity", 0.0)), 6),
                "threshold_used": (
                    round(float(result["threshold_used"]), 6)
                    if result.get("threshold_used") is not None
                    else None
                ),
                "detector_importance": {
                    k: round(float(v), 6)
                    for k, v in (result.get("detector_importance") or {}).items()
                },
                "ethical_gate": {
                    "passed": bool(gosnn.get("ethical_gate_passed", True)),
                    "sigma_immutable_score": (
                        round(float(gosnn["sigma_immutable_score"]), 6)
                        if gosnn.get("sigma_immutable_score") is not None
                        else None
                    ),
                    "sigma_immutable_threshold": (
                        round(float(gosnn["sigma_immutable_threshold"]), 6)
                        if gosnn.get("sigma_immutable_threshold") is not None
                        else None
                    ),
                    "backend": gosnn.get("sigma_immutable_backend"),
                },
                "mode": result.get("mode", "fusion"),
                "note": "flagship neuro-symbolic fusion (trained checkpoint + GOSNN + sigma_Immutable)",
            }
        )

    @staticmethod
    def _tool_tier_detect(args: dict[str, Any]) -> str:
        import numpy as np

        raw = args.get("data")
        if not isinstance(raw, list) or not raw:
            raise ToolError("'data' must be a non-empty numeric array")
        try:
            series = np.asarray(raw, dtype=float).ravel()
        except (ValueError, TypeError) as exc:
            raise ToolError(f"'data' is not a numeric series: {exc}") from exc
        labels_raw = args.get("labels")
        labels = None
        if labels_raw is not None:
            try:
                labels = np.asarray(labels_raw, dtype=int).ravel()
            except (ValueError, TypeError) as exc:
                raise ToolError(f"'labels' is not an integer array: {exc}") from exc
        subset = args.get("subset")
        if subset is not None and (
            not isinstance(subset, list) or not all(isinstance(s, str) for s in subset)
        ):
            raise ToolError("'subset' must be an array of detector-name strings")
        try:
            from omni_mercury_engine.detectors.detection_tier import run_tier_ensemble
        except Exception as exc:  # pragma: no cover - slim-install path
            raise ToolError(f"detector-tier stack unavailable in this environment: {exc}") from exc
        try:
            result = run_tier_ensemble(
                series,
                labels=labels,
                subset=tuple(subset) if subset else None,
                method=args.get("method"),
                contamination=float(args.get("contamination", 0.05)),
                conformal_alpha=(
                    float(args["conformal_alpha"])
                    if args.get("conformal_alpha") is not None
                    else None
                ),
            )
        except (ValueError, TypeError) as exc:
            raise ToolError(str(exc)) from exc
        return json.dumps(result)

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
        out = report.to_dict()
        # Verifier-in-the-loop: an extractive report can still surface a source's
        # oracle-refutable claim ("91 is prime", "the Collatz sequence of 27 never
        # reaches 1"). Guard the emitted text; in hard mode a refuted claim blocks
        # the report, in soft mode it is flagged. Unavailable checks never block.
        if report.available and not report.refused:
            emitted = " ".join(t for t in (out.get("summary", ""), out.get("document") or "") if t)
            guard = self._guard_emission(emitted, source="mercury_research")
            if not guard["allowed"]:
                raise ToolError(_verifier_block_message(guard["blocked"]))
            if guard["flagged"]:
                out["verifier_flags"] = guard["flagged"]
        return json.dumps(out)

    def _tool_answer(self, args: dict[str, Any]) -> str:
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ToolError("'question' must be a non-empty string")
        max_sources = int(args.get("max_sources", 3))
        answer = self._research_assistant().answer(question, max_sources=max_sources)
        # Guard the emitted answer the same way as a research report.
        guard = self._guard_emission(answer, source="mercury_answer")
        if not guard["allowed"]:
            raise ToolError(_verifier_block_message(guard["blocked"]))
        payload: dict[str, Any] = {"question": question, "answer": answer}
        if guard["flagged"]:
            payload["verifier_flags"] = guard["flagged"]
        return json.dumps(payload)

    @staticmethod
    def _guard_emission(text: str, *, source: str) -> dict[str, Any]:
        """Route a candidate emission through the verifier-in-the-loop.

        Returns ``{"allowed", "mode", "blocked", "flagged"}``. In ``hard`` mode a
        single oracle-refuted claim yields ``allowed=False`` with the refuted
        claims in ``blocked``; in ``soft`` mode they are returned in ``flagged``
        and the emission is allowed.

        Degradation is scoped precisely so a runtime fault can **never silently
        disable gating**:

        * If the verifier stack is genuinely **unavailable** -- a slim install
          where the module does not import -- the guard degrades honestly to allow
          (``mode="unavailable"``): an *unavailable* check never blocks (the loop
          refutes, it never guesses).
        * If the stack imports but the verifier **raises at runtime** (a bug, an
          oracle crash), that is *not* unavailability. Failing open there would
          silently turn hard-mode gating off, so the guard **fails closed**: it
          blocks in ``hard`` mode (surfacing the fault as a refusal,
          ``mode="verifier_error"``) and allow-but-flags in ``soft`` mode (whose
          contract is annotate-and-allow, ``mode="verifier_error_soft"``).
        """
        if not text or not text.strip():
            return {"allowed": True, "mode": "empty", "blocked": [], "flagged": []}
        # Genuine unavailability is a *missing module* only, so catch just
        # ModuleNotFoundError (the slim-install case). A broader ImportError --
        # e.g. "cannot import name VerifierMode" -- is a verifier bug, not
        # unavailability, and must fail closed rather than masquerade as
        # "unavailable" and disable gating; left uncaught it propagates to
        # call_tool, which surfaces it as an error (emission blocked).
        try:
            from omni_mercury_engine.intel.verifier_loop import VerifierLoop, VerifierMode
        except ModuleNotFoundError as exc:  # pragma: no cover - slim-install path
            logger.info(
                "verifier-in-the-loop unavailable (slim install); emission not gated (%s)", exc
            )
            return {"allowed": True, "mode": "unavailable", "blocked": [], "flagged": []}
        try:
            decision = VerifierLoop().guard_emission(text, source=source)
        except Exception:
            # A runtime fault in an *importable* verifier is a bug, not
            # unavailability -- fail closed rather than emit ungated.
            logger.exception(
                "verifier-in-the-loop runtime failure guarding %s; failing closed", source
            )
            err_claim = {
                "kind": "verifier_error",
                "claim_text": "",
                "status": "unavailable",
                "reason": "verifier-in-the-loop runtime failure (failing closed)",
                "checker": "verifier_loop",
            }
            if VerifierMode.from_env() is VerifierMode.HARD:
                return {
                    "allowed": False,
                    "mode": "verifier_error",
                    "blocked": [err_claim],
                    "flagged": [],
                }
            return {
                "allowed": True,
                "mode": "verifier_error_soft",
                "blocked": [],
                "flagged": [err_claim],
            }
        return {
            "allowed": decision.allowed,
            "mode": decision.mode.value,
            "blocked": [c.as_dict() for c in decision.blocked_claims],
            "flagged": [c.as_dict() for c in decision.flagged_claims],
        }

    @staticmethod
    def _tool_verify_claims(args: dict[str, Any]) -> str:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ToolError("'text' must be a non-empty string")
        try:
            from omni_mercury_engine.intel.verifier_loop import VerifierLoop
        except Exception as exc:  # pragma: no cover - slim-install path
            raise ToolError(f"verifier stack unavailable in this environment: {exc}") from exc
        decision = VerifierLoop().guard_emission(text, source="mercury_verify_claims")
        return json.dumps(
            {
                "allowed": decision.allowed,
                "mode": decision.mode.value,
                "n_claims": len(decision.verdicts),
                "n_refuted": len(decision.refuted),
                "verdicts": [v.as_dict() for v in decision.verdicts],
                "blocked": [c.as_dict() for c in decision.blocked_claims],
            }
        )

    @staticmethod
    def _tool_check_provenance(args: dict[str, Any]) -> str:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ToolError("'text' must be a non-empty string")
        sources = args.get("sources") or []
        if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            raise ToolError("'sources' must be an array of strings")
        # Validate the type rather than truthiness-coerce: ``verified`` asserts a
        # safety-relevant fact (the sources were independently checked) on a
        # possibly-hazardous boundary. ``bool(args.get("verified"))`` would treat a
        # non-bool truthy value -- notably the JSON string ``"false"`` -- as True,
        # wrongly asserting verification. The schema declares a boolean; enforce it
        # (fail closed on anything else) so a caller can never smuggle a truthy
        # non-bool into the "verified" attestation.
        verified = args.get("verified", False)
        if not isinstance(verified, bool):
            raise ToolError("'verified' must be a boolean (true or false)")
        try:
            from omni_mercury_engine.intel.provenance import (
                Provenance,
                ProvenanceOrigin,
                enforce_at_boundary,
            )
        except Exception as exc:  # pragma: no cover - slim-install path
            raise ToolError(f"provenance stack unavailable in this environment: {exc}") from exc
        prov = Provenance(
            origin=ProvenanceOrigin.EXTRACTIVE, sources=tuple(sources), verified=verified
        )
        decision = enforce_at_boundary(
            text, text=text, provenance=prov, source="mercury_check_provenance"
        )
        return json.dumps(decision.as_dict())

    @staticmethod
    def _tool_self_consistency(args: dict[str, Any]) -> str:
        from collections import Counter

        answers = args.get("answers")
        if not isinstance(answers, list) or not answers:
            raise ToolError("'answers' must be a non-empty array")
        try:
            from omni_mercury_engine.intel.self_consistency import (
                self_consistency_decision,
                vote_disagreement,
            )
        except Exception as exc:  # pragma: no cover - slim-install path
            raise ToolError(f"self-consistency stack unavailable: {exc}") from exc
        # Votes must be hashable; coerce structured answers to a canonical string.
        votes = [
            json.dumps(a, sort_keys=True) if isinstance(a, (dict, list)) else a for a in answers
        ]
        disagreement = vote_disagreement(votes)
        top, top_count = Counter(votes).most_common(1)[0]
        result: dict[str, Any] = {
            "n_samples": len(votes),
            "plurality_answer": top,
            "disagreement": round(float(disagreement), 6),
            "agreement": round(1.0 - float(disagreement), 6),
            "plurality_vote_fraction": round(top_count / len(votes), 6),
        }
        prob = args.get("prob")
        if prob is not None:
            try:
                dec = self_consistency_decision(float(prob), float(disagreement))
            except (TypeError, ValueError) as exc:
                raise ToolError(f"'prob' must be a number in [0,1]: {exc}") from exc
            result["decision"] = {
                "decision": dec.decision,
                "widened_prob": round(float(dec.widened_prob), 6),
                "abstained": dec.abstained,
            }
        return json.dumps(result)

    @staticmethod
    def _tool_value_metrics(args: dict[str, Any]) -> str:
        from omni_mercury_engine.intel.value_metrics import VALUE_METRICS, get_value_metric

        stream = args.get("stream")
        if stream:
            if not isinstance(stream, str):
                raise ToolError("'stream' must be a string")
            try:
                metric = get_value_metric(stream)
            except KeyError as exc:
                raise ToolError(str(exc)) from exc
            return json.dumps(metric.as_dict())
        return json.dumps({"streams": {k: v.as_dict() for k, v in VALUE_METRICS.items()}})

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


def _verifier_block_message(blocked: list[dict[str, Any]]) -> str:
    """Human-readable reason a verifier-in-the-loop refused an emission."""
    reasons = "; ".join(str(c.get("reason") or c.get("claim_text", "")) for c in blocked)
    return (
        f"verifier-in-the-loop blocked this emission: {len(blocked)} oracle-refuted "
        f"claim(s): {reasons}. Set MERCURY_VERIFIER_MODE=soft to flag instead of block."
    )


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
