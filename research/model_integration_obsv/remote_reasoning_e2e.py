# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end proof of the full Mercury reasoning chain served by Claude.

Where ``cloud_adapter_wire_proof.py`` proves the adapter in isolation, this proves
the *whole* path a Mercury operator uses when they wire Claude in as the
reasoning engine:

    RemoteReasoningBackend            (Mercury owns the loop + provenance)
      -> dual ethical gate            (benevolence + sigma_Immutable, fail-closed)
      -> FallbackLLMChain             (offline-first; Ollama -> cloud -> template)
      -> AnthropicCloudAdapter        (real request construction + parse)
      -> SafeHTTPClient               (SSRF-gated egress)
      -> Claude                       (api.anthropic.com)

It asserts, on the real Mercury code:

  1. Routing + provenance — with cloud enabled, a key present, and no local
     Ollama, the chain selects the Anthropic adapter and reports it truthfully
     (``backend.model == "cloud:anthropic"``); ``explain`` / ``propose_hypotheses``
     / ``synthesize_report`` all return provenance-stamped, gated Mercury shapes.
  2. Ethics enforcement is real, not decorative — the dual gate runs once per
     reasoning op with the correct boundary, and when it denies, the call
     raises and **no** network request is made (nothing is surfaced).
  3. Usage accounting threads through the chain into the shared ledger.
  4. Air-gap fail-closed — under ``MERCURY_OFFLINE`` a direct remote call
     raises rather than silently substituting a weaker local answer.
  5. Truthful fallback — with no key and no Ollama the chain serves the
     template and says so; it never claims to be Claude when it is not.

If ``ANTHROPIC_API_KEY`` is set, part 1 additionally performs a real Claude
``explain`` and prints the model's actual analyst prose.

Run: ``python research/model_integration_obsv/remote_reasoning_e2e.py``
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from omni_mercury_engine.cognitive.ethical_bounding import (
    EthicalConstraintViolationError,
)
from omni_mercury_engine.models.foundation.llm_adapter import (
    LLMConfig,
    LLMProvider,
)
from omni_mercury_engine.models.foundation.llm_usage import UsageLedger
from omni_mercury_engine.reasoning import backend as backend_mod
from omni_mercury_engine.reasoning.backends import RemoteReasoningBackend
from omni_mercury_engine.reasoning.schemas import ReasoningContext
from omni_mercury_engine.security import safe_http

_EXPLANATION_TEXT = (
    "The reading of 42.0 sits 6.2 standard deviations above the rolling mean of "
    "3.1, well outside any plausible sensor-noise band. Nothing in the evidence "
    "explains a jump of that size as benign, so the point should be treated as a "
    "genuine anomaly and escalated for review."
)


def _anthropic_payload(text: str) -> dict[str, Any]:
    return {
        "id": "msg_01ReasonChain",
        "type": "message",
        "role": "assistant",
        "model": _WIRE_FIXTURE_MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 190, "output_tokens": 71},
    }


def _passed(msg: str) -> None:
    print(f"  ✓ {msg}")


def _ctx() -> ReasoningContext:
    return ReasoningContext(
        summary="Point outlier flagged by the statistical detector tier",
        domain="infrastructure",
        evidence={"value": 42.0, "rolling_mean": 3.1, "rolling_std": 6.3, "z": 6.2},
        severity=0.7,
        anomaly_prob=0.91,
    )


# Offline-leg test target: the model id recorded in the wire fixture below.
# This is TEST DATA (a replayed real 200 payload), not a product default --
# Mercury ships no default model for any provider; operators name the model.
_WIRE_FIXTURE_MODEL = "claude-opus-4-8"


def part1_routing_provenance_and_usage() -> None:
    print("[PART 1] Routing + provenance + governed Mercury shapes (real chain)")
    live_key = os.environ.get("ANTHROPIC_API_KEY")
    live_model = os.environ.get("MERCURY_ANTHROPIC_MODEL")
    if live_key and not live_model:
        # Same operator contract as cloud_adapter_wire_proof.py PART 4:
        # Mercury ships no vendor-default model, so a live call is never
        # made with a model the operator did not name. The offline legs
        # below still run in full against the mocked boundary.
        print(
            "  ! ANTHROPIC_API_KEY is set but MERCURY_ANTHROPIC_MODEL is not; "
            "running the offline (mocked-boundary) legs only. Name the model "
            "to turn this into a live proof."
        )
        live_key = None
    ledger = UsageLedger()
    cfg = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        # The fixture id is only ever used against the mocked boundary; a
        # live run requires the operator-named MERCURY_ANTHROPIC_MODEL.
        model_name=live_model or _WIRE_FIXTURE_MODEL,
        api_key=live_key or "unit-test-placeholder-key",
        max_tokens=400,
    )
    backend = RemoteReasoningBackend(cloud_config=cfg, usage_ledger=ledger)

    assert backend.name == "remote", backend.name
    assert backend.is_offline is False
    assert (
        backend.model == "cloud:anthropic"
    ), f"chain must select + truthfully report the Anthropic adapter, got {backend.model!r}"
    _passed(f"chain routed to Claude and reports it truthfully: model={backend.model!r}")

    original = safe_http.SafeHTTPClient.post_json

    if live_key:
        explanation = backend.explain(_ctx())
        assert not explanation.text.startswith("API error:"), explanation.text
        _passed(f"LIVE Claude explanation: {explanation.text[:110]!r}...")
    else:
        payload = json.loads(json.dumps(_anthropic_payload(_EXPLANATION_TEXT)))
        safe_http.SafeHTTPClient.post_json = classmethod(  # type: ignore[assignment]
            lambda cls, url, _p=payload, **kw: _p
        )
        try:
            explanation = backend.explain(_ctx())
        finally:
            safe_http.SafeHTTPClient.post_json = original  # type: ignore[assignment]
        assert explanation.text == _EXPLANATION_TEXT, explanation.text
        _passed("explain() returned the model's prose through the full chain")

    assert explanation.backend == "remote", explanation.backend
    assert explanation.model == "cloud:anthropic", explanation.model
    assert explanation.gated is True, "the reasoning op must be governed by Mercury's gate"
    _passed(
        f"Explanation is provenance-stamped + gated "
        f"(backend={explanation.backend!r}, model={explanation.model!r}, gated={explanation.gated})"
    )

    # propose_hypotheses + synthesize_report over the same chain.
    if not live_key:
        hyp_payload = json.loads(
            json.dumps(
                _anthropic_payload("Sensor fault\nGenuine physical event\nData pipeline glitch")
            )
        )
        safe_http.SafeHTTPClient.post_json = classmethod(  # type: ignore[assignment]
            lambda cls, url, _p=hyp_payload, **kw: _p
        )
        try:
            hyps = backend.propose_hypotheses(_ctx())
            report = backend.synthesize_report(_ctx())
        finally:
            safe_http.SafeHTTPClient.post_json = original  # type: ignore[assignment]
        assert [h.statement for h in hyps] == [
            "Sensor fault",
            "Genuine physical event",
            "Data pipeline glitch",
        ], hyps
        _passed(f"propose_hypotheses() parsed {len(hyps)} candidate hypotheses (one per line)")
        assert report.title == "Mercury reasoning report: infrastructure", report.title
        assert report.backend == "remote" and report.model == "cloud:anthropic"
        _passed(f"synthesize_report() returned a provenance-stamped Report: {report.title!r}")

    totals = ledger.totals()
    assert totals["calls"] >= 1 and totals["total_tokens"] > 0, totals
    _passed(
        "usage ledger threaded through the chain "
        f"(calls={totals['calls']}, total_tokens={totals['total_tokens']})"
    )


def part2_ethics_gate_is_enforced() -> None:
    print("[PART 2] Dual ethical gate is real (invoked per op; denial surfaces nothing)")
    cfg = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name=_WIRE_FIXTURE_MODEL,
        api_key="unit-test-placeholder-key",
    )
    backend = RemoteReasoningBackend(cloud_config=cfg)

    calls: list[dict[str, Any]] = []
    network_hits = {"n": 0}
    original_gate = backend_mod.enforce_dual_ethical_gate
    original_post = safe_http.SafeHTTPClient.post_json

    def _record_gate(**kwargs: Any) -> None:
        calls.append(kwargs)

    def _count_post(cls: Any, url: str, **kw: Any) -> Any:
        network_hits["n"] += 1
        return _anthropic_payload("ok")

    backend_mod.enforce_dual_ethical_gate = _record_gate  # type: ignore[assignment]
    safe_http.SafeHTTPClient.post_json = classmethod(_count_post)  # type: ignore[assignment]
    try:
        backend.explain(_ctx())
        assert len(calls) == 1, calls
        assert calls[0]["boundary"] == "reasoning_backend.explain", calls[0]
        assert "infrastructure" in calls[0]["domain"], calls[0]
        assert calls[0]["severity"] == 0.7 and calls[0]["anomaly_prob"] == 0.91, calls[0]
        _passed("gate invoked once with correct boundary/domain/severity/anomaly_prob")

        # Now make the gate DENY and prove nothing is surfaced.
        def _deny_gate(**kwargs: Any) -> None:
            raise EthicalConstraintViolationError(
                action="unit:forced-denial",
                score=0.0,
                threshold=0.99,
                check="benevolence",
            )

        backend_mod.enforce_dual_ethical_gate = _deny_gate  # type: ignore[assignment]
        hits_before = network_hits["n"]
        raised = False
        try:
            backend.explain(_ctx())
        except EthicalConstraintViolationError:
            raised = True
        assert raised, "a gate denial must raise, not return content"
        assert (
            network_hits["n"] == hits_before
        ), "no network request may be made after a gate denial (nothing surfaced)"
        _passed("gate denial raises and makes zero network calls (fail-closed, no output)")
    finally:
        backend_mod.enforce_dual_ethical_gate = original_gate  # type: ignore[assignment]
        safe_http.SafeHTTPClient.post_json = original_post  # type: ignore[assignment]


def part3_airgap_fail_closed() -> None:
    print("[PART 3] Air-gap fail-closed (MERCURY_OFFLINE)")
    prev = os.environ.get("MERCURY_OFFLINE")
    os.environ["MERCURY_OFFLINE"] = "1"
    try:
        from omni_mercury_engine.reasoning.backends import (
            ReasoningBackendUnavailableError,
        )

        cfg = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name=_WIRE_FIXTURE_MODEL,
            api_key="unit-test-placeholder-key",
        )
        backend = RemoteReasoningBackend(cloud_config=cfg, ethics_enabled=False)
        # Under the air-gap the chain never even constructs the cloud adapter.
        assert backend.model == "template", backend.model
        _passed("under air-gap the chain refuses to construct a cloud adapter (model=template)")

        raised = False
        try:
            backend.explain(_ctx())
        except ReasoningBackendUnavailableError:
            raised = True
        assert raised, "a direct remote reasoning call under the air-gap must raise"
        _passed("direct remote reasoning under MERCURY_OFFLINE raises (no silent substitution)")
    finally:
        if prev is None:
            os.environ.pop("MERCURY_OFFLINE", None)
        else:
            os.environ["MERCURY_OFFLINE"] = prev


def part4_truthful_fallback() -> None:
    print("[PART 4] Truthful fallback (no key, no Ollama -> template, said plainly)")
    prev = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        cfg = LLMConfig(
            provider=LLMProvider.ANTHROPIC, model_name=_WIRE_FIXTURE_MODEL, api_key=None
        )
        backend = RemoteReasoningBackend(cloud_config=cfg, ethics_enabled=False)
        assert (
            backend.model == "template"
        ), f"with no key the chain must fall to template, not claim Claude, got {backend.model!r}"
        _passed(
            "no credential -> chain serves template and reports model='template' (no false claim)"
        )
    finally:
        if prev is not None:
            os.environ["ANTHROPIC_API_KEY"] = prev


def main() -> int:
    print("=" * 78)
    print("Mercury reasoning chain <- Claude: full end-to-end proof")
    print("=" * 78)
    part1_routing_provenance_and_usage()
    part2_ethics_gate_is_enforced()
    part3_airgap_fail_closed()
    part4_truthful_fallback()
    print("-" * 78)
    print("ALL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
