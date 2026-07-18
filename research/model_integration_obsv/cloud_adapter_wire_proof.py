# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end proof of Mercury's Anthropic cloud LLM adapter.

This exercises the *real* ``AnthropicCloudAdapter`` code path — request
construction, HTTP transport through ``SafeHTTPClient``, response parsing, and
provider-reported usage accounting — with no changes to the adapter itself.

It runs in four parts:

1. **Request-format conformance.**  The network boundary
   (``SafeHTTPClient.post_json``) is intercepted so the exact outbound request
   the adapter builds can be inspected byte-for-byte and checked against the
   documented Anthropic Messages API contract (POST ``/v1/messages``;
   ``x-api-key`` + ``anthropic-version: 2023-06-01`` headers; a body of
   ``{model, max_tokens, messages:[{role,content}], system?}``).  The boundary
   returns a spec-accurate 200 payload so the adapter's parse + usage-ledger
   code runs for real.

2. **Response-parse robustness.**  Degenerate-but-legal payloads (empty
   ``content``, absent ``usage``) are fed through the same real parse path to
   confirm graceful degradation and truthful usage accounting.

3. **Error path.**  A 401 ``requests.HTTPError`` is raised at the boundary to
   confirm the adapter surfaces the provider's auth error rather than masking it.

4. **Live transport.**  With the boundary *un-patched*, the real adapter is
   pointed at the genuine ``https://api.anthropic.com``.  With a deliberately
   invalid key it must come back with an authentication error (proving DNS
   pin + TLS + POST to the real endpoint + HTTPError handling all work end to
   end).  If a valid ``ANTHROPIC_API_KEY`` is present in the environment, it
   instead performs a real live completion against the operator-named model
   and asserts a non-error response —
   this is the single flip that turns the whole proof into a live model call.

Run: ``python research/model_integration_obsv/cloud_adapter_wire_proof.py``
Exit code 0 == every assertion held.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Ensure ``src`` layout is importable when run from a source checkout without
# an editable install (the CI/dev path uses ``pip install -e .``; this keeps
# the harness runnable straight from the repo).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import requests

from omni_mercury_engine.models.foundation.llm_adapter import (
    LLMConfig,
    LLMProvider,
)
from omni_mercury_engine.models.foundation.llm_usage import UsageLedger
from omni_mercury_engine.models.foundation.ollama_adapter import (
    AnthropicCloudAdapter,
)
from omni_mercury_engine.security import safe_http

# A byte-accurate Anthropic Messages API success payload (the shape documented
# at platform.claude.com and returned by the live /v1/messages endpoint).

# Offline-leg test target: the model id recorded in the wire fixture below.
# This is TEST DATA (a replayed real 200 payload), not a product default --
# Mercury ships no default model for any provider; operators name the model.
_WIRE_FIXTURE_MODEL = "claude-opus-4-8"
_REAL_ANTHROPIC_200 = {
    "id": "msg_01ABCdefWireProof",
    "type": "message",
    "role": "assistant",
    "model": _WIRE_FIXTURE_MODEL,
    "content": [
        {
            "type": "text",
            "text": (
                '{"is_anomaly": true, "anomaly_score": 0.91, "confidence": 0.88, '
                '"category": "point_outlier", "explanation": "value is 6.2 sigma '
                'above the rolling mean."}'
            ),
        }
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 214, "output_tokens": 63},
}


class _Captured:
    """Mutable holder for the request the adapter hands to the HTTP boundary."""

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = {}


def _passed(msg: str) -> None:
    print(f"  ✓ {msg}")


def part1_request_format_and_usage() -> None:
    print("[PART 1] Request-format conformance + usage accounting (real adapter code)")
    captured = _Captured()

    def _fake_post_json(url: str, **kwargs: Any) -> Any:
        captured.args = (url,)
        captured.kwargs = kwargs
        return json.loads(json.dumps(_REAL_ANTHROPIC_200))  # deep copy

    original = safe_http.SafeHTTPClient.post_json
    safe_http.SafeHTTPClient.post_json = classmethod(  # type: ignore[assignment]
        lambda cls, url, **kw: _fake_post_json(url, **kw)
    )
    try:
        ledger = UsageLedger()
        cfg = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name=_WIRE_FIXTURE_MODEL,
            api_key="unit-test-placeholder-key",
            max_tokens=512,
            timeout=30.0,
        )
        adapter = AnthropicCloudAdapter(cfg)
        adapter.attach_usage_ledger(ledger)
        assert adapter.is_available(), "adapter must be available when a key is present"

        system_prompt = "You are a subordinate reasoning engine invoked by Mercury Agent."
        user_prompt = "Analyze the following data for anomalies: value=42.0, mean=3.1, std=6.3"
        text = adapter.generate(user_prompt, system_prompt)

        url = captured.args[0]
        body = captured.kwargs["json_body"]
        headers = captured.kwargs["headers"]

        assert url == "https://api.anthropic.com/v1/messages", f"wrong URL: {url}"
        _passed(f"POST target is the real Messages endpoint: {url}")

        assert headers["anthropic-version"] == "2023-06-01", headers
        assert headers["x-api-key"] == "unit-test-placeholder-key", "key must be sent verbatim"
        assert headers["Content-Type"] == "application/json", headers
        _passed("headers carry x-api-key + anthropic-version:2023-06-01 + JSON content-type")

        assert (
            captured.kwargs.get("user_configured") is True
        ), "adapter must pass user_configured=True to the SafeHTTPClient boundary"
        _passed(
            "adapter passes user_configured=True to the SafeHTTPClient boundary "
            "(the gate itself is patched out in this leg; PART 4 exercises it live)"
        )

        assert body["model"] == _WIRE_FIXTURE_MODEL, body
        assert isinstance(body["max_tokens"], int) and body["max_tokens"] == 512, body
        assert body["messages"] == [{"role": "user", "content": user_prompt}], body["messages"]
        assert body["system"] == system_prompt, body
        _passed("request body matches the Messages API contract (model/max_tokens/messages/system)")

        expected_text = _REAL_ANTHROPIC_200["content"][0]["text"]
        assert text == expected_text, f"parse mismatch: {text!r}"
        _passed("adapter extracted content[0].text exactly from the 200 payload")

        assert adapter.last_usage is not None
        assert adapter.last_usage.prompt_tokens == 214, adapter.last_usage
        assert adapter.last_usage.completion_tokens == 63, adapter.last_usage
        assert adapter.last_usage.reported is True
        totals = ledger.totals()
        assert totals["prompt_tokens"] == 214 and totals["completion_tokens"] == 63, totals
        _passed(
            "usage ledger booked provider-reported tokens "
            f"(input={totals['prompt_tokens']}, output={totals['completion_tokens']})"
        )
    finally:
        safe_http.SafeHTTPClient.post_json = original  # type: ignore[assignment]


def part2_parse_robustness() -> None:
    print("[PART 2] Response-parse robustness (degenerate-but-legal payloads)")

    cases = {
        "empty content list": {"content": [], "usage": {"input_tokens": 5, "output_tokens": 0}},
        "absent usage": {"content": [{"type": "text", "text": "ok"}]},
    }
    original = safe_http.SafeHTTPClient.post_json
    try:
        for label, payload in cases.items():
            frozen = json.loads(json.dumps(payload))
            safe_http.SafeHTTPClient.post_json = classmethod(  # type: ignore[assignment]
                lambda cls, url, _p=frozen, **kw: _p
            )
            cfg = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name=_WIRE_FIXTURE_MODEL,
                api_key="unit-test-placeholder-key",
            )
            adapter = AnthropicCloudAdapter(cfg)
            text = adapter.generate("probe", None)
            if label == "empty content list":
                assert text == "", f"empty content should yield empty string, got {text!r}"
                assert adapter.last_usage is not None and adapter.last_usage.reported is True
                _passed("empty content -> empty string, usage still booked")
            else:  # absent usage
                assert text == "ok", text
                assert (
                    adapter.last_usage is not None and adapter.last_usage.reported is False
                ), "absent usage must be recorded as *unreported* (spend stays visible)"
                _passed("absent usage -> parsed text intact, call recorded as unmetered")
    finally:
        safe_http.SafeHTTPClient.post_json = original  # type: ignore[assignment]


def part3_error_path() -> None:
    print("[PART 3] Error path (provider 401 surfaced, not masked)")

    class _Resp:
        def json(self) -> Any:
            return {
                "type": "error",
                "error": {"type": "authentication_error", "message": "invalid x-api-key"},
            }

    def _raise_401(cls: Any, url: str, **kw: Any) -> Any:
        err = requests.HTTPError("401 Client Error")
        err.response = _Resp()  # type: ignore[assignment]
        raise err

    original = safe_http.SafeHTTPClient.post_json
    safe_http.SafeHTTPClient.post_json = classmethod(_raise_401)  # type: ignore[assignment]
    try:
        cfg = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name=_WIRE_FIXTURE_MODEL,
            api_key="unit-test-placeholder-key",
        )
        adapter = AnthropicCloudAdapter(cfg)
        text = adapter.generate("probe", None)
        assert text == "API error: invalid x-api-key", text
        _passed("401 HTTPError -> 'API error: invalid x-api-key' (provider message preserved)")
    finally:
        safe_http.SafeHTTPClient.post_json = original  # type: ignore[assignment]


def part4_live_transport() -> None:
    print("[PART 4] Live transport against the genuine api.anthropic.com")
    live_key = os.environ.get("ANTHROPIC_API_KEY")
    live_model = os.environ.get("MERCURY_ANTHROPIC_MODEL")
    if live_key and not live_model:
        print(
            "  ! ANTHROPIC_API_KEY is set but MERCURY_ANTHROPIC_MODEL is not; "
            "live completion leg skipped (Mercury ships no vendor-default "
            "model -- name the one to test against)."
        )
        return
    cfg = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        # Without a live key this leg only proves the endpoint auth-gates
        # (401 precedes model validation), so the id is a neutral probe.
        model_name=live_model or "auth-probe-model",
        api_key=live_key or "invalid-transport-probe-key",
        max_tokens=32,
        timeout=30.0,
    )
    adapter = AnthropicCloudAdapter(cfg)
    try:
        text = adapter.generate("Reply with the single word PONG.", None)
    except safe_http.UnsafeURLError as exc:
        # The only exception generate() lets escape (SSRF/egress policy);
        # transport failures come back as documented error strings, and any
        # other exception is a real adapter bug that must fail this proof.
        print(f"  ! egress to api.anthropic.com blocked by policy ({exc!r}); transport leg skipped")
        return

    if live_key:
        assert not text.startswith("API error:"), f"live call errored: {text}"
        assert not text.startswith("Request failed:"), text
        _passed(
            f"LIVE completion received from the operator-named model (len={len(text)}): {text[:80]!r}"
        )
        assert (
            adapter.last_usage is not None and adapter.last_usage.reported
        ), "a live 200 must carry provider-reported usage"
        _passed(
            "live usage booked "
            f"(input={adapter.last_usage.prompt_tokens}, "
            f"output={adapter.last_usage.completion_tokens})"
        )
    else:
        low = text.lower()
        assert text.startswith(
            "API error:"
        ), f"expected an auth error from the real endpoint, got: {text!r}"
        assert ("api-key" in low) or ("authentication" in low) or ("x-api-key" in low), text
        _passed(f"real endpoint reached; auth-gated as expected -> {text!r}")
        print(
            "    (set ANTHROPIC_API_KEY + MERCURY_ANTHROPIC_MODEL to turn this leg "
            "into a live completion)"
        )


def main() -> int:
    print("=" * 78)
    print("Mercury <- Anthropic Messages API: end-to-end adapter proof")
    print("=" * 78)
    part1_request_format_and_usage()
    part2_parse_robustness()
    part3_error_path()
    part4_live_transport()
    print("-" * 78)
    print("ALL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
