# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""One-command "is Mercury doing real LLM generation right now?" validator.

An operator's live LLM is *their* credential/endpoint to supply — an Anthropic
(or other cloud) API key, or a local Ollama server, exactly as any LLM-backed
system requires. This script answers the recurring question **"with whatever
backend I have, does Mercury's governed reasoning chain produce a real model
completion end-to-end?"** — and it upgrades itself automatically as backends
become available. No secret is hard-coded; nothing is mocked.

It probes both real backends Mercury ships an operator would use here:

* **Ollama** (offline-first, zero-cost, zero-secret) via ``LocalReasoningBackend``
  — the chain serves a local model if ``ollama serve`` is up and the model is
  pulled, else the deterministic builtin template.
* **Anthropic (Claude)** via ``RemoteReasoningBackend`` — live when
  ``ANTHROPIC_API_KEY`` is set.

For every backend that resolves to a *real model* (``ollama:*`` or ``cloud:*``)
it runs a genuine, ethics-gated ``explain()`` and prints the model, the model's
actual prose, and the provider-reported token usage. If only the template is
available it says so plainly and prints the exact runbook to turn a live model
on. Exit code: ``0`` if at least one real model answered, ``2`` if only the
template was available (so CI can require a live backend when it wants one).

Run: ``python research/model_integration_obsv/live_llm_smoke.py``
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from omni_mercury_engine.models.foundation.llm_adapter import LLMConfig, LLMProvider
from omni_mercury_engine.models.foundation.llm_usage import UsageLedger
from omni_mercury_engine.reasoning.backends import (
    LocalReasoningBackend,
    RemoteReasoningBackend,
)
from omni_mercury_engine.reasoning.schemas import ReasoningContext

_CTX = ReasoningContext(
    summary="Point outlier flagged by the statistical detector tier",
    domain="infrastructure",
    evidence={"value": 42.0, "rolling_mean": 3.1, "rolling_std": 6.3, "z": 6.2},
    severity=0.7,
    anomaly_prob=0.91,
)

_RUNBOOK = """\
No live model backend was reachable, so Mercury served its deterministic
builtin template (a real, honest offline fallback -- but not a model). To turn
on a live end-to-end proof, give Mercury ONE of:

  (A) A local Ollama model  [zero cost, zero secret -- recommended for dev]
        curl -fsSL https://ollama.com/install.sh | sh
        ollama serve &                 # loopback daemon on 127.0.0.1:11434
        ollama pull llama3.2:1b        # ~1.3 GB; any chat model works
      then re-run this script.

  (B) An Anthropic (Claude) API key
        # Create one at https://console.claude.com/ -> API keys.
        # There is no anonymous/free key: a key is billed to an account. You
        # can mint a WORKSPACE-scoped key with a spend cap so it is not your
        # personal umbrella key.
        export ANTHROPIC_API_KEY=sk-ant-...
        export MERCURY_ANTHROPIC_MODEL=claude-opus-4-8   # optional; the default
      then re-run this script.

Either one flips every leg below from 'template' to a real, ethics-gated model
completion. Mercury's integration is complete; the model endpoint is the
operator-supplied dependency it is designed to be.
"""


def _is_real_model(model_label: str) -> bool:
    """A chain that served a real model reports ``ollama:*`` or ``cloud:*``."""
    return model_label.startswith("ollama:") or model_label.startswith("cloud:")


def _exercise(title: str, backend: object) -> bool:
    """Run a live explain() if the backend resolved to a real model."""
    model = backend.model  # type: ignore[attr-defined]
    print(f"[{title}] active backend model = {model!r}")
    if not _is_real_model(model):
        print(f"  - no live model here (served {model!r}); skipping the live call")
        return False
    explanation = backend.explain(_CTX)  # type: ignore[attr-defined]
    if explanation.text.startswith(("API error:", "Request failed:")):
        print(f"  ! backend reachable but the call failed: {explanation.text}")
        return False
    print(
        f"  ✓ LIVE {model} completion (gated={explanation.gated}, backend={explanation.backend!r})"
    )
    print(f"    model said: {explanation.text.strip()[:200]!r}")
    return True


def main() -> int:
    print("=" * 78)
    print("Mercury governed reasoning chain -- live LLM smoke (auto-detects backend)")
    print("=" * 78)

    live_any = False

    # (A) Ollama / offline-first local chain.
    local_ledger = UsageLedger()
    local = LocalReasoningBackend(usage_ledger=local_ledger)
    live_any |= _exercise("Ollama / local", local)
    if local_ledger.totals()["calls"]:
        print(f"    local usage: {local_ledger.totals()}")

    # (B) Anthropic (Claude) cloud path.
    remote_ledger = UsageLedger()
    remote = RemoteReasoningBackend(
        cloud_config=LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name=os.environ.get("MERCURY_ANTHROPIC_MODEL", "claude-opus-4-8"),
        ),
        usage_ledger=remote_ledger,
    )
    live_any |= _exercise("Anthropic / Claude", remote)
    if remote_ledger.totals()["calls"]:
        print(f"    remote usage: {remote_ledger.totals()}")

    print("-" * 78)
    if live_any:
        print("RESULT: at least one REAL model produced a governed completion end-to-end.")
        return 0
    print("RESULT: no live model backend available; Mercury served the template.\n")
    print(_RUNBOOK)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
