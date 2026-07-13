# Model-integration observability: proving Mercury ⇄ Claude end-to-end

This directory is a committed, re-runnable proof that Mercury Agent's LLM
integration works end-to-end when an operator wires in the **Anthropic (Claude)
Messages API** as the reasoning engine — the exact scenario of *"a user wants to
use the Claude API with Mercury Agent."*

It answers one question honestly: **how much of the Claude integration actually
works, and what is the single thing standing between it and a live model call?**

## What is proven, and how

Mercury's cloud LLM adapters are SSRF-hardened: every request goes through
`SafeHTTPClient` with `user_configured=True`, which **blocks loopback and
private IPs and rejects plain `http://`**. That is correct security, and it has
a consequence for testing — the cloud adapter can only ever talk to a genuine
public endpoint, so a local mock server cannot stand in for Anthropic through
the real code path. The proof is therefore split into two honest halves:

| Leg | What is real | What is substituted | What it proves |
|---|---|---|---|
| **Request format** | The adapter's request construction (`AnthropicCloudAdapter.generate`), headers, body, SSRF gate | The network return value (a byte-accurate Anthropic 200 payload) | The bytes Mercury sends conform to the documented Messages API contract |
| **Response parse + usage** | The adapter's parse path + `UsageLedger` booking | same | Mercury extracts `content[0].text` and books provider-reported `input_tokens`/`output_tokens` |
| **Error path** | The adapter's `requests.HTTPError` handling | a raised 401 | Provider auth errors are surfaced, not masked |
| **Live transport** | **Everything** — DNS pin, TLS, POST to `https://api.anthropic.com/v1/messages`, HTTPError handling | **nothing** (real endpoint, invalid key) | The request genuinely reaches Claude and is auth-gated (proving transport + endpoint are real) |
| **Full reasoning chain** | `RemoteReasoningBackend` → dual ethics gate → `FallbackLLMChain` → `AnthropicCloudAdapter` | network return value only | Mercury's own reasoning surface routes to Claude, gates it, stamps provenance, accounts usage, and fails closed under the air-gap |

The network return value is the *only* thing substituted, and only where a
valid credential would otherwise be required. Every line of Mercury's own
integration code — request building, parsing, usage accounting, ethics gating,
provenance, air-gap enforcement — runs for real.

## The one flip to a live model call

Both scripts detect `ANTHROPIC_API_KEY`. With it set, the substituted network
leg is replaced by a **real Claude completion** — no code change, no different
path. That is the whole point: the integration is complete, and a valid key is
the single missing input.

```bash
# Fixture + live-transport mode (no key needed): proves everything except model weights
python research/model_integration_obsv/anthropic_wire_proof.py
python research/model_integration_obsv/mercury_claude_reasoning_e2e.py

# Live Claude mode: real model calls end-to-end
export ANTHROPIC_API_KEY=sk-ant-...            # your key
export MERCURY_ANTHROPIC_MODEL=claude-opus-4-8 # optional; this is the default
python research/model_integration_obsv/anthropic_wire_proof.py
python research/model_integration_obsv/mercury_claude_reasoning_e2e.py
```

Or through Mercury's public surface, exactly as an operator would:

```python
from omni_mercury_engine.models.foundation.llm_adapter import LLMConfig, LLMProvider
from omni_mercury_engine.models.foundation.llm_usage import UsageLedger
from omni_mercury_engine.reasoning.backends import RemoteReasoningBackend
from omni_mercury_engine.reasoning.schemas import ReasoningContext

ledger = UsageLedger()
backend = RemoteReasoningBackend(
    cloud_config=LLMConfig(provider=LLMProvider.ANTHROPIC, model_name="claude-opus-4-8"),
    usage_ledger=ledger,
)  # reads ANTHROPIC_API_KEY from the environment
print(backend.model)  # -> "cloud:anthropic" once a key is present and Ollama is absent
expl = backend.explain(ReasoningContext(summary="...", domain="infrastructure",
                                        evidence={...}, severity=0.7, anomaly_prob=0.9))
print(expl.text, ledger.totals())
```

## Scripts

- **`anthropic_wire_proof.py`** — the adapter in isolation (request format,
  parse robustness, error path, live transport). Exits `0` only if every
  assertion held.
- **`mercury_claude_reasoning_e2e.py`** — the full Mercury reasoning chain
  served by Claude (routing, provenance, ethics enforcement, usage accounting,
  air-gap fail-closed, truthful fallback). Exits `0` only if every assertion held.
- **`live_llm_smoke.py`** — the operator's recurring "is Mercury doing real LLM
  generation *right now*?" check. Auto-detects whatever real backend is present
  — a local **Ollama** server *or* an **Anthropic** key — runs a genuine
  ethics-gated `explain()` through Mercury's chain, and prints the model's
  actual prose + token usage. Exits `0` if a real model answered, `2` if only
  the deterministic template was available (and prints the runbook below).

## Runbook — turn on a live model proof (operator-supplied, no shared secret)

Like any LLM-backed system, the model endpoint is the operator's to provide.
Mercury needs **one** of these; `live_llm_smoke.py` flips to a live completion
the moment either is present:

**(A) Ollama — zero cost, zero secret (recommended for dev/CI):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &            # loopback daemon on 127.0.0.1:11434 (SafeHTTPClient loopback-gated)
ollama pull llama3.2:1b   # any chat model; ~1.3 GB
python research/model_integration_obsv/live_llm_smoke.py   # -> LIVE ollama:llama3.2:1b completion
```

**(B) Anthropic (Claude) key** — there is **no anonymous/free key**; create one at
`https://console.claude.com/` → API keys (a workspace-scoped key with a spend
cap keeps it off your personal account):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export MERCURY_ANTHROPIC_MODEL=claude-opus-4-8   # optional; the default
python research/model_integration_obsv/live_llm_smoke.py            # -> LIVE cloud:anthropic completion
python research/model_integration_obsv/anthropic_wire_proof.py      # PART 4 becomes a live completion
python research/model_integration_obsv/mercury_claude_reasoning_e2e.py
```

Any other shipped provider works the same way via its own key
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, `COHERE_API_KEY`, `DEEPSEEK_API_KEY`,
`XAI_API_KEY`, …) — see `models/llm_registry.PROVIDER_CATALOG`.

## Related fix shipped alongside this harness

`AnthropicCloudAdapter`'s fallback model default was
`claude-3-5-sonnet-20241022`, a model **retired on 2025-10-28** that now returns
`404 not_found_error`. Any operator who selected the Anthropic provider without
naming a model hit a hard 404 on the first call. The default is now
`claude-opus-4-8` (a current first-party model); operators still override it via
`LLMConfig.model_name`. See `src/omni_mercury_engine/models/foundation/ollama_adapter.py`.

## Results

<!-- RESULTS:START — filled in from the actual runs on this branch -->
Recorded on this branch (Python 3.11, `mercury-agent[ml,llm,api,benchmark,dev]`
plus the AMA-Cryptography v3.3.0 native PQC backend). **No live
`ANTHROPIC_API_KEY` was available in the validation environment**, so the two
scripts ran in fixture + live-transport mode; every assertion held.

`anthropic_wire_proof.py` — **ALL ASSERTIONS PASSED**
- POST target is `https://api.anthropic.com/v1/messages`
- headers carry `x-api-key` + `anthropic-version: 2023-06-01` + JSON content-type
- SafeHTTPClient SSRF gate engaged (`user_configured=True`)
- request body matches the Messages API contract (`model`/`max_tokens`/`messages`/`system`)
- `content[0].text` extracted exactly; usage ledger booked input=214 / output=63
- empty-content → empty string (usage still booked); absent-usage → unmetered
- 401 → `"API error: invalid x-api-key"` (provider message preserved)
- **live transport**: the real adapter reached the genuine `api.anthropic.com`
  and was auth-gated (`API error: invalid x-api-key`) — proving transport + the
  real endpoint work; only a valid key is missing.

`mercury_claude_reasoning_e2e.py` — **ALL ASSERTIONS PASSED**
- chain routed to Claude and reported it truthfully (`model='cloud:anthropic'`)
- `explain` / `propose_hypotheses` (3 parsed) / `synthesize_report` all returned
  provenance-stamped, gated Mercury shapes
- usage ledger threaded through the chain (calls=3, total_tokens=783)
- dual ethical gate invoked once per op with the correct
  boundary/domain/severity/anomaly_prob; a denial raised with **zero** network
  calls (fail-closed, nothing surfaced)
- air-gap: under `MERCURY_OFFLINE` the chain refused to construct a cloud
  adapter and a direct remote call raised
- truthful fallback: no key → `model='template'`, no false Claude claim

Bottom line: the entire Mercury⇄Claude integration is verified end-to-end
against real code and the real endpoint. The single missing input for a live
model completion is a valid `ANTHROPIC_API_KEY`.
<!-- RESULTS:END -->
