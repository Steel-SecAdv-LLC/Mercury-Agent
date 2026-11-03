# Interconnecting with Mercury — the MCP server

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

Mercury Agent is **interconnectable**: any AI system that speaks the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io) can link up to
it and run its capabilities, with no Mercury-specific client or SDK on the other
side. MCP is the contract; Mercury ships the server.

This is the universal path. If you are embedding Mercury in Python, the
in-process API (`from omni_mercury_engine.agentic import MercuryAgent`) and the
REST API (`mercury-agent serve`) are still there — but to let *another* AI system
(an MCP-capable desktop assistant, an IDE agent, an orchestrator) drive Mercury,
use the MCP server.

## Run it

```bash
mercury-agent mcp                       # serve MCP on stdio
# or, equivalently:
python -m omni_mercury_engine.mcp_server
```

The server speaks **newline-delimited JSON-RPC 2.0 over stdio** — the MCP stdio
transport. It is implemented with the Python **standard library only** (`json` +
`sys`): no `mcp` package, no web framework, **no new dependency**.

## Connect any MCP client

Point the client at the command. A typical MCP client config (the standard
`mcpServers` block most MCP clients accept) looks like:

```json
{
  "mcpServers": {
    "mercury": { "command": "mercury-agent", "args": ["mcp"] }
  }
}
```

The client performs the MCP handshake (`initialize` → `notifications/initialized`),
discovers Mercury's tools (`tools/list`), and invokes them (`tools/call`). The
`tools/list` payload is a self-describing **capability manifest** (each tool
carries a JSON-Schema `inputSchema`), also available programmatically via
`MercuryMCPServer().manifest()` for other transports.

## From the console and the API

`GET /api/v1/voice/backends` (anonymous, read-only) returns the same block for
each client (Claude Desktop and Claude Code, Cursor, VS Code with Copilot, any
stdio client) together with the ten language-model provider adapters the
engine ships, whether each provider's key is set on the server (never its
value), and whether the local Ollama runtime is reachable and with which
models. The console's **AI vendors** view (INTERFACES) shows the same: a
client card carries its configuration block and a Copy action; a provider
card carries the key variable's state, the Ollama models, the exact `export`
line and `mercury-agent voice --model` command. Keys are read by the engine
from its environment; the console never holds one.

Vendors that attach to *remote* MCP servers only (ChatGPT, Grok) cannot run
the stdio command; Mercury does not ship an HTTP MCP transport. Those vendors
reach Mercury the other way round, as model providers (`openai`, `xai`).

### Connecting a model provider from the console

An administrator (admin API key or session) connects a provider without
touching the engine host's environment: on the provider's card, **Connect**
stores the key (sealed at rest under `MERCURY_DATA_ENC_KEY` or
`AMA_MASTER_SEED`; never returned), the endpoint and the model
(`PUT /api/v1/voice/backends/{provider}`); **Test connection** makes one
read-only call and lists the models the provider answers with
(`POST /api/v1/voice/backends/{provider}/test`); **Use for the agent** attaches
the provider to the served voice and remembers it across restarts
(`POST /api/v1/voice/backends/select`; `DELETE` detaches); **Forget stored
settings** removes the stored key, endpoint and model (`DELETE
/api/v1/voice/backends/{provider}`). A change to the provider the voice is
using reaches the running voice at once, and a change that leaves it unable
to answer detaches it rather than keeping a model that cannot reply. The
twenty-one providers: OpenAI, Anthropic, xAI Grok, Google Gemini, Cohere, DeepSeek,
Mistral, Groq, Together, Fireworks, OpenRouter, Perplexity, Hugging Face,
Cursor, Azure OpenAI, any OpenAI-compatible endpoint, and the local Ollama,
LM Studio, vLLM and llama.cpp servers (loopback only, no key), plus the
built-in templates. What an attached model does is bounded: it composes the
free-text search answers from the records the retriever found and nothing
else; grounded answers (the record, the code, the narrative engine) never go
through it. `mercury-agent voice --model provider:model` attaches the same
way from the shell.

## Tools exposed

| Tool | What it does | Availability |
|---|---|---|
| `mercury_detect_anomaly` | Unsupervised batch anomaly detection over a numeric matrix (Mercury's statistical ensemble): per-row score, flag, threshold. | Always (numpy) |
| `mercury_detect_fusion` | Flagship neuro-symbolic anomaly detection over a numeric matrix via the full OmniMercuryEngine fusion path: trained fusion network, GOSNN scalar integration, and the `sigma_Immutable` hard ethical gate. Returns a calibrated probability, decision, severity, per-detector importances, and gate metadata. | ML-gated, fail-closed |
| `mercury_tier_detect` | Streaming detector-tier ensemble over a 1-D numeric series (torch-free): per-point calibrated probabilities and flags, cross-detector uncertainty, and — when `conformal_alpha` is set — flags with a distribution-free false-positive guarantee (FPR ≤ alpha). | Always (torch-free) |
| `mercury_localize_root_cause` | Graph-based root-cause attribution for a multivariate anomaly (torch-free): a reverse personalised random walk over a causal/service adjacency ranks which node originated the fault. | Always (torch-free) |
| `mercury_hazard_visualize` | Render a hazard detector's persisted diagnostics into a deterministic PNG (base64) or an RFC 7946 GeoJSON `FeatureCollection`; panels draw only what the detectors genuinely compute. | ML/hazard-gated |
| `mercury_score_ethics` | Score an action against Mercury's fail-closed benevolence/harm gate. | Always |
| `mercury_research` | Research a question on the open web → cited, extractive report (never fabricated). | Network-gated, fail-closed |
| `mercury_answer` | Answer a question with sentences extracted verbatim from sources, cited. | Network-gated, fail-closed |
| `mercury_write_document` | Generate a Markdown/HTML/text document from sections, benevolence-gated. | Always |
| `mercury_calibrate_confidence` | Fit the cross-validated, accept-gated confidence calibrator on (score, label) pairs and report measured out-of-fold ECE/Brier. | Always (numpy) |
| `mercury_verify_claims` | Route the checkable claims in a text through Mercury's oracle-validated verifiers (primality, Collatz, propositional tautology/SAT via a Tseitin→DPLL transform, physics dimensional consistency) and return a per-claim verdict. | Always |
| `mercury_check_provenance` | Enforce provenance at the output boundary: decide whether a candidate emission on a possibly hazardous topic may be emitted given its cited sources, using the weapons/mass-casualty gate to decide when attribution is required. | Always |
| `mercury_self_consistency` | Score N sampled reasoning-path answers for disagreement and apply the calibrated decision rule (widen confidence toward 0.5 with disagreement, abstain when the paths are too split). | Always |
| `mercury_value_metrics` | Return Mercury's intelligence-layer value board: each stream's declared, measured value metric with its baseline, target, and direction. | Always |

Every tool is **transparent and fail-closed**, exactly like the rest of Mercury:

- Outward tools (`research` / `answer` / `write_document`) pass the benevolence
  gate before acting and refuse harmful requests.
- The intel closed-loop tools (`verify_claims` / `check_provenance`) are the
  verifier-in-the-loop and the output-boundary provenance gate that Mercury
  applies to its own research/answer emissions before they leave the boundary.
- A tool whose backing stack is unavailable returns an MCP `isError` result
  explaining why — it never fabricates a capability it cannot deliver.
- The research tools degrade to a transparent "no sources reachable" offline
  rather than inventing answers.

## Offline / Ollama alignment

The MCP server inherits Mercury's offline-first posture:

- **Search** uses the [provider ladder](GENERAL_CAPABILITIES.md): point
  `MERCURY_SEARXNG_URL` at a self-hosted SearXNG (keyless) and Mercury's web
  research needs no third-party credential. Set `MERCURY_OFFLINE=1` for a hard
  air-gap.
- **Reasoning / harm classification** can ride Mercury's own local **Ollama**
  backend (`LocalReasoningBackend`, `reasoning_harm_classifier`) — no cloud call.

So an operator can run `mercury-agent mcp` behind a fully local stack (Ollama +
self-hosted SearXNG) and expose Mercury to any MCP-speaking AI with zero external
dependencies.

## Protocol notes

- Protocol version: `2024-11-05`.
- Methods: `initialize`, `ping`, `tools/list`, `tools/call`; the
  `notifications/initialized` notification is accepted (no response, per spec).
- Errors follow JSON-RPC 2.0 (`-32601` method-not-found, `-32602` invalid params,
  `-32700` parse error); tool-level failures come back as `tools/call` results
  with `isError: true` so the calling model can see and react to them.
