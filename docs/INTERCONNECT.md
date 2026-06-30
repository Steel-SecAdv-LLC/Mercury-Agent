# Interconnecting with Mercury — the MCP server

Mercury Agent is **interconnectable**: any AI system that speaks the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io) can link up to
it and run its capabilities, with no Mercury-specific client or SDK on the other
side. MCP is the contract; Mercury ships the server.

This is the universal path. If you are embedding Mercury in Python, the
in-process API (`from omni_mercury_engine.agentic import MercuryAgent`) and the
REST API (`mercury-agent serve`) are still there — but to let *another* AI system
(Claude Desktop/Code, an IDE agent, an orchestrator) drive Mercury, use the MCP
server.

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

Point the client at the command. For a Claude Desktop / Claude Code style client
config:

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

## Tools exposed

| Tool | What it does | Availability |
|---|---|---|
| `mercury_detect_anomaly` | Unsupervised batch anomaly detection over a numeric matrix (Mercury's statistical ensemble): per-row score, flag, threshold. | Always (numpy) |
| `mercury_score_ethics` | Score an action against Mercury's fail-closed benevolence/harm gate. | Always |
| `mercury_research` | Research a question on the open web → cited, extractive report (never fabricated). | Network-gated, fail-closed |
| `mercury_answer` | Answer a question with sentences extracted verbatim from sources, cited. | Network-gated, fail-closed |
| `mercury_write_document` | Generate a Markdown/HTML/text document from sections, benevolence-gated. | Always |
| `mercury_calibrate_confidence` | Fit the cross-validated, accept-gated confidence calibrator on (score, label) pairs and report measured out-of-fold ECE/Brier. | Always (numpy) |

Every tool is **honest and fail-closed**, exactly like the rest of Mercury:

- Outward tools (`research` / `answer` / `write_document`) pass the benevolence
  gate before acting and refuse harmful requests.
- A tool whose backing stack is unavailable returns an MCP `isError` result
  explaining why — it never fabricates a capability it cannot deliver.
- The research tools degrade to an honest "no sources reachable" offline rather
  than inventing answers.

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
