# Mercury General-Purpose Capabilities

Mercury's specialty is calibrated, multi-domain **anomaly detection**. But a free
and potentially life-saving agent should also be *generally useful* — able to
research the open web when one is reachable, read and synthesize sources, and
produce documents — not only score anomalies.

This document describes the general-purpose capability layer
(`omni_mercury_engine.agentic.capabilities`). It is built with the **Python
standard library + numpy only** — no new third-party dependencies and **no
language-model service**. Every behaviour is native, deterministic, and
fail-closed.

## What it is — and what it is not

| It **is** | It is **not** |
|---|---|
| A native, tool-using agent capability layer | A large language model / chatbot |
| Open-web fetch + HTML→text extraction (stdlib `urllib`/`html.parser`) | A scraping framework or headless browser |
| Best-effort keyless web search (DuckDuckGo html→lite chain) | A search-API client (no key, no SaaS dependency) |
| **Extractive** synthesis — ranks and quotes source sentences verbatim | A generative summarizer (it never paraphrases or invents text) |
| Markdown / HTML / plain-text document generation | A document editor or publishing pipeline |

Because Mercury ships no generative model, the assistant **extracts and
organizes** content sources actually contain and **cites** them — it does not
fabricate prose. This is a deliberate honesty contract.

## Components

- **`WebResearcher`** (`web_research.py`) — `fetch(url)`, `fetch_text(url)`,
  `extract_text(html)`, `search(query)`. Honours the environment's proxy/TLS via
  `urllib`. The transport is injectable, so the behaviour is fully testable
  offline. Fail-closed: a network error / non-OK status / disallowed scheme
  yields a `FetchResult` carrying the error (never a fabricated body); `search`
  returns `[]` with the reason logged.

### Search is a provider ladder, not a single scrape

Web search resolves through a **ranked ladder of providers**, tried in order,
each fail-closed. The recommended rungs are robust and operator-owned; HTML
scraping is the explicit last resort, **not** the default. From highest to
lowest priority:

1. **A keyed engine** — `brave_provider(api_key)` (Brave Search) or any
   `(query, max_results) -> [SearchResult]` you supply. Highest quality and
   contractually stable. Stdlib-only; no SDK dependency.
2. **A self-hosted SearXNG** — `searxng_provider(base_url)`. *Keyless* and
   *self-hostable*, so it adds no SaaS dependency and runs entirely under your
   control. This is the preferred default for an offline-leaning deployment:
   pair it with the local Ollama reasoning backend and Mercury's open-web
   research needs no third-party credential at all.
3. **Keyless DuckDuckGo HTML scrape** — a **best-effort fallback only**. Scrape
   endpoints rate-limit, serve challenges, and change markup without notice, so
   they are the bottom rung — enabled by default so a zero-config install still
   returns *something*, but never the recommended path. The fallback itself is
   robust (it queries the richer `html.duckduckgo.com` page, then the leaner
   `lite.duckduckgo.com`), and you can disable it entirely with
   `enable_ddg_fallback=False` for a provider-only posture.

Configure the ladder explicitly or from the environment:

```python
# Explicit (provider-first):
WebResearcher(search_providers=[brave_provider(key), searxng_provider(url)])

# From the environment (BRAVE_API_KEY / MERCURY_SEARXNG_URL /
# MERCURY_SEARCH_DDG_FALLBACK):
WebResearcher.from_env()
```

The legacy singular `WebResearcher(search_provider=…)` still fully replaces the
chain, for backward compatibility. Either way the contract is unchanged: any
failure (network, parse, or an exception from any provider) is fail-closed to
`[]` with the reason logged — Mercury never fabricates search hits.
- **`ExtractiveSynthesizer`** (`text_synthesis.py`) — numpy-only TF-IDF
  centroid sentence ranking (the LexRank/centroid family), `summarize`,
  `summarize_sources`, `keywords`, `relevance`. Returned summary sentences are
  copied verbatim from the input.
- **`DocumentGenerator`** (`document_generator.py`) — renders structured
  sections + cited sources to Markdown, HTML (escaped), or plain text.
  Deterministic, diffable output.
- **`GeneralAssistant`** (`assistant.py`) — orchestrates *research → synthesize →
  cited document*, plus `answer(question)` and `summarize_url(url)`.

## Ethics gate

Every outward action passes a **fail-closed harm gate** built on Mercury's own
`BenevolenceScorer` (the same scorer hardened in this PR with morphological +
curated-euphemism harm matching, an optional pluggable semantic classifier, and
severity/reversibility damping). The gate refuses an action when detected
harm or severity crosses `GeneralAssistant.HARM_REFUSAL_THRESHOLD` (0.5).

It gates on **detected harm**, not on a positive-benevolence floor: a benign,
neutral research query carries no harm and is permitted, while a query expressing
intent to injure/kill/destroy is refused. (The benevolence-floor model is
calibrated for detection *actions* and would false-reject all neutral research —
gating on harm is the correct, fail-closed semantics for a research/author
capability.)

## Usage

```python
from omni_mercury_engine.agentic.mercury_a_agent import MercuryAgent

agent = MercuryAgent(name="Mercury")
agent.enable_assistant()                     # native; no new deps, no LLM

report = agent.research("conformal prediction for anomaly detection")
if report.available:
    print(report.document.content)           # cited Markdown report
    for s in report.sources:                 # provenance for every source
        print(s["url"], s.get("relevance"))
else:
    print(report.note)                       # honest: refused / offline / no results

# Extractive answer (verbatim sentences from sources, cited):
print(agent.answer("what coverage guarantee does conformal prediction give?"))

# Document generation (Markdown/HTML/text), harm-gated:
doc = agent.write_document(
    "Incident note",
    [("Summary", "..."), ("Actions", "...")],
    fmt="markdown",
)
```

In a sandbox with no outbound network, `research`/`answer` return honestly with
`available=False` rather than fabricating results — the same fail-closed posture
as the rest of Mercury.
