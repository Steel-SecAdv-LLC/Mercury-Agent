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

### Why DuckDuckGo, and how search stays robust

Under the project's hard constraints — **no new dependency, no API key, standard
library only** — DuckDuckGo is the strongest general-web engine, because it
exposes *keyless* HTML endpoints we can `GET` and parse. Google (Programmable
Search JSON API), Bing (search API retired / keyed), and Brave (keyed) all
require an API key and account, i.e. a dependency *and* a credential. So the
keyless default is DuckDuckGo. Two design choices keep it from being fragile:

1. **Multi-endpoint chain.** `search()` queries the full `html.duckduckgo.com`
   page first (richer: titles + snippets), then falls back to the leaner
   `lite.duckduckgo.com` page, which is far more tolerant of non-browser
   clients. It returns the first endpoint that yields hits, so a challenge or
   empty body on one endpoint degrades to the other rather than to nothing.
2. **Pluggable provider.** `WebResearcher(search_provider=…)` accepts an
   injectable `(query, max_results) -> [SearchResult]`. A deployment that *does*
   hold a key for a higher-quality engine (Brave, Google, …) can slot it in and
   it fully replaces the built-in chain — **without this module taking a
   dependency**. The default stays keyless; the ceiling is whatever engine the
   operator chooses to supply.

Either way the contract is unchanged: any failure (network, parse, or an
exception from a custom provider) is fail-closed to `[]` with the reason logged
— Mercury never fabricates search hits.
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
`BenevolenceScorer` (the same scorer hardened in this PR with semantic harm
matching and severity/reversibility). The gate refuses an action when detected
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
