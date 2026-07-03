# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""General-purpose agent capabilities for Mercury.

Mercury's specialty is calibrated multi-domain anomaly detection, but a free,
potentially life-saving agent should also be *generally useful*: able to research
the open web when one is reachable, read and synthesize sources, and produce
documents -- not only score anomalies.

This package adds those capabilities using **only the Python standard library +
numpy** (no new third-party dependencies, no language-model service): every
behaviour here is native, deterministic, and fail-closed (it degrades honestly
when the network is unavailable rather than fabricating results).

Capabilities:
- :class:`~omni_mercury_engine.agentic.capabilities.web_research.WebResearcher`
  -- native HTTP fetch + HTML->text extraction + best-effort web search.
- :class:`~omni_mercury_engine.agentic.capabilities.text_synthesis.ExtractiveSynthesizer`
  -- numpy-only extractive summarization + keyword extraction (no LLM).
- :class:`~omni_mercury_engine.agentic.capabilities.document_generator.DocumentGenerator`
  -- Markdown / HTML / plain-text document and report generation.
- :class:`~omni_mercury_engine.agentic.capabilities.assistant.GeneralAssistant`
  -- ties them into a research -> synthesize -> document workflow, governed by
  the same fail-closed benevolence gate as the rest of Mercury.
"""

from __future__ import annotations

from omni_mercury_engine.agentic.capabilities.assistant import (
    GeneralAssistant,
    ResearchReport,
)
from omni_mercury_engine.agentic.capabilities.document_generator import (
    Document,
    DocumentGenerator,
)
from omni_mercury_engine.agentic.capabilities.text_synthesis import ExtractiveSynthesizer
from omni_mercury_engine.agentic.capabilities.web_research import (
    FetchResult,
    SearchProvider,
    SearchResult,
    WebResearcher,
)

__all__ = [
    "Document",
    "DocumentGenerator",
    "ExtractiveSynthesizer",
    "FetchResult",
    "GeneralAssistant",
    "ResearchReport",
    "SearchProvider",
    "SearchResult",
    "WebResearcher",
]
