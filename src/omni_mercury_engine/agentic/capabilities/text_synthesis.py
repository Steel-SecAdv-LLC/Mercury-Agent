# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extractive text synthesis -- numpy-only, no language model.

Mercury has no generative language model, so it does not *write* prose. What it
can do honestly is **extract**: rank the sentences a source already contains by
how central they are to the document, and return the most representative ones
verbatim. This is classic centroid/TF-based extractive summarization (the family
behind TextRank/LexRank), implemented here with numpy + the standard library so
there is no dependency and the output is deterministic.

The contract is honest: every returned sentence is copied verbatim from the
input -- nothing is paraphrased, invented, or hallucinated.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# A sentence gate maps one sentence to True (safe to emit) / False (redact).
# Injectable so this module stays dependency-free: the caller wires it to the
# weapons/mass-casualty output gate (assess_weapons_uplift) without this file
# importing the ethics layer.
SentenceGate = Callable[[str], bool]

#: Placeholder substituted for a redacted sentence, so the output is honest
#: about *where* content was withheld rather than silently dropping it.
REDACTION_NOTICE = "[redacted: operational content withheld by the harm gate]"

# A compact English stopword list (stdlib-only; no nltk dependency).
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "their",
        "his",
        "her",
        "our",
        "your",
        "my",
        "we",
        "you",
        "they",
        "he",
        "she",
        "them",
        "us",
        "i",
        "me",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "not",
        "no",
        "nor",
        "so",
        "than",
        "too",
        "very",
        "just",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "s",
        "t",
        "don",
        "now",
        "also",
        "which",
        "who",
        "whom",
        "what",
    ]
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[a-z][a-z'-]+")


@dataclass
class ExtractiveSynthesizer:
    """Rank-and-extract sentences/keywords from text (deterministic, no LLM).

    Args:
        min_sentence_chars: Sentences shorter than this are ignored as fragments.
        sentence_gate: Optional ``Callable[[str], bool]`` -- the pre-emission
            output gate (spec §5.3). The verbatim extractor is the single
            highest-risk component: a benign query can surface a source
            sentence that IS operational weapons procedure, and a faithful
            quoter would reproduce it. When set, every candidate sentence is
            passed through the gate before it can appear in a summary; a
            sentence the gate rejects (returns ``False``) is replaced with
            :data:`REDACTION_NOTICE`. Fail-closed: a gate that raises is
            treated as a rejection, never a pass. Default ``None`` keeps the
            synthesizer a pure extractor.
    """

    min_sentence_chars: int = 25
    sentence_gate: SentenceGate | None = None
    #: Largest span of consecutive sentences re-gated together as an assembled
    #: procedure (spec §5.3). 1 = per-sentence only; the default 3 also catches a
    #: procedure split across 2-3 sentences that each pass individually.
    cross_sentence_window: int = 3

    def _gate_ok(self, text: str) -> bool:
        """True when ``text`` passes the sentence gate. Fail-closed on error."""
        if self.sentence_gate is None:
            return True
        try:
            return bool(self.sentence_gate(text))
        except Exception:
            return False  # fail-closed: a gate error redacts, never passes.

    def _emit(self, sentence: str) -> str:
        """Return ``sentence`` verbatim, or the redaction notice if the gate rejects it."""
        if self.sentence_gate is None:
            return sentence
        return sentence if self._gate_ok(sentence) else REDACTION_NOTICE

    def _safe_flags(self, sentences: list[str]) -> list[bool]:
        """Per-sentence safety flags, tightened by a cross-sentence re-gate.

        A verbatim extractor can leak an operational procedure whose steps are
        individually innocuous but assemble into weapons content once emitted
        together ("Step 1 ..." / "Step 2 ..."). After the per-sentence pass, any
        run of ``2..cross_sentence_window`` **individually-safe** sentences whose
        concatenation trips the gate is redacted as an assembled procedure. Runs
        that already contain an individually-unsafe (redacted) sentence are left
        alone -- that sentence is handled per-sentence and its safe neighbours are
        not punished (so defensive context adjacent to a redacted step survives).
        """
        flags = [self._gate_ok(s) for s in sentences]
        if self.sentence_gate is None:
            return flags
        n = len(sentences)
        window = max(1, self.cross_sentence_window)
        for size in range(2, window + 1):
            for i in range(n - size + 1):
                span = range(i, i + size)
                if all(flags[j] for j in span) and not self._gate_ok(
                    " ".join(sentences[j] for j in span)
                ):
                    for j in span:
                        flags[j] = False
        return flags

    def _join(self, sentences: list[str]) -> str:
        """Join sentences through the output gate, collapsing adjacent redactions."""
        flags = self._safe_flags(sentences)
        out: list[str] = []
        for s, ok in zip(sentences, flags):
            emitted = s if ok else REDACTION_NOTICE
            if emitted == REDACTION_NOTICE and out and out[-1] == REDACTION_NOTICE:
                continue  # don't repeat the notice for consecutive redactions
            out.append(emitted)
        return " ".join(out)

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Split text into sentence-like spans (regex; punctuation-based)."""
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            return []
        parts = _SENTENCE_SPLIT.split(text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2]

    def keywords(self, text: str, top_k: int = 10) -> list[str]:
        """Return the ``top_k`` most frequent content words (stopword-filtered)."""
        counts = Counter(self._tokens(text))
        return [w for w, _ in counts.most_common(top_k)]

    def summarize(self, text: str, max_sentences: int = 5) -> str:
        """Return the most central sentences, verbatim, in original order.

        Each sentence is scored by the TF-IDF-weighted cosine similarity to the
        document centroid; the top ``max_sentences`` are returned in their
        original order so the summary reads coherently. Returns the input
        (trimmed) unchanged when it is already short.
        """
        sentences = [s for s in self.split_sentences(text) if len(s) >= self.min_sentence_chars]
        if len(sentences) <= max_sentences:
            if sentences:
                return self._join(sentences)
            # Short/unsplittable input: still gate the whole thing before emitting.
            trimmed = (text or "").strip()
            return self._emit(trimmed) if trimmed else ""

        # Build a sentence x vocabulary TF matrix, weight by IDF, score by cosine
        # similarity to the (IDF-weighted) document centroid.
        tokenized = [self._tokens(s) for s in sentences]
        vocab = sorted({w for toks in tokenized for w in toks})
        if not vocab:
            return self._join(sentences[:max_sentences])
        index = {w: i for i, w in enumerate(vocab)}
        n_sent = len(sentences)
        tf = np.zeros((n_sent, len(vocab)))
        for i, toks in enumerate(tokenized):
            for w in toks:
                tf[i, index[w]] += 1.0
        df = np.count_nonzero(tf > 0, axis=0)
        idf = np.log((1.0 + n_sent) / (1.0 + df)) + 1.0
        weighted = tf * idf
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = weighted / norms
        centroid = unit.mean(axis=0)
        c_norm = np.linalg.norm(centroid) or 1.0
        scores = unit @ (centroid / c_norm)

        top_idx = sorted(np.argsort(scores)[-max_sentences:].tolist())
        return self._join([sentences[i] for i in top_idx])

    def summarize_sources(self, sources: list[tuple[str, str]], max_sentences: int = 6) -> str:
        """Summarize across multiple ``(label, text)`` sources into one digest.

        Sentences are pooled across sources (longer sources naturally
        contribute more), then ranked as in :meth:`summarize`.
        """
        combined = " ".join(text for _, text in sources if text)
        return self.summarize(combined, max_sentences=max_sentences)

    def relevance(self, query: str, text: str) -> float:
        """Cosine relevance of ``text`` to ``query`` over content-word TF (0..1)."""
        q = Counter(self._tokens(query))
        d = Counter(self._tokens(text))
        if not q or not d:
            return 0.0
        vocab = set(q) | set(d)
        qv = np.array([q.get(w, 0) for w in vocab], dtype=float)
        dv = np.array([d.get(w, 0) for w in vocab], dtype=float)
        denom = (np.linalg.norm(qv) * np.linalg.norm(dv)) or 1.0
        return float(np.dot(qv, dv) / denom)


__all__ = ["ExtractiveSynthesizer"]
