# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Adapt a text-generation callable into a harm classifier for the ethics gate.

This is the *meaning-level* extension point for
:class:`omni_mercury_engine.cognitive.ethical_bounding.BenevolenceScorer`. The
gate's built-in evidence is keyword + char-trigram (morphological) + a curated
euphemism lexicon -- all deterministic and model-free. A deployment that wants
genuine semantic harm classification can plug in a model **it already runs**
(e.g. Mercury's own offline Ollama reasoning backend) via this adapter, with no
new dependency and no cloud call.

Pure stdlib -- no LLM/crypto imports -- so it loads anywhere the ethics gate
does. The model is consulted, never trusted: its score can only RAISE harm in
the gate (combined by ``max``), and any failure or unparseable reply yields
``0.0`` so an absent/weak model is never a safety regression.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Static, model-free instruction. The model returns ONLY a probability, so a
# weak/templated local model that cannot comply degrades to an unparseable reply
# -> 0.0 (the deterministic lexical harm in the ethics gate then governs), never
# a false "safe" verdict that LOWERS harm.
_HARM_CLASSIFIER_SYSTEM = (
    "You are a safety classifier. Given a request, output ONLY a single number "
    "between 0.00 and 1.00: the probability the request expresses intent to "
    "cause serious harm to a person. Output just the number, nothing else."
)


def reasoning_harm_classifier(
    generate: Callable[..., str], *, max_chars: int = 2000
) -> Callable[[str], float]:
    """Adapt a text-generation callable into a ``Callable[[str], float]`` harm score.

    The result is suitable for
    :class:`~omni_mercury_engine.cognitive.ethical_bounding.BenevolenceScorer`'s
    ``harm_classifier`` hook, giving the keyword/euphemism gate a genuine
    meaning-level layer using Mercury's *own* offline LLM substrate::

        from omni_mercury_engine.reasoning.backends import LocalReasoningBackend
        from omni_mercury_engine.cognitive.harm_classifier import reasoning_harm_classifier

        backend = LocalReasoningBackend()                 # Ollama-or-template, offline
        scorer = BenevolenceScorer(
            harm_classifier=reasoning_harm_classifier(backend._generate)
        )

    ``generate`` is any callable returning text -- ``FallbackLLMChain.generate``,
    an adapter's ``generate``, or a backend's ``_generate``. It is invoked with
    ``(prompt, system_prompt)`` and, on a ``TypeError``, retried with
    ``(prompt)`` so single-argument callables also work.

    Fail-safe: any exception or an unparseable reply yields ``0.0`` -- the
    classifier can only ever RAISE harm in the ethics gate, so a failed or absent
    model is never a safety regression.
    """

    def classify(text: str) -> float:
        prompt = (
            "Classify the harm intent of this request and output only the "
            f"probability number:\n\n{text[:max_chars]}"
        )
        try:
            try:
                reply = generate(prompt, _HARM_CLASSIFIER_SYSTEM)
            except TypeError:
                reply = generate(prompt)
        except Exception:
            return 0.0
        return _parse_probability(reply)

    return classify


def _parse_probability(reply: str) -> float:
    """Extract a probability in ``[0, 1]`` from a reply that is *just a number*.

    The classifier is instructed to output ONLY the number. Parsing is therefore
    strict: the reply must be a bare number (optionally with a ``%`` or trailing
    punctuation), or a short reply that clearly leads with one. A long/prose
    reply -- e.g. a template backend that echoes the input ("Received query
    (12 words)...") -- returns ``0.0`` rather than a spurious probability parsed
    out of arbitrary text. Fail-safe: the classifier must never *lower* harm, so
    an unparseable/prose reply contributes no evidence. A value in ``(1, 100]`` is
    treated as a 0-100 percentage and divided by 100; a value **above 100** is
    implausible for a probability classifier (e.g. a model that echoed a large
    number) and yields ``0.0`` rather than clamping to a spurious ``1.0``.
    """
    text = (reply or "").strip()
    if not text:
        return 0.0
    # Whole-reply number (allow a trailing % or single sentence-final period).
    match = re.fullmatch(r"(\d*\.?\d+)\s*%?\.?", text)
    if match is None and len(text) <= 16:
        # Short reply that leads with a number ("prob 0.85", "0.85 harm").
        match = re.match(r"[^\d]{0,6}(\d*\.?\d+)", text)
    if match is None:
        return 0.0
    try:
        value = float(match.group(1))
    except (ValueError, IndexError):
        return 0.0
    if value > 100.0:
        # Implausible for a 0.00-1.00 / 0-100 classifier -> not a probability.
        # Fail-safe: contribute no evidence rather than clamp to a spurious 1.0.
        return 0.0
    if value > 1.0:  # a model that answered on a 0-100 scale
        value /= 100.0
    return float(min(max(value, 0.0), 1.0))


#: Adapter names the default classifier trusts as a genuine semantic model. A
#: ``template`` fallback (no model) is NOT trusted -- its output is not a harm
#: probability -- so the default classifier contributes 0.0 under it.
_REAL_MODEL_PREFIXES = ("ollama", "cloud", "remote", "openai", "anthropic")

_DEFAULT_CACHE: dict[str, object] = {}


def _resolve_default_backend() -> object | None:
    """Lazily build Mercury's offline-first local reasoning backend (cached).

    Returns ``None`` if the reasoning stack cannot be constructed at all. Import
    and construction are deferred to first use so importing this module stays
    stdlib-only and cheap.
    """
    if "backend" in _DEFAULT_CACHE:
        return _DEFAULT_CACHE["backend"]
    backend: object | None = None
    try:
        from omni_mercury_engine.reasoning.backends import LocalReasoningBackend

        backend = LocalReasoningBackend()
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.info("default harm classifier: local reasoning backend unavailable (%s)", exc)
        backend = None
    _DEFAULT_CACHE["backend"] = backend
    return backend


def default_harm_classifier() -> Callable[[str], float]:
    """Mercury's own offline reasoning-backed harm classifier, wired by default.

    Returns a fail-open ``Callable[[str], float]`` backed by
    :class:`~omni_mercury_engine.reasoning.backends.LocalReasoningBackend`
    (Ollama-when-present, deterministic-template otherwise, always offline-safe).
    It contributes a harm probability **only when a genuine local/cloud model is
    actually serving**; under the template fallback (no model), a missing
    reasoning stack, or any error, it returns ``0.0``. Because the ethics gate
    combines the classifier by ``max``, this can only ever RAISE harm when a real
    semantic model is present and never regresses the deterministic lexical gate
    -- so it is safe to wire by default on the open-web/text surface without
    adding a hard dependency or a network call in air-gapped deployments.

    Disable entirely with ``MERCURY_DISABLE_DEFAULT_HARM_CLASSIFIER=1`` (e.g. to
    keep a surface strictly deterministic). The returned callable is cheap to
    obtain repeatedly (backend construction is cached).
    """

    def classify(text: str) -> float:
        if os.environ.get("MERCURY_DISABLE_DEFAULT_HARM_CLASSIFIER") == "1":
            return 0.0
        backend = _resolve_default_backend()
        if backend is None:
            return 0.0
        try:
            active = str(getattr(backend, "model", "")).lower()
            if not active.startswith(_REAL_MODEL_PREFIXES):
                return 0.0  # template / no real model: not a harm probability
            return reasoning_harm_classifier(backend._generate)(text)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - fail-open
            logger.info("default harm classifier failed (%s); contributing 0.0", exc)
            return 0.0

    return classify


__all__ = ["default_harm_classifier", "reasoning_harm_classifier"]
