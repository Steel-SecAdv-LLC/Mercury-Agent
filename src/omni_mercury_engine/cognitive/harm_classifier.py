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

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

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
    """Extract the first probability-like float in ``[0, 1]`` from ``reply``.

    Returns ``0.0`` when nothing parseable is present (fail-safe: the classifier
    must never *lower* harm, so an ambiguous reply contributes no evidence). A
    value above 1 is treated as a 0-100 scale and divided by 100.
    """
    match = re.search(r"\d*\.?\d+", reply or "")
    if not match:
        return 0.0
    try:
        value = float(match.group())
    except ValueError:
        return 0.0
    if value > 1.0:  # a model that answered on a 0-100 scale
        value /= 100.0
    return float(min(max(value, 0.0), 1.0))


__all__ = ["reasoning_harm_classifier"]
