# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Meaning-level harm classification for the ethics gate.

This is the *meaning-level* layer for
:class:`omni_mercury_engine.cognitive.ethical_bounding.BenevolenceScorer`. The
gate's built-in evidence is keyword + char-trigram (morphological) + a curated
euphemism lexicon -- all deterministic and model-free, and all bounded by what a
lexicon can enumerate.

Two sources can supply meaning above that lexical floor, and
:func:`default_harm_classifier` combines both by ``max``:

* **The shipped offline classifier** in
  :mod:`omni_mercury_engine.cognitive.meaning_level` -- a trained linear model
  over request-frame features, deterministic and stdlib-only. It ships with the
  package, so meaning-level coverage exists in **every** deployment, including
  air-gapped ones and CI, with no model server and no network call.
* **A served generative model** the deployment already runs (e.g. Mercury's
  offline Ollama reasoning backend), adapted by
  :func:`reasoning_harm_classifier`.

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
    # Build the reasoning->probability adapter once per backend, not per
    # classify() call: this is a hot safety-gate boundary, and
    # reasoning_harm_classifier() constructs a fresh closure each time.
    if backend is not None:
        try:
            _DEFAULT_CACHE["adapter"] = reasoning_harm_classifier(backend._generate)
        except Exception as exc:  # pragma: no cover - environment-dependent
            logger.info("default harm classifier: adapter construction failed (%s)", exc)
            _DEFAULT_CACHE["adapter"] = None
    else:
        _DEFAULT_CACHE["adapter"] = None
    return backend


def default_harm_classifier() -> Callable[[str], float]:
    """Mercury's meaning-level harm classifier, wired by default.

    Combines two independent meaning-level sources by ``max``:

    1. **The shipped offline classifier**
       (:mod:`omni_mercury_engine.cognitive.meaning_level`) -- a trained linear
       model over request-frame features. Deterministic, stdlib-only, no network
       call, no model server, and therefore available in **every** deployment
       including air-gapped ones and CI. This is what makes the meaning-level
       layer intrinsic rather than conditional on an operator running Ollama.
    2. **A served generative model**, when one happens to be running
       (:class:`~omni_mercury_engine.reasoning.backends.LocalReasoningBackend`,
       Ollama-when-present). Contributes only when a *genuine* local/cloud model
       is serving; under the template fallback it contributes ``0.0``, because a
       template's output is not a harm probability.

    Combining by ``max`` means adding either source can only ever RAISE harm --
    the gate itself also combines by ``max`` -- so neither can regress the
    deterministic lexical gate, and a deployment that runs a strong local model
    still benefits from it on top of the shipped floor.

    Disable entirely with ``MERCURY_DISABLE_DEFAULT_HARM_CLASSIFIER=1`` (e.g. to
    keep a surface strictly deterministic and lexical). The returned callable is
    cheap to obtain repeatedly (both the backend and the weight artifact are
    cached).
    """

    def classify(text: str) -> float:
        if os.environ.get("MERCURY_DISABLE_DEFAULT_HARM_CLASSIFIER") == "1":
            return 0.0
        score = _shipped_meaning_level_score(text)
        backend = _resolve_default_backend()
        if backend is None:
            return score
        try:
            active = str(getattr(backend, "model", "")).lower()
            if not active.startswith(_REAL_MODEL_PREFIXES):
                return score  # template / no real model: not a harm probability
            adapter = _DEFAULT_CACHE.get("adapter")
            if not callable(adapter):
                return score
            return max(score, float(adapter(text)))
        except Exception as exc:  # pragma: no cover - fail-open
            logger.info(
                "served harm classifier failed (%s); using shipped score only (%s)", exc, score
            )
            return score

    return classify


def _shipped_meaning_level_score(text: str) -> float:
    """Score ``text`` with the shipped offline model; ``0.0`` if unavailable.

    Imported lazily so this module keeps its stdlib-only, load-anywhere import
    contract for callers that never reach the classifier.
    """
    try:
        from omni_mercury_engine.cognitive.meaning_level import meaning_level_harm_classifier

        return float(meaning_level_harm_classifier()(text))
    except Exception as exc:  # pragma: no cover - fail-open
        logger.info("shipped meaning-level classifier unavailable (%s); contributing 0.0", exc)
        return 0.0


def served_model_available() -> bool:
    """True iff a *served generative model* backs :func:`default_harm_classifier`.

    Resolves Mercury's local reasoning backend and reports whether a real model
    (not the deterministic template fallback, and not disabled) is actually
    serving. This is now only *one* of the two meaning-level sources -- see
    :func:`real_harm_classifier_available` for whether meaning-level coverage
    exists at all.

    Fail-safe: any resolution/attribute error returns ``False``.
    """
    if os.environ.get("MERCURY_DISABLE_DEFAULT_HARM_CLASSIFIER") == "1":
        return False
    backend = _resolve_default_backend()
    if backend is None:
        return False
    try:
        active = str(getattr(backend, "model", "")).lower()
    except Exception:  # pragma: no cover - defensive
        return False
    return active.startswith(_REAL_MODEL_PREFIXES) and _DEFAULT_CACHE.get("adapter") is not None


def real_harm_classifier_available() -> bool:
    """True iff genuine meaning-level coverage backs :func:`default_harm_classifier`.

    Meaning-level coverage no longer requires an operator to be running a model.
    It is satisfied by **either**:

    * the shipped offline classifier
      (:mod:`omni_mercury_engine.cognitive.meaning_level`), which is present in
      every install and needs no network or model server, **or**
    * a served generative model (:func:`served_model_available`).

    The meaning-level routing rescue in
    :func:`~omni_mercury_engine.cognitive.ethical_bounding.assess_weapons_uplift`
    only cuts false-negatives when this is ``True``. Before the shipped model
    existed, that meant the weapons gate ran lexical-only in CI, air-gapped
    deployments and every default install -- a measured 0.744 held-out
    false-negative rate. See ``docs/WEAPONS_GATE_ADVERSARIAL_EVAL.md``.

    Fail-safe: returns ``False`` if neither source resolves.
    """
    if os.environ.get("MERCURY_DISABLE_DEFAULT_HARM_CLASSIFIER") == "1":
        return False
    try:
        from omni_mercury_engine.cognitive.meaning_level import meaning_level_available

        if meaning_level_available():
            return True
    except Exception:  # pragma: no cover - fail-safe
        pass
    return served_model_available()


def harm_classifier_posture() -> dict[str, bool]:
    """Report which meaning-level sources are active, for audit and CI logging.

    Keys: ``shipped_model`` (the offline classifier loaded), ``served_model``
    (a generative backend is serving), ``disabled`` (the kill switch is set),
    and ``meaning_level`` (coverage exists at all).
    """
    disabled = os.environ.get("MERCURY_DISABLE_DEFAULT_HARM_CLASSIFIER") == "1"
    shipped = False
    if not disabled:
        try:
            from omni_mercury_engine.cognitive.meaning_level import meaning_level_available

            shipped = meaning_level_available()
        except Exception:  # pragma: no cover - fail-safe
            shipped = False
    served = served_model_available()
    return {
        "disabled": disabled,
        "shipped_model": shipped,
        "served_model": served,
        "meaning_level": bool(shipped or served),
    }


__all__ = [
    "default_harm_classifier",
    "harm_classifier_posture",
    "real_harm_classifier_available",
    "reasoning_harm_classifier",
    "served_model_available",
]
