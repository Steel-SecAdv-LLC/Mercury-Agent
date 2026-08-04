# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury's own offline meaning-level harm classifier.

The weapons/mass-casualty gate in :mod:`.ethical_bounding` routes on two lexical
axes and then, for the cases the lexicons cannot decide, consults a
*meaning-level* ``harm_classifier``. Until now the only thing that could serve
that hook was a generative model (Ollama or a cloud backend). Wherever no model
is running -- CI, air-gapped deployments, and every default install -- the hook
returned ``0.0`` and the gate ran lexical-only, measured at a **0.744 held-out
false-negative rate** (``docs/WEAPONS_GATE_ADVERSARIAL_EVAL.md``).

This module closes that by making the meaning-level layer *intrinsic*: a small
linear classifier over lexical-frame features, trained offline by
``scripts/train_meaning_level_classifier.py`` on
``benchmarks/meaning_level_corpus.py`` and shipped as a JSON weight file. It is
deterministic, dependency-free (stdlib only, like the rest of the gate),
sub-millisecond, and available in every deployment -- including air-gapped ones,
because it makes no network call and loads no model server.

**What it actually learns.** Not "which hazard nouns are dangerous" -- Axis A
already routes those, and the training corpus deliberately pairs *every* hazard
noun with both offensive and defensive framings so noun identity carries no
label information. What separates the classes is the **request frame**: whether
the query seeks an operational procedure for producing/deploying the hazard, or
seeks to understand, detect, treat, regulate, or respond to it. That distinction
is what the lexical Axis-B regexes approximate with an enumerated verb list --
an open class that can never be finished -- and what this model learns as a
weighted frame pattern instead.

**Contract.** It is *consulted, never trusted*. It returns a probability in
``[0, 1]``; the gate combines it by ``max`` and it can only ever RAISE a
disposition, never lower one earned by lexical evidence, and never auto-refuse
on classifier-alone evidence (the routing rescue raises to ESCALATE -- human
review -- not to a refusal). Any failure -- missing weights, malformed JSON,
unreadable file -- yields ``0.0``, i.e. exactly the pre-existing lexical-only
behaviour, so a broken artifact is a *loss of the improvement*, never a safety
regression.

The weight file is auditable on purpose: it is plain JSON mapping human-readable
feature strings to weights, so a reviewer can read what the model keys on rather
than trusting an opaque blob.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.cognitive.harm_normalization import canonical_normalize

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

#: Shipped weight artifact. Declared in ``pyproject.toml``'s ``package-data`` so
#: a non-editable install carries it too (see the note there about the
#: σ_Immutable corpus, the same failure mode).
WEIGHTS_PATH = Path(__file__).with_name("meaning_level_weights.json")

#: Bumped whenever the feature extractor changes in a way that invalidates
#: previously trained weights. :func:`load_model` refuses a mismatched artifact
#: rather than scoring with features the model was never fit on.
FEATURE_VERSION = 1

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Words carrying no frame information. Kept deliberately tiny -- function words
#: like "to", "for", "how" and "me" are exactly the frame signal here ("how to
#: make" vs "how does ... work"), so the usual stopword list would delete the
#: very thing the model reads.
_DROP = frozenset({"a", "an", "the"})

#: Length at which a token is also emitted as a truncated prefix. This is the
#: model's morphology: "refines", "refining" and "refine" all share the prefix
#: "refin", so an inflection unseen in training still reaches a trained weight.
_PREFIX_LEN = 5


def tokenize(text: str) -> list[str]:
    """Return the canonical token sequence for ``text``.

    De-obfuscates first (:func:`~.harm_normalization.canonical_normalize`), so
    leetspeak, homoglyph spoofing and per-character spacing are folded away
    before tokenizing and the model never has to learn them.
    """
    return [t for t in _TOKEN_RE.findall(canonical_normalize(text)) if t not in _DROP]


def extract_features(text: str) -> set[str]:
    """Return the binary feature set for ``text``.

    Three families, all frame-oriented:

    * ``w:<token>`` -- unigram presence.
    * ``b:<t1>_<t2>`` -- adjacent-pair presence; this is what encodes a frame
      ("walk_me", "me_through", "how_to", "steps_to", "process_for").
    * ``p:<prefix>`` -- the first five characters of a longer token, giving
      inflectional transfer across verb forms.

    Presence, not count: the inputs are single short requests, where a repeated
    token says little and a raw count would let padding inflate a score.

    Note:
        Anything that *accumulates floats* over these features must iterate them
        in a fixed order -- use :func:`ordered_features`. Python randomizes
        string hashing per process, so set iteration order varies between runs;
        summing in a different order changes the result in the last bit, which
        is enough to make trained weights fail to reproduce byte-for-byte.
    """
    tokens = tokenize(text)
    feats = {f"w:{t}" for t in tokens}
    feats.update(f"p:{t[:_PREFIX_LEN]}" for t in tokens if len(t) > _PREFIX_LEN)
    feats.update(f"b:{a}_{b}" for a, b in pairwise(tokens))
    return feats


def ordered_features(text: str) -> tuple[str, ...]:
    """Return :func:`extract_features` in a fixed, process-independent order.

    Float addition is not associative, so an unordered iteration makes both
    scoring and training depend on Python's per-process string hash seed. Sorting
    costs microseconds on the ~30 features a request produces and buys exact
    reproducibility -- which is the property the trained artifact and the
    published measurements both rest on.
    """
    return tuple(sorted(extract_features(text)))


class MeaningLevelModel:
    """A trained linear harm-intent scorer.

    Attributes:
        weights: Feature string -> learned weight. Absent features contribute 0.
        bias: Intercept.
        feature_version: The :data:`FEATURE_VERSION` the weights were fit under.
        metadata: Free-form provenance recorded by the trainer (corpus sizes,
            hyperparameters, measured scores) -- carried so a deployment can
            report which model it is running.
    """

    __slots__ = ("bias", "feature_version", "metadata", "weights")

    def __init__(
        self,
        weights: dict[str, float],
        bias: float = 0.0,
        feature_version: int = FEATURE_VERSION,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the instance."""
        self.weights = weights
        self.bias = float(bias)
        self.feature_version = int(feature_version)
        self.metadata = metadata or {}

    def score(self, text: str) -> float:
        """Return ``P(offensive intent)`` in ``[0, 1]`` for ``text``."""
        z = self.bias + sum(self.weights.get(f, 0.0) for f in ordered_features(text))
        # Overflow-safe logistic: math.exp(710) raises OverflowError.
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-min(z, 60.0)))
        e = math.exp(max(z, -60.0))
        return e / (1.0 + e)

    def explain(self, text: str, top: int = 10) -> list[tuple[str, float]]:
        """Return the ``top`` highest-magnitude contributing features for ``text``.

        The audit surface: it answers "why did this score what it scored" in
        terms a reviewer can read, without re-running training.
        """
        contribs = [(f, self.weights[f]) for f in extract_features(text) if f in self.weights]
        contribs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return contribs[:top]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable artifact form."""
        return {
            "feature_version": self.feature_version,
            "bias": self.bias,
            "metadata": self.metadata,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MeaningLevelModel:
        """Rebuild a model from :meth:`to_dict` output.

        Raises:
            ValueError: if the artifact was fit under a different
                :data:`FEATURE_VERSION`, or carries no weights.
        """
        version = int(payload.get("feature_version", -1))
        if version != FEATURE_VERSION:
            raise ValueError(
                f"meaning-level weights are feature_version {version}, "
                f"but this build extracts version {FEATURE_VERSION} features; "
                "retrain with scripts/train_meaning_level_classifier.py"
            )
        weights = payload.get("weights")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("meaning-level weights artifact carries no weights")
        return cls(
            weights={str(k): float(v) for k, v in weights.items()},
            bias=float(payload.get("bias", 0.0)),
            feature_version=version,
            metadata=dict(payload.get("metadata") or {}),
        )


_MODEL_CACHE: dict[str, MeaningLevelModel | None] = {}


def load_model(path: Path | None = None, *, refresh: bool = False) -> MeaningLevelModel | None:
    """Load the shipped model, or ``None`` when it is unavailable/invalid.

    Cached after the first call: this sits on a hot safety-gate path and the
    artifact never changes within a process.

    Fail-open by design. A missing or corrupt artifact returns ``None``, the
    classifier then scores ``0.0``, and the gate falls back to exactly its
    pre-existing lexical-only behaviour -- degraded, never unsafe.
    """
    target = path or WEIGHTS_PATH
    key = str(target)
    if not refresh and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model: MeaningLevelModel | None = None
    try:
        with target.open(encoding="utf-8") as fh:
            model = MeaningLevelModel.from_dict(json.load(fh))
    except FileNotFoundError:
        logger.info("meaning-level weights not present at %s; classifier inactive", target)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("meaning-level weights unusable (%s); classifier inactive", exc)
    _MODEL_CACHE[key] = model
    return model


#: Kill switch. Set to ``"1"`` to force a strictly lexical, model-free posture.
#: Honoured *here*, at the single point every consumer resolves through, rather
#: than only in ``harm_classifier.default_harm_classifier`` -- the gate's own
#: default (``ethical_bounding.USE_SHIPPED_CLASSIFIER``) resolves straight to
#: this module, so a check that lived only in the other caller would leave the
#: documented switch silently inert on the path that actually matters.
DISABLE_ENV_VAR = "MERCURY_DISABLE_DEFAULT_HARM_CLASSIFIER"


def meaning_level_disabled() -> bool:
    """True when the kill switch is set. Read per call, so it can be toggled."""
    return os.environ.get(DISABLE_ENV_VAR) == "1"


def meaning_level_available() -> bool:
    """True when the shipped model can contribute a nonzero score.

    ``False`` when the kill switch is set, because a disabled classifier
    contributes nothing -- callers branch on this to decide whether
    meaning-level coverage exists, and "loadable but switched off" is not
    coverage.
    """
    return not meaning_level_disabled() and load_model() is not None


def meaning_level_harm_classifier() -> Any:
    """Return a ``Callable[[str], float]`` backed by the shipped model.

    Suitable for :class:`~.ethical_bounding.BenevolenceScorer`'s
    ``harm_classifier`` hook and for
    :func:`~.ethical_bounding.assess_weapons_uplift`. Returns ``0.0`` for every
    input when the model is unavailable or the kill switch is set -- in both
    cases the gate falls back to exactly its pre-existing lexical behaviour.
    """

    def classify(text: str) -> float:
        if meaning_level_disabled():
            return 0.0
        model = load_model()
        if model is None:
            return 0.0
        try:
            return model.score(text)
        except Exception as exc:  # pragma: no cover - fail-open
            logger.info("meaning-level scoring failed (%s); contributing 0.0", exc)
            return 0.0

    return classify


def score_many(texts: Iterable[str]) -> list[float]:
    """Score an iterable of texts (evaluation helper; same model, same cache)."""
    model = None if meaning_level_disabled() else load_model()
    if model is None:
        return [0.0 for _ in texts]
    return [model.score(t) for t in texts]


__all__ = [
    "DISABLE_ENV_VAR",
    "FEATURE_VERSION",
    "WEIGHTS_PATH",
    "MeaningLevelModel",
    "extract_features",
    "load_model",
    "meaning_level_available",
    "meaning_level_disabled",
    "meaning_level_harm_classifier",
    "ordered_features",
    "score_many",
    "tokenize",
]
