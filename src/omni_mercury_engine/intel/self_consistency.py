# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-consistency: an N-sample disagreement signal the calibrator can trust.

A single sampled reasoning path gives one answer and no honest sense of how
*load-bearing* that answer is. Drawing ``N`` independent paths and measuring how
much they **disagree** does: unanimous paths are a strong agreement signal, a
split vote is a red flag that the model is guessing. This module makes that
signal first-class and deterministic:

* :func:`self_consistency` draws ``N`` samples from a caller-supplied sampler
  (seeded for reproducibility) and returns a :class:`SelfConsistencyResult` --
  the plurality answer, the vote distribution, and a **disagreement** in
  ``[0, 1]`` (0 = unanimous, 1 = maximally split).
* :func:`vote_disagreement` / :func:`normalized_entropy` / :func:`dispersion`
  are the underlying metrics (categorical votes and continuous scores).
* The calibrator hooks -- :func:`widen_uncertainty` and
  :func:`self_consistency_decision` -- let a calibrated probability *consume* the
  disagreement: a high-disagreement prediction is pulled toward ``0.5`` (its
  confidence is not trusted) and, past a threshold, the decision rule abstains /
  escalates rather than committing.
* :func:`disagreement_error_auroc` measures the signal's worth -- how well
  disagreement ranks errored predictions above correct ones -- which is the
  stream's declared value metric (:data:`value_metrics.VALUE_METRICS`).

Nothing here calls a model directly; the sampler is injected, so the same code
drives a real reasoning model, the template path, or a deterministic test
double. Randomness is confined to a per-call :class:`numpy.random.Generator`
seeded by the caller, so a result is reproducible byte-for-byte.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: A sampler draws one reasoning-path answer given a seeded RNG, returning a
#: hashable answer (a label/string/int). :func:`self_consistency` always votes on
#: these categorically (via :func:`vote_disagreement`); it does not branch on the
#: answer type. The continuous :func:`dispersion` metric is a *separate* entry
#: point a caller applies directly to a sequence of floats in ``[0, 1]`` -- it is
#: not auto-selected from a sampler that happens to return floats.
Sampler = Callable[[np.random.Generator], Any]


@dataclass(frozen=True)
class SelfConsistencyResult:
    """The outcome of N-sample self-consistency for one item.

    Attributes:
        answer: The plurality (most-voted) answer across the samples.
        disagreement: Uncertainty in ``[0, 1]`` (0 unanimous, 1 maximally split).
        agreement: ``1 - disagreement`` -- the fraction-of-consensus signal.
        n_samples: How many paths were drawn.
        distribution: Vote count per distinct answer.
        support: The plurality answer's vote fraction (``max count / n``).
    """

    answer: Any
    disagreement: float
    agreement: float
    n_samples: int
    distribution: dict[Any, int]
    support: float

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping (answers rendered as strings).

        Both the plurality ``answer`` and the ``distribution`` keys are rendered
        with ``str`` so the mapping is always JSON-serializable even when the
        sampler returned a non-JSON answer type (a tuple, an int label, a custom
        object) -- matching this method's documented contract rather than leaking
        a raw ``Any`` the caller cannot ``json.dumps``.
        """
        return {
            "answer": str(self.answer),
            "disagreement": self.disagreement,
            "agreement": self.agreement,
            "n_samples": self.n_samples,
            "distribution": {str(k): v for k, v in self.distribution.items()},
            "support": self.support,
        }


def vote_disagreement(answers: Sequence[Any]) -> float:
    """Categorical vote disagreement: ``1 - (plurality count / n)``.

    ``0.0`` when all answers agree; approaches ``1.0`` as the vote fragments.
    An empty input is maximally uncertain (``1.0``, fail-closed): no evidence of
    agreement must never read as agreement.
    """
    n = len(answers)
    if n == 0:
        return 1.0
    top = Counter(answers).most_common(1)[0][1]
    return 1.0 - top / n


def normalized_entropy(answers: Sequence[Any]) -> float:
    """Shannon entropy of the vote distribution, normalized to ``[0, 1]``.

    Normalized by ``log(k)`` over the ``k`` *distinct* observed answers, so a
    uniform split is ``1.0`` regardless of ``k`` and a unanimous vote is ``0.0``.
    A single distinct answer (``k == 1``) has no entropy to normalize and returns
    ``0.0``. Complements :func:`vote_disagreement`: entropy is sensitive to the
    whole distribution's shape, not just the plurality mass.
    """
    n = len(answers)
    if n == 0:
        return 1.0
    counts = np.array(list(Counter(answers).values()), dtype=float)
    k = len(counts)
    if k <= 1:
        return 0.0
    probs = counts / counts.sum()
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / math.log(k)


def dispersion(values: Sequence[float]) -> float:
    """Disagreement of continuous scores in ``[0, 1]`` via normalized spread.

    Uses ``2 * std`` clipped to ``[0, 1]``: a degenerate all-equal set is ``0``;
    a maximal ``{0, 1}`` split has ``std = 0.5`` and maps to ``1.0``. This keeps
    the continuous path on the same ``0..1`` scale as :func:`vote_disagreement`.
    An empty input is ``1.0`` (fail-closed, as in :func:`vote_disagreement`).
    """
    if len(values) == 0:
        return 1.0
    arr = np.asarray(values, dtype=float)
    return float(np.clip(2.0 * np.std(arr), 0.0, 1.0))


def self_consistency(
    sampler: Sampler,
    n_samples: int = 5,
    *,
    seed: int | None = None,
) -> SelfConsistencyResult:
    """Draw ``n_samples`` reasoning paths and summarize their (dis)agreement.

    Args:
        sampler: Callable given a seeded :class:`numpy.random.Generator`,
            returning one hashable answer per call.
        n_samples: Number of independent paths to draw (must be >= 1).
        seed: Seed for the per-call RNG passed to the sampler. ``None`` uses OS
            entropy; a fixed value makes the whole result reproducible.

    Returns:
        A :class:`SelfConsistencyResult`.

    Raises:
        ValueError: if ``n_samples < 1`` (there is no consistency of zero paths).
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1; got {n_samples}")
    rng = np.random.default_rng(seed)
    answers = [sampler(rng) for _ in range(n_samples)]
    distribution = dict(Counter(answers))
    top_answer, top_count = Counter(answers).most_common(1)[0]
    disagree = vote_disagreement(answers)
    return SelfConsistencyResult(
        answer=top_answer,
        disagreement=disagree,
        agreement=1.0 - disagree,
        n_samples=n_samples,
        distribution=distribution,
        support=top_count / n_samples,
    )


# --------------------------------------------------------------------------- #
# Calibrator integration.
# --------------------------------------------------------------------------- #
def widen_uncertainty(prob: float, disagreement: float, *, strength: float = 1.0) -> float:
    """Pull a calibrated probability toward ``0.5`` in proportion to disagreement.

    A prediction the reasoning paths split on should not keep its sharp
    confidence. The map ``p' = 0.5 + (p - 0.5) * (1 - strength * disagreement)``
    shrinks the distance from the decision boundary by up to ``strength`` at full
    disagreement, never crossing the boundary (so it *widens* uncertainty, never
    flips a decision). ``strength`` is clamped to ``[0, 1]`` and ``disagreement``
    to ``[0, 1]``.

    Args:
        prob: The base calibrated probability in ``[0, 1]``.
        disagreement: The self-consistency disagreement in ``[0, 1]``.
        strength: How much full disagreement shrinks confidence (``[0, 1]``).

    Returns:
        The disagreement-widened probability in ``[0, 1]``.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    d = float(np.clip(disagreement, 0.0, 1.0))
    p = float(np.clip(prob, 0.0, 1.0))
    return 0.5 + (p - 0.5) * (1.0 - s * d)


@dataclass(frozen=True)
class ConsistencyDecision:
    """A calibrated decision that consulted the self-consistency signal.

    Attributes:
        decision: ``"positive"`` / ``"negative"`` (thresholded) or ``"abstain"``
            when disagreement exceeded the abstention threshold.
        widened_prob: The disagreement-widened probability actually thresholded.
        abstained: Whether the rule abstained on high disagreement.
        disagreement: The disagreement the rule saw.
    """

    decision: str
    widened_prob: float
    abstained: bool
    disagreement: float


def self_consistency_decision(
    prob: float,
    disagreement: float,
    *,
    decision_threshold: float = 0.5,
    abstain_above: float = 0.6,
    strength: float = 1.0,
) -> ConsistencyDecision:
    """Decision rule that abstains on high disagreement, else thresholds a widened p.

    This is the "the calibrator uses the disagreement signal in a decision rule"
    integration point:

    1. If ``disagreement >= abstain_above`` the paths are too split to commit --
       the rule **abstains** (defer to a heavier path or a human), rather than
       emit a confident-looking guess.
    2. Otherwise the probability is widened by :func:`widen_uncertainty` (so
       residual disagreement still costs confidence) and thresholded.

    Args:
        prob: Base calibrated probability in ``[0, 1]``.
        disagreement: Self-consistency disagreement in ``[0, 1]``.
        decision_threshold: Probability cut for positive vs negative.
        abstain_above: Disagreement at/above which the rule abstains.
        strength: Passed through to :func:`widen_uncertainty`.

    Returns:
        A :class:`ConsistencyDecision`.
    """
    d = float(np.clip(disagreement, 0.0, 1.0))
    widened = widen_uncertainty(prob, d, strength=strength)
    if d >= abstain_above:
        return ConsistencyDecision(
            decision="abstain", widened_prob=widened, abstained=True, disagreement=d
        )
    decision = "positive" if widened >= decision_threshold else "negative"
    return ConsistencyDecision(
        decision=decision, widened_prob=widened, abstained=False, disagreement=d
    )


def disagreement_error_auroc(disagreements: Sequence[float], errors: Sequence[int]) -> float:
    """AUROC of ``disagreement`` predicting ``error`` (the stream's value metric).

    ``errors[i]`` is ``1`` when prediction ``i`` was wrong, ``0`` when right. A
    good uncertainty signal assigns higher disagreement to the errors, so this
    AUROC exceeds ``0.5``. Returns ``0.5`` (uninformative) when either class is
    absent (AUROC is undefined without both an error and a correct row).

    Args:
        disagreements: The per-item disagreement scores.
        errors: Per-item binary error indicators (1 wrong, 0 correct).

    Returns:
        AUROC in ``[0, 1]`` of disagreement ranking errors above correct items.
    """
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    y = np.asarray(errors, dtype=float)
    d = np.asarray(disagreements, dtype=float)
    if y.size == 0 or len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, d))


__all__ = [
    "ConsistencyDecision",
    "Sampler",
    "SelfConsistencyResult",
    "disagreement_error_auroc",
    "dispersion",
    "normalized_entropy",
    "self_consistency",
    "self_consistency_decision",
    "vote_disagreement",
    "widen_uncertainty",
]
