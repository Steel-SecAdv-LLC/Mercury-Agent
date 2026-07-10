# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Counterfactual reasoning over anomaly *detections* (first-class detection path).

Turns any detection-path score function -- the statistical ensemble, the tier
ensemble blend, a symbolic rule-satisfaction score, the fusion serve path --
into an answer to "what is the smallest change to this point that flips the
detection?".  The seam is a plain callable ``(k, d) score matrix -> (k,)
anomaly scores`` so no detector has to be wrapped in a model class, and every
counterfactual is validated by *re-scoring through the real detector*, never a
surrogate.

Three thin adapters are provided for Mercury's production scorers:

* :func:`make_statistical_score_fn` -- scores a candidate row by substituting
  it into the original batch and re-running
  :meth:`~omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector.detect`,
  so batch-level guards (inversion guard, ensemble flip) see exactly what they
  would have seen in production.
* :func:`make_tier_score_fn` -- scores a candidate point value by substituting
  it into the original series and re-running the fitted
  :class:`~omni_mercury_engine.detectors.detection_tier.StreamingScoreEnsemble`.
* :func:`make_symbolic_score_fn` -- the
  :class:`~omni_mercury_engine.ml.symbolic_constraint.SymbolicConstraintModule`
  consensus over a per-detector score vector.

Every counterfactual is post-processed by a deterministic greedy minimization
pass (:func:`explain_detection_counterfactual`): each changed feature is
reverted singly (then in pairs, within an evaluation budget) and the revert is
kept only when the flip survives re-scoring.  ``minimal=True`` is set only
after a full single-revert sweep accepts nothing -- i.e. minimality w.r.t.
single-feature reverts is *verified*, not assumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.explainability.counterfactuals import (
    Counterfactual,
    FeatureConstraint,
    NonFiniteScoreError,
    create_counterfactual_generator,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "DETECTION_COUNTERFACTUAL_METHODS",
    "ChangedFeature",
    "DetectionCounterfactual",
    "explain_detection_counterfactual",
    "make_statistical_score_fn",
    "make_symbolic_score_fn",
    "make_tier_score_fn",
]

#: Every selectable search method.  ``prototype`` additionally requires
#: ``training_data``/``training_labels`` (real labelled points to draw
#: target-class prototypes from) and fails loudly without them.
DETECTION_COUNTERFACTUAL_METHODS: tuple[str, ...] = (
    "wachter",
    "dice",
    "growing_spheres",
    "prototype",
    "genetic",
)

#: Change-detection tolerances (match ``CounterfactualGenerator``'s
#: ``np.isclose`` usage so sparsity agrees with the generators' own view).
_RTOL = 1e-05
_ATOL = 1e-08


@dataclass
class ChangedFeature:
    """One feature the counterfactual changed.

    Attributes:
        name: Feature label (``feature_<i>`` when no names are supplied).
        old: Original value.
        new: Counterfactual value.
        delta: ``new - old``.
    """

    name: str
    old: float
    new: float
    delta: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "old": float(self.old),
            "new": float(self.new),
            "delta": float(self.delta),
        }


@dataclass
class DetectionCounterfactual:
    """A validated, minimized counterfactual for one detection decision.

    Attributes:
        original_x: The point as the detector saw it.
        counterfactual_x: The flipped (or best-effort) point.
        changed_features: Per-feature ``(name, old, new, delta)`` records.
        score_before: Real detector score of ``original_x`` (re-scored).
        score_after: Real detector score of ``counterfactual_x`` (re-scored).
        threshold: Decision threshold (flag semantics: ``score > threshold``).
        flipped: ``True`` iff ``score_after`` is on the other side of
            ``threshold`` from ``score_before`` -- verified by re-scoring.
        sparsity: Number of changed features.
        distance: L2 distance between original and counterfactual.
        method: Search method that produced the candidate.
        minimal: ``True`` only when a full single-feature revert sweep was
            re-scored and no revert preserved the flip (verified minimality
            w.r.t. single-feature reverts).  Always ``False`` when not flipped.
        n_score_evals: Real-detector evaluations spent (search + minimization).
        feature_names: Labels for every feature of ``original_x``.
    """

    original_x: np.ndarray[Any, Any]
    counterfactual_x: np.ndarray[Any, Any]
    changed_features: list[ChangedFeature]
    score_before: float
    score_after: float
    threshold: float
    flipped: bool
    sparsity: int
    distance: float
    method: str
    minimal: bool
    n_score_evals: int = 0
    feature_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation (surface payload)."""
        return {
            "original_x": [float(v) for v in self.original_x],
            "counterfactual_x": [float(v) for v in self.counterfactual_x],
            "changed_features": [c.to_dict() for c in self.changed_features],
            "score_before": float(self.score_before),
            "score_after": float(self.score_after),
            "threshold": float(self.threshold),
            "flipped": bool(self.flipped),
            "sparsity": int(self.sparsity),
            "distance": float(self.distance),
            "method": self.method,
            "minimal": bool(self.minimal),
            "n_score_evals": int(self.n_score_evals),
            "feature_names": list(self.feature_names),
        }


class _CountingScoreFn:
    """Wrap a raw detector score function with validation and an eval counter.

    The wrapped callable takes a ``(k, d)`` candidate matrix and must return
    ``k`` finite scores.  Anything else (NaN, wrong shape, non-numeric) raises
    ``ValueError`` immediately -- a score function that fabricates or drops
    values must never silently steer a counterfactual search.
    """

    def __init__(self, fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]) -> None:
        self._fn = fn
        self.n_evals = 0

    def __call__(self, candidates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        batch = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
        raw = self._fn(batch)
        scores = np.asarray(raw, dtype=np.float64).reshape(-1)
        if scores.shape[0] != batch.shape[0]:
            raise ValueError(
                f"detector_score_fn returned {scores.shape[0]} scores for "
                f"{batch.shape[0]} candidates"
            )
        if not np.all(np.isfinite(scores)):
            # NonFiniteScoreError (a ValueError) lets the gradient searches'
            # infeasibility barrier catch EXACTLY this case and repel the
            # optimizer, while every other contract violation still aborts.
            raise NonFiniteScoreError("detector_score_fn returned non-finite scores")
        self.n_evals += int(batch.shape[0])
        return scores

    def scalar(self, x: np.ndarray[Any, Any]) -> float:
        """Score a single point."""
        return float(self(x.reshape(1, -1))[0])


def _is_flipped(score: float, threshold: float, flagged_before: bool) -> bool:
    """Whether ``score`` sits on the opposite side of the flag boundary.

    Flag semantics across Mercury detectors are ``score > threshold``; the
    normal direction of a flip is therefore ``score <= threshold`` for a
    flagged point and ``score > threshold`` for an unflagged one.
    """
    if flagged_before:
        return score <= threshold
    return score > threshold


def _changed_indices(original: np.ndarray[Any, Any], candidate: np.ndarray[Any, Any]) -> list[int]:
    """Indices where the candidate genuinely differs from the original."""
    diff = ~np.isclose(original, candidate, rtol=_RTOL, atol=_ATOL)
    return [int(i) for i in np.flatnonzero(diff)]


def _singles_fixpoint(
    score: _CountingScoreFn,
    original: np.ndarray[Any, Any],
    current: np.ndarray[Any, Any],
    changed: list[int],
    threshold: float,
    flagged_before: bool,
) -> tuple[np.ndarray[Any, Any], list[int]]:
    """Revert single features until a full sweep keeps the flip for none.

    Deterministic order: largest ``|delta|`` first (ties by index) so the
    biggest spurious moves are undone first.  The terminating sweep -- one
    complete pass in which every remaining single revert was re-scored and
    none preserved the flip -- is exactly the verification that the result is
    minimal w.r.t. single-feature reverts.
    """
    current = current.copy()
    changed = list(changed)
    improved = True
    while improved:
        improved = False
        order = sorted(changed, key=lambda i: (-abs(float(current[i] - original[i])), i))
        for idx in order:
            trial = current.copy()
            trial[idx] = original[idx]
            if _is_flipped(score.scalar(trial), threshold, flagged_before):
                current = trial
                changed.remove(idx)
                improved = True
    return current, changed


def _greedy_minimize(
    score: _CountingScoreFn,
    original: np.ndarray[Any, Any],
    candidate: np.ndarray[Any, Any],
    threshold: float,
    flagged_before: bool,
    max_pair_evals: int,
) -> tuple[np.ndarray[Any, Any], bool]:
    """Deterministic greedy minimization of a *valid* counterfactual.

    Reverts changed features singly to a fixpoint, then tries pair reverts
    within ``max_pair_evals`` re-scorings (each accepted pair is followed by
    another singles fixpoint).  Returns the minimized point and ``True``:
    the closing singles sweep re-scored every remaining single revert and none
    preserved the flip, so minimality w.r.t. single-feature reverts is
    verified by construction.

    Args:
        score: Counting wrapper around the real detector score function.
        original: The point as detected.
        candidate: A candidate already verified to flip the detection.
        threshold: Decision threshold.
        flagged_before: Whether the original point was flagged.
        max_pair_evals: Re-scoring budget for the pair-revert stage.

    Returns:
        ``(minimized_candidate, True)``.
    """
    current, changed = _singles_fixpoint(
        score, original, candidate, _changed_indices(original, candidate), threshold, flagged_before
    )

    pair_evals = 0
    improved = True
    while improved and pair_evals < max_pair_evals and len(changed) >= 2:
        improved = False
        order = sorted(changed, key=lambda i: (-abs(float(current[i] - original[i])), i))
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                if pair_evals >= max_pair_evals:
                    break
                i, j = order[a], order[b]
                trial = current.copy()
                trial[i] = original[i]
                trial[j] = original[j]
                pair_evals += 1
                if _is_flipped(score.scalar(trial), threshold, flagged_before):
                    # Keep the pair revert, then re-close under single reverts.
                    current, changed = _singles_fixpoint(
                        score,
                        original,
                        trial,
                        [k for k in changed if k not in (i, j)],
                        threshold,
                        flagged_before,
                    )
                    improved = True
                    break
            if improved or pair_evals >= max_pair_evals:
                break

    # ``_singles_fixpoint`` always terminates with a complete clean sweep, so
    # the current point is verified single-revert minimal.
    return current, True


def _method_defaults(method: str) -> dict[str, Any]:
    """Per-method generator defaults tuned for detection-path score functions.

    Detection scores are typically compressed into ``[0, 1]``, so the Wachter
    proximity weight is kept small (validity first -- the minimization pass
    restores sparsity afterwards) and iteration counts are bounded because a
    real detector evaluation can be a full batch re-score.
    """
    if method == "wachter":
        return {"lambda_param": 0.05, "max_iterations": 500}
    if method == "dice":
        return {"proximity_weight": 0.2, "diversity_weight": 0.5, "max_iterations": 200}
    if method == "growing_spheres":
        return {"n_samples": 400, "step_size": 0.25, "max_iterations": 60}
    if method == "prototype":
        return {"n_prototypes": 5}
    if method == "genetic":
        return {}
    raise ValueError(
        f"Unknown detection counterfactual method: {method!r}; "
        f"expected one of {DETECTION_COUNTERFACTUAL_METHODS}"
    )


def _search_attempts(method: str, kwargs: dict[str, Any], spread: float) -> list[dict[str, Any]]:
    """Deterministic escalation schedule for the search-method parameters.

    The gradient-based searches (Wachter, DiCE) can stall on detection scores
    for two structural reasons: (a) the proximity term balances a weak
    far-from-boundary gradient at a stationary point short of the flip --
    Wachter et al.'s own remedy is to anneal the proximity weight; (b)
    piecewise-constant score functions (e.g. the tier's ECDF calibration)
    carry zero gradient, where only restarts seeded wide enough to *land*
    across the boundary can succeed.  Each attempt is tried in order and the
    first that yields a re-scored valid candidate wins, so results stay
    deterministic for a fixed seed.

    Args:
        method: Search method name.
        kwargs: Resolved generator kwargs for the first attempt.
        spread: Feature scale of the instance (drives restart jitter).

    Returns:
        Ordered list of generator-kwarg dicts (length 1 for the sampling
        searches, which have no proximity/gradient pathology).
    """
    if method == "wachter":
        lam = float(kwargs.get("lambda_param", 0.05))
        base_init = float(kwargs.get("init_scale", 0.1))
        return [
            dict(kwargs),
            {**kwargs, "lambda_param": lam * 0.1},
            {**kwargs, "lambda_param": 0.0, "init_scale": max(base_init, 0.3 * spread)},
            {**kwargs, "lambda_param": 0.0, "init_scale": max(base_init, 1.0 * spread)},
        ]
    if method == "dice":
        prox = float(kwargs.get("proximity_weight", 0.2))
        base_init = float(kwargs.get("init_scale", 0.1))
        return [
            dict(kwargs),
            {**kwargs, "proximity_weight": prox * 0.1},
            {**kwargs, "proximity_weight": 0.0, "init_scale": max(base_init, 0.3 * spread)},
            {**kwargs, "proximity_weight": 0.0, "init_scale": max(base_init, 1.0 * spread)},
        ]
    return [dict(kwargs)]


def explain_detection_counterfactual(
    detector_score_fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
    x: np.ndarray[Any, Any] | Sequence[float],
    threshold: float,
    feature_names: Sequence[str] | None = None,
    method: str = "wachter",
    *,
    feature_constraints: list[FeatureConstraint] | None = None,
    training_data: np.ndarray[Any, Any] | None = None,
    training_labels: np.ndarray[Any, Any] | None = None,
    seed: int = 0,
    n_restarts: int = 4,
    boundary_sharpness: float = 8.0,
    max_pair_evals: int = 200,
    **method_kwargs: Any,
) -> DetectionCounterfactual:
    """Explain a detection decision with a validated, minimized counterfactual.

    Works against ANY detection-path score function: the callable receives a
    ``(k, d)`` candidate matrix and returns ``k`` anomaly scores with the
    universal Mercury flag semantics ``score > threshold`` => anomalous.  The
    flip target is the *normal direction*: a flagged point is driven to
    ``score <= threshold`` ("what would have made this normal"), an unflagged
    point to ``score > threshold`` ("what would it take to trip detection").

    Correctness and minimality are both enforced by re-scoring through the
    real ``detector_score_fn`` -- the search method's own validity claim is
    never trusted:

    * ``flipped`` is recomputed from ``score_after`` vs ``threshold``.
    * every counterfactual is post-processed by a deterministic greedy
      minimization pass (single-feature reverts to a fixpoint, then pair
      reverts within ``max_pair_evals``), and ``minimal=True`` is set only
      when a complete re-scored single-revert sweep preserved no flip.

    Args:
        detector_score_fn: Real detection scorer ``(k, d) -> (k,)``.  Higher
            score = more anomalous.  Non-finite or mis-shaped outputs raise.
        x: The 1-D point whose detection is being explained.
        threshold: Decision threshold used by the detector for this point.
        feature_names: Optional labels (default ``feature_<i>``).
        method: ``"wachter"`` | ``"dice"`` | ``"growing_spheres"`` |
            ``"prototype"`` | ``"genetic"``.
        feature_constraints: Optional per-feature mutability/bound constraints
            forwarded to the search method.
        training_data: Real labelled points (required for ``prototype``).
        training_labels: 0/1 anomaly labels for ``training_data`` (required
            for ``prototype``; 0 = normal).
        seed: Deterministic seed -- the same seed, point and score function
            reproduce the same counterfactual bit-for-bit.
        n_restarts: Search restarts / candidate count requested from the
            generator (the best re-scored valid candidate is kept).
        boundary_sharpness: Slope of the sigmoid that maps raw scores onto the
            generators' probability contract ``p >= 0.5 <=> score > threshold``
            (only used *inside* the search; validation is on raw scores).
        max_pair_evals: Re-scoring budget for the pair-revert minimization
            stage.
        **method_kwargs: Extra generator parameters (override the tuned
            defaults, e.g. ``max_iterations`` or ``population_size``).

    Returns:
        A :class:`DetectionCounterfactual`.  When no candidate flips the
        detection, the closest candidate is still returned with
        ``flipped=False`` and ``minimal=False`` -- never a fabricated success.

    Raises:
        ValueError: On an unknown method, a non-1-D/non-finite ``x``, a
            non-finite threshold, a missing prototype training set, or a
            score function that violates its contract.
    """
    if method not in DETECTION_COUNTERFACTUAL_METHODS:
        raise ValueError(
            f"Unknown detection counterfactual method: {method!r}; "
            f"expected one of {DETECTION_COUNTERFACTUAL_METHODS}"
        )
    point = np.asarray(x, dtype=np.float64).reshape(-1)
    if point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError("x must be a non-empty, finite 1-D point")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if method == "prototype" and (training_data is None or training_labels is None):
        raise ValueError(
            "method='prototype' requires training_data and training_labels "
            "(real labelled points to draw target-class prototypes from)"
        )

    names = (
        [str(n) for n in feature_names]
        if feature_names is not None
        else [f"feature_{i}" for i in range(point.size)]
    )
    if len(names) != point.size:
        raise ValueError(f"feature_names length {len(names)} != feature count {point.size}")

    score = _CountingScoreFn(detector_score_fn)
    score_before = score.scalar(point)
    flagged_before = score_before > threshold
    # Generator probability contract: p >= 0.5 <=> score > threshold.  The
    # sigmoid keeps a usable gradient near the boundary for the optimizing
    # searches; every accept/reject decision below uses the RAW score.
    thr = float(threshold)
    sharp = float(boundary_sharpness)

    def _pred(candidates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return 1.0 / (1.0 + np.exp(-(score(candidates) - thr) * sharp))

    target_class = 0 if flagged_before else 1

    kwargs = _method_defaults(method)
    kwargs.update(method_kwargs)
    labels_arr = (
        np.asarray(training_labels).astype(int).ravel() if training_labels is not None else None
    )
    spread = float(np.maximum(1.0, np.abs(point)).mean())

    # Re-score every candidate through the real detector and keep the closest
    # genuinely-flipping one (deterministic tie-break by candidate order);
    # escalate through the annealing schedule until one attempt flips.
    best: np.ndarray[Any, Any] | None = None
    best_distance = np.inf
    fallback: np.ndarray[Any, Any] | None = None
    fallback_gap = np.inf
    for attempt_kwargs in _search_attempts(method, kwargs, spread):
        generator = create_counterfactual_generator(
            _pred,
            method=method,
            training_data=training_data,
            training_labels=labels_arr,
            feature_names=names,
            feature_constraints=feature_constraints,
            seed=seed,
            **attempt_kwargs,
        )
        cf_set = generator.generate(point, target_class=target_class, n_counterfactuals=n_restarts)
        candidates: list[Counterfactual] = list(cf_set.counterfactuals)
        for cf in candidates:
            cand = np.asarray(cf.counterfactual, dtype=np.float64).reshape(-1)
            if cand.shape != point.shape or not np.all(np.isfinite(cand)):
                continue
            s = score.scalar(cand)
            if _is_flipped(s, thr, flagged_before):
                dist = float(np.linalg.norm(cand - point))
                if dist < best_distance:
                    best, best_distance = cand, dist
            else:
                gap = abs(s - thr)
                if gap < fallback_gap:
                    fallback, fallback_gap = cand, gap
        if best is not None:
            break

    if best is not None:
        minimized, minimal = _greedy_minimize(
            score, point, best, thr, flagged_before, max_pair_evals
        )
        chosen = minimized
    else:
        chosen = fallback if fallback is not None else point.copy()
        minimal = False

    score_after = score.scalar(chosen)
    flipped = _is_flipped(score_after, thr, flagged_before)
    changed_idx = _changed_indices(point, chosen)
    changed = [
        ChangedFeature(
            name=names[i],
            old=float(point[i]),
            new=float(chosen[i]),
            delta=float(chosen[i] - point[i]),
        )
        for i in changed_idx
    ]

    return DetectionCounterfactual(
        original_x=point,
        counterfactual_x=chosen,
        changed_features=changed,
        score_before=score_before,
        score_after=score_after,
        threshold=thr,
        flipped=flipped,
        sparsity=len(changed),
        distance=float(np.linalg.norm(chosen - point)),
        method=method,
        minimal=bool(minimal and flipped),
        n_score_evals=score.n_evals,
        feature_names=names,
    )


def make_statistical_score_fn(
    detector: Any,
    context: np.ndarray[Any, Any],
    row_index: int,
) -> Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    """Adapt a fitted ``MercuryAnomalyDetector`` to the counterfactual seam.

    Scores a candidate row by substituting it for ``context[row_index]`` and
    re-running the detector's real :meth:`detect` path over the whole batch,
    then reading the score at ``row_index``.  This is deliberately NOT a
    row-local surrogate: batch-level behaviour (inversion guard, ensemble
    flip, residual filter) sees exactly the batch production saw, with one
    row hypothetically changed.  One full ``detect()`` per candidate.

    Args:
        detector: A fitted
            :class:`~omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector`.
        context: The 2-D batch the detection came from.
        row_index: Row being explained.

    Returns:
        Score function ``(k, d) -> (k,)`` over candidate replacement rows.

    Raises:
        ValueError: If ``context`` is not 2-D or ``row_index`` out of range.
    """
    base = np.array(context, dtype=np.float64, copy=True)
    if base.ndim != 2:
        raise ValueError(f"context must be 2-D, got shape {base.shape}")
    n_rows, n_features = base.shape
    if not 0 <= int(row_index) < n_rows:
        raise ValueError(f"row_index {row_index} out of range for {n_rows} context rows")
    idx = int(row_index)

    def _score(candidates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        cands = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
        if cands.shape[1] != n_features:
            raise ValueError(f"candidate width {cands.shape[1]} != context width {n_features}")
        out = np.empty(cands.shape[0], dtype=np.float64)
        for j in range(cands.shape[0]):
            batch = base.copy()
            batch[idx] = cands[j]
            result = detector.detect(batch)
            out[j] = float(np.asarray(result["scores"], dtype=np.float64)[idx])
        return out

    return _score


def make_tier_score_fn(
    ensemble: Any,
    series: np.ndarray[Any, Any],
    index: int,
    window_radius: int = 3,
) -> tuple[Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]], np.ndarray[Any, Any], list[str]]:
    """Adapt a fitted tier ``StreamingScoreEnsemble`` to the counterfactual seam.

    The tier's detectors are CONTEXTUAL (streaming / state-space members score
    a point against its neighborhood), so the counterfactual feature space is
    the window ``series[index-r .. index+r]`` rather than the single value: a
    candidate ``(k, w)`` matrix of replacement windows is scored by
    substituting each into the original series and re-running the fitted
    ensemble's real :meth:`score` path (all members + calibration + combiner).
    The greedy minimizer then prunes the window back to exactly the neighbors
    that must change — a genuinely contextual explanation ("this detection
    flips only if points 81–83 also come down").

    Args:
        ensemble: A fitted
            :class:`~omni_mercury_engine.detectors.detection_tier.StreamingScoreEnsemble`.
        series: The original 1-D series the detection came from.
        index: The point being explained.
        window_radius: Neighborhood half-width (clamped to the series bounds).

    Returns:
        ``(score_fn, x_window, feature_names)`` — the score function
        ``(k, w) -> (k,)``, the original window values, and per-position
        names (``series[i]``).

    Raises:
        ValueError: If ``series`` is not 1-D, ``index`` out of range, or
            ``window_radius`` negative.
    """
    base = np.asarray(series, dtype=np.float64).reshape(-1)
    if base.size == 0:
        raise ValueError("series must be a non-empty 1-D array")
    if not 0 <= int(index) < base.size:
        raise ValueError(f"index {index} out of range for series of length {base.size}")
    if window_radius < 0:
        raise ValueError(f"window_radius must be >= 0, got {window_radius}")
    pos = int(index)
    lo = max(0, pos - int(window_radius))
    hi = min(base.size, pos + int(window_radius) + 1)
    names = [f"series[{i}]" for i in range(lo, hi)]
    x_window = base[lo:hi].copy()

    def _score(candidates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        cands = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
        if cands.shape[1] != hi - lo:
            raise ValueError(f"tier candidates must have width {hi - lo}, got {cands.shape[1]}")
        out = np.empty(cands.shape[0], dtype=np.float64)
        for j in range(cands.shape[0]):
            modified = base.copy()
            modified[lo:hi] = cands[j]
            out[j] = float(np.asarray(ensemble.score(modified), dtype=np.float64)[pos])
        return out

    return _score, x_window, names


def make_symbolic_score_fn(
    module: Any,
) -> Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    """Adapt a ``SymbolicConstraintModule`` consensus to the counterfactual seam.

    The feature space is the per-detector score vector; the score is the
    module's grounded ``Consensus`` predicate
    (:meth:`~omni_mercury_engine.ml.symbolic_constraint.SymbolicConstraintModule.predict`),
    i.e. "how strongly do the rules see joint detector support for an
    anomaly".  Candidates are clamped to ``[0, 1]`` first -- the module's own
    ``forward`` applies the same clamp, so the adapter scores exactly what the
    rule engine would see.

    Args:
        module: A
            :class:`~omni_mercury_engine.ml.symbolic_constraint.SymbolicConstraintModule`.

    Returns:
        Score function ``(k, n_detectors) -> (k,)`` of consensus probabilities.
    """
    import torch

    def _score(candidates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        cands = np.clip(np.atleast_2d(np.asarray(candidates, dtype=np.float64)), 0.0, 1.0)
        with torch.no_grad():
            probs = module.predict(torch.as_tensor(cands, dtype=torch.float32))
        return np.asarray(probs.cpu().numpy(), dtype=np.float64).reshape(-1)

    return _score
