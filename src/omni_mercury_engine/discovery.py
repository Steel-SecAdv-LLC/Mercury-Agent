# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Discovery artifacts: what a candidate finding rests on, in a form that can be re-run.

A discovery instrument that reports a verdict and a sentence is asking to be
believed.  Each instrument here reports an **artifact** instead: the claim,
the named procedure that produced it, the numbers that procedure was given,
and a digest binding them together.  Whoever holds the record can run the
same procedure on the same numbers and see whether the claim comes out --
without the process that made it, and without trusting it.

Three instruments, three artifacts:

``novelty``
    "This observation is unlike the reference."  Carries the observation, the
    reference mean and scale, and the reference distances it was ranked
    among.  Replays from the record alone; against a reference of your own
    (:func:`replay`, ``reference=``) it answers whether the observation is
    still novel there.

``causal``
    "Within this window, X leads Y."  A Granger F-test per ordered pair of
    columns, corrected for the lags it searched (Bonferroni) and for the
    pairs it tested (Holm), so the chance of *any* false edge in one analysis
    is at most ``alpha``.  Carries the sufficient statistics of every pair
    tested, so the corrected verdict replays from the record alone; given the
    data (``data=``, checked against the recorded digest) the statistics
    themselves are recomputed.

``hypothesis``
    A named conclusion that follows from premises, each a measured verdict
    with its statistic and cut.  Carries the premises and the derivation
    steps, names the weakest premise and the value at which it fails, and
    replays against the other artifacts in the same record: when a premise's
    own artifact no longer holds, neither does the hypothesis.

The procedures are the single implementation of their arithmetic: the
instruments in :mod:`omni_mercury_engine.cognitive` call them, so a replay is
the production computation, not a re-description of it.

Everything is plain JSON.  Vectors longer than :data:`MAX_FEATURES` are not
carried (the record stays bounded); such an artifact says so and why.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import linalg, stats

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "CANNOT_REPLAY",
    "FAILS",
    "HOLDS",
    "MAX_FEATURES",
    "MIN_REFERENCE_STD",
    "SCHEMA",
    "SERVED_REPLAY_BOUNDS",
    "GrangerStats",
    "ReplayBounds",
    "causal_artifact",
    "causal_artifact_from_analysis",
    "contaminated_rows",
    "granger_analysis",
    "granger_family",
    "granger_pair",
    "holm_adjust",
    "hypothesis_artifact",
    "ledger_form",
    "level_stability",
    "matrix_digest",
    "novelty_artifact",
    "rank_p_value",
    "replay",
    "replay_record",
    "rms_standardized_distance",
    "sealed",
    "verdict",
]

SCHEMA = "mercury.discovery/v1"

#: Replay outcomes. HOLDS: the claim reproduces. FAILS: it does not (the
#: numbers changed, the artifact was altered, or it is false against what was
#: supplied). CANNOT_REPLAY: the artifact does not carry enough to decide, and
#: says why -- never read as HOLDS.
HOLDS = "HOLDS"
FAILS = "FAILS"
CANNOT_REPLAY = "CANNOT_REPLAY"

#: A feature whose reference standard deviation is at or below this has no
#: scale to standardize by and is left out of the distance.
MIN_REFERENCE_STD = 1e-12

#: Longest vector an artifact carries inline.
MAX_FEATURES = 256

#: Robust z-score beyond which a value contaminates its time step.
TRIM_K = 4.0

#: Level below which a window's level is declared unstable (Bonferroni across columns).
STABILITY_ALPHA = 0.001

#: Fewest usable rows the stability check can judge (three blocks of ten).
MIN_ASSESSABLE_ROWS = 30


@dataclass(frozen=True)
class ReplayBounds:
    """The largest causal finding a replay will recompute.

    A record is whatever the caller sends.  Its causal artifact names the
    width and the lag it was computed at, and recomputing one costs time that
    grows with the square of the width and with the lag: at 1,000 rows, 64
    columns took 4.2 s at lag 3 and did not finish in ten minutes at lag 50.
    A replay that serves other people therefore refuses anything larger than
    these and says so; an offline replay (``bounds=None``) is the caller's own
    time and is not bounded.

    Attributes:
        max_columns: Widest window analysed.
        max_lag: Longest lag tried.
    """

    max_columns: int
    max_lag: int

    @property
    def max_pairs(self) -> int:
        """Ordered pairs among ``max_columns`` columns."""
        return self.max_columns * (self.max_columns - 1)

    def exceeded_by(self, inputs: Mapping[str, Any]) -> str | None:
        """Why a causal artifact's inputs are over these bounds, or ``None``."""
        shape = [int(v) for v in inputs["input_shape"]]
        columns = shape[1] if len(shape) == 2 else 0
        lag, pairs = int(inputs["max_lag"]), len(inputs["tests"])
        if 0 < columns <= self.max_columns and 0 < lag <= self.max_lag and pairs <= self.max_pairs:
            return None
        return (
            f"this replay recomputes at most {self.max_columns} columns at lags up to "
            f"{self.max_lag} ({self.max_pairs} pairs); the artifact declares {columns} "
            f"column(s) at lag {lag} with {pairs} pair(s). Replay it offline: mercury-agent replay"
        )


#: What a served replay accepts: the shipped instrument analyses the five
#: leading columns at lags up to five, which costs 1.35 s to recompute at the
#: largest window the flagship route takes (65,536 rows; measured 2026-11-21).
SERVED_REPLAY_BOUNDS = ReplayBounds(max_columns=5, max_lag=5)

#: Relative tolerance for "the recomputed number is the recorded number".
_RTOL = 1e-9
_ATOL = 1e-12


# --------------------------------------------------------------------------- #
# Procedures: the one implementation of each instrument's arithmetic.
# --------------------------------------------------------------------------- #
def rms_standardized_distance(
    observation: Any, mean: Any, std: Any, min_std: float = MIN_REFERENCE_STD
) -> float | None:
    """Root-mean-square standardized deviation of ``observation`` from a reference.

    A diagonal Mahalanobis distance: features whose reference standard
    deviation is at or below ``min_std`` are left out.

    Args:
        observation: The vector scored.
        mean: The reference mean, same length.
        std: The reference standard deviation, same length.
        min_std: Smallest scale a feature may be standardized by.

    Returns:
        The distance, or ``None`` when no feature has a usable scale.
    """
    x = np.asarray(observation, dtype=float).reshape(-1)
    mu = np.asarray(mean, dtype=float).reshape(-1)
    sd = np.asarray(std, dtype=float).reshape(-1)
    if not (x.shape == mu.shape == sd.shape) or x.size == 0:
        return None
    valid = sd > min_std
    if not np.any(valid):
        return None
    z = (x[valid] - mu[valid]) / sd[valid]
    return float(np.sqrt(np.mean(np.square(z))))


def rank_p_value(distance: float, calibration: Any) -> float:
    """The share of reference distances at least as large as ``distance``.

    ``(1 + #{c >= distance}) / (n + 1)``: distribution-free, and at or below
    ``alpha`` for no more than a fraction ``alpha`` of observations that are
    exchangeable with the reference.

    Args:
        distance: The observation's standardized distance.
        calibration: The reference distances.

    Returns:
        The p-value in ``(0, 1]``; ``1.0`` when there is no reference.
    """
    scores = np.asarray(calibration, dtype=float).reshape(-1)
    return float((1 + int(np.count_nonzero(scores >= distance))) / (scores.size + 1))


@dataclass(frozen=True)
class GrangerStats:
    """One ordered pair's Granger test at the lag with the smallest p-value.

    Attributes:
        lag: The lag reported.
        lags_tested: How many lags were tried (what the lag correction multiplies by).
        rss_restricted: Residual sum of squares of Y on its own lags.
        rss_unrestricted: Residual sum of squares of Y on its own lags and X's.
        df1: Numerator degrees of freedom (the X lags added).
        df2: Denominator degrees of freedom.
        f_statistic: The nested-model F statistic.
        p_raw: Its p-value at that lag, uncorrected.
        p_lag_adjusted: ``min(1, lags_tested * p_raw)``: corrected for having
            searched the lags for the smallest p-value.
        partial_r2: ``(rss_restricted - rss_unrestricted) / rss_restricted``:
            the share of Y's unexplained variance that X's lags explain.
    """

    lag: int
    lags_tested: int
    rss_restricted: float
    rss_unrestricted: float
    df1: int
    df2: int
    f_statistic: float
    p_raw: float
    p_lag_adjusted: float
    partial_r2: float


def _f_test(rss_r: float, rss_u: float, df1: int, df2: int) -> tuple[float, float]:
    """The nested-model F statistic and its upper-tail p-value."""
    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
    return float(f_stat), float(stats.f.sf(f_stat, df1, df2))


def granger_pair(x: Any, y: Any, max_lag: int, clean: Any = None) -> GrangerStats | None:
    """Test whether ``x`` Granger-causes ``y``, honestly about the lags it searched.

    For each lag up to ``max_lag`` it compares Y regressed on its own lags
    with Y regressed on its own lags and X's (an F-test), and reports the lag
    with the smallest p-value.  That search is a multiple comparison: the
    smallest of several p-values is not a p-value.  ``p_lag_adjusted``
    multiplies it by the number of lags tried (Bonferroni), which is.

    Args:
        x: The candidate cause series.
        y: The candidate effect series, same length.
        max_lag: Largest lag to try.
        clean: Optional boolean per time step (:func:`contaminated_rows`,
            negated).  A regression row is used only when its own time step
            and every lagged time step it reads are clean.

    Returns:
        The statistics at the reported lag, or ``None`` when no lag could be
        tested (series too short, or a degenerate fit).
    """
    xs = np.asarray(x, dtype=float).reshape(-1)
    ys = np.asarray(y, dtype=float).reshape(-1)
    n = xs.size
    if n != ys.size or n <= max_lag + 2:
        return None
    usable = None if clean is None else np.asarray(clean, dtype=bool).reshape(-1)
    best: tuple[float, int, float, float, int, int, float] | None = None
    tested = 0
    for lag in range(1, max_lag + 1):
        if n <= 2 * lag + 2:
            continue
        target = ys[lag:]
        own = np.column_stack([ys[lag - i - 1 : n - i - 1] for i in range(lag)])
        other = np.column_stack([xs[lag - i - 1 : n - i - 1] for i in range(lag)])
        restricted = np.column_stack([np.ones(target.size), own])
        unrestricted = np.column_stack([restricted, other])
        if usable is not None:
            keep = np.ones(target.size, dtype=bool)
            for back in range(lag + 1):
                keep &= usable[lag - back : n - back]
            target, restricted, unrestricted = target[keep], restricted[keep], unrestricted[keep]
        rows = target.size
        df1 = unrestricted.shape[1] - restricted.shape[1]
        df2 = rows - unrestricted.shape[1]
        if df2 <= 0:
            continue
        try:
            beta_r = linalg.lstsq(restricted, target)[0]
            rss_r = float(np.sum((target - restricted @ beta_r) ** 2))
            beta_u = linalg.lstsq(unrestricted, target)[0]
            rss_u = float(np.sum((target - unrestricted @ beta_u) ** 2))
        except (linalg.LinAlgError, ValueError):
            continue
        if rss_u <= 0 or rss_r <= 0:
            continue
        f_stat, p_raw = _f_test(rss_r, rss_u, df1, df2)
        if not math.isfinite(p_raw):
            continue
        tested += 1
        if best is None or p_raw < best[0]:
            best = (p_raw, lag, rss_r, rss_u, df1, df2, f_stat)
    if best is None:
        return None
    p_raw, lag, rss_r, rss_u, df1, df2, f_stat = best
    return GrangerStats(
        lag=lag,
        lags_tested=tested,
        rss_restricted=rss_r,
        rss_unrestricted=rss_u,
        df1=df1,
        df2=df2,
        f_statistic=f_stat,
        p_raw=p_raw,
        p_lag_adjusted=min(1.0, tested * p_raw),
        partial_r2=max(0.0, (rss_r - rss_u) / rss_r),
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjustment: family-wise valid under any dependence.

    Args:
        p_values: The family's p-values, in any order.

    Returns:
        The adjusted p-values in the same order.  Rejecting those at or below
        ``alpha`` keeps the chance of any false rejection at or below ``alpha``.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted


def contaminated_rows(matrix: Any, k: float = TRIM_K) -> Any:
    """Time steps at which any column lies more than ``k`` robust deviations from its median.

    A flagged window usually holds a shock that moves several columns at
    once.  Left in, those few rows make every column "predict" every other:
    measured on independent columns with a six-row common shift, the family
    test reported an edge in 82-100% of analyses.  Excluding the rows brings
    that to 1.5-6% (``artifacts/causal_discovery.json``).

    Args:
        matrix: ``(time, variables)``.
        k: Robust z-score (median and 1.4826 x MAD) beyond which a value
            contaminates its time step.

    Returns:
        A boolean array over time steps, ``True`` where contaminated.
    """
    data = np.asarray(matrix, dtype=float)
    median = np.median(data, axis=0)
    scale = np.median(np.abs(data - median), axis=0) * 1.4826
    scale = np.where(scale < MIN_REFERENCE_STD, np.inf, scale)
    return (np.abs(data - median) / scale > k).any(axis=1)


def level_stability(matrix: Any, alpha: float = STABILITY_ALPHA) -> dict[str, Any]:
    """Whether each column's level is the same across consecutive blocks of the window.

    Precedence can only be read from a window whose level is stable.  When
    every column steps to a new level together (a regime change, a trend, a
    wandering series) each one's past predicts each other's future and the
    F-test finds edges that are not there: 100% of analyses, measured.

    The window is cut into up to ten consecutive blocks of at least ten rows.
    Per column, the spread of the block means is compared with what the
    within-block variation (with its serial correlation, by a Bartlett
    long-run variance) allows: a chi-square test, Bonferroni across columns.

    This gate is deliberately fail-closed and it is not free.  It declines
    every shifted window measured, and also 5-8% of stable independent
    windows and 77-92% of strongly autocorrelated ones (AR(1) 0.7), because
    short blocks understate a persistent series' long-run variance.  A
    declined window gets no edge at all, never a wrong one.

    Args:
        matrix: ``(time, variables)``, contaminated rows already removed.
        alpha: Level below which the window is declared unstable.

    Returns:
        ``{"stable": bool, "p_adjusted": float | None, "blocks": int,
        "alpha": float}`` and, when not stable, ``"reason"``.
    """
    data = np.asarray(matrix, dtype=float)
    rows = data.shape[0]
    blocks = min(10, rows // 10)
    report: dict[str, Any] = {"stable": False, "p_adjusted": None, "blocks": blocks, "alpha": alpha}
    if blocks < 3:
        report["reason"] = (
            f"{rows} usable rows are too few to check the level is stable "
            f"(needs at least {MIN_ASSESSABLE_ROWS})"
        )
        return report
    length = rows // blocks
    smallest = 1.0
    for column in range(data.shape[1]):
        series = data[: blocks * length, column].reshape(blocks, length)
        means = series.mean(axis=1)
        residual = (series - means[:, None]).reshape(-1)
        size = residual.size
        bandwidth = min(length - 1, int(np.floor(4 * (size / 100.0) ** (2.0 / 9.0))))
        long_run = float(residual @ residual) / size
        for lag in range(1, bandwidth + 1):
            weight = 1.0 - lag / (bandwidth + 1.0)
            long_run += 2.0 * weight * float(residual[lag:] @ residual[:-lag]) / size
        if long_run <= 0:
            continue  # a constant column has a level, and it is stable
        statistic = length * float(np.sum((means - series.mean()) ** 2)) / long_run
        smallest = min(smallest, float(stats.chi2.sf(statistic, blocks - 1)))
    adjusted = min(1.0, smallest * data.shape[1])
    report["p_adjusted"] = adjusted
    report["stable"] = adjusted >= alpha
    if not report["stable"]:
        report["reason"] = (
            "the level of at least one column differs between blocks of the window "
            f"(p={adjusted:.2g} < {alpha}); precedence cannot be read from a window "
            "whose level is not stable"
        )
    return report


def granger_family(
    matrix: Any, max_lag: int, names: Sequence[str] | None = None, clean: Any = None
) -> list[dict[str, Any]]:
    """Every ordered pair of columns, tested and corrected as one family.

    Args:
        matrix: ``(time, variables)``.
        max_lag: Largest lag to try.
        names: Column names; ``X0``, ``X1``, ... when omitted.
        clean: Optional boolean per time step; see :func:`granger_pair`.

    Returns:
        One entry per ordered pair that could be tested, in column order, each
        with the pair's statistics and ``p_adjusted`` (lag-corrected, then
        Holm across the pairs).
    """
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"granger_family expects a 2-D matrix, got shape {data.shape}")
    count = data.shape[1]
    labels = list(names) if names is not None else [f"X{i}" for i in range(count)]
    entries: list[dict[str, Any]] = []
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            found = granger_pair(data[:, i], data[:, j], max_lag, clean)
            if found is None:
                continue
            entries.append(
                {
                    "source": labels[i],
                    "target": labels[j],
                    "lag": found.lag,
                    "lags_tested": found.lags_tested,
                    "rss_restricted": found.rss_restricted,
                    "rss_unrestricted": found.rss_unrestricted,
                    "df1": found.df1,
                    "df2": found.df2,
                }
            )
    return _correct_family(entries)


def granger_analysis(
    matrix: Any, max_lag: int, names: Sequence[str] | None = None
) -> dict[str, Any]:
    """The whole causal analysis of one window: what was set aside, whether it could be read, what was found.

    Contaminated time steps are set aside (:func:`contaminated_rows`); the
    rest must have a stable level (:func:`level_stability`) or the window is
    not assessed; then every ordered pair is tested as one family
    (:func:`granger_family`).

    Args:
        matrix: ``(time, variables)``.
        max_lag: Largest lag to try.
        names: Column names.

    Returns:
        ``{"assessed", "tests", "rows", "rows_excluded", "trim_k",
        "stability"}`` and, when not assessed, ``"not_assessed_because"``.
        ``tests`` is empty when the window was not assessed.
    """
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"granger_analysis expects a 2-D matrix, got shape {data.shape}")
    contaminated = contaminated_rows(data)
    stability = level_stability(data[~contaminated])
    analysis: dict[str, Any] = {
        "assessed": bool(stability["stable"]),
        "tests": [],
        "rows": int(data.shape[0]),
        "rows_excluded": int(np.count_nonzero(contaminated)),
        "trim_k": TRIM_K,
        "stability": {k: stability[k] for k in ("p_adjusted", "blocks", "alpha")},
    }
    if not stability["stable"]:
        analysis["not_assessed_because"] = stability["reason"]
        return analysis
    analysis["tests"] = granger_family(data, max_lag, names, ~contaminated)
    return analysis


def _correct_family(tests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """From each pair's sufficient statistics to its family-corrected p-value."""
    out: list[dict[str, Any]] = []
    lag_adjusted: list[float] = []
    for test in tests:
        rss_r, rss_u = float(test["rss_restricted"]), float(test["rss_unrestricted"])
        df1, df2 = int(test["df1"]), int(test["df2"])
        f_stat, p_raw = _f_test(rss_r, rss_u, df1, df2)
        p_lag = min(1.0, int(test["lags_tested"]) * p_raw)
        lag_adjusted.append(p_lag)
        out.append(
            {
                "source": str(test["source"]),
                "target": str(test["target"]),
                "lag": int(test["lag"]),
                "lags_tested": int(test["lags_tested"]),
                "rss_restricted": rss_r,
                "rss_unrestricted": rss_u,
                "df1": df1,
                "df2": df2,
                "f_statistic": f_stat,
                "p_raw": p_raw,
                "p_lag_adjusted": p_lag,
                "partial_r2": max(0.0, (rss_r - rss_u) / rss_r),
            }
        )
    for entry, p_adjusted in zip(out, holm_adjust(lag_adjusted), strict=True):
        entry["p_adjusted"] = p_adjusted
    return out


def matrix_digest(matrix: Any) -> str:
    """SHA-256 of a matrix as float64 little-endian with its shape: what ``data=`` is checked against."""
    data = np.ascontiguousarray(np.asarray(matrix, dtype="<f8"))
    sha = hashlib.sha256()
    sha.update(repr(data.shape).encode("ascii"))
    sha.update(data.tobytes())
    return sha.hexdigest()


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def _seal(kind: str, procedure: str, claim: dict[str, Any], inputs: dict[str, Any]) -> str:
    """The digest that binds a claim to the procedure and the numbers it came from."""
    body = {"kind": kind, "procedure": procedure, "claim": claim, "inputs": inputs}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact(
    kind: str,
    procedure: str,
    claim: dict[str, Any],
    inputs: dict[str, Any],
    challenge: dict[str, Any],
    replayable: str,
    reason: str | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": kind,
        "procedure": procedure,
        "claim": claim,
        "inputs": inputs,
        "challenge": challenge,
        "replayable": replayable,
        "digest": _seal(kind, procedure, claim, inputs),
    }
    if reason:
        artifact["not_replayable_because"] = reason
    return artifact


def novelty_artifact(
    *,
    observation: Sequence[float] | None,
    reference_mean: Sequence[float] | None,
    reference_std: Sequence[float] | None,
    reference_n: int,
    calibration: Sequence[float] | None,
    alpha: float | None,
    standardized_distance: float,
    p_value: float | None,
    is_novel: bool,
    disclose: bool = True,
) -> dict[str, Any] | None:
    """The novelty verdict as an artifact, or ``None`` when no verdict was measured.

    Args:
        observation: The scored vector.
        reference_mean: The reference mean it was scored against.
        reference_std: The reference standard deviation.
        reference_n: Observations behind that reference.
        calibration: The reference distances the p-value is a rank among.
        alpha: The level of the verdict.
        standardized_distance: The measured distance.
        p_value: Its rank p-value.
        is_novel: The verdict.
        disclose: Whether the reference may be carried.  A reference built
            from several callers' observations is theirs, not the reader's:
            with ``False`` the artifact carries digests in place of the
            vectors, and says it cannot be replayed and why.

    Returns:
        The artifact, or ``None`` when the verdict was undetermined (warm-up):
        there is then no claim to carry.
    """
    if (
        observation is None
        or reference_mean is None
        or reference_std is None
        or calibration is None
        or alpha is None
        or p_value is None
    ):
        return None
    claim = {
        "is_novel": bool(is_novel),
        "p_value": float(p_value),
        "standardized_distance": float(standardized_distance),
    }
    challenge = {
        "statement": (
            "Novel means the observation's distance ranks in the top alpha of the "
            "distances the reference observations themselves had."
        ),
        "withdrawn_if": f"p_value > {alpha}",
        "how": (
            "Replay from this record to check the arithmetic; replay with a reference "
            "of your own to ask whether it is novel there."
        ),
    }
    withheld = None
    if not disclose:
        withheld = (
            "the reference is shared by unauthenticated callers and is not disclosed; "
            "call with credentials for a reference of your own, or run locally"
        )
    elif len(observation) > MAX_FEATURES:
        withheld = (
            f"{len(observation)} features exceed the {MAX_FEATURES}-feature bound a record carries"
        )
    if withheld is not None:
        inputs: dict[str, Any] = {
            "features": len(observation),
            "reference_n": int(reference_n),
            "calibration_n": len(calibration),
            "alpha": float(alpha),
            "observation_digest": matrix_digest(observation),
            "reference_digest": matrix_digest([reference_mean, reference_std]),
        }
        return _artifact(
            "novelty",
            "rank_of_rms_standardized_distance/v1",
            claim,
            inputs,
            challenge,
            "no",
            withheld,
        )
    inputs = {
        "observation": [float(v) for v in observation],
        "reference_mean": [float(v) for v in reference_mean],
        "reference_std": [float(v) for v in reference_std],
        "reference_n": int(reference_n),
        "calibration": [float(v) for v in calibration],
        "alpha": float(alpha),
        "min_std": MIN_REFERENCE_STD,
    }
    return _artifact(
        "novelty", "rank_of_rms_standardized_distance/v1", claim, inputs, challenge, "from_record"
    )


def causal_artifact(
    matrix: Any, *, max_lag: int, alpha: float, names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Run the causal analysis of a window and return it as an artifact.

    Args:
        matrix: ``(time, variables)``: the window analysed.
        max_lag: Largest lag tried.
        alpha: Family-wise level.
        names: Column names.

    Returns:
        The artifact.  A window that could not be assessed still yields one:
        its claim says so, and why, and names no edge.
    """
    data = np.asarray(matrix, dtype=float)
    return causal_artifact_from_analysis(
        granger_analysis(data, max_lag, names),
        input_shape=data.shape,
        input_digest=matrix_digest(data),
        max_lag=max_lag,
        alpha=alpha,
    )


def causal_artifact_from_analysis(
    analysis: Mapping[str, Any],
    *,
    input_shape: Sequence[int],
    input_digest: str,
    max_lag: int,
    alpha: float,
) -> dict[str, Any]:
    """The causal artifact for an analysis that has already been run.

    Args:
        analysis: What :func:`granger_analysis` returned for the window.
        input_shape: The window's shape.
        input_digest: Its :func:`matrix_digest`.
        max_lag: Largest lag tried.
        alpha: Family-wise level.

    Returns:
        The artifact.
    """
    tests = list(analysis.get("tests") or [])
    inputs: dict[str, Any] = {
        "input_digest": input_digest,
        "input_shape": [int(v) for v in input_shape],
        "max_lag": int(max_lag),
        "alpha": float(alpha),
        "corrections": ["bonferroni_over_lags", "holm_over_pairs"],
        "rows_excluded": int(analysis.get("rows_excluded", 0)),
        "trim_k": float(analysis.get("trim_k", TRIM_K)),
        "stability": dict(analysis.get("stability") or {}),
        "tests": [
            {
                key: test[key]
                for key in (
                    "source",
                    "target",
                    "lag",
                    "lags_tested",
                    "rss_restricted",
                    "rss_unrestricted",
                    "df1",
                    "df2",
                )
            }
            for test in tests
        ],
    }
    assessed = bool(analysis.get("assessed"))
    claim: dict[str, Any] = {
        "assessed": assessed,
        "edges": _edges(tests, alpha) if assessed else [],
        "pairs_tested": len(tests),
    }
    if not assessed:
        claim["not_assessed_because"] = str(analysis.get("not_assessed_because", "not assessed"))
    challenge = {
        "statement": (
            "An edge means X's past improves the prediction of Y beyond Y's own past, "
            "in the ordinary rows of this window, after correcting for the lags searched "
            "and the pairs tested. It is predictive precedence, not proof of mechanism. "
            "A window whose level is not stable is not assessed at all."
        ),
        "withdrawn_if": f"p_adjusted > {alpha}",
        "how": (
            "Replay from this record to re-derive every corrected p-value from the "
            "recorded sums of squares; replay with the data to recompute the sums of "
            "squares themselves; test a fresh window to see whether the edge recurs."
        ),
    }
    return _artifact(
        "causal", "granger_f_bonferroni_holm/v2", claim, inputs, challenge, "from_record"
    )


def ledger_form(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """An artifact as a ledger keeps it: the claim and the digest, without the inputs.

    The inputs are the caller's numbers (an observation, the sums of squares
    of their window).  They go back to the caller in the response; a ledger
    that outlives the request keeps what was claimed and the digest that
    identifies the full artifact, so one presented later can be matched to it.

    Args:
        artifact: A full discovery artifact.

    Returns:
        ``{"schema", "kind", "procedure", "claim", "digest", "replayable":
        "no", "not_replayable_because"}``.
    """
    return {
        "schema": artifact.get("schema"),
        "kind": artifact.get("kind"),
        "procedure": artifact.get("procedure"),
        "claim": artifact.get("claim"),
        "digest": artifact.get("digest"),
        "replayable": "no",
        "not_replayable_because": (
            "the ledger keeps the claim and its digest; the full artifact was "
            "returned with the decision"
        ),
    }


def _edges(tests: Sequence[Mapping[str, Any]], alpha: float) -> list[dict[str, Any]]:
    return [
        {
            "source": t["source"],
            "target": t["target"],
            "lag": t["lag"],
            "partial_r2": t["partial_r2"],
            "p_adjusted": t["p_adjusted"],
        }
        for t in tests
        if t["p_adjusted"] <= alpha
    ]


def _margin(statistic: float, op: str, cut: float) -> float | None:
    """How far a premise is from failing, as a ratio: at or above 1 it holds."""
    if op == ">=":
        return statistic / cut if cut > 0 else None
    if op == "<=":
        return cut / max(statistic, 1e-300) if cut > 0 else None
    return None


def verdict(statistic: float, op: str, cut: float) -> bool:
    """Whether ``statistic`` satisfies ``op cut`` (``>=`` or ``<=``)."""
    if op == ">=":
        return statistic >= cut
    if op == "<=":
        return statistic <= cut
    raise ValueError(f"unknown comparison {op!r}")


def hypothesis_artifact(
    *,
    conclusion: str,
    statement: str,
    premises: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """A derived conclusion as an artifact: its premises, its steps, its weakest link.

    Each premise is a measured verdict: ``{"id", "statement", "statistic",
    "op", "cut", "source"}`` where ``source`` names what measured it
    (``"detection"``, ``"novelty"`` or ``"causal"``, the latter two being
    artifacts in the same record).

    Args:
        conclusion: The derived conclusion's identifier.
        statement: The conclusion in a sentence.
        premises: The measured verdicts it rests on.
        steps: The derivation: ``{"rule", "premises", "conclusion"}`` each.

    Returns:
        The artifact, or ``None`` without premises or steps.
    """
    if not premises or not steps:
        return None
    listed: list[dict[str, Any]] = []
    for premise in premises:
        statistic, cut, op = float(premise["statistic"]), float(premise["cut"]), str(premise["op"])
        entry = {
            "id": str(premise["id"]),
            "statement": str(premise["statement"])[:240],
            "statistic": statistic,
            "op": op,
            "cut": cut,
            "source": str(premise["source"]),
        }
        for key in ("edge",):
            if key in premise:
                entry[key] = premise[key]
        listed.append(entry)
    inputs = {
        "premises": listed,
        "steps": [
            {
                "rule": str(step["rule"]),
                "premises": [str(p) for p in step["premises"]],
                "conclusion": str(step["conclusion"]),
            }
            for step in steps
        ],
    }
    holding = all(verdict(p["statistic"], p["op"], p["cut"]) for p in listed)
    claim = {"conclusion": str(conclusion), "statement": str(statement)[:240], "holds": holding}
    return _artifact(
        "hypothesis",
        "conjunction_of_measured_verdicts/v1",
        claim,
        inputs,
        _weakest_link(listed),
        "from_record",
    )


def _weakest_link(premises: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The premise closest to failing, and the value at which it fails."""
    ranked = [
        (margin, premise)
        for premise in premises
        if (margin := _margin(premise["statistic"], premise["op"], premise["cut"])) is not None
    ]
    challenge: dict[str, Any] = {
        "statement": (
            "The conclusion follows only while every premise holds. It carries no "
            "probability of its own: it is exactly as good as its weakest premise."
        ),
        "how": (
            "Replay each premise's own artifact; the conclusion is withdrawn the moment one fails."
        ),
    }
    if ranked:
        margin, weakest = min(ranked, key=lambda item: item[0])
        fails = "<" if weakest["op"] == ">=" else ">"
        challenge["weakest_link"] = {
            "premise": weakest["id"],
            "statement": weakest["statement"],
            "margin": margin,
            "source": weakest["source"],
        }
        challenge["withdrawn_if"] = f"{weakest['id']}: statistic {fails} {weakest['cut']}"
    return challenge


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_RTOL, abs_tol=_ATOL)


def _report(artifact: Mapping[str, Any], status: str, detail: str, **extra: Any) -> dict[str, Any]:
    report = {
        "kind": artifact.get("kind"),
        "status": status,
        "detail": detail,
        "recorded": artifact.get("claim"),
    }
    report.update(extra)
    return report


def sealed(artifact: Any) -> bool:
    """Whether an artifact's digest matches its own claim, procedure and inputs."""
    return hasattr(artifact, "get") and artifact.get("schema") == SCHEMA and _sealed(artifact)


def _sealed(artifact: Mapping[str, Any]) -> bool:
    try:
        expected = _seal(
            str(artifact["kind"]),
            str(artifact["procedure"]),
            dict(artifact["claim"]),
            dict(artifact["inputs"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return expected == artifact.get("digest")


def _reference_from_rows(rows: Any, window: int) -> tuple[Any, Any, list[float], int]:
    """A reference built from rows the way the instrument builds its own.

    Each row is scored against the rows before it and then folded in, so the
    reference distances are out-of-sample, as in production.
    """
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[0] < 3:
        raise ValueError("a reference needs at least three rows of the observation's width")
    # Running mean and sum of squared deviations (Welford), exactly as the
    # instrument keeps them, so each row costs its own width: scoring every
    # row against a freshly recomputed mean and deviation of all the rows
    # before it took time quadratic in the rows (measured at 16 columns:
    # 0.72 s for 4,000 rows, 13.4 s for 16,000).
    mean = np.zeros(data.shape[1])
    squares = np.zeros(data.shape[1])
    distances: list[float] = []
    for count, row in enumerate(data):
        if count >= 2:
            found = rms_standardized_distance(row, mean, np.sqrt(squares / (count - 1)))
            if found is not None:
                distances.append(found)
        delta = row - mean
        mean = mean + delta / (count + 1)
        squares = squares + delta * (row - mean)
    rows_seen = int(data.shape[0])
    return mean, np.sqrt(squares / (rows_seen - 1)), distances[-window:], rows_seen


def _replay_novelty(artifact: Mapping[str, Any], reference: Any) -> dict[str, Any]:
    if artifact.get("replayable") != "from_record":
        return _report(
            artifact, CANNOT_REPLAY, str(artifact.get("not_replayable_because", "not replayable"))
        )
    inputs, claim = artifact["inputs"], artifact["claim"]
    observation = inputs["observation"]
    alpha = float(inputs["alpha"])
    if reference is not None:
        try:
            mean, std, calibration, count = _reference_from_rows(
                reference, max(len(inputs["calibration"]), 20)
            )
        except ValueError as exc:
            return _report(artifact, CANNOT_REPLAY, str(exc))
        if np.asarray(mean).size != len(observation):
            return _report(
                artifact,
                CANNOT_REPLAY,
                f"the reference has {np.asarray(mean).size} features; the observation has {len(observation)}",
            )
        if len(calibration) < 20:
            return _report(
                artifact,
                CANNOT_REPLAY,
                "the reference yields fewer than 20 distances to rank among",
            )
        distance = rms_standardized_distance(observation, mean, std, float(inputs["min_std"]))
        if distance is None:
            return _report(artifact, CANNOT_REPLAY, "the reference has no feature with a scale")
        p_value = rank_p_value(distance, calibration)
        novel = p_value <= alpha
        recomputed = {
            "is_novel": novel,
            "p_value": p_value,
            "standardized_distance": distance,
            "reference_rows": count,
        }
        if novel == bool(claim["is_novel"]):
            return _report(
                artifact,
                HOLDS,
                f"against the supplied reference the verdict is the same (p={p_value:.4g}, alpha={alpha})",
                recomputed=recomputed,
            )
        return _report(
            artifact,
            FAILS,
            f"against the supplied reference the verdict changes: is_novel={novel} (p={p_value:.4g}, alpha={alpha})",
            recomputed=recomputed,
        )
    distance = rms_standardized_distance(
        observation, inputs["reference_mean"], inputs["reference_std"], float(inputs["min_std"])
    )
    if distance is None:
        return _report(artifact, FAILS, "the recorded reference has no feature with a scale")
    p_value = rank_p_value(distance, inputs["calibration"])
    recomputed = {
        "is_novel": p_value <= alpha,
        "p_value": p_value,
        "standardized_distance": distance,
    }
    same = (
        _close(distance, float(claim["standardized_distance"]))
        and _close(p_value, float(claim["p_value"]))
        and recomputed["is_novel"] == bool(claim["is_novel"])
    )
    return _report(
        artifact,
        HOLDS if same else FAILS,
        (
            f"distance {distance:.6g} ranks at p={p_value:.4g} among {len(inputs['calibration'])} "
            f"reference distances (alpha={alpha}): is_novel={recomputed['is_novel']}"
            if same
            else "the recorded numbers do not reproduce the recorded verdict"
        ),
        recomputed=recomputed,
    )


def _same_edges(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> bool:
    if len(a) != len(b):
        return False
    for left, right in zip(a, b, strict=True):
        if (left["source"], left["target"], int(left["lag"])) != (
            right["source"],
            right["target"],
            int(right["lag"]),
        ):
            return False
        if not _close(float(left["p_adjusted"]), float(right["p_adjusted"])):
            return False
        if not _close(float(left["partial_r2"]), float(right["partial_r2"])):
            return False
    return True


def _replay_causal(
    artifact: Mapping[str, Any], data: Any, bounds: ReplayBounds | None = None
) -> dict[str, Any]:
    inputs, claim = artifact["inputs"], artifact["claim"]
    if bounds is not None:
        over = bounds.exceeded_by(inputs)
        if over is not None:
            return _report(artifact, CANNOT_REPLAY, over)
    alpha = float(inputs["alpha"])
    assessed = bool(claim.get("assessed", True))
    if data is not None:
        matrix = np.asarray(data, dtype=float)
        shape = [int(v) for v in inputs["input_shape"]]
        if matrix.ndim == 2 and matrix.shape[0] == shape[0] and matrix.shape[1] > shape[1]:
            # The instrument analyses the leading columns of what it was given.
            matrix = matrix[:, : shape[1]]
        if matrix_digest(matrix) != inputs["input_digest"]:
            return _report(
                artifact,
                CANNOT_REPLAY,
                "the supplied data is not the window this artifact was computed on (digest differs)",
            )
        analysis = granger_analysis(matrix, int(inputs["max_lag"]))
        edges = _edges(analysis["tests"], alpha) if analysis["assessed"] else []
        recomputed = {
            "assessed": analysis["assessed"],
            "edges": edges,
            "pairs_tested": len(analysis["tests"]),
            "rows_excluded": analysis["rows_excluded"],
        }
        same = analysis["assessed"] == assessed and _same_edges(edges, claim["edges"])
        if not same:
            detail = "recomputing from the data does not give the recorded finding"
        elif not assessed:
            detail = f"recomputed from the data: not assessed ({analysis['not_assessed_because']})"
        else:
            detail = (
                f"recomputed from the data: {analysis['rows_excluded']} contaminated row(s) set "
                f"aside, {len(analysis['tests'])} pairs tested, {len(edges)} edge(s) at alpha={alpha}"
            )
        return _report(artifact, HOLDS if same else FAILS, detail, recomputed=recomputed)
    if not assessed:
        same = not claim["edges"] and not inputs["tests"]
        return _report(
            artifact,
            HOLDS if same else FAILS,
            (
                f"not assessed, and no edge is claimed: {claim.get('not_assessed_because', '')}"
                if same
                else "the record says the window was not assessed, yet it carries tests or edges"
            ),
            recomputed={"assessed": False, "edges": [], "pairs_tested": 0},
        )
    tests = _correct_family(inputs["tests"])
    edges = _edges(tests, alpha)
    same = _same_edges(edges, claim["edges"]) and len(tests) == int(claim["pairs_tested"])
    return _report(
        artifact,
        HOLDS if same else FAILS,
        (
            f"re-derived from the recorded sums of squares: {len(tests)} pairs, {len(edges)} edge(s) at alpha={alpha}"
            if same
            else "the recorded statistics do not give the recorded edges"
        ),
        recomputed={"assessed": True, "edges": edges, "pairs_tested": len(tests)},
    )


def _replay_hypothesis(
    artifact: Mapping[str, Any], siblings: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    inputs, claim = artifact["inputs"], artifact["claim"]
    failed: list[str] = []
    checked: list[dict[str, Any]] = []
    for premise in inputs["premises"]:
        statistic = float(premise["statistic"])
        source = str(premise["source"])
        sibling = siblings.get(source)
        note = "as recorded"
        if sibling is not None:
            if sibling["status"] == CANNOT_REPLAY:
                return _report(
                    artifact,
                    CANNOT_REPLAY,
                    f"premise {premise['id']} rests on the {source} artifact, which could not be replayed",
                )
            recomputed = sibling.get("recomputed") or {}
            if source == "novelty" and "p_value" in recomputed:
                statistic, note = float(recomputed["p_value"]), "from the replayed novelty artifact"
            elif source == "causal":
                wanted = premise.get("edge") or {}
                match = [
                    e
                    for e in recomputed.get("edges", [])
                    if e["source"] == wanted.get("source") and e["target"] == wanted.get("target")
                ]
                if match:
                    statistic, note = (
                        float(match[0]["p_adjusted"]),
                        "from the replayed causal artifact",
                    )
                else:
                    statistic, note = 1.0, "the edge is absent from the replayed causal artifact"
        holds = verdict(statistic, str(premise["op"]), float(premise["cut"]))
        checked.append({"id": premise["id"], "statistic": statistic, "holds": holds, "basis": note})
        if not holds:
            failed.append(str(premise["id"]))
    known = {p["id"] for p in inputs["premises"]}
    derived: set[str] = set()
    for step in inputs["steps"]:
        if not all(p in known or p in derived for p in step["premises"]):
            return _report(
                artifact, FAILS, f"step {step['rule']} uses a premise the artifact does not carry"
            )
        derived.add(step["conclusion"])
    if claim["conclusion"] not in derived:
        return _report(artifact, FAILS, "the steps do not derive the recorded conclusion")
    holds_now = not failed
    if holds_now != bool(claim["holds"]) or not holds_now:
        return _report(
            artifact,
            FAILS,
            (
                f"withdrawn: premise(s) {', '.join(failed)} no longer hold"
                if failed
                else "the premises hold but the record says the conclusion does not"
            ),
            recomputed={"holds": holds_now, "premises": checked},
        )
    return _report(
        artifact,
        HOLDS,
        f"all {len(checked)} premise(s) hold and the steps derive {claim['conclusion']}",
        recomputed={"holds": True, "premises": checked},
    )


def replay(
    artifact: Any,
    *,
    data: Any = None,
    reference: Any = None,
    siblings: Mapping[str, Mapping[str, Any]] | None = None,
    bounds: ReplayBounds | None = None,
) -> dict[str, Any]:
    """Run an artifact's procedure again and say whether its claim comes out.

    Args:
        artifact: A discovery artifact (the mapping an instrument produced).
        data: For a causal artifact, the window it was computed on; the
            statistics are then recomputed from it.  Checked against the
            recorded digest.
        reference: For a novelty artifact, rows of a reference of your own;
            the question becomes whether the observation is novel there.
        siblings: For a hypothesis, the replay reports of the artifacts its
            premises rest on, by kind.
        bounds: The largest causal finding to recompute; anything larger is
            CANNOT_REPLAY with the reason.  ``None`` (offline) is unbounded.

    Returns:
        ``{"kind", "status", "detail", "recorded", "recomputed"}`` with status
        :data:`HOLDS`, :data:`FAILS` or :data:`CANNOT_REPLAY`.  Total: a
        malformed artifact is a report, not an exception.
    """
    if not hasattr(artifact, "get") or artifact.get("schema") != SCHEMA:
        return {
            "kind": None,
            "status": CANNOT_REPLAY,
            "detail": f"not a {SCHEMA} artifact",
            "recorded": None,
        }
    if not _sealed(artifact):
        return _report(
            artifact,
            FAILS,
            "the artifact does not match its own digest: it was altered or is incomplete",
        )
    kind = artifact.get("kind")
    try:
        if kind == "novelty":
            return _replay_novelty(artifact, reference)
        if kind == "causal":
            return _replay_causal(artifact, data, bounds)
        if kind == "hypothesis":
            return _replay_hypothesis(artifact, siblings or {})
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return _report(artifact, FAILS, f"the artifact is malformed: {type(exc).__name__}: {exc}")
    return _report(artifact, CANNOT_REPLAY, f"unknown artifact kind {kind!r}")


def replay_record(
    record: Any,
    *,
    data: Any = None,
    reference: Any = None,
    bounds: ReplayBounds | None = None,
) -> list[dict[str, Any]]:
    """Replay every discovery artifact a decision record carries.

    The novelty and causal artifacts are replayed first; the hypothesis is
    then replayed against their reports, so a premise that fails takes the
    conclusion with it.

    A whole detection result carries its artifacts twice, in the cognitive
    analysis and in the decision record built from it.  Both copies are read:
    if they differ, that artifact FAILS, whichever copy is the altered one.

    Args:
        record: A ``DecisionRecord`` as a mapping (its ``to_dict()``), a
            detection result, or a cognitive payload: wherever the artifacts
            are, under ``discovery``.
        data: See :func:`replay`.
        reference: See :func:`replay`.
        bounds: See :func:`replay`.

    Returns:
        One report per artifact found, in the order novelty, causal,
        hypothesis.  Empty when the record carries none.
    """
    copies = _find_artifacts(record)
    reports: dict[str, dict[str, Any]] = {}
    for kind in ("novelty", "causal", "hypothesis"):
        found = copies.get(kind)
        if not found:
            continue
        if any(_canonical(other) != _canonical(found[0]) for other in found[1:]):
            reports[kind] = _report(
                found[0],
                FAILS,
                "the record carries copies of this artifact that differ from each other",
            )
        elif kind == "hypothesis":
            reports[kind] = replay(found[0], siblings=reports)
        else:
            reports[kind] = replay(found[0], data=data, reference=reference, bounds=bounds)
    return [reports[kind] for kind in ("novelty", "causal", "hypothesis") if kind in reports]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


#: Where a record may carry its ``discovery`` block.
_DISCOVERY_PATHS: tuple[tuple[str, ...], ...] = (
    ("discovery",),
    ("signals", "cognitive", "discovery"),
    ("cognitive", "discovery"),
    ("cognitive_analysis", "discovery"),
    ("decision", "signals", "cognitive", "discovery"),
)


def _find_artifacts(record: Any) -> dict[str, list[Mapping[str, Any]]]:
    """Every copy of each discovery artifact in a record, by kind."""
    copies: dict[str, list[Mapping[str, Any]]] = {}
    if not hasattr(record, "get"):
        return copies
    for path in _DISCOVERY_PATHS:
        node: Any = record
        for key in path:
            node = node.get(key) if hasattr(node, "get") else None
        if not hasattr(node, "get"):
            continue
        for kind, artifact in node.items():
            if hasattr(artifact, "get") and artifact.get("schema") == SCHEMA:
                copies.setdefault(str(kind), []).append(artifact)
    return copies
