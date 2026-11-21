# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does the causal instrument report edges that are there, and only those? Measured.

The orchestrator's causal step is a Granger F-test over every ordered pair of
the window's leading columns.  It shipped reporting its findings in every
decision record as ``causal_factors`` with a ``strength``.  Nothing had ever
measured how often it finds an edge where there is none.

This harness does, on windows whose truth is known because they are
generated: independent columns (no edge exists), independent columns under
the disturbances a flagged window actually has (a short common shock, a
sustained common shift, a common trend, a random walk), and windows with one
planted edge (``X0`` drives ``X1`` at lag 1, coefficient 0.5).

Six procedures are measured on the same windows:

``prior``
    What shipped: the lag with the smallest p-value, compared with 0.05
    uncorrected; every pair tested independently.
``corrected``
    Bonferroni over the lags searched, Holm over the pairs tested.
``corrected_trimmed``
    As above, on the rows left after setting aside contaminated time steps.
``shipped``
    As above, and a window whose level is not stable is not assessed.
``first_difference`` (refused)
    ``corrected_trimmed`` on first differences.
``residual_guard`` (refused)
    ``corrected_trimmed``, declining a window when its autoregression's
    residuals shift between blocks.

Bars, set before the corrected procedures were run:

* no edge exists: an analysis reports one in at most 7.5% of plain-noise
  windows and at most 10% of disturbed ones;
* one planted edge: found in at least 90% of windows of 200 rows, with any
  other edge in at most 10%;
* coverage: at most 15% of stable windows declined.

``shipped`` meets the first two everywhere.  It does **not** meet the
coverage bar on strongly autocorrelated windows, most of which it declines:
that is recorded as the open problem it is, not as a pass.  It was adopted
because a declined window yields no edge, and the procedures that decline
nothing report false edges in up to 100% of shifted windows.

Writes ``artifacts/causal_discovery.json``.
``tests/cognitive/test_causal_discovery_honest.py`` pins the result.

Run from the repository root::

    python research/cognitive/harness_causal_discovery.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from scipy import linalg, stats

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omni_mercury_engine.discovery import (
    contaminated_rows,
    granger_analysis,
    granger_family,
    granger_pair,
)

if TYPE_CHECKING:
    from collections.abc import Callable

ARTIFACT_PATH = _ROOT / "artifacts" / "causal_discovery.json"
SEED = 20261121
TRIALS = 200
MAX_LAG = 5
ALPHA = 0.05
TRUE_EDGE = ("X0", "X1")

Matrix: TypeAlias = np.ndarray[Any, Any]
Edges = set[tuple[str, str]]


# --------------------------------------------------------------------------- #
# Procedures.  Each returns (edges, assessed).
# --------------------------------------------------------------------------- #
def _names(data: Matrix) -> list[str]:
    return [f"X{i}" for i in range(data.shape[1])]


def prior(data: Matrix) -> tuple[Edges, bool]:
    """The procedure that shipped: best lag by p-value, no correction of any kind."""
    names = _names(data)
    edges: Edges = set()
    for i, source in enumerate(names):
        for j, target in enumerate(names):
            if i == j:
                continue
            found = granger_pair(data[:, i], data[:, j], MAX_LAG)
            if found is not None and found.p_raw < ALPHA:
                edges.add((source, target))
    return edges, True


def _significant(tests: list[dict[str, Any]]) -> Edges:
    return {(t["source"], t["target"]) for t in tests if t["p_adjusted"] <= ALPHA}


def corrected(data: Matrix) -> tuple[Edges, bool]:
    return _significant(granger_family(data, MAX_LAG)), True


def corrected_trimmed(data: Matrix) -> tuple[Edges, bool]:
    return _significant(granger_family(data, MAX_LAG, clean=~contaminated_rows(data))), True


def shipped(data: Matrix) -> tuple[Edges, bool]:
    analysis = granger_analysis(data, MAX_LAG)
    return _significant(analysis["tests"]), bool(analysis["assessed"])


def first_difference(data: Matrix) -> tuple[Edges, bool]:
    return corrected_trimmed(np.diff(data, axis=0))


def _residual_shift_p(series: Matrix, clean: Matrix) -> float | None:
    """Block-mean test on AR(MAX_LAG) residuals (the refused guard)."""
    n = series.size
    target = series[MAX_LAG:]
    own = np.column_stack([series[MAX_LAG - k - 1 : n - k - 1] for k in range(MAX_LAG)])
    keep = np.ones(target.size, dtype=bool)
    for back in range(MAX_LAG + 1):
        keep &= clean[MAX_LAG - back : n - back]
    design = np.column_stack([np.ones(target.size), own])[keep]
    kept = target[keep]
    if kept.size < 30:
        return None
    residual = kept - design @ linalg.lstsq(design, kept)[0]
    blocks = min(10, residual.size // 10)
    if blocks < 3:
        return None
    length = residual.size // blocks
    variance = float(residual @ residual) / (residual.size - design.shape[1])
    if variance <= 0:
        return None
    means = residual[: blocks * length].reshape(blocks, length).mean(axis=1)
    return float(stats.chi2.sf(length * float(np.sum(means**2)) / variance, blocks - 1))


def residual_guard(data: Matrix) -> tuple[Edges, bool]:
    clean = ~contaminated_rows(data)
    p_values = [_residual_shift_p(data[:, c], clean) for c in range(data.shape[1])]
    if any(p is None for p in p_values):
        return set(), False
    if min(1.0, min(p for p in p_values if p is not None) * len(p_values)) < 0.01:
        return set(), False
    return _significant(granger_family(data, MAX_LAG, clean=clean)), True


PROCEDURES: dict[str, Callable[[Matrix], tuple[Edges, bool]]] = {
    "prior": prior,
    "corrected": corrected,
    "corrected_trimmed": corrected_trimmed,
    "shipped": shipped,
    "first_difference": first_difference,
    "residual_guard": residual_guard,
}


# --------------------------------------------------------------------------- #
# Windows whose truth is known.
# --------------------------------------------------------------------------- #
def _noise(rng: np.random.Generator, n: int, d: int) -> Matrix:
    return rng.normal(size=(n, d))


def _ar1(rng: np.random.Generator, n: int, d: int, phi: float) -> Matrix:
    shocks = rng.normal(size=(n, d))
    out = np.zeros((n, d))
    for t in range(1, n):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


def _shift(rng: np.random.Generator, n: int, d: int, size: float, width: int) -> Matrix:
    out = _noise(rng, n, d)
    start = int(rng.integers(10, n - width - 10))
    out[start : start + width] += size
    return out


def _planted(rng: np.random.Generator, n: int, d: int, shock: float = 0.0) -> Matrix:
    out = _noise(rng, n, d)
    out[1:, 1] += 0.5 * out[:-1, 0]
    if shock:
        start = int(rng.integers(10, n - 16))
        out[start : start + 6] += shock
    return out


#: name -> (generator, "none" when no edge exists | "planted", kind for the bars)
SCENARIOS: dict[str, tuple[Callable[[np.random.Generator], Matrix], str, str]] = {
    "noise_n200_d5": (lambda r: _noise(r, 200, 5), "none", "plain"),
    "noise_n60_d4": (lambda r: _noise(r, 60, 4), "none", "plain"),
    "ar1_0.7_n200_d5": (lambda r: _ar1(r, 200, 5, 0.7), "none", "plain"),
    "common_shock_9sd_6rows_n200_d5": (lambda r: _shift(r, 200, 5, 9.0, 6), "none", "disturbed"),
    "common_shift_3sd_40rows_n200_d5": (lambda r: _shift(r, 200, 5, 3.0, 40), "none", "disturbed"),
    "common_shift_2sd_90rows_n200_d5": (lambda r: _shift(r, 200, 5, 2.0, 90), "none", "disturbed"),
    "common_trend_n200_d5": (
        lambda r: _noise(r, 200, 5) + np.linspace(0.0, 4.0, 200)[:, None],
        "none",
        "disturbed",
    ),
    "random_walk_n200_d5": (lambda r: np.cumsum(_noise(r, 200, 5), axis=0), "none", "disturbed"),
    "planted_n200_d4": (lambda r: _planted(r, 200, 4), "planted", "planted"),
    "planted_with_shock_n200_d4": (lambda r: _planted(r, 200, 4, 6.0), "planted", "planted"),
    "planted_n100_d4": (lambda r: _planted(r, 100, 4), "planted", "planted_short"),
}

BARS = {
    "false_edge_plain_max": 0.075,
    "false_edge_disturbed_max": 0.10,
    "planted_found_min_n200": 0.90,
    "planted_other_edge_max": 0.10,
    "stable_windows_declined_max": 0.15,
}


def measure() -> dict[str, dict[str, dict[str, float]]]:
    results: dict[str, dict[str, dict[str, float]]] = {}
    for index, (name, (generate, truth, _kind)) in enumerate(SCENARIOS.items()):
        rng = np.random.default_rng(SEED + index)
        windows = [generate(rng) for _ in range(TRIALS)]
        results[name] = {}
        for label, procedure in PROCEDURES.items():
            false_edge = found = declined = 0
            for window in windows:
                edges, assessed = procedure(window)
                declined += not assessed
                if truth == "planted":
                    found += TRUE_EDGE in edges
                    false_edge += bool(edges - {TRUE_EDGE})
                else:
                    false_edge += bool(edges)
            row = {"false_edge": false_edge / TRIALS, "not_assessed": declined / TRIALS}
            if truth == "planted":
                row["true_edge"] = found / TRIALS
            results[name][label] = row
    return results


def judge(results: dict[str, dict[str, dict[str, float]]], label: str) -> dict[str, Any]:
    """Each bar, for one procedure: the worst value seen and whether it is met."""
    worst_plain = max(
        results[n][label]["false_edge"] for n, s in SCENARIOS.items() if s[2] == "plain"
    )
    worst_disturbed = max(
        results[n][label]["false_edge"] for n, s in SCENARIOS.items() if s[2] == "disturbed"
    )
    planted = [n for n, s in SCENARIOS.items() if s[2] == "planted"]
    found = min(results[n][label]["true_edge"] for n in planted)
    other = max(results[n][label]["false_edge"] for n, s in SCENARIOS.items() if s[1] == "planted")
    declined = max(
        results[n][label]["not_assessed"] for n, s in SCENARIOS.items() if s[2] == "plain"
    )
    return {
        "false_edge_plain": {
            "worst": worst_plain,
            "met": worst_plain <= BARS["false_edge_plain_max"],
        },
        "false_edge_disturbed": {
            "worst": worst_disturbed,
            "met": worst_disturbed <= BARS["false_edge_disturbed_max"],
        },
        "planted_found_n200": {"worst": found, "met": found >= BARS["planted_found_min_n200"]},
        "planted_other_edge": {"worst": other, "met": other <= BARS["planted_other_edge_max"]},
        "stable_windows_declined": {
            "worst": declined,
            "met": declined <= BARS["stable_windows_declined_max"],
        },
    }


def main() -> int:
    results = measure()
    verdicts = {label: judge(results, label) for label in PROCEDURES}
    artifact = {
        "schema": "mercury.research.causal_discovery/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "seed": SEED,
            "trials_per_scenario": TRIALS,
            "max_lag": MAX_LAG,
            "alpha": ALPHA,
            "planted_edge": "X0 -> X1 at lag 1, coefficient 0.5",
            "scenarios": {n: {"truth": s[1], "kind": s[2]} for n, s in SCENARIOS.items()},
        },
        "bars": BARS,
        "results": results,
        "verdicts": verdicts,
        "adopted": "shipped",
        "refused": ["first_difference", "residual_guard"],
        "open_problem": (
            "shipped declines most strongly autocorrelated windows (the coverage bar is "
            "not met there). What is missing is a test of level stability that is valid "
            "under autocorrelation and still has power against a shift inside the window: "
            "the residual guard is valid and has none; the block-mean check has power and "
            "over-declines."
        ),
    }
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    header = f"{'scenario':34s}" + "".join(f"{label[:17]:>19s}" for label in PROCEDURES)
    print("false-edge rate / not assessed / true edge")
    print(header)
    for name in SCENARIOS:
        cells = []
        for label in PROCEDURES:
            row = results[name][label]
            cell = f"{row['false_edge']:.2f}/{row['not_assessed']:.2f}"
            if "true_edge" in row:
                cell += f"/{row['true_edge']:.2f}"
            cells.append(f"{cell:>19s}")
        print(f"{name:34s}" + "".join(cells))
    for label, verdict in verdicts.items():
        unmet = [bar for bar, value in verdict.items() if not value["met"]]
        print(f"{label:20s} bars not met: {unmet or 'none'}")
    print(f"written: {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
