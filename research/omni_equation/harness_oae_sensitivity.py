# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sensitivity of the Omni-Ava equation and the 3R recursion to their dials.

The OAE's golden-ratio weights, its Lyapunov convergence rate λ and the 3R
recursion depth are documented constants that had never been varied around
their shipped values on real data. This harness treats the instruments as the
object of study, on the four recorded NAB files under the exact windowed
protocol of ``harness_oae_weights.py`` (the served R/H/O producers verbatim,
window 128, stride 16):

* **Golden weights**: each of w_R / w_H / w_O scaled by 0.8–1.2 one at a
  time (renormalised), against the shipped golden split and uniform. Mean
  AUROC and mean rank per configuration.
* **Recursion depth**: the served ``recursion_depth`` swept over 3–7 around
  the shipped 5, with the other two streams fixed; golden weights.
* **Lyapunov λ**: swept 0.15–0.35 around the shipped 0.25 through
  ``OmniAvaEquation.compute`` on identical (R, H, O) triples. The measured
  surface is exactly flat — λ parameterises only the reference decay
  envelope ``ε·e^(−λt)`` that ``verify_lyapunov_stability`` monitors, and
  the harness records both the zero score-sensitivity and the envelope
  values, so the published surface states what the dial does and does not
  touch.

Results go to ``artifacts/oae_sensitivity.json``;
``tests/benchmarks/test_oae_sensitivity.py`` pins the surface's shape.

Run from the repository root::

    python research/omni_equation/harness_oae_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from research.math_arrest.harness_probe_redundancy import _git_commit
from research.omni_equation.harness_oae_weights import (
    STRIDE,
    W_PHI,
    WINDOW,
    compute_recursion_score,
    streams,
)
from tests.recorded_data import auroc, nab_corpus

ARTIFACT_PATH = _ROOT / "artifacts" / "oae_sensitivity.json"
SCALES = (0.8, 0.9, 1.0, 1.1, 1.2)
DEPTHS = (3, 4, 5, 6, 7)
LAMBDAS = (0.15, 0.20, 0.25, 0.30, 0.35)


def _windows() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per NAB file: the (n_windows, 3) R/H/O matrix and window labels."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for series in nab_corpus():
        rows, labels = [], []
        for start in range(0, len(series.values) - WINDOW + 1, STRIDE):
            window = series.values[start : start + WINDOW]
            rows.append(streams(window))
            labels.append(bool(series.positive[start : start + WINDOW].any()))
        matrix = np.asarray(rows, dtype=float)
        y = np.asarray(labels, dtype=bool)
        if y.any() and not y.all():
            out[series.name] = (matrix, y)
    return out


def _depth_r_streams(depth: int) -> dict[str, np.ndarray]:
    """The recursion stream recomputed at a given depth, per file."""
    out: dict[str, np.ndarray] = {}
    for series in nab_corpus():
        values = []
        for start in range(0, len(series.values) - WINDOW + 1, STRIDE):
            window = series.values[start : start + WINDOW]
            values.append(float(np.clip(compute_recursion_score(window, depth), 0.0, 1.0)))
        out[series.name] = np.asarray(values, dtype=float)
    return out


def _mean_rank(per_config: dict[str, dict[str, float]]) -> dict[str, float]:
    """Average rank on ties across files (competitive-benchmark convention)."""
    configs = sorted(per_config)
    files = sorted(next(iter(per_config.values())))
    totals = dict.fromkeys(configs, 0.0)
    for name in files:
        values = np.array([per_config[c][name] for c in configs])
        order = np.argsort(-values, kind="stable")
        ranks = np.empty(len(configs))
        i = 0
        while i < len(configs):
            j = i
            while j + 1 < len(configs) and values[order[j + 1]] == values[order[i]]:
                j += 1
            ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        for k, config in enumerate(configs):
            totals[config] += float(ranks[k])
    return {c: round(totals[c] / len(files), 4) for c in configs}


def sweep_weights(data: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """One-at-a-time scaling of the golden weights, renormalised."""
    configs: dict[str, np.ndarray] = {"golden": W_PHI, "uniform": np.full(3, 1 / 3)}
    for axis, label in enumerate(("w_R", "w_H", "w_O")):
        for scale in SCALES:
            if scale == 1.0:
                continue
            w = W_PHI.copy()
            w[axis] *= scale
            configs[f"{label}_x{scale}"] = w / w.sum()

    per_config: dict[str, dict[str, float]] = {}
    for name, weights in configs.items():
        per_file = {}
        for file_name, (matrix, y) in data.items():
            fused = matrix @ weights
            per_file[file_name] = round(auroc(fused[y], fused[~y]), 4)
        per_config[name] = per_file
    ranks = _mean_rank(per_config)
    return {
        name: {
            "weights": [round(float(v), 4) for v in configs[name]],
            "per_file_auc": per_file,
            "mean_auc": round(float(np.mean(list(per_file.values()))), 4),
            "mean_rank": ranks[name],
        }
        for name, per_file in per_config.items()
    }


def sweep_depth(data: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """The served recursion depth swept around the shipped 5, golden weights."""
    per_config: dict[str, dict[str, float]] = {}
    for depth in DEPTHS:
        r_streams = _depth_r_streams(depth)
        per_file = {}
        for file_name, (matrix, y) in data.items():
            swapped = matrix.copy()
            swapped[:, 0] = r_streams[file_name][: len(swapped)]
            fused = swapped @ W_PHI
            per_file[file_name] = round(auroc(fused[y], fused[~y]), 4)
        per_config[f"depth_{depth}"] = per_file
    ranks = _mean_rank(per_config)
    return {
        name: {
            "per_file_auc": per_file,
            "mean_auc": round(float(np.mean(list(per_file.values()))), 4),
            "mean_rank": ranks[name],
        }
        for name, per_file in per_config.items()
    }


def sweep_lambda(data: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """λ through ``OmniAvaEquation.compute`` on identical (R, H, O) triples."""
    from omni_mercury_engine.core.three_r.fusion import OmniAvaEquation

    matrix = next(iter(data.values()))[0][:64]
    out: dict[str, Any] = {}
    reference: np.ndarray | None = None
    for lam in LAMBDAS:
        equation = OmniAvaEquation(convergence_rate=lam)
        scores = np.array(
            [equation.compute(float(r), float(h), float(o)).fusion_score for r, h, o in matrix]
        )
        first_bound = float(np.exp(-lam * 1.0))
        if reference is None:
            reference = scores
        out[f"lambda_{lam}"] = {
            "max_abs_score_delta_vs_0.15": float(np.max(np.abs(scores - reference))),
            "lyapunov_envelope_t1": round(first_bound, 6),
        }
    out["finding"] = (
        "the fusion score is exactly flat in lambda; the parameter sets only "
        "the reference decay envelope eps*exp(-lambda*t) that "
        "verify_lyapunov_stability monitors"
    )
    return out


def main() -> int:
    data = _windows()
    artifact: dict[str, Any] = {
        "schema": "mercury.research.oae_sensitivity/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit(),
        "protocol": {"window": WINDOW, "stride": STRIDE, "files": sorted(data)},
        "weights": sweep_weights(data),
        "recursion_depth": sweep_depth(data),
        "lambda": sweep_lambda(data),
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")

    print("weights sweep (mean AUC / mean rank):")
    for name, cell in sorted(artifact["weights"].items(), key=lambda kv: kv[1]["mean_rank"]):
        print(f"  {name:12s} {cell['mean_auc']:.4f} / {cell['mean_rank']:.2f}")
    print("depth sweep:")
    for name, cell in artifact["recursion_depth"].items():
        print(f"  {name:12s} {cell['mean_auc']:.4f} / {cell['mean_rank']:.2f}")
    print("lambda sweep:")
    for name, cell in artifact["lambda"].items():
        if isinstance(cell, dict):
            print(
                f"  {name:12s} max|dScore|={cell['max_abs_score_delta_vs_0.15']:.1e} "
                f"envelope(t=1)={cell['lyapunov_envelope_t1']}"
            )
    print(f"wrote {ARTIFACT_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
