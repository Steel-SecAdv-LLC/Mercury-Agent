# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does locally-adaptive 3R recursion depth beat the fixed served depth?

Item: make recursion depth and the Lyapunov λ functions of local signal
properties, keep the adaptation only where it improves Mercury's own metrics.

λ is settled before any sweep runs: ``artifacts/oae_sensitivity.json`` holds
the direct measurement that the fused score is *exactly flat* in λ (it
parameterises only the stability monitor's reference envelope), so no
adaptation of λ can improve a detection metric by construction, and none is
adopted. Depth is the live question — the global sweep measured a flat
0.7962–0.7992 across depths 3–7, and this harness asks whether a per-window
depth chosen from the window's own spectral entropy (the Resonance stream's
quantity: the normalised Shannon entropy of the FFT magnitude distribution)
does better than the served fixed depth 5:

    depth(window) = 3 + round(4 * (1 - H_spectral))   in [3, 7]

— structured windows (low entropy) recurse deeper, broadband windows stay
shallow. Same windowed NAB protocol as the sensitivity harness; mean AUROC,
mean rank against every fixed depth, and the R-stream wall-clock latency are
all reported, and the verdict (adopt only on improvement) is recorded in
``artifacts/oae_adaptive_depth.json``.

Run from the repository root::

    python research/omni_equation/harness_adaptive_depth.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from research.math_arrest.harness_probe_redundancy import _git_commit
from research.omni_equation.harness_oae_sensitivity import DEPTHS, _mean_rank, _windows
from research.omni_equation.harness_oae_weights import W_PHI, WINDOW, compute_recursion_score
from tests.recorded_data import auroc, nab_corpus

ARTIFACT_PATH = _ROOT / "artifacts" / "oae_adaptive_depth.json"
DEPTH_MIN, DEPTH_MAX = min(DEPTHS), max(DEPTHS)


def spectral_entropy(window: np.ndarray[Any, Any]) -> float:
    """Normalised Shannon entropy of the window's FFT magnitude distribution."""
    magnitudes = np.abs(np.fft.rfft(window - window.mean()))[1:]
    power = magnitudes**2
    total = float(power.sum())
    if total <= 0.0 or power.size <= 1:
        return 1.0
    p = power / total
    p = p[p > 0.0]
    return float(-(p * np.log(p)).sum() / np.log(power.size))


def adaptive_depth(window: np.ndarray[Any, Any]) -> int:
    """The local rule: structured windows recurse deeper, broadband shallow."""
    h = spectral_entropy(window)
    return int(
        np.clip(DEPTH_MIN + round((DEPTH_MAX - DEPTH_MIN) * (1.0 - h)), DEPTH_MIN, DEPTH_MAX)
    )


def r_streams() -> (
    tuple[dict[str, dict[int, np.ndarray]], dict[str, np.ndarray], dict[str, float], dict[str, Any]]
):
    """Fixed-depth and adaptive R streams per file, with wall-clock latency."""
    fixed: dict[str, dict[int, np.ndarray]] = {}
    adaptive: dict[str, np.ndarray] = {}
    latency: dict[str, float] = {f"depth_{d}": 0.0 for d in DEPTHS}
    latency["adaptive"] = 0.0
    depth_usage: list[int] = []
    from research.omni_equation.harness_oae_weights import STRIDE

    for series in nab_corpus():
        windows = [
            series.values[start : start + WINDOW]
            for start in range(0, len(series.values) - WINDOW + 1, STRIDE)
        ]
        fixed[series.name] = {}
        for depth in DEPTHS:
            start_time = time.perf_counter()
            fixed[series.name][depth] = np.array(
                [float(np.clip(compute_recursion_score(w, depth), 0.0, 1.0)) for w in windows]
            )
            latency[f"depth_{depth}"] += time.perf_counter() - start_time
        start_time = time.perf_counter()
        values = []
        for w in windows:
            depth = adaptive_depth(w)
            depth_usage.append(depth)
            values.append(float(np.clip(compute_recursion_score(w, depth), 0.0, 1.0)))
        adaptive[series.name] = np.array(values)
        latency["adaptive"] += time.perf_counter() - start_time
    usage = {
        "mean_depth": round(float(np.mean(depth_usage)), 3),
        "depth_histogram": {
            int(d): int(c) for d, c in zip(*np.unique(depth_usage, return_counts=True))
        },
    }
    return fixed, adaptive, {k: round(v, 3) for k, v in latency.items()}, usage


def main() -> int:
    data = _windows()
    fixed, adaptive, latency, usage = r_streams()

    per_config: dict[str, dict[str, float]] = {}
    for depth in DEPTHS:
        per_file = {}
        for name, (matrix, y) in data.items():
            swapped = matrix.copy()
            swapped[:, 0] = fixed[name][depth][: len(swapped)]
            fused = swapped @ W_PHI
            per_file[name] = round(auroc(fused[y], fused[~y]), 4)
        per_config[f"depth_{depth}"] = per_file
    per_file = {}
    for name, (matrix, y) in data.items():
        swapped = matrix.copy()
        swapped[:, 0] = adaptive[name][: len(swapped)]
        fused = swapped @ W_PHI
        per_file[name] = round(auroc(fused[y], fused[~y]), 4)
    per_config["adaptive"] = per_file

    ranks = _mean_rank(per_config)
    summary: dict[str, dict[str, Any]] = {
        config: {
            "per_file_auc": files,
            "mean_auc": round(float(np.mean(list(files.values()))), 4),
            "mean_rank": ranks[config],
            "latency_s": latency[config],
        }
        for config, files in per_config.items()
    }
    # Adoption bar: the adaptation must clear the depth dimension's own
    # measured noise band (the global sweep's 3-7 spread is 0.003 of mean
    # AUROC), not merely edge the served depth inside it. The first run
    # measured adaptive at rank 3.25 vs depth_5's 4.88 with +0.0017 mean
    # AUROC and 41% lower R-stream latency — but fixed depth_3 outranked
    # both, and every difference sits inside the flat band, so adopting the
    # adaptive rule on that margin would be noise-chasing.
    NOISE_MARGIN = 0.005
    adopted = (
        ranks["adaptive"] < ranks["depth_5"]
        and summary["adaptive"]["mean_auc"] > summary["depth_5"]["mean_auc"] + NOISE_MARGIN
    )
    artifact = {
        "schema": "mercury.research.oae_adaptive_depth/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit(),
        "rule": "depth = 3 + round(4 * (1 - spectral_entropy)), clipped to [3, 7]",
        "lambda_verdict": (
            "no adaptation: the fused score is exactly flat in lambda "
            "(artifacts/oae_sensitivity.json), so adapting it cannot improve "
            "any detection metric"
        ),
        "results": summary,
        "adaptive_depth_usage": usage,
        "noise_margin": NOISE_MARGIN,
        "adopted": adopted,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")

    for config, cell in sorted(summary.items(), key=lambda kv: kv[1]["mean_rank"]):
        print(
            f"  {config:10s} mean AUC {cell['mean_auc']:.4f} rank {cell['mean_rank']:5.2f} "
            f"latency {cell['latency_s']:6.3f}s"
        )
    print(f"adaptive depth usage: {usage}")
    print(f"adopted: {adopted}")
    print(f"wrote {ARTIFACT_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
