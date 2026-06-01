"""
Mercury Agent - Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GNU GPL v3

Formal-verification soundness: does the dormant interval-bound propagator
produce *sound* output certificates for a network over an input region?

`formal_verification.py` was orphaned and un-revivable by anomaly-AUC -- it emits
satisfiability proofs / certified bounds, not anomaly scores. The right metric for
a verifier is **soundness**: when it certifies that a network's output stays
within an interval over a whole input box, that certificate must *always* contain
the true output range (a sound verifier never certifies a false bound). The
secondary metric is **tightness**: a sound-but-vacuous verifier (certifying
(-inf, inf)) is useless, so we also measure how close the certified interval is to
the true sampled range.

This is the third non-AUC measurement framework (after causal recovery and
explanation fidelity). It is self-contained: random small ReLU networks and random
input boxes provide ground truth (the true output range is found by dense
sampling), against which `IntervalBoundPropagator`'s certificate is checked.

Pre-registered bar: the propagator is *validated* if it is **100% sound** across
all random cases (its interval always contains the densely-sampled true range)
and the mean tightness ratio is finite (non-vacuous certificates).

Usage::

    python -m benchmarks.formal_verification_soundness --n-cases 200 \\
        --out artifacts/formal_verification_soundness.json
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

IN_DIM = 4
HIDDEN = 8
OUT_DIM = 1
N_SAMPLES = 5000  # dense samples to estimate the true output range over the box


def _random_net(rng: np.random.Generator) -> dict[str, np.ndarray[Any, Any]]:
    return {
        "W1": rng.normal(0, 1, (IN_DIM, HIDDEN)) / np.sqrt(IN_DIM),
        "b1": rng.normal(0, 0.5, HIDDEN),
        "W2": rng.normal(0, 1, (HIDDEN, OUT_DIM)) / np.sqrt(HIDDEN),
        "b2": rng.normal(0, 0.5, OUT_DIM),
    }


def _forward(x: np.ndarray[Any, Any], net: dict[str, np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
    h = np.maximum(x @ net["W1"] + net["b1"], 0.0)
    return h @ net["W2"] + net["b2"]


def _ibp_bounds(
    lo: np.ndarray[Any, Any], hi: np.ndarray[Any, Any], net: dict[str, np.ndarray[Any, Any]]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    from omni_mercury_engine.cognitive.formal_verification import IntervalBoundPropagator

    prop = IntervalBoundPropagator()
    l1, u1 = prop.propagate_linear((lo, hi), net["W1"], net["b1"])
    l1, u1 = prop.propagate_relu((l1, u1))
    l2, u2 = prop.propagate_linear((l1, u1), net["W2"], net["b2"])
    return np.atleast_1d(l2), np.atleast_1d(u2)


def _true_bounds(
    lo: np.ndarray[Any, Any],
    hi: np.ndarray[Any, Any],
    net: dict[str, np.ndarray[Any, Any]],
    rng: np.random.Generator,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    # Dense sample of the input box (interior + the 2^d corners, where ReLU
    # extrema for a monotone-ish composition tend to live).
    samples = rng.uniform(lo, hi, size=(N_SAMPLES, IN_DIM))
    corners = np.array(
        [[lo[i] if (c >> i) & 1 else hi[i] for i in range(IN_DIM)] for c in range(2**IN_DIM)]
    )
    out = _forward(np.vstack([samples, corners]), net)
    return out.min(axis=0), out.max(axis=0)


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cases", type=int, default=200)
    parser.add_argument("--out", default="artifacts/formal_verification_soundness.json", type=str)
    args = parser.parse_args()

    print("Formal-verification soundness (interval bound propagation vs sampled truth)")
    print("-" * 80)

    unsound = 0
    tightness: list[float] = []
    eps = 1e-9
    for case in range(args.n_cases):
        rng = np.random.default_rng(case)
        net = _random_net(rng)
        center = rng.normal(0, 1, IN_DIM)
        radius = rng.uniform(0.1, 1.0, IN_DIM)
        lo, hi = center - radius, center + radius
        l_ibp, u_ibp = _ibp_bounds(lo, hi, net)
        l_true, u_true = _true_bounds(lo, hi, net, rng)
        # Sound iff the certificate contains the (sampled) true range.
        if np.any(l_ibp > l_true + 1e-6) or np.any(u_ibp < u_true - 1e-6):
            unsound += 1
        true_w = float(np.mean(u_true - l_true)) + eps
        ibp_w = float(np.mean(u_ibp - l_ibp))
        tightness.append(ibp_w / true_w)

    soundness_rate = 1.0 - unsound / max(1, args.n_cases)
    mean_tightness = float(np.mean(tightness))
    median_tightness = float(np.median(tightness))
    passed = bool(soundness_rate == 1.0 and np.isfinite(mean_tightness))
    verdict = {
        "n_cases": args.n_cases,
        "soundness_rate": soundness_rate,
        "unsound_cases": unsound,
        "mean_tightness_ratio": mean_tightness,
        "median_tightness_ratio": median_tightness,
        "passed": passed,
        "verdict": (
            "VALIDATED -- interval bound propagation is sound on every case "
            f"(certificate always contains the true range); mean tightness {mean_tightness:.2f}x"
            if passed
            else f"UNSOUND -- {unsound} certificate(s) did not contain the true range"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"  cases={args.n_cases}  soundness_rate={soundness_rate:.3f}  unsound={unsound}")
    print(f"  tightness ratio: mean={mean_tightness:.2f}x  median={median_tightness:.2f}x  (1.0 = exact)")
    print("-" * 80)
    print(f"VERDICT: {verdict['verdict']}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
