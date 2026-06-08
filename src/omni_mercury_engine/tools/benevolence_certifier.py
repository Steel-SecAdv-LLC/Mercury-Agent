# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.tools._base import Certificate, DependencyMissing, run_tool

_SCHEMA = "mercury.tools.benevolence_certifier/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.benevolence_certifier",
        description=(
            "Pre-deployment gate: run a curated probe set through the trained "
            "σ_Immutable checkpoint and assert score >= threshold on every probe."
        ),
    )
    parser.add_argument(
        "--probes",
        default=None,
        help=(
            "Optional .npy file with a (N, INPUT_DIM) matrix of feature vectors. "
            "If omitted, synthetic ethical-band probes are generated from --seed."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=64,
        help="Number of synthetic probes (ignored when --probes is given).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for synthetic probe generation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Score threshold; defaults to the gate's configured threshold "
            "(SIGMA_IMMUTABLE_DEFAULT 0.93).  Use --threshold 0.99 for the "
            "BENEVOLENCE_IMMUTABLE band."
        ),
    )
    parser.add_argument(
        "--pass-rate",
        type=float,
        default=1.0,
        help=(
            "Required fraction of probes that must pass (default 1.0). "
            "Operators occasionally relax this for noisy real-world probe sets."
        ),
    )
    return parser


def _ethical_band_probes(n: int, seed: int) -> npt.NDArray[np.float32]:
    """Synthesise ``n`` positive (ethical-band) probes.

    Reproduces the exact labelling rule used by
    :func:`omni_mercury_engine.security.sigma_immutable_corpus.generate_corpus`
    for the positive class: critical-ethical columns drawn from
    ``U[threshold, 2.0]`` and the remaining used columns from ``U[0, 2]``.
    """
    from omni_mercury_engine.security.sigma_immutable_corpus import (
        CORPUS_ETHICAL_DIMS,
        CORPUS_INPUT_DIM,
        CORPUS_THRESHOLD,
        CORPUS_USED_DIM,
    )

    rng = np.random.default_rng(seed)
    out = np.zeros((n, CORPUS_INPUT_DIM), dtype=np.float32)
    for i in range(n):
        out[i, :CORPUS_ETHICAL_DIMS] = rng.uniform(
            CORPUS_THRESHOLD, 2.0, CORPUS_ETHICAL_DIMS
        ).astype(np.float32)
        out[i, CORPUS_ETHICAL_DIMS:CORPUS_USED_DIM] = rng.uniform(
            0.0, 2.0, CORPUS_USED_DIM - CORPUS_ETHICAL_DIMS
        ).astype(np.float32)
    return out


def _collect(args: argparse.Namespace) -> Certificate:
    try:
        from omni_mercury_engine.security.sigma_immutable_gate import (
            SigmaImmutableGate,
        )
    except ImportError as exc:
        raise DependencyMissing(
            f"σ_Immutable gate import failed (torch likely missing): {exc}"
        ) from exc

    if args.probes:
        probes = np.load(args.probes, allow_pickle=False)
        if probes.ndim != 2:
            raise ValueError(f"--probes must be 2-D, got shape {probes.shape}")
        source = f"file:{args.probes}"
    else:
        probes = _ethical_band_probes(args.n, args.seed)
        source = f"synthetic(n={args.n}, seed={args.seed})"

    SigmaImmutableGate.reset_for_tests()
    if args.threshold is not None:
        gate = SigmaImmutableGate(threshold=float(args.threshold))
    else:
        gate = SigmaImmutableGate()

    effective_threshold = float(gate._threshold)  # accessor matches gate's clamp

    scores: list[float] = []
    passes: list[bool] = []
    for row in probes:
        ev = gate.evaluate(np.asarray(row))
        scores.append(float(ev.score))
        passes.append(bool(ev.passes))

    total = len(scores)
    n_pass = sum(passes)
    pass_rate = n_pass / total if total else 0.0

    body: dict[str, Any] = {
        "source": source,
        "probe_count": total,
        "threshold_effective": effective_threshold,
        "required_pass_rate": float(args.pass_rate),
        "observed_pass_rate": pass_rate,
        "n_pass": n_pass,
        "n_fail": total - n_pass,
        "score_min": float(np.min(scores)) if scores else 0.0,
        "score_mean": float(np.mean(scores)) if scores else 0.0,
        "score_max": float(np.max(scores)) if scores else 0.0,
    }

    warnings: list[str] = []
    if total and pass_rate + 1e-12 < args.pass_rate:
        warnings.append(f"observed pass-rate {pass_rate:.4f} below required {args.pass_rate:.4f}")
        status = "fail"
    else:
        status = "ok"

    return Certificate(
        tool="benevolence_certifier",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
