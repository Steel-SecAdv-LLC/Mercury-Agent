#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Type-I error and power of the Concealed Information Test, by simulation.

Each simulated session has ``--questions`` questions of ``--items`` items (one
probe, the rest irrelevant), each presented ``--repetitions`` times, with all
three channels recorded. Under the null every amplitude is independent
standard normal, so the probe is exchangeable with the irrelevant items. Under
an alternative the probe's amplitude is shifted by ``d`` standard deviations,
on the ``--effect-channels`` only (skin conductance by default; the other
channels stay pure noise), in the direction that
:data:`~omni_mercury_engine.truth_decipher.CHANNEL_SIGN` scores as a response
(larger skin conductance, heart-rate deceleration, shorter respiration line
length), on every repetition.

Both prespecified primary tests are measured: the exact rank test (the
default) and the item-relabelling permutation test. Rejection rates carry 95%
Wilson intervals. A valid test rejects at most ``alpha`` of null sessions; the
rank test's statistic is discrete and counts ties against the probe, so its
null rate sits below ``alpha``.

Usage::

    python benchmarks/cit_calibration.py [--sessions 2000] [--effects 0.25 0.5 1.0]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from omni_mercury_engine.truth_decipher import (
    CHANNEL_SIGN,
    CIT_CONSENT_SCOPE,
    Channel,
    CITConfig,
    CITQuestion,
    ConcealedInformationTest,
    ConsentRecord,
    wilson_interval,
)

RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "cit_calibration.json"
CONSENT = ConsentRecord("sim-subject", 0.0, frozenset({CIT_CONSENT_SCOPE}), "simulation")


def _session(
    rng: np.random.Generator,
    questions: int,
    items: int,
    repetitions: int,
    effect: float,
    effect_channels: frozenset[Channel],
) -> list[CITQuestion]:
    out = []
    for q in range(questions):
        responses: dict[Channel, tuple[tuple[float, ...], ...]] = {}
        for channel in Channel:
            amplitudes = rng.standard_normal((repetitions, items))
            if channel in effect_channels:
                amplitudes[:, 0] += CHANNEL_SIGN[channel] * effect
            responses[channel] = tuple(tuple(float(v) for v in row) for row in amplitudes)
        out.append(
            CITQuestion(
                question_id=f"q{q}",
                probe="probe",
                irrelevants=tuple(f"irrelevant-{i}" for i in range(1, items)),
                responses=responses,
            )
        )
    return out


def rejection_rate(
    *,
    sessions: int,
    questions: int,
    items: int,
    repetitions: int,
    effect: float,
    effect_channels: frozenset[Channel],
    primary_test: str,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """Return the share of simulated sessions whose primary test rejects at ``alpha``.

    Args:
        sessions: Simulated sessions.
        questions: Questions per session.
        items: Items per question (probe included).
        repetitions: Presentations of each item.
        effect: Probe shift in standard deviations (0 is the null).
        effect_channels: Channels the shift applies to.
        primary_test: ``"rank"`` or ``"permutation"``.
        alpha: Significance level.
        seed: Seed for the simulated amplitudes.

    Returns:
        ``{rejected, sessions, rate, wilson_95}``.
    """
    rng = np.random.default_rng(seed)
    cit = ConcealedInformationTest(
        CITConfig(alpha=alpha, primary_test=primary_test, seed=seed)  # type: ignore[arg-type]
    )
    rejected = 0
    for s in range(sessions):
        result = cit.run(
            _session(rng, questions, items, repetitions, effect, effect_channels),
            consent=CONSENT,
            subject_pseudonym="sim-subject",
            administered_at=float(s + 1),
        )
        if result.abstained:
            raise RuntimeError(f"session {s} abstained: {result.abstention_reason}")
        p = result.rank_test_p if primary_test == "rank" else result.permutation_p
        rejected += int(p is not None and p <= alpha)
    interval = wilson_interval(rejected, sessions, 0.95)
    return {
        "rejected": rejected,
        "sessions": sessions,
        "rate": rejected / sessions,
        "wilson_95": [interval.lower, interval.upper],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the simulation and write ``benchmarks/results/cit_calibration.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", type=int, default=2000)
    parser.add_argument("--questions", type=int, default=6)
    parser.add_argument("--items", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--effects", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument(
        "--effect-channels",
        nargs="+",
        choices=[c.value for c in Channel],
        default=[Channel.SKIN_CONDUCTANCE.value],
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    design = {
        "questions": args.questions,
        "items": args.items,
        "repetitions": args.repetitions,
        "channels": [c.value for c in Channel],
        "alpha": args.alpha,
        "effects_sd": args.effects,
        "effect_channels": args.effect_channels,
        "sessions": args.sessions,
        "seed": args.seed,
    }
    results: dict[str, Any] = {}
    started = time.perf_counter()
    effect_channels = frozenset(Channel(c) for c in args.effect_channels)
    for test in ("rank", "permutation"):
        for offset, effect in enumerate((0.0, *args.effects)):
            label = "null" if effect == 0.0 else f"d={effect:g}"
            results[f"{test}/{label}"] = rejection_rate(
                sessions=args.sessions,
                questions=args.questions,
                items=args.items,
                repetitions=args.repetitions,
                effect=effect,
                effect_channels=effect_channels,
                primary_test=test,
                alpha=args.alpha,
                seed=args.seed + offset,
            )
            print(f"{test:<12} {label:<8} {results[f'{test}/{label}']['rate']:.4f}")
    document = {
        "schema": "mercury.benchmarks.cit_calibration/v1",
        "design": design,
        "results": results,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "python": platform.python_version(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
