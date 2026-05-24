"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Benchmark: the advantage of oracle-grounded mystery scalars over a credulous baseline.

The thesis of the verifier family is that grounding a scalar in an independent oracle lets the
system *reject fabrications a credulous (model-trusting) baseline would accept*.  This benchmark
quantifies that on a labelled set of true and fabricated claims across every tier and writes a
committed JSON artifact so the delta is reproducible.

Two deltas are reported:
  * fabrication-detection rate: credulous baseline (accept every claim) vs. oracle-grounded;
  * grounded mystery scalars: the previously empty categories (0) vs. the count the registry
    grounds from the true claims.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    reset_global_network,
)
from omni_mercury_engine.verifiers import (
    collatz,
    goldbach,
    lean_theorem,
    paradox,
    physics,
    twin_primes,
)
from omni_mercury_engine.verifiers.goldbach import GoldbachCertificate
from omni_mercury_engine.verifiers.paradox import ParadoxDefenseCertificate
from omni_mercury_engine.verifiers.propositional import iff, var
from omni_mercury_engine.verifiers.registry import MysteryRegistry
from omni_mercury_engine.verifiers.twin_primes import TwinPrimeCertificate

_DEFAULT_OUT = ROOT / "benchmarks" / "mystery_verification_results.json"

# A correct Collatz trajectory for n=6, and a tampered copy with one illegal final step.
_COLLATZ_OK = (6, 3, 10, 5, 16, 8, 4, 2, 1)
_COLLATZ_TAMPERED = (6, 3, 10, 5, 16, 8, 4, 2, 99)


@dataclass(frozen=True)
class Claim:
    """A labelled claim: whether it is actually true, and what the oracle decides."""

    tier: str
    label: str
    is_true: bool
    oracle_accepts: bool


def _build_claims() -> list[Claim]:
    """Construct the labelled true/fabricated claim set across every verifier tier."""
    bogus_defense = ParadoxDefenseCertificate(
        name="bogus",
        naive=iff(var("X"), ~var("X")),
        defense=(frozenset({var("A")}), frozenset({~var("A")})),
    )
    claims = [
        Claim(
            "number_theory",
            "goldbach 100=3+97",
            True,
            goldbach.verify_certificate(GoldbachCertificate(100, 3, 97)).valid,
        ),
        Claim(
            "number_theory",
            "goldbach 100=9+91",
            False,
            goldbach.verify_certificate(GoldbachCertificate(100, 9, 91)).valid,
        ),
        Claim(
            "number_theory",
            "twin (11,13)",
            True,
            twin_primes.verify_certificate(TwinPrimeCertificate(11)).valid,
        ),
        Claim(
            "number_theory",
            "twin (7,9)",
            False,
            twin_primes.verify_certificate(TwinPrimeCertificate(7)).valid,
        ),
        Claim(
            "dynamical",
            "collatz 6 trajectory",
            True,
            collatz.verify_trajectory(6, _COLLATZ_OK).valid,
        ),
        Claim(
            "dynamical",
            "collatz 6 tampered",
            False,
            collatz.verify_trajectory(6, _COLLATZ_TAMPERED).valid,
        ),
        Claim(
            "physics",
            "E=mc^2",
            True,
            physics.verify_relation(physics.mass_energy_equivalence()).valid,
        ),
        Claim(
            "physics",
            "E=mc",
            False,
            physics.verify_relation(physics.dimensionally_wrong_mass_energy()).valid,
        ),
        Claim(
            "paradox", "liar defense", True, paradox.verify_defense(paradox.liar_paradox()).valid
        ),
        Claim("paradox", "bogus defense", False, paradox.verify_defense(bogus_defense).valid),
    ]
    if lean_theorem.lean_available():
        claims.append(
            Claim(
                "theorem",
                "lean 2+2=4",
                True,
                lean_theorem.verify_lean_proof(lean_theorem.KNOWN_THEOREM).valid,
            )
        )
        claims.append(
            Claim(
                "theorem",
                "lean 2+2=5",
                False,
                lean_theorem.verify_lean_proof(lean_theorem.FALSE_THEOREM).valid,
            )
        )
    return claims


def _confusion(claims: list[Claim], predict: Callable[[Claim], bool]) -> dict[str, int]:
    """Return the TP/FP/TN/FN counts for a strategy's acceptance function."""
    tp = fp = tn = fn = 0
    for c in claims:
        accepted = predict(c)
        if accepted and c.is_true:
            tp += 1
        elif accepted and not c.is_true:
            fp += 1
        elif not accepted and not c.is_true:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _metrics(cm: dict[str, int]) -> dict[str, float | int]:
    """Derive accuracy and fabrication-detection rate from a confusion matrix."""
    total = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
    false_total = cm["tn"] + cm["fp"]
    return {
        **cm,
        "accuracy": (cm["tp"] + cm["tn"]) / total if total else 0.0,
        "fabrication_detection_rate": cm["tn"] / false_total if false_total else 0.0,
    }


def _grounded_scalar_count() -> int:
    """Run the registry over the true claims and count distinct grounded mystery scalars."""
    reset_global_network()
    reg = MysteryRegistry(GlobalOmniScalarNetwork())
    reg.submit_goldbach(100)
    reg.submit_twin_prime(11)
    reg.submit_collatz(27)
    reg.submit_physics(physics.mass_energy_equivalence())
    reg.submit_physics(physics.newtons_second_law())
    reg.submit_paradox(paradox.liar_paradox())
    reg.submit_paradox(paradox.russell_paradox())
    if lean_theorem.lean_available():
        reg.submit_theorem(lean_theorem.KNOWN_THEOREM, name="two_plus_two")
    keys = {e.scalar_name for e in reg.ledger if e.registered}
    reset_global_network()
    return len(keys)


def run() -> dict[str, object]:
    """Run the benchmark and return the result document."""
    claims = _build_claims()
    baseline = _metrics(_confusion(claims, lambda _c: True))  # credulous: accept every claim
    oracle = _metrics(_confusion(claims, lambda c: c.oracle_accepts))
    grounded = _grounded_scalar_count()
    return {
        "benchmark": "mystery_verification",
        "lean_available": lean_theorem.lean_available(),
        "claim_count": len(claims),
        "claims": [
            {
                "tier": c.tier,
                "label": c.label,
                "is_true": c.is_true,
                "oracle_accepts": c.oracle_accepts,
            }
            for c in claims
        ],
        "credulous_baseline": baseline,
        "oracle_grounded": oracle,
        "delta": {
            "fabrication_detection_rate": oracle["fabrication_detection_rate"]
            - baseline["fabrication_detection_rate"],
            "accuracy": oracle["accuracy"] - baseline["accuracy"],
            "grounded_mystery_scalars_before": 0,
            "grounded_mystery_scalars_after": grounded,
        },
    }


def main() -> int:
    """CLI entry point: run the benchmark and write the JSON artifact."""
    parser = argparse.ArgumentParser(description="Oracle-grounded mystery verification benchmark")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args()

    results = run()
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    oracle = results["oracle_grounded"]
    baseline = results["credulous_baseline"]
    assert isinstance(oracle, dict) and isinstance(baseline, dict)
    print(f"claims: {results['claim_count']} | lean_available={results['lean_available']}")
    print(
        f"credulous baseline : accuracy={baseline['accuracy']:.2f} "
        f"fabrication_detection={baseline['fabrication_detection_rate']:.2f}"
    )
    print(
        f"oracle-grounded    : accuracy={oracle['accuracy']:.2f} "
        f"fabrication_detection={oracle['fabrication_detection_rate']:.2f}"
    )
    delta = results["delta"]
    assert isinstance(delta, dict)
    print(
        f"delta              : +{delta['fabrication_detection_rate']:.2f} fabrication detection; "
        f"grounded mystery scalars {delta['grounded_mystery_scalars_before']} -> "
        f"{delta['grounded_mystery_scalars_after']}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
