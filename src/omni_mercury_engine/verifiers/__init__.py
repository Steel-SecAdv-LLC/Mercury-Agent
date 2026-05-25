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

"""Oracle-validated verifiers that ground GOSNN scalars in independently checkable truth.

Every verifier follows the same shape -- a certificate adjudicated by an oracle independent of
any model, with the resulting scalar's value decided by the verdict. Three tiers are covered:

* number-theory / dynamical instances: :mod:`goldbach`, :mod:`twin_primes`, :mod:`collatz`
  (shared oracle: :mod:`primality`)
* physical-law consistency: :mod:`physics` (shared oracle: :mod:`dimensional`)
* formal proof and logical consistency: :mod:`lean_theorem`, :mod:`paradox`
  (shared oracle: :mod:`propositional`)

:class:`~omni_mercury_engine.verifiers.registry.MysteryRegistry` orchestrates all of them with
provenance tracking and a bounded, σ_Immutable-safe scalar footprint.
"""

from omni_mercury_engine.verifiers import (
    collatz,
    dimensional,
    goldbach,
    lean_theorem,
    paradox,
    physics,
    propositional,
    twin_primes,
)
from omni_mercury_engine.verifiers.primality import _is_prime_trial, is_prime
from omni_mercury_engine.verifiers.registry import LedgerEntry, MysteryRegistry
from omni_mercury_engine.verifiers.three_state import (
    KNOWN_UNDECIDABLE_IN_GENERAL,
    ThreeState,
    three_state_of,
)

__all__ = [
    "KNOWN_UNDECIDABLE_IN_GENERAL",
    "LedgerEntry",
    "MysteryRegistry",
    "ThreeState",
    "_is_prime_trial",
    "collatz",
    "dimensional",
    "goldbach",
    "is_prime",
    "lean_theorem",
    "paradox",
    "physics",
    "propositional",
    "three_state_of",
    "twin_primes",
]
