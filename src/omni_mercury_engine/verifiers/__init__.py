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

Each verifier follows the same shape -- a certificate adjudicated by an oracle that is
independent of any model, with the resulting scalar's value decided by the verdict:

* :mod:`primality`    -- shared deterministic primality oracle
* :mod:`goldbach`     -- Goldbach partition instances (number-theory tier)
* :mod:`twin_primes`  -- twin-prime pair instances (number-theory tier)
* :mod:`collatz`      -- Collatz trajectory instances (dynamical tier, semi-decidable)
* :mod:`lean_theorem` -- theorems checked by the Lean 4 kernel (formal-proof tier)
"""

from omni_mercury_engine.verifiers import collatz, goldbach, lean_theorem, twin_primes
from omni_mercury_engine.verifiers.primality import _is_prime_trial, is_prime

__all__ = [
    "_is_prime_trial",
    "collatz",
    "goldbach",
    "is_prime",
    "lean_theorem",
    "twin_primes",
]
