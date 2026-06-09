# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Primality oracle shared by the number-theory verifiers.

Deterministic Miller-Rabin with a fixed witness set, exact for every ``n < 3.317 * 10**24``
(Sorenson & Webster).  Plain integer arithmetic -- no model, no learned weights, no network
call.  ``is_prime`` is cross-checked against ``_is_prime_trial`` (naive trial division) in the
test-suite, so the oracle is itself validated by a second independent method.
"""

from __future__ import annotations

# Deterministic Miller-Rabin witnesses: exact primality test for all n < 3.317e24.
_MR_WITNESSES: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    """Deterministic primality test (Miller-Rabin, exact for n < 3.317e24)."""
    if n < 2:
        return False
    for w in _MR_WITNESSES:
        if n % w == 0:
            return n == w
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_WITNESSES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _is_prime_trial(n: int) -> bool:
    """Naive trial-division primality test.

    Obviously correct but slow; used only to independently validate :func:`is_prime`.
    """
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True
