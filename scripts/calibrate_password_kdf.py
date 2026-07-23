#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure scrypt password-KDF cost on the current hardware.

The right scrypt parameters are a *measurement*, not a constant: the target is
the slowest setting whose latency the login path can afford (OWASP guidance:
well under ~1 s interactive), taken from the accepted parameter ladder
(``n=2^17,r=8,p=1`` … ``n=2^13,r=8,p=10`` — all ≈ equal attacker cost, trading
memory for parallelism). This script times each rung on the machine it runs
on plus the currently configured ``MERCURY_SCRYPT_*`` setting, and prints a
table so an operator can pick with data instead of folklore.

Reference numbers (Hetzner CCX23 / AMD EPYC 9454P vCPU, Python 3.12, measured
2026-07 via this script, median of 5): n=2^15,r=8,p=3 → ~85 ms / 32 MiB —
the shipped default. On slower hosts drop to n=2^14,r=8,p=5 (16 MiB) before
touching r.

Usage::

    python scripts/calibrate_password_kdf.py [--rounds 5]
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import statistics
import time

#: The OWASP-accepted scrypt ladder: equal attacker cost, decreasing memory.
LADDER: list[tuple[int, int, int]] = [
    (2**17, 8, 1),
    (2**16, 8, 2),
    (2**15, 8, 3),
    (2**14, 8, 5),
    (2**13, 8, 10),
]


def time_setting(n: int, r: int, p: int, rounds: int) -> tuple[float, float]:
    """Return (median_ms, mem_mib) for one scrypt setting over ``rounds`` runs."""
    password = secrets.token_bytes(16)
    salt = secrets.token_bytes(16)
    maxmem = 128 * n * r + 2 * 1024 * 1024
    samples: list[float] = []
    for _ in range(rounds):
        start = time.perf_counter()
        hashlib.scrypt(password, salt=salt, n=n, r=r, p=p, maxmem=maxmem, dklen=32)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), (128 * n * r) / (1024 * 1024)


def main(argv: list[str] | None = None) -> int:
    """Time the ladder plus the configured setting; print the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=5, help="timing rounds per setting")
    args = parser.parse_args(argv)

    from omni_mercury_engine.api.passwords import SCRYPT_N, SCRYPT_P, SCRYPT_R

    configured = (SCRYPT_N, SCRYPT_R, SCRYPT_P)
    settings = LADDER if configured in LADDER else [*LADDER, configured]

    print(f"{'n':>8} {'r':>3} {'p':>3} {'memory':>9} {'median':>10}  note")
    for n, r, p in settings:
        median_ms, mem_mib = time_setting(n, r, p, args.rounds)
        note = "<- configured (MERCURY_SCRYPT_*)" if (n, r, p) == configured else ""
        print(f"{n:>8} {r:>3} {p:>3} {mem_mib:>7.0f}MiB {median_ms:>8.1f}ms  {note}")
    print(
        "\nPick the slowest rung comfortably under your login-latency budget "
        "(OWASP: well under ~1s) and set MERCURY_SCRYPT_N/R/P accordingly."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
