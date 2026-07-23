#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate the random hex keys Mercury's operator env vars expect.

A first-class, dependency-free alternative to ``openssl rand -hex N``: the
random bytes come from Python's stdlib :mod:`secrets` (the OS CSPRNG —
``getrandom(2)`` / ``/dev/urandom``), so provisioning Mercury's secrets needs
**no third-party tool downloaded or invoked** — not OpenSSL, not anything.

The values are inert random hex; nothing here touches AMA, the network, or the
filesystem. Each Mercury key env var and its required length:

* ``AMA_MASTER_SEED``     — 64 bytes (derives JWT + TOTP-sealing keys fleet-wide)
* ``JWT_SECRET_KEY``      — 32 bytes (only if not deriving from the master seed)
* ``MERCURY_DATA_ENC_KEY``— 32 bytes (TOTP at-rest sealing; ditto)
* ``API_KEY_HASH_SALT``   — 32 bytes (API-key hashing; required in production)

Usage::

    python scripts/generate_secret_key.py                 # one 32-byte key
    python scripts/generate_secret_key.py --bytes 64      # a 64-byte master seed
    python scripts/generate_secret_key.py --all           # every var, ready to export
"""

from __future__ import annotations

import argparse
import secrets

#: (env var, byte length) for the ``--all`` block.
_KEYS: tuple[tuple[str, int], ...] = (
    ("AMA_MASTER_SEED", 64),
    ("API_KEY_HASH_SALT", 32),
    ("MERCURY_DATA_ENC_KEY", 32),
    ("JWT_SECRET_KEY", 32),
)


def generate(num_bytes: int) -> str:
    """Return ``num_bytes`` of CSPRNG output as a lowercase hex string."""
    return secrets.token_hex(num_bytes)


def main(argv: list[str] | None = None) -> int:
    """Print one key, or an exportable block of every Mercury key var."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bytes",
        type=int,
        default=32,
        help="length of the single key in bytes (default 32; use 64 for a master seed)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="print an export line for every Mercury secret env var",
    )
    args = parser.parse_args(argv)

    if args.all:
        for name, length in _KEYS:
            print(f"export {name}={generate(length)}")
        return 0

    if args.bytes < 16:
        parser.error("refusing to generate a key shorter than 16 bytes")
    print(generate(args.bytes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
