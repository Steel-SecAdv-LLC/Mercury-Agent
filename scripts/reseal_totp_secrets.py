#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rotate the at-rest encryption key for stored TOTP secrets.

Mercury seals 2FA (TOTP) secrets at rest under ``MERCURY_DATA_ENC_KEY`` (or a
key derived from ``AMA_MASTER_SEED``) — see
:mod:`omni_mercury_engine.api.secret_sealer`. The startup maintenance sweep
already upgrades *plaintext* secrets to sealed ones, but it cannot rotate from
one at-rest key to another: a sealed value only opens under the exact key it was
sealed with, so simply changing ``MERCURY_DATA_ENC_KEY`` would make every
enrolled account's second factor unopenable (fail-closed → nobody can pass 2FA).

This script performs the rotation safely. For each account it:

* **Already sealed under the NEW key** → left untouched (so the script is
  idempotent and safe to re-run, and a partial run resumes cleanly).
* **Sealed under the OLD key** → unsealed with the old key and re-sealed with the
  new key, bound to the same account id (the AAD), then written back.
* **Plaintext** → sealed under the new key (same upgrade the sweep performs).
* **Openable under neither key** → reported as a failure and left untouched
  (tampering, or the wrong old key was supplied) — never overwritten.

Usage::

    # Rotate from the retiring key to the new one (durable store required):
    export MERCURY_KEYSTORE_PATH=/var/lib/mercury/mercury.db
    export MERCURY_DATA_ENC_KEY=<new 64-hex key>        # the key to seal UNDER
    export MERCURY_DATA_ENC_KEY_OLD=<old 64-hex key>    # the key to unseal WITH
    python scripts/reseal_totp_secrets.py

    # Preview without writing:
    python scripts/reseal_totp_secrets.py --dry-run

    # Keys on the command line instead of the environment:
    python scripts/reseal_totp_secrets.py --new-key <hex> --old-key <hex>

Generate keys with ``python scripts/generate_secret_key.py`` (stdlib CSPRNG; no
external tool). Nothing here touches the network. The exit code is non-zero if
any secret could not be opened under either key, so an operator or CI run
notices rows that need attention.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

from omni_mercury_engine.api.identity_store import (
    IdentityStore,
    build_identity_store,
    identity_store_is_durable,
)
from omni_mercury_engine.api.secret_sealer import (
    DATA_ENC_KEY_ENV,
    SealedSecretError,
    SecretSealer,
)

#: Environment variable holding the *retiring* key (the one to unseal with).
DATA_ENC_KEY_OLD_ENV = "MERCURY_DATA_ENC_KEY_OLD"


@dataclass
class ResealReport:
    """Outcome of a re-seal pass."""

    total_accounts: int = 0
    with_totp: int = 0
    resealed: int = 0
    sealed_plaintext: int = 0
    already_current: int = 0
    #: Account ids whose secret opened under neither key (needs attention).
    failed: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def changed(self) -> int:
        """How many secrets this pass rewrote (or would, under ``--dry-run``)."""
        return self.resealed + self.sealed_plaintext


def reseal_totp_secrets(
    store: IdentityStore,
    *,
    new_sealer: SecretSealer,
    old_sealer: SecretSealer | None = None,
    dry_run: bool = False,
) -> ResealReport:
    """Re-seal every TOTP secret in ``store`` under ``new_sealer``.

    Args:
        store: The identity store to migrate (``iter_accounts`` /
            ``update_account``).
        new_sealer: Sealer holding the NEW key — every rewritten secret is
            sealed with this.
        old_sealer: Sealer holding the OLD (retiring) key, used to open secrets
            sealed under it. ``None`` means "no old key available": sealed
            values that do not already open under the new key are reported as
            failures rather than being lost.
        dry_run: When ``True``, classify every secret but write nothing.

    Returns:
        A :class:`ResealReport` describing what happened (or would happen).
    """
    report = ResealReport(dry_run=dry_run)
    for account in store.iter_accounts():
        report.total_accounts += 1
        secret = account.totp_secret
        if not secret:
            continue
        report.with_totp += 1
        aad = account.id

        if SecretSealer.is_sealed(secret):
            # Idempotency: if it already opens under the new key, this row was
            # migrated on a prior run (or the keys are identical) — leave it.
            try:
                new_sealer.unseal(secret, aad=aad)
                report.already_current += 1
                continue
            except SealedSecretError:
                pass
            # Otherwise it must open under the old key to be re-sealed.
            if old_sealer is None:
                report.failed.append(aad)
                continue
            try:
                plaintext = old_sealer.unseal(secret, aad=aad)
            except SealedSecretError:
                report.failed.append(aad)
                continue
            if not dry_run:
                account.totp_secret = new_sealer.seal(plaintext, aad=aad)
                store.update_account(account)
            report.resealed += 1
        else:
            # Legacy plaintext secret: seal it under the new key.
            if not dry_run:
                account.totp_secret = new_sealer.seal(secret, aad=aad)
                store.update_account(account)
            report.sealed_plaintext += 1

    return report


def _sealer_from_hex(label: str, raw: str) -> SecretSealer:
    """Build a stable :class:`SecretSealer` from a 64-hex-char key."""
    try:
        key = bytes.fromhex(raw)
    except ValueError as exc:
        raise SystemExit(
            f"{label} must be hex (generate with: python scripts/generate_secret_key.py)"
        ) from exc
    if len(key) != 32:
        raise SystemExit(f"{label} must decode to exactly 32 bytes (64 hex chars)")
    return SecretSealer(key, key_is_stable=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rotate the at-rest encryption key for stored TOTP secrets."
    )
    parser.add_argument(
        "--new-key",
        default=os.getenv(DATA_ENC_KEY_ENV, "").strip(),
        help=f"NEW key to seal under (64 hex chars). Defaults to ${DATA_ENC_KEY_ENV}.",
    )
    parser.add_argument(
        "--old-key",
        default=os.getenv(DATA_ENC_KEY_OLD_ENV, "").strip(),
        help=f"OLD key to unseal with (64 hex chars). Defaults to ${DATA_ENC_KEY_OLD_ENV}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify every secret and print the plan without writing anything.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)

    if not identity_store_is_durable():
        print(
            f"Refusing to run: no durable identity store configured (${'MERCURY_KEYSTORE_PATH'} "
            "is unset). Re-sealing an in-memory store would rewrite rows that vanish on exit.",
            file=sys.stderr,
        )
        return 2

    if not args.new_key:
        print(
            f"A new key is required (--new-key or ${DATA_ENC_KEY_ENV}).",
            file=sys.stderr,
        )
        return 2

    new_sealer = _sealer_from_hex("--new-key", args.new_key)
    old_sealer = _sealer_from_hex("--old-key", args.old_key) if args.old_key else None
    if old_sealer is None:
        print(
            f"No old key supplied (--old-key or ${DATA_ENC_KEY_OLD_ENV}); only plaintext "
            "secrets can be sealed. Secrets sealed under a different key will be reported "
            "as failures.",
            file=sys.stderr,
        )

    store = build_identity_store()
    report = reseal_totp_secrets(
        store, new_sealer=new_sealer, old_sealer=old_sealer, dry_run=args.dry_run
    )

    verb = "would reseal" if args.dry_run else "resealed"
    print(f"accounts scanned:        {report.total_accounts}")
    print(f"accounts with 2FA:       {report.with_totp}")
    print(f"{verb} (old→new key):   {report.resealed}")
    print(f"{verb} (plaintext→new): {report.sealed_plaintext}")
    print(f"already under new key:   {report.already_current}")
    print(f"could not open:          {len(report.failed)}")
    if report.failed:
        print("  affected account ids (unopenable under either key):", file=sys.stderr)
        for account_id in report.failed:
            print(f"    {account_id}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
