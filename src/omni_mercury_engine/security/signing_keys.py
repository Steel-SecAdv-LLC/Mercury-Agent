# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Owner-only ML-DSA-65 key files for the records Mercury signs at rest.

The adjudication ledger and its exports can carry ML-DSA-65 signatures, but
until now a signing key could only be passed in from Python, so nothing that
ran as a service (API, CLI, MCP) could sign. This module gives the key a home:
a JSON file readable by its owner only, named by an environment variable.

File format (``mercury.signing-key/v1``)::

    {"schema": "mercury.signing-key/v1", "algorithm": "ML-DSA-65",
     "public_key_hex": "...", "secret_key_hex": "..."}

Rules, each enforced on load:

* On POSIX the file must not be readable or writable by group or others
  (mode ``0o600`` or stricter); a looser file is refused, not repaired, the
  same rule ``scripts/gcn_kafka_keepalive.py`` applies to its credentials.
* The pair must match: a signature made with the secret key must verify
  under the public key. A file whose halves were swapped or edited is
  refused before anything is signed with it.
* Generation creates the file exclusively (``O_EXCL``) at mode ``0o600``, so
  an existing key is never overwritten.

Environment variables:

``MERCURY_ADJUDICATION_SIGNING_KEY``
    Path to the key file the adjudication surfaces sign ledger records and
    exports with.
``MERCURY_ADJUDICATION_PUBLIC_KEY``
    Hex ML-DSA-65 public key the ledger and exports are pinned to on read.
    When set, unsigned or differently signed records are refused.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni_mercury_engine.security.crypto_api import MLDSAProvider

KEY_FILE_SCHEMA = "mercury.signing-key/v1"
ALGORITHM = "ML-DSA-65"
ADJUDICATION_SIGNING_KEY_ENV = "MERCURY_ADJUDICATION_SIGNING_KEY"
ADJUDICATION_PUBLIC_KEY_ENV = "MERCURY_ADJUDICATION_PUBLIC_KEY"
_PAIR_CHECK_MESSAGE = b"mercury.signing-key/v1 pair check"


class SigningKeyError(ValueError):
    """A key file is missing, too permissive, malformed or mismatched."""


def _provider() -> MLDSAProvider:
    # Imported on use: AMA is required to sign, not to import this module.
    from omni_mercury_engine.security.crypto_api import MLDSAProvider

    return MLDSAProvider()


def generate_keyfile(path: str | Path) -> str:
    """Create a new ML-DSA-65 key file at ``path`` (mode 0600) and return the public key hex.

    Raises:
        FileExistsError: ``path`` already exists; keys are never overwritten.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pair = _provider().generate_keypair()
    document = {
        "schema": KEY_FILE_SCHEMA,
        "algorithm": ALGORITHM,
        "public_key_hex": pair.public_key.hex(),
        "secret_key_hex": pair.secret_key.hex(),
    }
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return pair.public_key.hex()


def load_keypair(path: str | Path) -> tuple[bytes, bytes]:
    """Load ``(public_key, secret_key)`` from an owner-only key file.

    Raises:
        SigningKeyError: The file is missing, readable by others, malformed,
            of another algorithm, or its halves do not form a pair.
    """
    target = Path(path).expanduser()
    try:
        info = target.stat()
    except OSError as exc:
        raise SigningKeyError(f"signing key file {target} is not readable: {exc}") from exc
    if os.name == "posix" and info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SigningKeyError(
            f"signing key file {target} is accessible to group or others "
            f"(mode {stat.S_IMODE(info.st_mode):o}); run: chmod 600 {target}"
        )
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
        if document.get("schema") != KEY_FILE_SCHEMA or document.get("algorithm") != ALGORITHM:
            raise SigningKeyError(f"{target} is not a {KEY_FILE_SCHEMA} {ALGORITHM} key file")
        public_key = bytes.fromhex(document["public_key_hex"])
        secret_key = bytes.fromhex(document["secret_key_hex"])
    except SigningKeyError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SigningKeyError(f"signing key file {target} is malformed: {exc}") from exc
    provider = _provider()
    signature = provider.sign(_PAIR_CHECK_MESSAGE, secret_key)
    if not provider.verify(_PAIR_CHECK_MESSAGE, signature, public_key):
        raise SigningKeyError(
            f"signing key file {target}: the secret key does not match the public key"
        )
    return public_key, secret_key


def keypair_from_env(
    variable: str = ADJUDICATION_SIGNING_KEY_ENV, environ: dict[str, str] | None = None
) -> tuple[bytes, bytes] | None:
    """Load the key file named by ``variable``; ``None`` when it is unset."""
    value = (os.environ if environ is None else environ).get(variable, "").strip()
    return load_keypair(value) if value else None


def pinned_public_key_from_env(
    variable: str = ADJUDICATION_PUBLIC_KEY_ENV, environ: dict[str, str] | None = None
) -> str | None:
    """Return the pinned public key hex from ``variable`` (lower case), or ``None``.

    Raises:
        SigningKeyError: The value is not hex.
    """
    value = (os.environ if environ is None else environ).get(variable, "").strip().lower()
    if not value:
        return None
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise SigningKeyError(f"{variable} is not a hex public key") from exc
    return value


__all__ = [
    "ADJUDICATION_PUBLIC_KEY_ENV",
    "ADJUDICATION_SIGNING_KEY_ENV",
    "ALGORITHM",
    "KEY_FILE_SCHEMA",
    "SigningKeyError",
    "generate_keyfile",
    "keypair_from_env",
    "load_keypair",
    "pinned_public_key_from_env",
]
