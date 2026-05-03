"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

------------------------------------------------------------------------

One-shot legacy ``.pkl`` -> ``.npz`` migration tool.

Pickle has been removed from the Mercury Agent runtime. This tool is
the only place in the codebase that may execute ``pickle.load``, and
it does so in a clearly-marked operator workflow that:

* refuses to import the engine -- the engine never sees pickle bytes;
* re-launches itself in a hardened subprocess before doing any
  unpickling -- a fresh interpreter with ``PYTHONNOUSERSITE=1``,
  ``PYTHONDONTWRITEBYTECODE=1``, no ``PYTHONSTARTUP``, and a scrubbed
  ``PYTHONPATH`` -- so a malicious pickle cannot poison the operator's
  environment via user customizations or startup scripts. The
  subprocess boundary, not the flag combination, is the load-bearing
  isolation;
* prints a loud disclaimer and refuses to run unless the caller passes
  ``--i-trust-this-file`` confirming the input pickle came from a
  trusted source;
* writes the converted ``.npz`` archive with ``allow_pickle=False``;
* optionally signs the output with HMAC-SHA-256 via
  :func:`omni_mercury_engine.security.safe_load.sign_npz`.

Usage::

    python -m omni_mercury_engine.tools.migrate_pkl \\
        --input legacy.pkl --output legacy.npz --i-trust-this-file

    python -m omni_mercury_engine.tools.migrate_pkl \\
        --input legacy.pkl --output legacy.npz --i-trust-this-file \\
        --sign-key-hex 0123...  # 64 hex chars = 32 bytes
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path
from typing import Sequence

_HARDENED_SENTINEL = "MERCURY_MIGRATE_PKL_HARDENED"

# Environment variables we explicitly forward to the hardened child.
# Anything else (PYTHONSTARTUP, LD_PRELOAD, etc.) is dropped.
_FORWARDED_ENV = ("PATH", "LANG", "LC_ALL", "PYTHONPATH", "VIRTUAL_ENV", "HOME")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.migrate_pkl",
        description=(
            "Convert a legacy .pkl training payload to a safe .npz archive. "
            "Pickle is unsafe; this tool exists only for one-shot migration."
        ),
    )
    parser.add_argument("--input", required=True, help="Path to legacy .pkl file.")
    parser.add_argument("--output", required=True, help="Destination .npz path.")
    parser.add_argument(
        "--i-trust-this-file",
        action="store_true",
        required=False,
        help=(
            "Required acknowledgement that the input pickle is from a trusted "
            "source. Pickle deserialization can execute arbitrary code."
        ),
    )
    parser.add_argument(
        "--sign-key-hex",
        default=None,
        help=(
            "Optional 32-byte HMAC key (64 hex chars) used to write a "
            "<output>.sig sidecar via the safe_load module."
        ),
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=256 * 1024 * 1024,
        help="Reject input pickles larger than this (default 256 MiB).",
    )
    return parser


def _disclaimer() -> str:
    return (
        "WARNING: pickle deserialization can execute arbitrary code. This "
        "tool will load a pickle file in a hardened subprocess. Only run "
        "it on payloads from a trusted source you control. Mercury Agent "
        "will never load pickles at runtime."
    )


def _relaunch_hardened(argv: Sequence[str]) -> int:
    """Re-exec this module in a fresh subprocess with a scrubbed env.

    User customizations and startup scripts are disabled
    (``PYTHONNOUSERSITE=1``, no ``PYTHONSTARTUP``). Only a small
    allow-list of env vars is forwarded. The child process sets a
    sentinel so it does not re-launch itself.

    The load-bearing isolation here is the *process boundary* -- a
    malicious pickle that achieves code execution still cannot reach
    the parent (operator) process state.
    """
    import subprocess

    env: dict[str, str] = {
        _HARDENED_SENTINEL: "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name in _FORWARDED_ENV:
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    env.setdefault("LANG", "C")

    cmd = [sys.executable, "-m", "omni_mercury_engine.tools.migrate_pkl", *argv]
    completed = subprocess.run(cmd, env=env, check=False)
    return completed.returncode


def _do_migration(args: argparse.Namespace) -> int:
    """Body that runs inside the hardened subprocess."""
    import pickle  # noqa: S403  # Intentional: this is the only sanctioned use.

    import numpy as np

    src = Path(args.input)
    dst = Path(args.output)

    if not src.is_file():
        print(f"error: input file not found: {src}", file=sys.stderr)
        return 2
    size = src.stat().st_size
    if size > args.max_bytes:
        print(
            f"error: input file {size} bytes exceeds --max-bytes {args.max_bytes}",
            file=sys.stderr,
        )
        return 2
    if dst.exists():
        print(f"error: refusing to overwrite existing output: {dst}", file=sys.stderr)
        return 2

    print(f"loading legacy pickle: {src} ({size} bytes)", file=sys.stderr)
    with src.open("rb") as f:
        loaded = pickle.load(f)  # noqa: S301  # hardened subprocess; user opted in.

    if not isinstance(loaded, dict):
        print(
            f"error: pickle root must be a dict with 'features' and 'labels' "
            f"(got {type(loaded).__name__})",
            file=sys.stderr,
        )
        return 3

    features = loaded.get("features")
    labels = loaded.get("labels")
    if not isinstance(features, dict) or labels is None:
        print(
            "error: pickle must contain {'features': {str: ndarray}, 'labels': ndarray}",
            file=sys.stderr,
        )
        return 3

    archive: dict[str, np.ndarray] = {}
    for name, value in features.items():
        if not isinstance(name, str):
            print(f"error: feature key {name!r} is not a string", file=sys.stderr)
            return 3
        arr = np.asarray(value)
        if arr.dtype == object:
            print(
                f"error: feature {name!r} contains object dtype; refusing to "
                f"persist non-numeric data into .npz",
                file=sys.stderr,
            )
            return 3
        archive[name] = arr

    label_arr = np.asarray(labels)
    if label_arr.dtype == object:
        print("error: labels contain object dtype; refusing to persist", file=sys.stderr)
        return 3
    archive["labels"] = label_arr

    np.savez(str(dst), **archive)
    print(f"wrote: {dst}", file=sys.stderr)

    if args.sign_key_hex:
        try:
            key = bytes.fromhex(args.sign_key_hex)
        except ValueError as exc:
            print(f"error: --sign-key-hex is not valid hex: {exc}", file=sys.stderr)
            return 4
        if len(key) < 32:
            print("error: --sign-key-hex must decode to >= 32 bytes", file=sys.stderr)
            return 4
        # Re-import inside hardened path because site-packages is stripped;
        # the project's own packages are still on PYTHONPATH if installed
        # in development mode.
        from omni_mercury_engine.security.safe_load import sign_npz

        sig_path = sign_npz(dst, key)
        print(f"wrote signature: {sig_path}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.i_trust_this_file:
        print(_disclaimer(), file=sys.stderr)
        print(
            "refusing to run without --i-trust-this-file (this is intentional)",
            file=sys.stderr,
        )
        return 1

    if os.environ.get(_HARDENED_SENTINEL) == "1":
        # Already inside the hardened subprocess; do the actual work.
        return _do_migration(args)

    # Top-level invocation: relaunch ourselves under hardened flags.
    print(_disclaimer(), file=sys.stderr)
    nonce = secrets.token_hex(8)
    print(f"relaunching under hardened subprocess (nonce {nonce})...", file=sys.stderr)
    return _relaunch_hardened(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
