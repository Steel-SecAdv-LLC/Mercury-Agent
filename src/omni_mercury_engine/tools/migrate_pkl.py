r"""
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
it does so in a clearly-bounded operator workflow that:

* refuses to import the engine -- the engine never sees pickle bytes;
* re-launches itself in a hardened subprocess before unpickling. The
  hardening is environment-based, not flag-based, because ``-S -E -I``
  strip the project's own editable install and break legitimate
  operator workflows. Concretely the child runs with:

    - ``PYTHONNOUSERSITE=1`` (no user-site ``sitecustomize``)
    - ``PYTHONDONTWRITEBYTECODE=1`` (no stray ``.pyc`` files)
    - no ``PYTHONSTARTUP``, no ``PYTHONPATH``, no ``LD_PRELOAD`` (any env
      var not in :data:`_FORWARDED_ENV` is dropped)

  The subprocess boundary, not flag combinations, is the load-bearing
  isolation -- a malicious pickle that achieves code execution in the
  child still cannot reach the parent (operator) process state;
* writes the converted archive via ``numpy.savez``. ``numpy.savez``
  itself does not expose an ``allow_pickle`` parameter on the write
  path, so we enforce the "no pickle in the output" contract by
  rejecting any object-dtype array **before** the write happens (see
  ``_do_migration``). The resulting ``.npz`` is then loadable through
  :func:`omni_mercury_engine.security.safe_load.safe_load_training_data`
  with ``allow_pickle=False`` enforced on read;
* optionally signs the output with HMAC-SHA-256 via
  :func:`omni_mercury_engine.security.safe_load.sign_npz`.

Usage::

    python -m omni_mercury_engine.tools.migrate_pkl \\
        --input legacy.pkl --output legacy.npz

    python -m omni_mercury_engine.tools.migrate_pkl \\
        --input legacy.pkl --output legacy.npz \\
        --sign-key-hex 0123...  # 64 hex chars = 32 bytes
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_HARDENED_SENTINEL = "MERCURY_MIGRATE_PKL_HARDENED"

# Environment variables we explicitly forward to the hardened child.
# Anything else (PYTHONSTARTUP, PYTHONPATH, LD_PRELOAD, etc.) is dropped
# so a malicious pickle cannot weaponise import-path injection or a
# poisoned ``sitecustomize``. ``VIRTUAL_ENV`` is forwarded so a venv's
# own ``site-packages`` (which the editable install resolves through the
# venv's ``sys.path``, not via ``PYTHONPATH``) remains discoverable.
#
# ``LD_LIBRARY_PATH`` / ``DYLD_LIBRARY_PATH`` are forwarded so the AMA
# Cryptography native C library (``libama_cryptography.so``) remains
# resolvable when AMA is installed from a build tree (CI's standard
# install path) rather than via a system package manager. INVARIANT-7
# in ``ama_cryptography.crypto_api`` refuses to operate without the
# native HMAC/HKDF/SHA3 backends, so the hardened child must inherit
# the same loader search path the parent used to satisfy that invariant.
# The path is set by the operator's deployment environment, not by
# untrusted pickle content, so this preserves the threat model: the
# load-bearing isolation is the process boundary, not env-flag scrubbing.
# ``LD_PRELOAD`` is *not* forwarded; that is the actual injection vector.
_FORWARDED_ENV = (
    "PATH",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    "HOME",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    # AMA Cryptography honours ``AMA_NO_CYTHON`` and ``AMA_REQUIRE_REAL_PQC``
    # at import time; without them the hardened child would refuse to load
    # the native backend in CI / containerised production deployments.
    "AMA_NO_CYTHON",
    "AMA_REQUIRE_REAL_PQC",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.migrate_pkl",
        description=(
            "Convert a legacy .pkl training payload to a safe .npz archive. "
            "Runs in a hardened subprocess; the engine itself never loads pickles."
        ),
    )
    parser.add_argument("--input", required=True, help="Path to legacy .pkl file.")
    parser.add_argument("--output", required=True, help="Destination .npz path.")
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


def _banner() -> str:
    return (
        "migrate_pkl: converting legacy pickle in a hardened subprocess "
        "(PYTHONNOUSERSITE=1, scrubbed env). Mercury Agent itself never "
        "loads pickles at runtime."
    )


def _relaunch_hardened(argv: Sequence[str]) -> int:
    """
    Re-exec this module in a fresh subprocess with a scrubbed env.

    User customizations and startup scripts are disabled (``PYTHONNOUSERSITE=1``, no
    ``PYTHONSTARTUP``). Only a small allow-list of env vars is forwarded. The child process sets a
    sentinel so it does not re-launch itself.

    The load-bearing isolation here is the *process boundary* -- a malicious pickle that achieves
    code execution still cannot reach the parent (operator) process state.
    """
    import subprocess  # nosec B404

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
    # B603/S603: command list is fully constructed from sys.executable plus the
    # fixed module path of this tool plus argparse-validated args. There is no
    # shell interpolation and no shell=True.
    completed = subprocess.run(cmd, env=env, check=False)  # nosec B603
    return completed.returncode


def _do_migration(args: argparse.Namespace) -> int:
    """Body that runs inside the hardened subprocess."""
    # B403: pickle is intentionally used here -- this is the one-shot
    # operator migration tool whose entire purpose is to read a legacy
    # .pkl payload. The engine itself never imports pickle.
    import pickle  # nosec B403

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
        # B301: this is THE sanctioned pickle.load call in the codebase.
        # It only runs in the hardened subprocess. The engine itself
        # never reaches this code path; see module docstring.
        loaded = pickle.load(f)  # nosec B301

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

    # Historical note: prior numpy stubs declared the second positional of
    # ``np.savez`` as ``compress: bool`` while the runtime API explicitly
    # accepts named ``ndarray`` kwargs.  Newer numpy stubs (2.x) correctly
    # type the keyword form, so no ignore is needed.
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
        # Lazy import: the migration tool only depends on the project's
        # own ``security.safe_load`` module when --sign-key-hex is
        # actually used. Keeping the import here means the tool's
        # baseline import surface stays minimal (just stdlib + numpy)
        # for the common no-signing path. The hardened-subprocess
        # contract is environment-based, not flag-based -- venv /
        # system ``site-packages`` is *not* stripped, so this import
        # works whenever Mercury Agent is installed.
        from omni_mercury_engine.security.safe_load import sign_npz

        sig_path = sign_npz(dst, key)
        print(f"wrote signature: {sig_path}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if os.environ.get(_HARDENED_SENTINEL) == "1":
        # Already inside the hardened subprocess; do the actual work.
        return _do_migration(args)

    # Top-level invocation: relaunch ourselves in the hardened subprocess.
    # Forward whatever argv the caller supplied (test harnesses pass an
    # explicit list; CLI invocations leave argv=None and we fall back to
    # sys.argv[1:]). The earlier code unconditionally used sys.argv[1:],
    # which silently broke programmatic ``main([...])`` calls.
    print(_banner(), file=sys.stderr)
    nonce = secrets.token_hex(8)
    print(f"relaunching under hardened subprocess (nonce {nonce})...", file=sys.stderr)
    forwarded = list(argv) if argv is not None else sys.argv[1:]
    return _relaunch_hardened(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
