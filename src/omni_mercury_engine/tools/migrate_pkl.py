# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""One-shot legacy ``.pkl`` -> ``.npz`` migration tool.

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
* uses :class:`_RestrictedUnpickler` rather than the bare
  ``pickle.load``. ``find_class`` whitelists only the globals required
  to reconstruct numpy arrays plus a small set of basic Python
  builtins; anything else (``os.system``, ``subprocess.Popen``, any
  ``builtins.eval`` / ``builtins.exec`` / ``builtins.__import__`` /
  ``posix.*`` / ``nt.*`` reference, ``codecs.encode`` reduce-chain
  tricks) raises ``pickle.UnpicklingError`` *before* the global is
  resolved. This downgrades the threat from "arbitrary code
  execution on attempt 1" to "smuggle a malicious construction
  through ``numpy.core.multiarray._reconstruct``" -- defence in
  depth with the subprocess boundary, not a replacement for it;
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
from typing import TYPE_CHECKING, Any

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

# Stable exit code for ``_RestrictedUnpickler.find_class`` refusing a
# global that is not in the allow-list.  Distinct from:
#   0  success
#   2  input file errors (missing / oversize / would-overwrite)
#   3  payload shape / dtype errors
#   4  output write failures
# A dedicated code lets tests pin the refusal path without depending on
# the implementation-dependent unhandled-exception exit code.
_EXIT_RESTRICTED_UNPICKLER_REFUSAL = 5


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
    """Re-exec this module in a fresh subprocess with a scrubbed env.

    User customizations and startup scripts are disabled (``PYTHONNOUSERSITE=1``, no
    ``PYTHONSTARTUP``). Only a small allow-list of env vars is forwarded. The child process sets a
    sentinel so it does not re-launch itself.

    The load-bearing isolation here is the *process boundary* -- a malicious pickle that achieves
    code execution still cannot reach the parent (operator) process state.
    """
    from omni_mercury_engine.security.safe_exec import python_module

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

    # safe_exec.python_module validates the module name, pins
    # shell=False, and is the only annotated subprocess.run site in
    # src/.
    completed = python_module(
        "omni_mercury_engine.tools.migrate_pkl",
        args=list(argv),
        env=env,
        check=False,
    )
    return completed.returncode


#: Globals the restricted unpickler is willing to resolve. Every
#: entry is ``(module, qualname)``. The list is intentionally minimal:
#: just enough to round-trip a ``{'features': {str: ndarray}, 'labels':
#: ndarray}`` payload. Numpy 1.x stored arrays via
#: ``numpy.core.multiarray._reconstruct`` + ``numpy.ndarray`` +
#: ``numpy.dtype``; numpy 2.x adds the ``numpy._core`` re-export path,
#: which we accept too so legacy pickles produced under either numpy
#: ABI migrate cleanly. Builtins are limited to inert containers and
#: scalars -- ``eval`` / ``exec`` / ``__import__`` / ``getattr`` /
#: ``compile`` / ``open`` are deliberately omitted so a reduce-chain
#: cannot bootstrap into arbitrary code through ``builtins``.
#:
#: Pickle-protocol note (the ``_frombuffer`` entries): ``ndarray``
#: has *two* reduce paths.  Under protocol <= 4 (Python <= 3.13's
#: ``pickle.DEFAULT_PROTOCOL``) numpy serialises a contiguous array
#: through ``multiarray._reconstruct`` + a ``BINSTRING`` state.  Under
#: protocol 5 -- which Python 3.14 makes the *default* -- numpy takes
#: the PEP-574 zero-copy path instead, emitting
#: ``numpy._core.numeric._frombuffer`` (numpy 1.x: ``numpy.core.numeric``)
#: with a ``PickleBuffer``.  Both reconstructors are inert array
#: builders (buffer -> ndarray; no OS / import / attribute access), so
#: allow-listing ``_frombuffer`` alongside ``_reconstruct`` is the same
#: security posture while letting the tool migrate pickles written by
#: *any* supported interpreter.  Omitting it made every array-bearing
#: payload refuse with exit 5 the moment the default protocol flipped
#: to 5 on Python 3.14 (regression pinned in
#: ``tests/tools/test_migrate_pkl.py::TestProtocol5Reconstruction``).
_ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        # Numpy 1.x array reconstruction surface.
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        # Numpy 2.x renamed the private module to ``numpy._core``.
        # ``numpy.core.*`` still exists as a thin alias on 2.x but
        # pickles produced under 2.x reference the new path.
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        # Protocol-5 (Python 3.14 default) zero-copy array path. numpy
        # 2.x emits ``numpy._core.numeric._frombuffer``; numpy 1.x the
        # ``numpy.core.numeric`` alias. Same inert buffer->ndarray
        # builder as ``_reconstruct``; see the protocol note above.
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.numeric", "_frombuffer"),
        # Protocol 0/1/2 array-state reconstruction. Under the OLD pickle
        # protocols (0-2) — the protocols legacy operator pickles are most
        # likely to be in — numpy does NOT store the array databuffer as
        # raw bytes; it stores it as a latin-1 ``str`` and reconstructs the
        # bytes on load via ``_codecs.encode(state_str, 'latin1')``. Without
        # this entry EVERY array-bearing payload written at protocol <= 2
        # refused with exit 5, silently defeating the tool's entire purpose
        # for the oldest pickles (regression pinned in
        # tests/tools/test_migrate_pkl.py::TestAllProtocolsRoundTrip).
        #
        # Security: ``_codecs.encode`` is an INERT byte/string transform. It
        # cannot import, exec, spawn, or touch the OS, and it takes only
        # ``(obj, encoding)`` where ``encoding`` selects a registered codec —
        # all stdlib codecs are pure transforms and the restricted unpickler
        # exposes no ``codecs.register`` to add a hostile one. With no
        # ``eval``/``exec``/``__import__``/``getattr`` in this allow-list its
        # output (bytes/str) cannot be chained into code execution. Allowing
        # it keeps the deny-by-default posture: one more inert reconstruction
        # primitive, not an execution primitive. (The module docstring's
        # "codecs.encode reduce-chain tricks" caveat refers to chains that
        # also need an exec gadget — none is reachable here.)
        ("_codecs", "encode"),
        # Inert builtins -- no callable that touches the OS, the
        # import system, or arbitrary attribute lookup.
        ("builtins", "dict"),
        ("builtins", "list"),
        ("builtins", "tuple"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "str"),
        ("builtins", "bytes"),
        ("builtins", "bytearray"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "bool"),
        ("builtins", "complex"),
        # Pickle emits this for ``None`` in some protocols.
        ("builtins", "NoneType"),
    }
)


def _make_restricted_unpickler() -> Any:
    """Construct the ``_RestrictedUnpickler`` class, deferring the ``pickle`` import.

    ``pickle`` is imported here (not at module top) so that the only
    site in ``src/`` that touches the pickle module lives inside the
    hardened-subprocess code path. The class is created lazily on
    each call rather than at import time because the only caller is
    ``_do_migration``, which itself only runs inside the relaunched
    child process; building it at import time would force the parent
    process to import pickle even when no migration is happening.
    """
    # ``pickle`` is required to read legacy operator payloads;
    # ``_RestrictedUnpickler`` below overrides ``find_class`` to only
    # resolve globals in ``_ALLOWED_GLOBALS``. The import is reachable
    # only from inside the hardened subprocess.
    import pickle  # nosec B403 - restricted-unpickler scope; hardened-subprocess only; see _ALLOWED_GLOBALS + module docstring

    class _RestrictedUnpicklerRefusal(pickle.UnpicklingError):
        """A global outside ``_ALLOWED_GLOBALS`` was rejected.

        Subclass of ``UnpicklingError`` so existing ``except`` blocks
        in third-party code keep working, but distinguishable from the
        same exception raised by truncated / malformed pickle data.
        Only this subclass should map to exit code
        ``_EXIT_RESTRICTED_UNPICKLER_REFUSAL`` -- a corrupt-input
        ``UnpicklingError`` is an input/shape error and belongs on
        the same exit-code path as the rest of the schema checks.
        """

    class _RestrictedUnpickler(pickle.Unpickler):
        """``pickle.Unpickler`` that refuses any global not in the allow-list.

        Overrides :meth:`pickle.Unpickler.find_class` -- the single
        choke point through which every ``GLOBAL`` / ``STACK_GLOBAL``
        opcode resolves a callable. Returning a callable from
        ``find_class`` is what lets a vanilla unpickler invoke
        ``os.system`` via ``__reduce__``; refusing it here is what
        stops the RCE before it starts.
        """

        def find_class(self, module: str, name: str) -> Any:
            if (module, name) in _ALLOWED_GLOBALS:
                return super().find_class(module, name)
            raise _RestrictedUnpicklerRefusal(
                f"_RestrictedUnpickler: refusing global '{module}.{name}'; "
                f"not in migrate_pkl allow-list."
            )

    return _RestrictedUnpickler, _RestrictedUnpicklerRefusal


def _do_migration(args: argparse.Namespace) -> int:
    """Body that runs inside the hardened subprocess."""
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
    restricted_unpickler_cls, restricted_refusal_exc = _make_restricted_unpickler()
    # ``_make_restricted_unpickler`` imports ``pickle`` lazily and
    # returns the restricted subclass plus the policy-refusal
    # exception class.  We re-import ``pickle`` here only to bind
    # ``pickle.UnpicklingError`` for the corrupt-input branch below.
    # The subprocess-only import boundary documented in the module
    # docstring is preserved (this code path runs exclusively inside
    # the hardened relaunch).
    import pickle  # nosec B403 - re-import to bind UnpicklingError; same hardened-subprocess scope as line 259

    try:
        with src.open("rb") as f:
            # This is THE sanctioned pickle.load call in the codebase.
            # It only runs in the hardened subprocess **and** via
            # ``_RestrictedUnpickler`` (find_class whitelisted). The
            # engine itself never reaches this code path; see module
            # docstring.
            loaded = restricted_unpickler_cls(
                f
            ).load()  # nosec B301 - load() goes through _RestrictedUnpickler.find_class allow-list; subprocess + restricted unpickler is defence-in-depth
    except restricted_refusal_exc as exc:
        # Policy refusal: ``_RestrictedUnpickler.find_class`` rejected
        # a global outside ``_ALLOWED_GLOBALS`` (``os.system`` /
        # ``subprocess.Popen`` / ``builtins.eval`` and friends).
        # Surface as one concise stderr line and the documented
        # stable exit code so callers/tests can distinguish "refused"
        # from "corrupt input".
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_RESTRICTED_UNPICKLER_REFUSAL
    except pickle.UnpicklingError as exc:
        # Corrupt / truncated / malformed pickle data. This is NOT a
        # policy refusal -- ``find_class`` was never reached. Map to
        # the input/shape error exit code (3) so callers can tell
        # corrupt-input apart from refused-global.
        print(f"error: input pickle is malformed: {exc}", file=sys.stderr)
        return 3

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

    archive: dict[str, Any] = {}
    for name, value in features.items():
        if not isinstance(name, str):
            print(f"error: feature key {name!r} is not a string", file=sys.stderr)
            return 3
        arr = np.asarray(value)
        # ``dtype == object`` only matches the plain ``dtype('O')``; a
        # structured dtype whose fields contain objects (e.g.
        # ``[('a', 'O'), ('b', 'i4')]``) reports ``dtype != object`` yet
        # ``numpy.savez`` still serialises the object members through
        # pickle.  ``dtype.hasobject`` is True for both shapes, so it is
        # the right "any pickle in the output" predicate and the only
        # one that makes the docstring guarantee hold.
        if arr.dtype.hasobject:
            print(
                f"error: feature {name!r} contains object dtype (or structured "
                "dtype with object members); refusing to persist non-numeric "
                "data into .npz",
                file=sys.stderr,
            )
            return 3
        archive[name] = arr

    label_arr = np.asarray(labels)
    if label_arr.dtype.hasobject:
        print(
            "error: labels contain object dtype (or structured dtype with "
            "object members); refusing to persist",
            file=sys.stderr,
        )
        return 3
    archive["labels"] = label_arr

    # Guard against archive keys that collide with ``numpy.savez``'s
    # kw-only parameters.  On numpy 2.x ``savez`` declares
    # ``allow_pickle: bool = True`` between ``*args`` and ``**kwds``,
    # so a feature literally named ``"allow_pickle"`` would be
    # silently routed to that parameter (and reject non-bool values
    # at runtime, or coerce a truthy ndarray to ``True`` with no
    # warning).  On numpy 1.x (project floor ``>=1.24``) the kwarg
    # has no special meaning and would be silently stored as a 0-D
    # array named ``allow_pickle`` in the .npz — phantom data the
    # loader does not expect.  Either way is wrong: reject before
    # the call.
    _RESERVED_SAVEZ_KWARGS = frozenset({"allow_pickle"})
    collisions = _RESERVED_SAVEZ_KWARGS & archive.keys()
    if collisions:
        print(
            f"error: feature key(s) {sorted(collisions)} collide with "
            f"numpy.savez reserved kwargs; rename before persisting",
            file=sys.stderr,
        )
        return 3

    # ``**archive`` is typed ``dict[str, Any]`` (not the stricter
    # ``dict[str, np.ndarray[Any, Any]]``) so the spread satisfies numpy 2.x's
    # ``savez(file, *args, allow_pickle: bool, **kwds: ArrayLike)``
    # stub — the kw-only ``allow_pickle: bool`` slot accepts ``Any``,
    # which matches ``bool``.  The runtime contract that values are
    # ``ndarray`` is enforced by the ``np.asarray`` + object-dtype
    # filter above, and the reserved-kwarg guard above ensures
    # ``allow_pickle`` is never the key being passed.
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
