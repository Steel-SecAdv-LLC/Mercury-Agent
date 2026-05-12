"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

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

safe_exec -- single subprocess gate for Mercury Agent.

This module owns the only ``subprocess.run`` call in ``src/`` that
bandit can flag.  The argv list is validated before the call:

* the argv must be a non-empty sequence of strings (no string -- no
  shell);
* ``shell=False`` is enforced;
* the executable path must be absolute (``sys.executable`` and the
  output of ``shutil.which()`` both satisfy this) and must exist on
  disk before the call -- so a typo or a deleted binary fails loudly
  here instead of inside the child;
* the child env is whatever the caller passes.  ``env=None`` inherits
  the parent's full environment, exactly like ``subprocess.run`` does;
  callers that need scrubbing build the dict explicitly (see
  ``tools/migrate_pkl.py:_relaunch_hardened`` for the hardened
  template).  This module does not invent an implicit allowlist
  because the right scrub varies per caller.

Every place in Mercury Agent that needs to run another process
imports ``safe_exec`` from here.  The current callers are
``tools/migrate_pkl.py`` (re-launches itself in a hardened subprocess
before unpickling) and ``loaders/base.py`` (reads ``git rev-parse
HEAD`` for provenance).
"""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class UnsafeSubprocessError(ValueError):
    """The argv / env supplied to safe_exec violated the gate."""


def _validate_argv(argv: Sequence[str]) -> list[str]:
    """Validate the argv list and return a defensive copy.

    Raises:
        UnsafeSubprocessError: argv is not a sequence of non-empty
            NUL-free strings, or the first element is not an absolute
            path that exists on disk.
    """
    if isinstance(argv, (str, bytes)):
        raise UnsafeSubprocessError(
            "safe_exec: argv must be a sequence of strings, not a single "
            "string. Passing a string would invoke a shell via posix "
            "argument parsing in some downstream callers."
        )
    argv_list = list(argv)
    if not argv_list:
        raise UnsafeSubprocessError("safe_exec: argv must be non-empty.")
    for i, arg in enumerate(argv_list):
        if not isinstance(arg, str):
            raise UnsafeSubprocessError(f"safe_exec: argv[{i}] is {type(arg).__name__}, not str.")
        if "\x00" in arg:
            raise UnsafeSubprocessError(f"safe_exec: argv[{i}] contains a NUL byte; refusing.")
    executable = argv_list[0]
    if not executable:
        raise UnsafeSubprocessError("safe_exec: argv[0] is empty.")
    # The executable must be an absolute path. Refusing relative names
    # blocks $PATH-resolution attacks where a hostile cwd could shadow
    # a system binary.
    exec_path = Path(executable)
    if not exec_path.is_absolute():
        raise UnsafeSubprocessError(
            f"safe_exec: argv[0]='{executable}' must be an absolute path. "
            "Use sys.executable or shutil.which() output."
        )
    # And it must actually exist on disk. A missing executable would
    # surface inside subprocess as FileNotFoundError; failing here
    # gives the caller a single, consistent exception type.
    if not exec_path.exists():
        raise UnsafeSubprocessError(f"safe_exec: argv[0]='{executable}' does not exist on disk.")
    return argv_list


@overload
def safe_exec(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = ...,
    check: bool = ...,
    capture_output: bool = ...,
    text: Literal[True] = ...,
    timeout: float | None = ...,
    cwd: str | None = ...,
) -> subprocess.CompletedProcess[str]: ...


@overload
def safe_exec(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = ...,
    check: bool = ...,
    capture_output: bool = ...,
    text: Literal[False],
    timeout: float | None = ...,
    cwd: str | None = ...,
) -> subprocess.CompletedProcess[bytes]: ...


def safe_exec(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    check: bool = False,
    capture_output: bool = False,
    text: bool = True,
    timeout: float | None = None,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    """Run a subprocess after validating argv and pinning shell=False.

    Args:
        argv: Command as a list of strings.  ``argv[0]`` must be an
            absolute path that exists.  Strings (which would trigger
            shell invocation in some downstream callers) are rejected.
        env: Environment for the child.  ``None`` inherits the parent's
            full environment (same semantics as ``subprocess.run``).
            A supplied mapping is the *complete* env for the child.
        check: Raise ``CalledProcessError`` on non-zero exit.
        capture_output: Capture stdout/stderr.
        text: Decode stdout/stderr as text.  When True the return type
            is ``CompletedProcess[str]``; when False it is
            ``CompletedProcess[bytes]`` (overloads narrow this).
        timeout: Kill the child after this many seconds.
        cwd: Working directory for the child.

    Returns:
        The ``CompletedProcess`` from ``subprocess.run``.

    Raises:
        UnsafeSubprocessError: argv validation failed.
        subprocess.CalledProcessError: ``check=True`` and the child
            exited non-zero.
        subprocess.TimeoutExpired: ``timeout`` elapsed before exit.
    """
    argv_list = _validate_argv(argv)
    env_dict: dict[str, str] | None = dict(env) if env is not None else None
    # B603/S603: argv is a list, shell is False, every element has been
    # validated to be a non-empty NUL-free string and argv[0] is an
    # absolute path that exists on disk.  This is the only subprocess
    # call site in src/; all other callers route through here.
    return subprocess.run(  # noqa: S603  # nosec B603
        argv_list,
        shell=False,
        env=env_dict,
        check=check,
        capture_output=capture_output,
        text=text,
        timeout=timeout,
        cwd=cwd,
    )


def python_module(
    module: str,
    args: Sequence[str] = (),
    *,
    env: Mapping[str, str] | None = None,
    check: bool = False,
    capture_output: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Re-exec ``python -m <module> <args>`` via :func:`safe_exec`.

    Convenience for the migrate_pkl-style "relaunch myself in a
    hardened subprocess" pattern.  Uses ``sys.executable`` so the
    same interpreter (and therefore the same venv) is used.

    Output is always returned as text -- callers that need raw bytes
    should use :func:`safe_exec` directly with ``text=False`` so the
    overloads can narrow the return type.
    """
    if not module or not all(c.isalnum() or c in "._" for c in module):
        raise UnsafeSubprocessError(
            f"safe_exec.python_module: module name '{module}' must match "
            "[A-Za-z0-9._]+ to be safe as a -m argument."
        )
    argv = [sys.executable, "-m", module, *args]
    return safe_exec(
        argv,
        env=env,
        check=check,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
    )


__all__ = [
    "UnsafeSubprocessError",
    "python_module",
    "safe_exec",
]
