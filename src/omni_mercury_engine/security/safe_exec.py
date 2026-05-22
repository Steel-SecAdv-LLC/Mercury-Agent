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

import errno
import os
import signal
import subprocess  # nosec B404 - this module IS the subprocess gate; sole annotated subprocess.run lives below
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Whether the host kernel exposes POSIX session APIs (``os.setsid`` is the
# capability indicator the stdlib uses internally for ``start_new_session``).
# Windows does not provide ``setsid``; on that platform we fall back to
# ``taskkill /T /F`` for tree-kill semantics.
_POSIX_SESSION = hasattr(os, "setsid") and hasattr(os, "killpg")


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
        if not arg:
            # The contract is "non-empty strings".  Empty strings in
            # argv[1:] are almost always a bug (a caller built a list
            # like ``[..., maybe_flag or "", ...]``) and they confuse
            # downstream tools (git treats "" as the empty pathspec
            # which silently matches everything).  Refuse here.
            raise UnsafeSubprocessError(f"safe_exec: argv[{i}] is empty.")
        if "\x00" in arg:
            raise UnsafeSubprocessError(f"safe_exec: argv[{i}] contains a NUL byte; refusing.")
    executable = argv_list[0]
    # The executable must be an absolute path. Refusing relative names
    # blocks $PATH-resolution attacks where a hostile cwd could shadow
    # a system binary.
    exec_path = Path(executable)
    if not exec_path.is_absolute():
        raise UnsafeSubprocessError(
            f"safe_exec: argv[0]='{executable}' must be an absolute path. "
            "Use sys.executable or shutil.which() output."
        )
    # And it must actually exist on disk AND be an executable regular
    # file. A missing or non-executable target would otherwise surface
    # inside ``subprocess.run`` as a ``FileNotFoundError``,
    # ``PermissionError``, or ``OSError`` -- and the gate documents
    # ``UnsafeSubprocessError`` as the single failure type for argv
    # validation. Checking ``is_file()`` rejects directories and
    # special files (sockets, devices); checking ``os.access(X_OK)``
    # rejects regular files without the executable bit set.
    import os as _os

    if not exec_path.exists():
        raise UnsafeSubprocessError(f"safe_exec: argv[0]='{executable}' does not exist on disk.")
    if not exec_path.is_file():
        raise UnsafeSubprocessError(
            f"safe_exec: argv[0]='{executable}' is not a regular file "
            "(directory, socket, or device); refuse to invoke as an executable."
        )
    if not _os.access(exec_path, _os.X_OK):
        raise UnsafeSubprocessError(
            f"safe_exec: argv[0]='{executable}' is not executable "
            "(missing X bit for the running user)."
        )
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


def _kill_process_tree(proc: subprocess.Popen[bytes] | subprocess.Popen[str]) -> None:
    """Best-effort tree-kill of ``proc`` and any descendants.

    POSIX: when ``proc`` was launched with ``start_new_session=True`` the
    child is the leader of its own session/process group, so
    ``os.killpg(pid, SIGKILL)`` reaps the child *and* every grandchild
    that was forked under it. This closes the orphan-grandchild gap left
    by ``Popen.kill()``, which only signals the immediate child.

    Windows: there is no analogue of ``killpg``. We invoke
    ``taskkill /F /T /PID <pid>`` which the Windows process API uses to
    walk the process snapshot and terminate the parent plus every
    descendant. ``taskkill`` is documented in MS-DOCS as the supported
    primitive for tree-kill from outside the process; we fall back to a
    plain ``Popen.kill`` if it is unavailable (e.g. constrained Nano
    images) so we still make a best effort.

    This function never raises: a failure to signal is logged via the
    return path (the caller still re-raises ``TimeoutExpired``) but does
    not mask the original timeout.
    """
    if proc.pid is None or proc.poll() is not None:
        # Either the child never started or it already exited; nothing
        # to clean up.
        return

    if _POSIX_SESSION:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            # Process already reaped — racing with normal exit; benign.
            return
        except OSError as exc:
            # ESRCH means the group is already gone (same race as above).
            # Anything else means we have no privilege to signal — fall
            # back to the direct kill on the immediate child so we at
            # least don't leave the parent running.
            if exc.errno != errno.ESRCH:
                proc.kill()
        return

    # Windows path. Resolve the taskkill absolute path first so the gate
    # cannot be redirected by a hostile $PATH entry (which is precisely
    # what S607 protects against). Fall back to a plain ``Popen.kill`` if
    # the resolver cannot find taskkill (constrained Nano images, custom
    # paths, etc.) so we still make a best effort.
    import shutil as _shutil  # local import: only needed on the Windows branch

    taskkill = _shutil.which("taskkill")
    if not taskkill:
        proc.kill()
        return
    try:
        subprocess.run(  # noqa: S603  # nosec B603 - taskkill resolved via shutil.which(); argv is a literal with no shell expansion
            [taskkill, "/F", "/T", "/PID", str(proc.pid)],
            shell=False,
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError):
        # taskkill became unreachable between resolution and exec —
        # fall back to immediate-child kill.
        proc.kill()


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
        timeout: Kill the child *and any orphaned grandchildren* after
            this many seconds.  Unlike ``subprocess.run(timeout=...)``,
            which only signals the immediate child via ``Popen.kill()``
            (leaving forked grandchildren reparented to PID 1), this
            gate launches the child in its own session/process group
            on POSIX (``start_new_session=True``) and tree-kills via
            ``os.killpg`` on timeout.  On Windows the equivalent is
            ``taskkill /F /T``.  See :func:`_kill_process_tree`.
        cwd: Working directory for the child.

    Returns:
        The ``CompletedProcess`` from ``subprocess.run``.

    Raises:
        UnsafeSubprocessError: argv validation failed.
        subprocess.CalledProcessError: ``check=True`` and the child
            exited non-zero.
        subprocess.TimeoutExpired: ``timeout`` elapsed before exit.
            Before the exception propagates, the entire process group
            (POSIX) or process tree (Windows) rooted at the child has
            been signalled for termination.
    """
    argv_list = _validate_argv(argv)
    env_dict: dict[str, str] | None = dict(env) if env is not None else None
    # argv is a list, shell is False, every element has been validated
    # to be a non-empty NUL-free string, and argv[0] is an absolute path
    # that exists on disk. This is the only subprocess call site in
    # src/; all other callers route through here.
    #
    # We use Popen + communicate(timeout=...) instead of subprocess.run
    # so we can interpose tree-kill semantics on timeout. The behaviour
    # otherwise matches subprocess.run exactly: same return type, same
    # exceptions, same kwargs.
    stdout_pipe = subprocess.PIPE if capture_output else None
    stderr_pipe = subprocess.PIPE if capture_output else None
    proc: subprocess.Popen[str] | subprocess.Popen[bytes] = subprocess.Popen(  # noqa: S603  # nosec B603 - argv pre-validated (no shell, no relative paths, no empty args, no NUL); sole annotated Popen in src/
        argv_list,
        shell=False,
        env=env_dict,
        stdout=stdout_pipe,
        stderr=stderr_pipe,
        text=text,
        cwd=cwd,
        # POSIX: leader of new session+process group so killpg() can
        # reap orphaned grandchildren. No-op on platforms without
        # setsid; on Windows we set CREATE_NEW_PROCESS_GROUP instead so
        # taskkill /T sees a coherent tree.
        start_new_session=_POSIX_SESSION,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            if not _POSIX_SESSION and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP")
            else 0
        ),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Reap the entire process group/tree before the partial output
        # is drained, then re-raise with the captured partials so the
        # caller gets the same observable behaviour as subprocess.run.
        _kill_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = b"", b""
            proc.wait(timeout=5)
        raise subprocess.TimeoutExpired(
            cmd=argv_list,
            timeout=timeout if timeout is not None else 0.0,
            output=stdout,
            stderr=stderr,
        ) from None
    except BaseException:
        # Any other escape (KeyboardInterrupt, system exit) must not
        # leak the child or its grandchildren.
        _kill_process_tree(proc)
        proc.wait()
        raise

    retcode = proc.poll()
    assert retcode is not None
    completed: subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes] = (
        subprocess.CompletedProcess(
            args=argv_list,
            returncode=retcode,
            stdout=stdout,
            stderr=stderr,
        )
    )
    if check and retcode != 0:
        raise subprocess.CalledProcessError(retcode, argv_list, output=stdout, stderr=stderr)
    return completed


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
