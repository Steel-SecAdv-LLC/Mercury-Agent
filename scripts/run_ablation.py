#!/usr/bin/env python3
"""Execute a Mercury Agent ablation experiment with a Lyapunov pre-gate.

This script is the canonical entry-point for running configurations under
``configs/`` that declare (a) an ablation experiment to execute and (b)
a Lyapunov decay claim to certify before the experiment runs.  The
pre-gate is non-negotiable: experiments are only launched when the
declared ``lambda`` is actually achievable by the supplied ``(A, P)``
matrices (or, for non-linear systems, by the supplied
``lyapunov_samples``).

Exit codes
----------
* ``0`` -- success: Lyapunov gate passed and the experiment completed.
* ``2`` -- usage error: ``--config`` path not found, ``--timeout`` value
  invalid, or PyYAML import failure.  On these exits the JSON report is
  intentionally NOT written, because there is no meaningful result to
  serialise; downstream tooling that polls for ``--out`` should treat
  the file's absence as the explicit "no run attempted" signal.
* ``3`` -- Lyapunov gate failed; experiment was *not* launched.  The
  JSON report IS written so CI dashboards can render the failed
  certificate's ``computed_lambda`` / ``claimed_lambda`` directly.
* ``4`` -- config has no ``run_command``; nothing to execute (still
  considered a soft failure so CI surfaces the missing wiring).  JSON
  report is written.
* ``124`` -- experiment exceeded ``--timeout`` seconds; mirrors GNU
  ``timeout(1)`` convention so wrapper scripts can detect the case
  without parsing stderr.  JSON report is written with
  ``run_timed_out=true``.
* otherwise -- the exit code of the experiment subprocess.

Examples
--------
::

    # Single-purpose Lyapunov config (canonical 2x2 surrogate).
    python scripts/run_ablation.py \\
        --config configs/lyapunov_canonical.yaml \\
        --out artifacts/lyapunov_check.json \\
        --skip-run

    # Multi-variant ablation config with a nested `lyapunov:` block.
    python scripts/run_ablation.py \\
        --config configs/ablation_3r_lyapunov.yaml \\
        --out artifacts/ablation_result.json \\
        --timeout 1800
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from tools.lyapunov_validator import validate_lyapunov_from_config

# GNU timeout(1) exit code for "command exceeded the time limit" so CI
# wrappers can detect timeout vs. genuine subprocess failure.
_TIMEOUT_EXIT_CODE: int = 124


def _write_result(out_path: Path, result: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))


def _run_command(cmd: str, cwd: Path, timeout: float | None) -> dict[str, Any]:
    """Run ``cmd`` in ``cwd`` and return its exit code plus timing data.

    ``shell=True`` is used deliberately because configs declare full
    command strings (matching Mercury Agent's existing benchmark runner
    convention).  Configs are part of the repository trust boundary --
    only repository contributors can author them, and they are reviewed
    on the same code path as ``src/``.  The subprocess inherits stdin
    from the parent (so an experiment that expects ``tty`` interaction
    fails fast rather than hanging) and inherits stdout/stderr (so CI
    logs remain readable in real time).

    Process-group isolation: the subprocess is started in a *new
    session* (``start_new_session=True``), which on POSIX creates a
    fresh process group rooted at the shell PID.  On timeout the entire
    group is signalled via ``os.killpg(SIGTERM)`` followed by
    ``SIGKILL`` if the shell's children fail to exit within a short
    grace window.  Without this, ``subprocess.run(..., shell=True,
    timeout=...)`` only kills the shell itself; any python /
    long-running child the shell spawned would survive as an orphan and
    keep consuming CI runner resources after the wrapper reported
    rc=124.  On non-POSIX platforms (Windows), ``start_new_session`` is
    a no-op and we fall back to Popen.terminate(), accepting the
    weaker guarantee documented by ``subprocess``.

    A non-None ``timeout`` enforces a wall-clock bound: the subprocess
    is killed on expiry and the function returns the canonical
    ``timeout(1)`` exit code (``124``) plus a ``timed_out=True`` flag.
    """
    print(f"RUN: {cmd}", flush=True)
    import os
    import signal
    import time as _time

    started = _time.monotonic()
    timed_out = False
    proc = subprocess.Popen(  # noqa: S602 - documented trust boundary
        cmd,
        shell=True,
        cwd=str(cwd),
        start_new_session=True,
    )
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        # Signal the whole process group so the shell's children die too.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover
            proc.terminate()
        # Grace window for graceful exit.
        try:
            rc = proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover
                proc.kill()
            proc.wait()
            rc = _TIMEOUT_EXIT_CODE
        else:
            # The group exited during the SIGTERM grace window; report
            # the wrapper's canonical timeout exit code, not whatever
            # rc the shell happened to return.
            rc = _TIMEOUT_EXIT_CODE
        print(
            f"ERROR: run_command exceeded --timeout={timeout!r}s; killed.",
            file=sys.stderr,
        )
    elapsed = _time.monotonic() - started
    return {
        "rc": int(rc),
        "elapsed_s": elapsed,
        "timed_out": timed_out,
        "timeout_s": float(timeout) if timeout is not None else None,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an ablation with a Lyapunov pre-gate.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to ablation/lyapunov YAML config.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/ablation_result.json"),
        help="Where to write the machine-readable JSON result.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only run the Lyapunov pre-gate; do not execute run_command.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Wall-clock timeout in seconds for the experiment subprocess. "
            "Defaults to no timeout. Exceeding the timeout exits with 124 "
            "(GNU timeout(1) convention)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg_path: Path = args.config
    out_path: Path = args.out

    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    if args.timeout is not None and args.timeout <= 0:
        print(
            f"ERROR: --timeout must be positive (got {args.timeout!r}).",
            file=sys.stderr,
        )
        return 2

    valid, details = validate_lyapunov_from_config(cfg_path)
    result: dict[str, Any] = {
        "config": str(cfg_path),
        "lyapunov_valid": bool(valid),
        "lyapunov_details": details,
    }

    if not valid:
        print(
            "Lyapunov validation failed; aborting experiment " f"(details={details})",
            file=sys.stderr,
        )
        _write_result(out_path, result)
        return 3

    if args.skip_run:
        result["skipped_run"] = True
        _write_result(out_path, result)
        return 0

    try:
        import yaml  # PyYAML is declared in pyproject.toml core deps
    except ImportError as exc:
        print(
            "ERROR: PyYAML is required to read the `run_command` field but "
            f"is not installed in this environment ({exc}). Install with "
            "`pip install pyyaml>=6.0` or re-run with `--skip-run` to gate "
            "only on the Lyapunov certificate.",
            file=sys.stderr,
        )
        return 2

    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    run_cmd = cfg.get("run_command") if isinstance(cfg, dict) else None
    if not run_cmd:
        print(
            "No run_command in config; Lyapunov gate passed but nothing to "
            "execute. Add a `run_command:` field to enable experiment "
            "execution.",
            file=sys.stderr,
        )
        result["run_command"] = None
        _write_result(out_path, result)
        return 4

    run_info = _run_command(str(run_cmd), cwd=_REPO_ROOT, timeout=args.timeout)
    result["run_command"] = str(run_cmd)
    result["run_rc"] = run_info["rc"]
    result["run_elapsed_s"] = run_info["elapsed_s"]
    result["run_timed_out"] = run_info["timed_out"]
    result["run_timeout_s"] = run_info["timeout_s"]
    _write_result(out_path, result)
    return int(run_info["rc"])


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    raise SystemExit(main())
