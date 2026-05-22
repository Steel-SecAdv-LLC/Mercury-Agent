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
* ``2`` -- usage error: config path not found.
* ``3`` -- Lyapunov gate failed; experiment was *not* launched.
* ``4`` -- config has no ``run_command``; nothing to execute (still
  considered a soft failure so CI surfaces the missing wiring).
* otherwise -- the exit code of the experiment subprocess.

Results are always written as JSON to the ``--out`` path so downstream
CI jobs can post deterministic regression reports.

Examples
--------
::

    python scripts/run_ablation.py \\
        --config configs/lyapunov_canonical.yaml \\
        --out artifacts/lyapunov_check.json

    python scripts/run_ablation.py \\
        --config configs/ablation_3r_lyapunov.yaml \\
        --out artifacts/ablation_result.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from tools.lyapunov_validator import validate_lyapunov_from_config  # noqa: E402


def _write_result(out_path: Path, result: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))


def _run_command(cmd: str, cwd: Path) -> int:
    """Run ``cmd`` in ``cwd`` and return its exit code.

    ``shell=True`` is used deliberately because configs declare full
    command strings (matching Mercury Agent's existing benchmark runner
    convention).  Configs must therefore only be authored by trusted
    repository contributors -- the same trust boundary that already
    applies to ``configs/*.yaml``.
    """
    print(f"RUN: {cmd}", flush=True)
    return subprocess.call(cmd, shell=True, cwd=str(cwd))  # noqa: S602


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg_path: Path = args.config
    out_path: Path = args.out

    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    valid, details = validate_lyapunov_from_config(cfg_path)
    result: Dict[str, Any] = {
        "config": str(cfg_path),
        "lyapunov_valid": bool(valid),
        "lyapunov_details": details,
    }

    if not valid:
        print(
            "Lyapunov validation failed; aborting experiment "
            f"(details={details})",
            file=sys.stderr,
        )
        _write_result(out_path, result)
        return 3

    if args.skip_run:
        result["skipped_run"] = True
        _write_result(out_path, result)
        return 0

    import yaml  # lazy: pyyaml is already a runtime dependency

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

    rc = _run_command(str(run_cmd), cwd=_REPO_ROOT)
    result["run_command"] = str(run_cmd)
    result["run_rc"] = int(rc)
    _write_result(out_path, result)
    return int(rc)


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    raise SystemExit(main())
