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

Operator tool: dump the current ~209 omni-scalar values as JSON (127
operational + 82 diagnostic measurement scalars).

The README's GOSNN section makes structural claims about the σ band
of omni-scalars but an operator can't inspect them today.  This tool
constructs a ``GlobalOmniScalarNetwork``, captures the current scalar
state, and emits it as JSON for review.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.gosnn_scalar_dump/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.gosnn_scalar_dump",
        description="Dump the current omni-scalar vector of a GOSNN instance.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used to drive any random init.")
    parser.add_argument(
        "--probe-input",
        action="store_true",
        help="Run a synthetic detect_anomaly() to update the scalars before dumping.",
    )
    return parser


def _collect(args: argparse.Namespace) -> Certificate:
    from omni_mercury_engine.tools._base import DependencyMissing

    try:
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
        )
    except ImportError as exc:
        raise DependencyMissing(f"GOSNN import failed: {exc}") from exc

    net = GlobalOmniScalarNetwork()

    if args.probe_input:
        rng = np.random.default_rng(args.seed)
        X = rng.standard_normal((8, 16)).astype(np.float64)
        # ``GlobalOmniScalarNetwork`` exposes ``evaluate`` on the
        # numpy-vector surface but historically also ``detect_anomaly``
        # on the torch surface; we resolve via ``getattr`` so the dump
        # remains usable across GOSNN revisions and silently skips when
        # neither entrypoint is present.  Probe failure must never
        # block the scalar dump.
        probe = getattr(net, "detect_anomaly", None) or getattr(net, "evaluate", None)
        if callable(probe):
            try:
                probe(X)
            except Exception:
                pass

    # Best-effort scalar extraction — GOSNN exposes a few candidate
    # attributes; we try them all and report whichever yields a numeric
    # vector.  The redundant scan is intentional: it keeps this tool
    # robust to GOSNN internal refactors.
    scalars: dict[str, list[float]] = {}
    bands: dict[str, dict[str, float]] = {}
    for attr in (
        "scalars",
        "omni_scalars",
        "_omni_scalars",
        "_scalars",
        "state",
        "_state_vector",
    ):
        val = getattr(net, attr, None)
        if val is None:
            continue
        try:
            arr = np.asarray(val, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            continue
        if arr.size == 0:
            continue
        scalars[attr] = arr.tolist()
        bands[attr] = {
            "count": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }

    body: dict[str, Any] = {
        "class": type(net).__name__,
        "module": type(net).__module__,
        "scalar_sources": sorted(scalars),
        "scalars": scalars,
        "bands": bands,
    }
    warnings: list[str] = []
    if not scalars:
        warnings.append(
            "no omni-scalar attribute found; GOSNN structural contract may "
            "have changed — update gosnn_scalar_dump's attribute probe list"
        )
    status = "ok" if scalars else "warn"
    return Certificate(
        tool="gosnn_scalar_dump",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
