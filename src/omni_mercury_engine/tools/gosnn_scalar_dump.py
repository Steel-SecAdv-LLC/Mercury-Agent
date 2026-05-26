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
    parser.add_argument(
        "--include-named",
        action="store_true",
        help="Include named scalar values grouped by operational vs diagnostic band.",
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

    operational_named: dict[str, dict[str, float]] = {}
    diagnostic_named: dict[str, dict[str, float]] = {}
    total_named_operational = 0
    total_named_diagnostic = 0
    if args.include_named:
        for group in net.scalar_groups:
            operational = net._operational_scalars_for(group)
            diagnostic = net._metric_only_scalars_for(group)
            operational_named[group.value] = {
                name: float(value) for name, value in operational.items()
            }
            diagnostic_named[group.value] = {
                name: float(value) for name, value in diagnostic.items()
            }
            total_named_operational += len(operational)
            total_named_diagnostic += len(diagnostic)

    operational_vector = np.asarray(list(net._collect_all_scalars().values()), dtype=np.float64)
    scalars: dict[str, list[float]] = {
        "operational_vector": operational_vector.tolist(),
    }
    bands: dict[str, dict[str, float]] = {
        "operational_vector": {
            "count": int(operational_vector.size),
            "min": float(operational_vector.min()) if operational_vector.size else 0.0,
            "max": float(operational_vector.max()) if operational_vector.size else 0.0,
            "mean": float(operational_vector.mean()) if operational_vector.size else 0.0,
            "std": float(operational_vector.std()) if operational_vector.size else 0.0,
        }
    }

    # Best-effort legacy scalar extraction — GOSNN revisions may expose
    # additional vector attributes. Include them after the canonical
    # operational vector when they exist.
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
    if args.include_named:
        body["named_scalars"] = {
            "operational": operational_named,
            "diagnostic": diagnostic_named,
            "counts": {
                "operational": total_named_operational,
                "diagnostic": total_named_diagnostic,
                "registered": total_named_operational + total_named_diagnostic,
            },
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
