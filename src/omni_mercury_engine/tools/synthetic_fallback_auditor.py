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

Operator tool: synthetic-fallback auditor.

Scans a benchmark results JSON for any dataset using >50% synthetic
data and flags it.  README already says this triggers a warning at
runtime; this tool enforces it post-hoc on the recorded benchmark
artefact so a CI gate can fail-closed on stealthy synthetic-fallback
contamination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.synthetic_fallback_auditor/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.synthetic_fallback_auditor",
        description=(
            "Scan benchmark results for >50% synthetic data and flag. "
            "Enforces the README's stated rule post-hoc."
        ),
    )
    parser.add_argument("results", help="Path to benchmark results JSON.")
    parser.add_argument(
        "--max-synthetic-fraction",
        type=float,
        default=0.5,
        help="Flag any dataset above this synthetic fraction (default 0.5).",
    )
    return parser


def _iter_datasets(doc: Any) -> list[tuple[str, dict[str, Any]]]:
    """Yield ``(dataset_name, record)`` pairs from a variety of shapes."""
    out: list[tuple[str, dict[str, Any]]] = []
    if isinstance(doc, dict) and "datasets" in doc and isinstance(doc["datasets"], dict):
        for k, v in doc["datasets"].items():
            if isinstance(v, dict):
                out.append((str(k), v))
        return out
    if isinstance(doc, dict) and "results" in doc and isinstance(doc["results"], list):
        for row in doc["results"]:
            if isinstance(row, dict) and "dataset" in row:
                out.append((str(row["dataset"]), row))
        return out
    if isinstance(doc, dict):
        # detector -> dataset -> record
        for detector, datasets in doc.items():
            if isinstance(datasets, dict):
                for ds, rec in datasets.items():
                    if isinstance(rec, dict):
                        out.append((f"{detector}/{ds}", rec))
    return out


def _synthetic_fraction(rec: dict[str, Any]) -> float | None:
    for k in (
        "synthetic_fraction",
        "synthetic_ratio",
        "synthetic_pct",
        "fallback_fraction",
    ):
        v = rec.get(k)
        if isinstance(v, (int, float)):
            return float(v) / (100.0 if k.endswith("pct") else 1.0)
    n_total = rec.get("n") or rec.get("samples") or rec.get("n_samples")
    n_synth = rec.get("n_synthetic") or rec.get("synthetic_samples")
    if isinstance(n_total, (int, float)) and isinstance(n_synth, (int, float)) and n_total > 0:
        return float(n_synth) / float(n_total)
    return None


def _collect(args: argparse.Namespace) -> Certificate:
    doc = json.loads(Path(args.results).read_text())
    rows = _iter_datasets(doc)
    flagged: list[dict[str, Any]] = []
    inspected: list[dict[str, Any]] = []
    unknown: list[str] = []
    for name, rec in rows:
        frac = _synthetic_fraction(rec)
        if frac is None:
            unknown.append(name)
            continue
        inspected.append({"dataset": name, "synthetic_fraction": frac})
        if frac > args.max_synthetic_fraction:
            flagged.append({"dataset": name, "synthetic_fraction": frac})

    body: dict[str, Any] = {
        "results_path": args.results,
        "threshold": args.max_synthetic_fraction,
        "row_count": len(rows),
        "inspected": inspected,
        "unknown_synthetic_attribution": unknown,
        "flagged": flagged,
    }
    warnings: list[str] = []
    for row in flagged:
        warnings.append(
            f"{row['dataset']}: synthetic_fraction={row['synthetic_fraction']:.2f} "
            f"exceeds {args.max_synthetic_fraction}"
        )
    if unknown:
        warnings.append(
            f"{len(unknown)} datasets had no synthetic-fraction field — "
            "post-hoc audit cannot vouch for those rows"
        )
    status = "fail" if flagged else ("warn" if unknown else "ok")
    return Certificate(
        tool="synthetic_fallback_auditor",
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
