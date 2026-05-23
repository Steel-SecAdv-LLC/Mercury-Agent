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

Operator tool: human-readable diff between two benchmark JSON files.

Mercury's README auto-regenerates an AUC/F1 block from the canonical
benchmark output, but there is no general per-dataset / per-detector
diff between two arbitrary runs.  This tool fills that gap so an
operator can::

    python -m omni_mercury_engine.tools.benchmark_diff prev.json current.json

and immediately see which detectors regressed on which datasets, with
both the absolute and relative delta.

The expected JSON shape is loose by design so both
``benchmarks/baseline_results.json`` and any ad-hoc per-run dump can be
diffed.  We accept any of::

    {"detector_name": {"dataset_name": {"auc": 0.97, "f1": 0.83, ...}}}
    {"detector_name": {"dataset_name": 0.97}}
    {"results": [{"detector": ..., "dataset": ..., "auc": ..., "f1": ...}, ...]}

— and normalise to the first shape internally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.benchmark_diff/v1"
_METRIC_KEYS = ("auc", "f1", "precision", "recall", "ap", "auprc")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.benchmark_diff",
        description=(
            "Diff two benchmark JSON files and print a per-dataset / "
            "per-detector regression report."
        ),
    )
    parser.add_argument("previous", help="Path to the baseline benchmark JSON.")
    parser.add_argument("current", help="Path to the new benchmark JSON.")
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.005,
        help=(
            "Absolute drop in any metric below which a row is flagged as a "
            "regression (default 0.005)."
        ),
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when any regression is detected.",
    )
    return parser


def _normalise(doc: Any) -> dict[str, dict[str, dict[str, float]]]:
    """Coerce a benchmark JSON into ``{detector: {dataset: {metric: value}}}``."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    if isinstance(doc, dict) and "results" in doc and isinstance(doc["results"], list):
        for row in doc["results"]:
            det = str(row.get("detector", "unknown"))
            ds = str(row.get("dataset", "unknown"))
            metrics = {
                k: float(row[k])
                for k in _METRIC_KEYS
                if k in row and isinstance(row[k], (int, float))
            }
            if metrics:
                out.setdefault(det, {})[ds] = metrics
        return out
    if isinstance(doc, dict):
        for det, datasets in doc.items():
            if not isinstance(datasets, dict):
                continue
            for ds, val in datasets.items():
                if isinstance(val, dict):
                    metrics = {
                        k: float(val[k])
                        for k in _METRIC_KEYS
                        if k in val and isinstance(val[k], (int, float))
                    }
                    if metrics:
                        out.setdefault(det, {})[ds] = metrics
                elif isinstance(val, (int, float)):
                    out.setdefault(det, {})[ds] = {"auc": float(val)}
        return out
    raise ValueError(f"unsupported benchmark JSON shape: {type(doc).__name__}")


def _collect(args: argparse.Namespace) -> Certificate:
    prev_doc = json.loads(Path(args.previous).read_text())
    curr_doc = json.loads(Path(args.current).read_text())
    prev = _normalise(prev_doc)
    curr = _normalise(curr_doc)

    rows: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    additions: list[dict[str, Any]] = []
    removals: list[dict[str, Any]] = []

    all_detectors = set(prev) | set(curr)
    for det in sorted(all_detectors):
        p_datasets = prev.get(det, {})
        c_datasets = curr.get(det, {})
        for ds in sorted(set(p_datasets) | set(c_datasets)):
            p_m = p_datasets.get(ds)
            c_m = c_datasets.get(ds)
            if p_m is None:
                additions.append({"detector": det, "dataset": ds, "current": c_m})
                continue
            if c_m is None:
                removals.append({"detector": det, "dataset": ds, "previous": p_m})
                continue
            for metric in set(p_m) | set(c_m):
                if metric in p_m and metric in c_m:
                    delta = c_m[metric] - p_m[metric]
                    rel = delta / p_m[metric] if p_m[metric] else 0.0
                    row = {
                        "detector": det,
                        "dataset": ds,
                        "metric": metric,
                        "previous": p_m[metric],
                        "current": c_m[metric],
                        "delta": delta,
                        "relative_delta": rel,
                    }
                    rows.append(row)
                    if delta < -args.regression_threshold:
                        regressions.append(row)
                    elif delta > args.regression_threshold:
                        improvements.append(row)

    body: dict[str, Any] = {
        "previous": args.previous,
        "current": args.current,
        "regression_threshold": args.regression_threshold,
        "total_rows": len(rows),
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "added_rows": additions,
        "removed_rows": removals,
        "regressions": regressions,
        "improvements": improvements,
    }

    warnings: list[str] = []
    for r in regressions:
        warnings.append(
            f"{r['detector']}/{r['dataset']}.{r['metric']}: "
            f"{r['previous']:.4f} → {r['current']:.4f} (Δ={r['delta']:+.4f})"
        )

    if regressions and args.fail_on_regression:
        status = "fail"
    elif regressions:
        status = "warn"
    else:
        status = "ok"

    return Certificate(
        tool="benchmark_diff",
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
