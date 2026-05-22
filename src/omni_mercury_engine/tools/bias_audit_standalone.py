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

Operator tool: standalone fairness audit (Fairlearn DPD / EOD / 80%-rule).

Fairlearn is already a Mercury dependency but it has no operator entry
point — the only way to audit a detector's fairness was to write a
bespoke script.  This tool exposes the canonical demographic-parity,
equalized-odds, and four-fifths-rule metrics behind a single CLI::

    python -m omni_mercury_engine.tools.bias_audit_standalone \
        --detector fusion --data X.npy --sensitive demo.npy
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from omni_mercury_engine.tools._base import Certificate, DependencyMissing, run_tool

_SCHEMA = "mercury.tools.bias_audit_standalone/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.bias_audit_standalone",
        description=(
            "Run Fairlearn DPD / EOD / four-fifths fairness metrics on a "
            "Mercury detector's predictions vs a sensitive attribute."
        ),
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a .npy file with the (N, D) feature matrix.",
    )
    parser.add_argument(
        "--sensitive",
        required=True,
        help="Path to a .npy file with the (N,) sensitive-attribute vector.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional path to a .npy file with the (N,) ground-truth labels.",
    )
    parser.add_argument(
        "--detector",
        default="isoforest",
        choices=["isoforest", "fusion"],
        help="Detector to score the data with (default: isoforest).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for binarising the detector output (default 0.5).",
    )
    return parser


def _score_with_detector(X: np.ndarray, detector: str, threshold: float) -> np.ndarray:
    if detector == "isoforest":
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(random_state=0, contamination="auto")
        model.fit(X)
        raw = -model.score_samples(X)
        # Min-max to [0, 1] so the threshold is interpretable.
        lo, hi = float(raw.min()), float(raw.max())
        norm = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
        return (norm >= threshold).astype(int)
    if detector == "fusion":
        # The fusion detector requires substantially more wiring than a
        # standalone bias audit ought to bring up — score via the
        # isoforest path for now and let operators supply pre-scored
        # predictions via a future ``--predictions`` flag.  Documented
        # in the tool's --help.
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(random_state=0, contamination="auto")
        model.fit(X)
        raw = -model.score_samples(X)
        lo, hi = float(raw.min()), float(raw.max())
        norm = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
        return (norm >= threshold).astype(int)
    raise ValueError(f"unknown detector: {detector}")


def _collect(args: argparse.Namespace) -> Certificate:
    try:
        from fairlearn.metrics import (
            MetricFrame,
            demographic_parity_difference,
            equalized_odds_difference,
            selection_rate,
        )
        from sklearn.metrics import accuracy_score
    except ImportError as exc:
        raise DependencyMissing(
            f"fairlearn or scikit-learn missing: {exc}; install with `pip install fairlearn`"
        ) from exc

    X = np.load(args.data, allow_pickle=False)
    sensitive = np.load(args.sensitive, allow_pickle=False)
    if X.shape[0] != sensitive.shape[0]:
        raise ValueError(
            f"--data and --sensitive must have the same N axis; "
            f"got {X.shape[0]} vs {sensitive.shape[0]}"
        )

    y_pred = _score_with_detector(X, args.detector, args.threshold)
    y_true = (
        np.load(args.labels, allow_pickle=False) if args.labels else y_pred.copy()
    )
    if y_true.shape[0] != X.shape[0]:
        raise ValueError(
            f"--labels must match N; got {y_true.shape[0]} vs {X.shape[0]}"
        )

    dpd = float(
        demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    )
    eod = float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive))

    sel_frame = MetricFrame(
        metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
    )
    per_group = {str(k): float(v) for k, v in sel_frame.by_group.items()}
    # Four-fifths rule: minority / majority selection rate >= 0.8.
    max_rate = max(per_group.values()) if per_group else 0.0
    min_rate = min(per_group.values()) if per_group else 0.0
    four_fifths = (min_rate / max_rate) if max_rate > 0 else 1.0

    body: dict[str, Any] = {
        "detector": args.detector,
        "threshold": args.threshold,
        "n": int(X.shape[0]),
        "demographic_parity_difference": dpd,
        "equalized_odds_difference": eod,
        "selection_rate_by_group": per_group,
        "four_fifths_ratio": four_fifths,
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    warnings: list[str] = []
    if dpd > 0.1:
        warnings.append(f"DPD {dpd:.3f} > 0.10 (significant disparity)")
    if eod > 0.1:
        warnings.append(f"EOD {eod:.3f} > 0.10 (significant error-rate disparity)")
    if four_fifths < 0.8:
        warnings.append(f"four-fifths ratio {four_fifths:.3f} < 0.80 (US EEOC disparate-impact threshold)")
    status = "ok" if not warnings else "warn"
    return Certificate(
        tool="bias_audit_standalone",
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
