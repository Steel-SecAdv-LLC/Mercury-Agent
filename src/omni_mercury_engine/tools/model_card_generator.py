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

Operator tool: Google-style model card generator.

Given a fitted detector (provided via importable module:class form),
emits a model card in JSON + Markdown covering training data,
performance metrics, fairness audit, limitations, and intended use.
Mercury's ethical-AI governance claim makes model cards the de facto
evidence format.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.tools._base import Certificate, atomic_write_text, run_tool

_SCHEMA = "mercury.tools.model_card_generator/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.model_card_generator",
        description=(
            "Generate a Google-style model card for a Mercury detector. "
            "The card is emitted as JSON in the certificate body and "
            "(when --markdown is supplied) as a Markdown file on disk."
        ),
    )
    parser.add_argument(
        "--detector",
        required=True,
        help="Detector identifier (module.path:ClassName).",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional .npy file with the (N, D) feature matrix used to compute metrics.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional .npy file with (N,) ground-truth labels.",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Path to write a human-readable Markdown model card.",
    )
    parser.add_argument(
        "--intended-use",
        default="Anomaly detection within Mercury Agent ethical safeguards.",
        help="Free-text intended use statement.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the card but do NOT write --markdown.",
    )
    parser.add_argument(
        "--evidence-dir",
        default=None,
        help=(
            "Optional directory containing sibling certificates (benevolence_calibration_report, "
            "fairness_subgroup_explorer, adversarial_probe).  When present they are spliced "
            "into the card's metrics / fairness sections as signed evidence."
        ),
    )
    parser.add_argument(
        "--limitations",
        default=(
            "Mercury's σ_Immutable and benevolence gates remain authoritative; "
            "this detector's scores are inputs, not decisions."
        ),
        help="Free-text limitations statement.",
    )
    return parser


def _load_class(identifier: str) -> Any:
    if ":" not in identifier:
        raise ValueError(f"--detector must be module.path:ClassName, got {identifier!r}")
    module_path, class_name = identifier.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _compute_metrics(
    detector_obj: Any, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64] | None
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    score_fn = (
        getattr(detector_obj, "score_samples", None)
        or getattr(detector_obj, "decision_function", None)
        or getattr(detector_obj, "predict", None)
    )
    if score_fn is None:
        out["error"] = "detector exposes neither score_samples, decision_function, nor predict"
        return out
    try:
        raw = np.asarray(score_fn(X), dtype=np.float64).ravel()
    except Exception as exc:
        out["error"] = f"scoring failed: {type(exc).__name__}: {exc}"
        return out

    out["score_summary"] = {
        "n": int(raw.size),
        "min": float(raw.min()),
        "max": float(raw.max()),
        "mean": float(raw.mean()),
        "std": float(raw.std()),
    }
    if y is not None:
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_recall_fscore_support,
            roc_auc_score,
        )

        try:
            out["roc_auc"] = float(roc_auc_score(y, raw))
            out["average_precision"] = float(average_precision_score(y, raw))
        except ValueError as exc:
            out["roc_auc_error"] = str(exc)
        thresh = float(np.median(raw))
        y_pred = (raw >= thresh).astype(int)
        out["threshold_used"] = thresh
        out["f1"] = float(f1_score(y, y_pred, zero_division=0))
        p, r, f, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
        out["precision"] = float(p)
        out["recall"] = float(r)
    return out


def _card_to_markdown(card: dict[str, Any]) -> str:
    lines = [f"# Model Card — {card['detector']}", ""]
    lines += ["## Model details", ""]
    for k, v in card["model_details"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Intended use", "", card["intended_use"], ""]
    lines += ["## Limitations", "", card["limitations"], ""]
    lines += ["## Metrics", "", "```json", json.dumps(card["metrics"], indent=2), "```", ""]
    if card.get("fairness"):
        lines += ["## Fairness", "", "```json", json.dumps(card["fairness"], indent=2), "```", ""]
    lines += ["## Provenance", ""]
    for k, v in card["provenance"].items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) + "\n"


def _collect(args: argparse.Namespace) -> Certificate:
    cls = _load_class(args.detector)
    detector_obj: Any
    try:
        detector_obj = cls()
    except TypeError:
        # Some detectors require explicit args — fail gracefully.
        return Certificate(
            tool="model_card_generator",
            schema=_SCHEMA,
            status="warn",
            body={"detector": args.detector, "error": "detector class requires constructor args"},
            warnings=[
                "instantiate the detector outside this tool and use the future --pickle flag"
            ],
        )

    X: npt.NDArray[np.float64] | None = None
    y: npt.NDArray[np.float64] | None = None
    if args.data:
        X = np.load(args.data, allow_pickle=False)
        if hasattr(detector_obj, "fit"):
            try:
                detector_obj.fit(X)
            except Exception:
                pass
    if args.labels:
        y = np.load(args.labels, allow_pickle=False)

    card: dict[str, Any] = {
        "detector": args.detector,
        "model_details": {
            "class_name": cls.__name__,
            "module": cls.__module__,
            "docstring": (cls.__doc__ or "").strip().splitlines()[:5],
        },
        "intended_use": args.intended_use,
        "limitations": args.limitations,
        "metrics": _compute_metrics(detector_obj, X, y) if X is not None else {},
        "fairness": {},
        "provenance": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    if args.evidence_dir:
        ev_dir = Path(args.evidence_dir)
        if ev_dir.is_dir():
            evidence: dict[str, Any] = {}
            for sibling in sorted(ev_dir.glob("*.json")):
                try:
                    blob = json.loads(sibling.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                schema = blob.get("schema", "")
                if not schema.startswith("mercury.tools."):
                    continue
                tool_name = schema.removeprefix("mercury.tools.").split("/")[0]
                if tool_name in {
                    "benevolence_calibration_report",
                    "fairness_subgroup_explorer",
                    "adversarial_probe",
                    "oae_dimensionality_probe",
                }:
                    evidence[tool_name] = {
                        "status": blob.get("status"),
                        "body": blob.get("body"),
                        "path": str(sibling),
                    }
            if evidence:
                card["evidence"] = evidence

    if args.markdown and not args.dry_run:
        atomic_write_text(Path(args.markdown), _card_to_markdown(card))

    return Certificate(
        tool="model_card_generator",
        schema=_SCHEMA,
        status="ok",
        body={"detector": args.detector, "markdown_path": args.markdown, "card": card},
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
