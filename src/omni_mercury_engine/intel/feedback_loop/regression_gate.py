# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The OOF/adversarial regression gate: a merge-blocker on the candidate model.

Human verification authorizes a retrain; it does **not** vouch for the resulting
model. A candidate refit on ingested audit + red-team data can silently regress
calibration or -- worse, if the feedback was poisoned -- drop adversarial recall.
This gate is the merge-blocker that catches both: it fits a *candidate* model on
``base corpus + queue examples``, fits a *baseline* on the base corpus alone, and
**refuses the candidate** unless it stays within the Tier-0 regression margins on
out-of-fold calibration (ECE/Brier/AUROC) and held-out adversarial recall.

It deliberately reuses the shipped rolling-corpus evaluator
(``benchmarks/rolling_corpus_eval.py``) -- the same OOF folds, the same
adversarial holdout, the same :data:`MARGINS` the ``ci/rolling-corpus-eval``
Tier-0 lane enforces -- so a candidate can never pass a *weaker* bar than the
foundation already requires. A poisoned candidate (offensive examples mislabeled
benign) shifts the decision boundary, its adversarial recall falls past the
margin, and the gate blocks it: that is the measured value
(:data:`value_metrics.VALUE_METRICS['closed_feedback_loop']`).
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from omni_mercury_engine.intel.feedback_loop.labeling import LabeledExample

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _rolling_corpus_eval() -> Any:
    """Import the shipped rolling-corpus evaluator (single source of OOF truth).

    The evaluator lives under ``benchmarks/`` (with its ``calibration_brief``
    metric core); this puts both on ``sys.path`` once and returns the module, so
    the regression gate computes byte-identical OOF/adversarial numbers to the
    Tier-0 ``ci/rolling-corpus-eval`` lane.
    """
    for sub in ("benchmarks", "benchmarks/calibration_brief"):
        p = str(_REPO_ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import rolling_corpus_eval  # type: ignore[import-not-found]

    return rolling_corpus_eval


@dataclass(frozen=True)
class CandidateReport:
    """OOF + adversarial metrics for one fitted model."""

    label: str
    n_train: int
    oof_ece: float
    oof_brier: float
    oof_auroc: float
    adversarial_recall: float
    adversarial_fn_rate: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return {
            "label": self.label,
            "n_train": self.n_train,
            "oof_ece": self.oof_ece,
            "oof_brier": self.oof_brier,
            "oof_auroc": self.oof_auroc,
            "adversarial_recall": self.adversarial_recall,
            "adversarial_fn_rate": self.adversarial_fn_rate,
        }


@dataclass(frozen=True)
class RegressionVerdict:
    """The gate's decision on a candidate vs the baseline."""

    accepted: bool
    violations: tuple[str, ...]
    baseline: CandidateReport
    candidate: CandidateReport

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return {
            "accepted": self.accepted,
            "violations": list(self.violations),
            "baseline": self.baseline.as_dict(),
            "candidate": self.candidate.as_dict(),
        }


def _example_rows(examples: list[LabeledExample]) -> list[dict[str, Any]]:
    """Render labeled examples as corpus rows (``text``/``label``)."""
    return [{"text": e.text, "label": e.label} for e in examples]


def _report(label: str, rows: list[dict[str, Any]], rce: Any) -> CandidateReport:
    """Compute the OOF + adversarial report for a corpus ``rows`` set."""
    x, y = rce._features_and_labels(rows)
    texts = [r["text"] for r in rows]
    kfold = rce.kfold_oof(x, y, texts)
    adversarial = rce.adversarial_holdout(x, y)
    return CandidateReport(
        label=label,
        n_train=len(rows),
        oof_ece=float(kfold["ece"]),
        oof_brier=float(kfold["brier"]),
        oof_auroc=float(kfold["auroc"]),
        adversarial_recall=float(adversarial.get("recall", float("nan"))),
        adversarial_fn_rate=float(adversarial.get("fn_rate", float("nan"))),
    )


def load_base_corpus() -> list[dict[str, Any]]:
    """Load the authoritative weapons-gate corpus rows."""
    rce = _rolling_corpus_eval()
    rows: list[dict[str, Any]] = rce._load_jsonl(rce.CORPUS_PATH)
    if not rows:
        raise RuntimeError(f"empty base corpus at {rce.CORPUS_PATH}")
    return rows


def evaluate_candidate(
    examples: list[LabeledExample],
    *,
    base_rows: list[dict[str, Any]] | None = None,
) -> RegressionVerdict:
    """Fit baseline + candidate models and adjudicate the candidate.

    Args:
        examples: The human-verified labeled examples to fold into the candidate.
        base_rows: The base corpus rows (loaded from the authoritative corpus if
            omitted).

    Returns:
        A :class:`RegressionVerdict`. ``accepted`` is ``False`` when the candidate
        regresses OOF calibration or adversarial recall beyond the Tier-0
        :data:`MARGINS`, or when any metric is NaN (fail-closed).
    """
    rce = _rolling_corpus_eval()
    base = list(base_rows) if base_rows is not None else load_base_corpus()
    augmented = base + _example_rows(examples)

    baseline = _report("baseline", base, rce)
    candidate = _report("candidate", augmented, rce)
    verdict = gate_reports(baseline, candidate, margins=rce.MARGINS)
    return verdict


def gate_reports(
    baseline: CandidateReport,
    candidate: CandidateReport,
    *,
    margins: dict[str, float],
) -> RegressionVerdict:
    """Compare a candidate to a baseline under the Tier-0 regression margins.

    ``margins`` follows the rolling-corpus convention: a positive margin means
    "may rise by at most this" (ECE/Brier), a negative margin means "may fall by
    at most this" (AUROC/recall). A NaN candidate metric fails closed.
    """
    violations: list[str] = []
    checks = {
        "oof_ece": (baseline.oof_ece, candidate.oof_ece),
        "oof_brier": (baseline.oof_brier, candidate.oof_brier),
        "oof_auroc": (baseline.oof_auroc, candidate.oof_auroc),
        "adversarial_recall": (baseline.adversarial_recall, candidate.adversarial_recall),
    }
    for key, (base_val, cand_val) in checks.items():
        margin = margins.get(key)
        if margin is None:
            continue
        if math.isnan(cand_val) or math.isnan(base_val):
            violations.append(f"{key}: NaN metric (fail-closed)")
            continue
        delta = cand_val - base_val
        if margin > 0 and delta > margin:
            violations.append(
                f"{key} regressed: {cand_val:.4f} > baseline {base_val:.4f} + {margin}"
            )
        elif margin < 0 and delta < margin:
            violations.append(
                f"{key} regressed: {cand_val:.4f} < baseline {base_val:.4f} - {abs(margin)}"
            )
    return RegressionVerdict(
        accepted=not violations,
        violations=tuple(violations),
        baseline=baseline,
        candidate=candidate,
    )


def fit_candidate_weights(
    examples: list[LabeledExample], *, base_rows: list[dict[str, Any]] | None = None
) -> np.ndarray:
    """Fit and return the candidate logistic weights ``[bias, w0, w1, w2]``.

    This is the staged model artifact -- the deterministic full-corpus fit on
    ``base + examples`` (the same ``_fit_logistic`` the OOF evaluator uses).
    """
    rce = _rolling_corpus_eval()
    base = list(base_rows) if base_rows is not None else load_base_corpus()
    rows = base + _example_rows(examples)
    x, y = rce._features_and_labels(rows)
    weights: np.ndarray = rce._fit_logistic(x, y)
    return weights


__all__ = [
    "CandidateReport",
    "RegressionVerdict",
    "evaluate_candidate",
    "fit_candidate_weights",
    "gate_reports",
    "load_base_corpus",
]
