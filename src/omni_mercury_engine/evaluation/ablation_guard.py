# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reusable confound guard for paired ablations (WS-B follow-on).

During PR #262 two ablation *designs* produced a spurious **+0.48 "KEEP"** for
the differentiable domain encoder. Both were the same confound: on imbalanced
datasets a weak/degenerate arm collapsed to an **inverted ranking** (ROC-AUC
well below 0.5), so the paired delta measured the *other arm's collapse*, not the
treatment. Those designs were caught and rejected by hand. This module promotes
that catch into a **reusable, tested guard** so a confounded "improvement" cannot
be reported again -- by this repo's ablations or any future one.

The core invariant: a paired ROC-AUC comparison is only meaningful if **both
arms produce non-degenerate rankings**. An arm with AUC < ~0.5 has learned an
inverted (or random) ordering; any delta involving it is dominated by that
artifact. :func:`check_ablation_confound` reports such a comparison as
*confounded*, and :func:`confound_free_or_quarantine` turns a confounded KEEP
into a forced QUARANTINE.

This is symmetric-rigor tooling: it guards against unearned *optimism* (a fake
KEEP from a collapsed baseline) exactly as the conservative noise thresholds
guard against over-reading small positive deltas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# An arm whose ROC-AUC is below this has learned an inverted/degenerate ranking.
# 0.5 is chance; we allow a small band for finite-sample noise around chance and
# only flag clearly-inverted arms (the confound was AUC ~0.05-0.45, not ~0.49).
INVERSION_FLOOR = 0.45


@dataclass
class ConfoundReport:
    """Result of auditing a paired ablation for the inverted-ranking confound."""

    confounded: bool
    reasons: list[str] = field(default_factory=list)
    n_degenerate_baseline: int = 0
    n_degenerate_treatment: int = 0
    n_pairs: int = 0
    inversion_floor: float = INVERSION_FLOOR

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "confounded": self.confounded,
            "reasons": self.reasons,
            "n_degenerate_baseline": self.n_degenerate_baseline,
            "n_degenerate_treatment": self.n_degenerate_treatment,
            "n_pairs": self.n_pairs,
            "inversion_floor": self.inversion_floor,
        }


def check_ablation_confound(
    baseline_aucs: Sequence[float],
    treatment_aucs: Sequence[float],
    *,
    inversion_floor: float = INVERSION_FLOOR,
    max_degenerate_fraction: float = 0.0,
) -> ConfoundReport:
    """Audit a set of paired per-seed AUCs for the inverted-ranking confound.

    Args:
        baseline_aucs / treatment_aucs: paired per-seed test ROC-AUCs (same
            length, same order).
        inversion_floor: AUC at/below which an arm is treated as
            inverted/degenerate (default :data:`INVERSION_FLOOR`).
        max_degenerate_fraction: the largest fraction of seeds an arm may have
            below the floor before the comparison is declared confounded. The
            default ``0.0`` means *any* inverted seed confounds the comparison
            (the strict default appropriate for KEEP decisions).

    Returns:
        A :class:`ConfoundReport`. ``confounded=True`` means the paired delta is
        not trustworthy and must not be read as evidence for the treatment.
    """
    n = min(len(baseline_aucs), len(treatment_aucs))
    if n == 0:
        return ConfoundReport(
            confounded=True,
            reasons=["no paired AUCs to compare"],
            inversion_floor=inversion_floor,
        )

    base_bad = [i for i in range(n) if baseline_aucs[i] < inversion_floor]
    treat_bad = [i for i in range(n) if treatment_aucs[i] < inversion_floor]
    reasons: list[str] = []
    if len(base_bad) / n > max_degenerate_fraction:
        reasons.append(
            f"baseline arm inverted (AUC<{inversion_floor}) on "
            f"{len(base_bad)}/{n} seeds {[round(baseline_aucs[i], 3) for i in base_bad]} "
            "-- a paired delta measures this collapse, not the treatment"
        )
    if len(treat_bad) / n > max_degenerate_fraction:
        reasons.append(
            f"treatment arm inverted (AUC<{inversion_floor}) on "
            f"{len(treat_bad)}/{n} seeds {[round(treatment_aucs[i], 3) for i in treat_bad]}"
        )
    return ConfoundReport(
        confounded=bool(reasons),
        reasons=reasons,
        n_degenerate_baseline=len(base_bad),
        n_degenerate_treatment=len(treat_bad),
        n_pairs=n,
        inversion_floor=inversion_floor,
    )


def confound_free_or_quarantine(
    cleared: bool,
    report: ConfoundReport,
) -> tuple[bool, str]:
    """Combine a raw verdict with the confound report.

    A KEEP (``cleared=True``) on a confounded comparison is downgraded to a
    forced QUARANTINE: an improvement built on a collapsed arm is not real.

    Returns ``(final_cleared, note)``.
    """
    if report.confounded:
        return False, (
            "FORCED QUARANTINE -- comparison is confounded: " + "; ".join(report.reasons)
        )
    if cleared:
        return True, "KEEP -- confound-free and clears the bar"
    return False, "QUARANTINE -- confound-free but does not clear the bar"


__all__ = [
    "INVERSION_FLOOR",
    "ConfoundReport",
    "check_ablation_confound",
    "confound_free_or_quarantine",
]
