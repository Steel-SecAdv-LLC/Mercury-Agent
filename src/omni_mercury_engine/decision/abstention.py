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

The abstention policy: a first-class "don't-know" gate over calibrated confidence.

This is the decision half of pillar *a*. It turns a :class:`ConfidenceSignal`
into a typed :class:`Decision`, and -- crucially -- can return
:attr:`Verdict.ABSTAIN` when the calibrated evidence does not commit. That
closes the long-standing gap where the engine's ``is_anomaly`` was always a bare
``anomaly_prob > threshold`` (two outcomes, no honest deferral).

Two decision paths, both deterministic and provenance-complete:

* **Conformal path (preferred).** When a conformal label set is present its
  ``set_size`` *already* encodes the three outcomes with a distribution-free
  coverage guarantee: a singleton is a confident call, ``{normal, anomaly}`` is
  genuine uncertainty (abstain), and the empty set is an atypical point neither
  class explains (abstain + novelty). The policy simply *honours* that set rather
  than overriding it with a point threshold.
* **Calibrated-band fallback.** With only a calibrated point probability, the
  policy abstains inside an explicit ``(negative, positive)`` indecision band and
  commits outside it.

Every abstention maps to :attr:`ThreeState.UNAVAILABLE` -- "decidable in
principle, not produced this run" -- and records *what would decide it*, so the
honesty is actionable, not a dead end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from omni_mercury_engine.decision.types import Decision, Verdict, verdict_to_three_state

if TYPE_CHECKING:
    from omni_mercury_engine.decision.confidence import ConfidenceSignal

__all__ = [
    "POLICY_VERSION",
    "AbstentionPolicy",
    "AbstentionThresholds",
]

#: Stable identifier stamped onto every decision for audit reproducibility.
POLICY_VERSION = "abstention/v1"


@dataclass(frozen=True)
class AbstentionThresholds:
    """Operating points for the calibrated-band fallback (no conformal set).

    The band ``(negative, positive)`` is the explicit indecision region: a
    calibrated ``P(anomaly)`` strictly inside it yields :attr:`Verdict.ABSTAIN`.
    Defaults are deliberately conservative and symmetric about ``0.5``; a
    deployment may widen the band (more abstentions, higher precision on the
    commits) or tune it asymmetrically against a domain's error costs.

    Attributes:
        positive: ``P(anomaly) >= positive`` commits to :attr:`Verdict.POSITIVE`.
        negative: ``P(anomaly) <= negative`` commits to :attr:`Verdict.NEGATIVE`.

    Raises:
        ValueError: If the thresholds are not ``0 <= negative <= positive <= 1``.
    """

    positive: float = 0.70
    negative: float = 0.30

    def __post_init__(self) -> None:
        if not 0.0 <= self.negative <= self.positive <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= negative <= positive <= 1, got "
                f"negative={self.negative}, positive={self.positive}"
            )

    @property
    def band_width(self) -> float:
        """Width of the abstention band ``positive - negative``."""
        return self.positive - self.negative


class AbstentionPolicy:
    """Map calibrated confidence to a typed decision with an honest abstention.

    The policy is **pure and deterministic**: the same :class:`ConfidenceSignal`
    always yields the same :class:`Decision`, which is what makes the audit trail
    verifiable. No randomness, no hidden state.

    Args:
        thresholds: Band operating points for the point-probability fallback.
        require_conformal: When ``True`` and a signal carries no conformal set,
            the policy abstains rather than fall back to the band -- the strict,
            coverage-guaranteed posture for high-stakes domains.
    """

    def __init__(
        self,
        thresholds: AbstentionThresholds | None = None,
        *,
        require_conformal: bool = False,
    ) -> None:
        self.thresholds = thresholds or AbstentionThresholds()
        self.require_conformal = require_conformal

    def decide(self, signal: ConfidenceSignal) -> Decision:
        """Decide a single signal, returning a grounded call or an honest abstention.

        Args:
            signal: The normalised calibrated-confidence input.

        Returns:
            A :class:`Decision`; :attr:`Decision.abstained` is ``True`` when the
            don't-know gate fired.
        """
        if signal.has_conformal:
            return self._decide_conformal(signal)
        if self.require_conformal:
            return self._abstain(
                signal,
                reason=(
                    "no conformal prediction set available and require_conformal=True; "
                    "abstaining rather than committing on an unguaranteed point estimate"
                ),
                would_decide="calibrate a conformal classifier (calibrate_fusion_conformal)",
            )
        return self._decide_band(signal)

    def decide_batch(self, signals: list[ConfidenceSignal]) -> list[Decision]:
        """Decide a list of signals (convenience wrapper over :meth:`decide`)."""
        return [self.decide(signal) for signal in signals]

    # ------------------------------------------------------------------
    # Conformal path: honour the set the coverage guarantee already produced.
    # ------------------------------------------------------------------
    def _decide_conformal(self, signal: ConfidenceSignal) -> Decision:
        pset = signal.prediction_set
        assert pset is not None  # guarded by caller via has_conformal
        prob = signal.anomaly_probability
        coverage = signal.coverage
        decisiveness = abs(2.0 * prob - 1.0)

        if pset == (1,):
            return self._commit(
                signal,
                verdict=Verdict.POSITIVE,
                margin=decisiveness,
                reason=(
                    f"conformal singleton {{anomaly}} at coverage "
                    f"{_fmt(coverage)}: normal ruled out"
                ),
            )
        if pset == (0,):
            return self._commit(
                signal,
                verdict=Verdict.NEGATIVE,
                margin=decisiveness,
                reason=(
                    f"conformal singleton {{normal}} at coverage "
                    f"{_fmt(coverage)}: anomaly ruled out"
                ),
            )
        if pset == ():
            return self._abstain(
                signal,
                reason=(
                    f"empty conformal set at coverage {_fmt(coverage)}: atypical point "
                    "neither class explains (novel / out-of-distribution)"
                ),
                would_decide="extend the calibration set to cover this region of input space",
                novelty=True,
            )
        # pset == (0, 1)
        return self._abstain(
            signal,
            reason=(
                f"conformal set {{normal, anomaly}} at coverage {_fmt(coverage)}: "
                "both labels admissible -- genuine uncertainty"
            ),
            would_decide="more calibration data or a lower target coverage would sharpen the set",
        )

    # ------------------------------------------------------------------
    # Band fallback: explicit indecision region on the calibrated point.
    # ------------------------------------------------------------------
    def _decide_band(self, signal: ConfidenceSignal) -> Decision:
        prob = signal.anomaly_probability
        positive = self.thresholds.positive
        negative = self.thresholds.negative

        if prob >= positive:
            denom = max(1.0 - positive, 1e-12)
            margin = min(1.0, (prob - positive) / denom)
            return self._commit(
                signal,
                verdict=Verdict.POSITIVE,
                margin=margin,
                reason=(f"calibrated P(anomaly)={prob:.3f} >= positive threshold {positive:.3f}"),
            )
        if prob <= negative:
            denom = max(negative, 1e-12)
            margin = min(1.0, (negative - prob) / denom)
            return self._commit(
                signal,
                verdict=Verdict.NEGATIVE,
                margin=margin,
                reason=(f"calibrated P(anomaly)={prob:.3f} <= negative threshold {negative:.3f}"),
            )
        return self._abstain(
            signal,
            reason=(
                f"calibrated P(anomaly)={prob:.3f} in abstention band "
                f"({negative:.3f}, {positive:.3f})"
            ),
            would_decide="a sharper calibrated probability (or a conformal set) would commit",
        )

    # ------------------------------------------------------------------
    # Decision constructors -- single place that stamps provenance + state.
    # ------------------------------------------------------------------
    def _commit(
        self,
        signal: ConfidenceSignal,
        *,
        verdict: Verdict,
        margin: float,
        reason: str,
    ) -> Decision:
        return Decision(
            verdict=verdict,
            state=verdict_to_three_state(verdict),
            confidence=signal.anomaly_probability,
            margin=float(margin),
            reason=reason,
            policy=POLICY_VERSION,
            prediction_set=signal.prediction_set,
            coverage=signal.coverage,
            novelty=signal.is_novel,
            provenance=self._provenance(signal),
        )

    def _abstain(
        self,
        signal: ConfidenceSignal,
        *,
        reason: str,
        would_decide: str,
        novelty: bool = False,
    ) -> Decision:
        provenance = self._provenance(signal)
        provenance["would_decide"] = would_decide
        return Decision(
            verdict=Verdict.ABSTAIN,
            state=verdict_to_three_state(Verdict.ABSTAIN),
            confidence=signal.anomaly_probability,
            margin=0.0,
            reason=reason,
            policy=POLICY_VERSION,
            prediction_set=signal.prediction_set,
            coverage=signal.coverage,
            novelty=novelty or signal.is_novel,
            provenance=provenance,
        )

    def _provenance(self, signal: ConfidenceSignal) -> dict[str, object]:
        return {
            "confidence_source": signal.source.value,
            "thresholds": {
                "positive": self.thresholds.positive,
                "negative": self.thresholds.negative,
            },
            "require_conformal": self.require_conformal,
            "signal": signal.as_metadata(),
        }


def _fmt(coverage: float | None) -> str:
    """Format an optional coverage level for human-readable reasons."""
    return "unknown" if coverage is None else f"{coverage:.2f}"
