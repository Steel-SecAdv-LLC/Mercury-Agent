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

Source-agnostic calibrated-confidence input for the decision layer.

The decision layer must not care *which* calibration produced a number, only
that the number is honest. :class:`ConfidenceSignal` is the normalised carrier:
a calibrated ``P(anomaly)`` plus, when available, the conformal label set whose
``set_size`` already names the three outcomes that matter (confident singleton /
uncertain two-label / atypical empty set).

The adapters here read the engine surfaces that exist **today on main** --
``OmniMercuryEngine.detect_with_fusion`` (``anomaly_prob`` + optional
``conformal`` sub-dict) and ``score_fusion_conformal`` -- and are
**forward-compatible** with PR #278's richer surface: when a result carries the
additive ``calibrated_probabilities`` key (Beta-MCA), it is preferred over the
raw temperature-scaled point automatically, with no code change here required.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ConfidenceSignal",
    "ConfidenceSource",
    "confidence_batch_from_conformal_scores",
    "confidence_from_conformal",
    "confidence_from_engine_result",
]

# Float fuzz tolerance: probabilities just outside ``[0, 1]`` from rounding are
# snapped in; anything further out is a real error and is rejected.
_PROB_TOLERANCE = 1e-9


class ConfidenceSource(Enum):
    """Which calibrated surface produced a :class:`ConfidenceSignal`."""

    #: A conformal label set drove the signal (carries a coverage guarantee).
    CONFORMAL = "conformal"

    #: A calibrated point probability (temperature scaling, or Beta-MCA when
    #: PR #278's ``calibrated_probabilities`` is present).
    CALIBRATED_PROBABILITY = "calibrated_probability"

    #: A reconciled operating point (forward-compatible hook for #278's
    #: ``reconciled_operating_point``); falls back cleanly when absent.
    RECONCILED = "reconciled"


@dataclass(frozen=True)
class ConfidenceSignal:
    """Normalised, source-agnostic calibrated confidence for one sample.

    Attributes:
        anomaly_probability: Calibrated ``P(anomaly)`` in ``[0, 1]``.
        prediction_set: Conformal label set over ``{0, 1}`` when available
            (``(1,)`` confident anomaly, ``(0,)`` confident normal, ``(0, 1)``
            uncertain, ``()`` atypical/novel), else ``None`` for a point-only
            signal.
        coverage: Target conformal coverage when ``prediction_set`` is present.
        source: The :class:`ConfidenceSource` that produced this signal.
        provenance: JSON-friendly structured context (which key was read, etc.).

    Raises:
        ValueError: If ``anomaly_probability`` is outside ``[0, 1]`` (beyond
            float fuzz), or ``prediction_set`` contains labels outside ``{0, 1}``
            or has duplicates.
    """

    anomaly_probability: float
    prediction_set: tuple[int, ...] | None = None
    coverage: float | None = None
    source: ConfidenceSource = ConfidenceSource.CALIBRATED_PROBABILITY
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        prob = float(self.anomaly_probability)
        if prob < -_PROB_TOLERANCE or prob > 1.0 + _PROB_TOLERANCE:
            raise ValueError(
                f"anomaly_probability must be in [0, 1], got {self.anomaly_probability!r}"
            )
        clamped = min(1.0, max(0.0, prob))
        # Frozen dataclass: normalise the stored value in place.
        object.__setattr__(self, "anomaly_probability", clamped)

        if self.prediction_set is not None:
            labels = tuple(int(x) for x in self.prediction_set)
            if any(lbl not in (0, 1) for lbl in labels):
                raise ValueError(
                    f"prediction_set labels must be in {{0, 1}}, got {self.prediction_set!r}"
                )
            if len(set(labels)) != len(labels):
                raise ValueError(f"prediction_set must not repeat labels: {self.prediction_set!r}")
            object.__setattr__(self, "prediction_set", tuple(sorted(labels)))

    @property
    def has_conformal(self) -> bool:
        """Whether a conformal label set is available to drive the decision."""
        return self.prediction_set is not None

    @property
    def is_novel(self) -> bool:
        """Whether an empty conformal set flagged an atypical (novel/OOD) point."""
        return self.prediction_set == ()

    def as_metadata(self) -> dict[str, object]:
        """Return a JSON-friendly mapping describing this signal."""
        return {
            "anomaly_probability": self.anomaly_probability,
            "prediction_set": (
                list(self.prediction_set) if self.prediction_set is not None else None
            ),
            "coverage": self.coverage,
            "source": self.source.value,
            **self.provenance,
        }


def _coerce_prediction_set(raw: Any) -> tuple[int, ...]:
    """Coerce a label-set-like value (list/tuple/ndarray) to a tuple of ints."""
    return tuple(int(x) for x in raw)


def confidence_from_engine_result(result: Mapping[str, Any]) -> ConfidenceSignal:
    """Normalise a single-sample engine result dict into a :class:`ConfidenceSignal`.

    Accepts the ``OmniMercuryEngine.detect_with_fusion`` shape on ``main``
    (``anomaly_prob`` plus an optional ``conformal`` sub-dict). Forward-compatible
    with PR #278: when ``calibrated_probabilities`` is present it is preferred
    over the raw ``anomaly_prob`` point, and a ``reconciled_operating_point``
    mapping carrying a ``"probability"`` is honoured if present.

    Args:
        result: A single-sample detection result mapping.

    Returns:
        The normalised :class:`ConfidenceSignal`.

    Raises:
        KeyError: If no calibrated probability field can be found.
    """
    provenance: dict[str, object] = {}
    source = ConfidenceSource.CALIBRATED_PROBABILITY

    probability: float | None = None
    reconciled = result.get("reconciled_operating_point")
    if isinstance(reconciled, Mapping) and "probability" in reconciled:
        probability = float(reconciled["probability"])
        source = ConfidenceSource.RECONCILED
        provenance["probability_source"] = "reconciled_operating_point"
    elif "calibrated_probabilities" in result:
        calibrated = result["calibrated_probabilities"]
        if isinstance(calibrated, Sequence) and len(calibrated) > 0:
            probability = float(calibrated[0])
            provenance["probability_source"] = "calibrated_probabilities"
    if probability is None:
        if "anomaly_prob" not in result:
            raise KeyError(
                "engine result has no calibrated probability "
                "('anomaly_prob' / 'calibrated_probabilities' / "
                "'reconciled_operating_point') to decide from"
            )
        probability = float(result["anomaly_prob"])
        provenance.setdefault("probability_source", "anomaly_prob")

    prediction_set: tuple[int, ...] | None = None
    coverage: float | None = None
    conformal = result.get("conformal")
    if isinstance(conformal, Mapping) and conformal.get("prediction_set") is not None:
        prediction_set = _coerce_prediction_set(conformal["prediction_set"])
        cov = conformal.get("coverage")
        coverage = float(cov) if cov is not None else None
        source = ConfidenceSource.CONFORMAL
        provenance["conformal_set_size"] = int(conformal.get("set_size", len(prediction_set)))

    return ConfidenceSignal(
        anomaly_probability=probability,
        prediction_set=prediction_set,
        coverage=coverage,
        source=source,
        provenance=provenance,
    )


def confidence_from_conformal(
    probability: float,
    prediction_set: Sequence[int],
    coverage: float,
    *,
    provenance: dict[str, object] | None = None,
) -> ConfidenceSignal:
    """Build a conformal-sourced signal from an explicit probability and label set.

    Args:
        probability: Calibrated ``P(anomaly)``.
        prediction_set: Conformal label set over ``{0, 1}``.
        coverage: Target conformal coverage the set guarantees.
        provenance: Optional structured context.

    Returns:
        A :class:`ConfidenceSignal` tagged :attr:`ConfidenceSource.CONFORMAL`.
    """
    return ConfidenceSignal(
        anomaly_probability=float(probability),
        prediction_set=_coerce_prediction_set(prediction_set),
        coverage=float(coverage),
        source=ConfidenceSource.CONFORMAL,
        provenance=provenance or {},
    )


def confidence_batch_from_conformal_scores(out: Mapping[str, Any]) -> list[ConfidenceSignal]:
    """Normalise a ``score_fusion_conformal`` batch into per-sample signals.

    Args:
        out: The mapping returned by ``OmniMercuryEngine.score_fusion_conformal``
            (keys ``probabilities``, ``prediction_sets``, ``coverage``).

    Returns:
        One :class:`ConfidenceSignal` per sample, conformal-sourced.

    Raises:
        KeyError: If the required batch keys are absent.
    """
    probabilities = out["probabilities"]
    prediction_sets = out["prediction_sets"]
    coverage = float(out["coverage"])
    signals: list[ConfidenceSignal] = []
    for index, (prob, pset) in enumerate(zip(probabilities, prediction_sets)):
        signals.append(
            confidence_from_conformal(
                float(prob),
                _coerce_prediction_set(pset),
                coverage,
                provenance={"batch_index": index},
            )
        )
    return signals
