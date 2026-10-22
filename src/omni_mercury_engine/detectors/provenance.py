# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decision provenance for detector alarm rules.

Every detector decision rule (a published threshold, a calibrated operating
point, a trained model's alert bar) must be traceable to its source.  A
:class:`DecisionSource` records where the rule comes from, and the
enforcement test (``tests/core/test_decision_source.py``) fails when a
detector class reachable from ``DETECTOR_MANIFEST`` declares none.

The three evidentiary kinds, in descending order of external authority:

``published``
    The rule is a value taken from an authoritative publication (a
    peer-reviewed paper, an operational agency specification, a standard).
    ``citation`` is mandatory and must name the document precisely enough
    for a reviewer to check the number.

``calibrated``
    The rule is an operating point calibrated on identified data, carrying
    the false-alarm rate measured at that point.  ``false_alarm_rate`` and
    ``calibration_data`` are mandatory.

``trained``
    The rule is a trained model's operating point, selected by a recorded
    procedure.  ``operating_point`` is mandatory and names the checkpoint or
    the selection record.

``experimental``
    No published threshold exists and no labelled corpus is available yet to
    calibrate one.  The description must say so plainly; downstream surfaces
    may present such verdicts only as experimental.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Valid values for :attr:`DecisionSource.kind`.
DECISION_SOURCE_KINDS: frozenset[str] = frozenset(
    {"published", "calibrated", "trained", "experimental"}
)


class WrappedDecisionSource:
    """Lazy class attribute serving the wrapped detector's provenance.

    Adapter entries in the manifest wrap a domain detector whose module is
    deliberately imported lazily (inside methods) to keep adapter import
    cheap.  Declaring ``decision_source = WrappedDecisionSource("pkg.mod:Cls")``
    on the adapter keeps that laziness while single-sourcing provenance on
    the domain class: reading the attribute imports the target and returns
    its ``decision_source``.
    """

    def __init__(self, target: str) -> None:
        """Record the ``module.path:ClassName`` target to delegate to."""
        if ":" not in target:
            raise ValueError(f"target must be 'module.path:ClassName', got {target!r}")
        self._target = target

    def __get__(self, obj: object, objtype: type | None = None) -> object:
        """Resolve the wrapped class and return its decision_source."""
        import importlib

        module_path, class_name = self._target.split(":", 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        source = getattr(cls, "decision_source", None)
        if source is None:
            raise AttributeError(f"{self._target} declares no decision_source to delegate to")
        return source

    def __repr__(self) -> str:
        """Show the delegation target."""
        return f"WrappedDecisionSource({self._target!r})"


@dataclass(frozen=True)
class DecisionSource:
    """Provenance record for one detector decision rule.

    Attributes:
        kind: One of ``published``, ``calibrated``, ``trained``,
            ``experimental`` (see module docstring).
        description: What the rule decides and how the source backs it.
        citation: The authoritative document (mandatory for ``published``).
        false_alarm_rate: Measured false-alarm rate at the calibrated
            operating point (mandatory for ``calibrated``).
        calibration_data: The identified dataset the operating point was
            calibrated on (mandatory for ``calibrated``).
        operating_point: The checkpoint or selection record that fixes a
            trained rule's alert bar (mandatory for ``trained``).
    """

    kind: str
    description: str
    citation: str | None = None
    false_alarm_rate: float | None = None
    calibration_data: str | None = None
    operating_point: str | None = None

    def __post_init__(self) -> None:
        """Refuse records that do not carry their kind's mandatory evidence."""
        if self.kind not in DECISION_SOURCE_KINDS:
            raise ValueError(
                f"DecisionSource.kind must be one of {sorted(DECISION_SOURCE_KINDS)}, "
                f"got {self.kind!r}"
            )
        if not self.description or not self.description.strip():
            raise ValueError("DecisionSource.description must be non-empty")
        if self.kind == "published" and not (self.citation and self.citation.strip()):
            raise ValueError("a 'published' DecisionSource must carry a citation")
        if self.kind == "calibrated":
            if self.false_alarm_rate is None or not 0.0 <= self.false_alarm_rate <= 1.0:
                raise ValueError(
                    "a 'calibrated' DecisionSource must carry its measured "
                    f"false_alarm_rate in [0, 1], got {self.false_alarm_rate!r}"
                )
            if not (self.calibration_data and self.calibration_data.strip()):
                raise ValueError("a 'calibrated' DecisionSource must name its calibration_data")
        if self.kind == "trained" and not (self.operating_point and self.operating_point.strip()):
            raise ValueError(
                "a 'trained' DecisionSource must name its operating_point "
                "(checkpoint or selection record)"
            )
