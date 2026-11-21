# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What proves a number was measured.

A result that says "measured" with a bare boolean is a claim nobody can check.
Mercury carried several: ``epistemic_measured``, ``aleatoric_measured`` and
``measured`` flags that a constructor could set, and that *defaulted to True*,
so a result built without running any estimator reported its placeholder
``0.0`` as a measurement.

Here a quantity is measured only when it carries the artifact that proves it:

* **how** it was measured (a named method from :data:`METHODS`);
* **on how many** observations (draws, ensemble members, residuals, a
  calibration set, a reference set) — never fewer than the method needs;
* **against what**: the SHA-256 of the exact numbers it was measured on, so
  the claim is bound to content and a re-run can be compared with it;
* and, where the quantity is a verdict against a cut, the **statistic** and
  the **threshold** it was cut at.

``measured`` is then *derived* — ``evidence is not None`` — and can no longer
be declared.  A :class:`MeasurementEvidence` cannot be built empty or
malformed (it raises), and a record that claims a measurement whose evidence
does not parse reads as unmeasured (:meth:`MeasurementEvidence.from_dict`
returns ``None``).

The digest is over IEEE-754 little-endian doubles with each array's shape, so
it is identical on every platform: see :func:`digest_of`.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["METHODS", "MeasurementEvidence", "digest_of"]

#: Every way a quantity may be measured, with the fewest observations that
#: make it one. A variance needs two; the others carry the floor of the
#: estimator that produces them.
METHODS: dict[str, int] = {
    # Epistemic: spread across stochastic forward passes / draws / members.
    "mc_dropout": 2,
    "mc_sampling": 2,
    "ensemble_variance": 2,
    # Aleatoric: spread of the drawn samples, or an accumulated residual history.
    "sample_variance": 2,
    "heteroscedastic_residuals": 2,
    # Confidence mapped through a calibrator fitted on observed outcomes.
    "fitted_calibrator": 2,
    # Expected calibration error over observed (confidence, outcome) pairs.
    "expected_calibration_error": 20,
    # Novelty: the rank of a standardized distance among reference distances.
    # Twenty is the fewest that can resolve a p-value of 0.05.
    "conformal_rank": 20,
}

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def digest_of(*arrays: Any) -> str:
    """SHA-256 over the exact numbers a quantity was measured on.

    Each argument is taken as float64, little-endian, C-contiguous, and is
    preceded by its shape, so two stacks with the same numbers in a different
    arrangement do not collide and the digest does not depend on the machine.

    Args:
        *arrays: Array-likes of numbers (the sample stack, the residual
            history, the reference mean and scale, ...).

    Returns:
        The hex digest (64 lowercase characters).
    """
    sha = hashlib.sha256()
    for item in arrays:
        array = np.ascontiguousarray(np.asarray(item, dtype="<f8"))
        sha.update(repr(array.shape).encode("ascii"))
        sha.update(array.tobytes())
    return sha.hexdigest()


def _finite_or_none(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ValueError(f"{name} must be finite, got {as_float}")
    return as_float


@dataclass(frozen=True)
class MeasurementEvidence:
    """The artifact that proves one quantity was measured.

    Attributes:
        method: How it was measured; a key of :data:`METHODS`.
        n: Observations behind it — draws, members, residuals, the
            calibration-set size, or the reference-set size.
        digest: SHA-256 (:func:`digest_of`) of the numbers it was measured on.
        statistic: For a verdict, the measured value that was compared
            (a novelty verdict's standardized distance).
        threshold: For a verdict, the cut the statistic was compared with.

    Raises:
        ValueError: If the method is not one of :data:`METHODS`, ``n`` is
            below what the method needs, the digest is not a SHA-256, or a
            statistic or threshold is not a finite number.  Evidence that
            proves nothing cannot be constructed.
    """

    method: str
    n: int
    digest: str
    statistic: float | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        """Refuse evidence that proves nothing."""
        if self.method not in METHODS:
            raise ValueError(
                f"unknown measurement method {self.method!r}; one of {sorted(METHODS)}"
            )
        if isinstance(self.n, bool) or not isinstance(self.n, (int, np.integer)):
            raise ValueError(f"n must be an integer count, got {type(self.n).__name__}")
        floor = METHODS[self.method]
        if int(self.n) < floor:
            raise ValueError(
                f"{self.method} needs at least {floor} observations to be a measurement, got {self.n}"
            )
        if not isinstance(self.digest, str) or not _DIGEST_RE.fullmatch(self.digest):
            raise ValueError("digest must be the SHA-256 hex of the measured numbers")
        object.__setattr__(self, "n", int(self.n))
        object.__setattr__(self, "statistic", _finite_or_none("statistic", self.statistic))
        object.__setattr__(self, "threshold", _finite_or_none("threshold", self.threshold))

    @classmethod
    def from_samples(
        cls,
        method: str,
        *arrays: Any,
        n: int | None = None,
        statistic: float | None = None,
        threshold: float | None = None,
    ) -> MeasurementEvidence:
        """Evidence computed from the numbers themselves, so it cannot drift from them.

        Args:
            method: How the quantity was measured.
            *arrays: The numbers it was measured on; their digest is taken here.
            n: Observations behind it; defaults to the first array's leading
                dimension.
            statistic: See :class:`MeasurementEvidence`.
            threshold: See :class:`MeasurementEvidence`.

        Returns:
            The evidence.

        Raises:
            ValueError: If no array is given, or the evidence is not valid.
        """
        if not arrays:
            raise ValueError("evidence needs the numbers it was measured on")
        count = int(np.asarray(arrays[0]).shape[0]) if n is None else n
        return cls(
            method=method,
            n=count,
            digest=digest_of(*arrays),
            statistic=statistic,
            threshold=threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON form; ``statistic`` / ``threshold`` appear only when present."""
        payload: dict[str, Any] = {"method": self.method, "n": self.n, "digest": self.digest}
        if self.statistic is not None:
            payload["statistic"] = self.statistic
        if self.threshold is not None:
            payload["threshold"] = self.threshold
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> MeasurementEvidence | None:
        """Read evidence back from a record, or ``None`` when it proves nothing.

        Total by design: a missing, mistyped or invalid payload is not an
        error, it is the absence of evidence — and so of a measurement.

        Args:
            payload: The mapping :meth:`to_dict` produced, or anything else.

        Returns:
            The evidence, or ``None``.
        """
        if not hasattr(payload, "get"):
            return None
        mapping: Mapping[str, Any] = payload
        try:
            return cls(
                method=mapping.get("method"),  # type: ignore[arg-type]
                n=mapping.get("n"),  # type: ignore[arg-type]
                digest=mapping.get("digest"),  # type: ignore[arg-type]
                statistic=mapping.get("statistic"),
                threshold=mapping.get("threshold"),
            )
        except (ValueError, TypeError):
            return None
