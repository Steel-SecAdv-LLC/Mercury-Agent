# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail-closed checks for the physical inputs of the domain hazard detectors.

A hazard detector that receives a value it cannot interpret must refuse it
rather than score it. Without these checks a NaN fell through severity ladders
to their top bucket (earthquake "great", hurricane Category 5), a temperature in
kelvin read as degrees Celsius, and an empty gauge record read as "at flood
stage". Each check names the quantity and the unit the detector expects, so the
caller can correct the input. Pure NumPy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from omni_mercury_engine.core.exceptions import InvalidObservationError

__all__ = ["InvalidObservationError", "finite_array", "finite_scalar"]


def _range_text(lo: float | None, hi: float | None, unit: str) -> str:
    """Human-readable physical range for an error message."""
    if lo is not None and hi is not None:
        return f"[{lo:g}, {hi:g}] {unit}"
    if lo is not None:
        return f">= {lo:g} {unit}"
    return f"<= {hi:g} {unit}"


def finite_scalar(
    value: Any,
    name: str,
    *,
    unit: str,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    """Return ``value`` as a float, refusing non-numeric, non-finite or out-of-range input.

    Args:
        value: The observation.
        name: Quantity name for the error message.
        unit: Unit the detector expects (named in the error message).
        lo: Inclusive physical lower bound, if any.
        hi: Inclusive physical upper bound, if any.

    Returns:
        The value as ``float``.

    Raises:
        InvalidObservationError: If the value is not a finite number in range.
    """
    if isinstance(value, bool):
        raise InvalidObservationError(f"{name} must be a number in {unit}, got a bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidObservationError(
            f"{name} must be a number in {unit}, got {type(value).__name__}"
        ) from exc
    if not np.isfinite(number):
        raise InvalidObservationError(f"{name} must be finite, got {number}")
    if (lo is not None and number < lo) or (hi is not None and number > hi):
        raise InvalidObservationError(
            f"{name}={number:g} is outside the physical range {_range_text(lo, hi, unit)}; "
            "check the unit"
        )
    return number


def finite_array(
    values: Any,
    name: str,
    *,
    unit: str,
    ndim: int | tuple[int, ...] | None = None,
    min_size: int = 1,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray[Any, Any]:
    """Return ``values`` as a float64 array, refusing malformed or non-physical input.

    Args:
        values: Array-like observation (a torch tensor is detached first).
        name: Quantity name for the error message.
        unit: Unit the detector expects.
        ndim: Required number of dimensions (or allowed set), if any.
        min_size: Minimum number of elements.
        lo: Inclusive physical lower bound applied to every element, if any.
        hi: Inclusive physical upper bound applied to every element, if any.

    Returns:
        A ``float64`` copy of the data.

    Raises:
        InvalidObservationError: On non-numeric data, wrong dimensionality, too
            few elements, any non-finite element, or an element out of range.
    """
    detach = getattr(values, "detach", None)
    if callable(detach):
        values = detach().cpu().numpy()
    try:
        arr = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidObservationError(
            f"{name} must be a numeric array in {unit}, got {type(values).__name__}"
        ) from exc
    allowed = (ndim,) if isinstance(ndim, int) else ndim
    if allowed is not None and arr.ndim not in allowed:
        raise InvalidObservationError(
            f"{name} must have {' or '.join(str(d) for d in allowed)} dimension(s), "
            f"got shape {arr.shape}"
        )
    if arr.size < min_size:
        raise InvalidObservationError(f"{name} needs at least {min_size} value(s), got {arr.size}")
    if not np.isfinite(arr).all():
        bad = int(arr.size - np.isfinite(arr).sum())
        raise InvalidObservationError(
            f"{name} has {bad} non-finite value(s); clean or interpolate the record first"
        )
    if arr.size and (
        (lo is not None and float(arr.min()) < lo) or (hi is not None and float(arr.max()) > hi)
    ):
        raise InvalidObservationError(
            f"{name} spans [{float(arr.min()):g}, {float(arr.max()):g}], outside the physical "
            f"range {_range_text(lo, hi, unit)}; check the unit"
        )
    return arr
