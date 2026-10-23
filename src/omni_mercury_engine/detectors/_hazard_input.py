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

from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "InputGrader",
    "InvalidObservationError",
    "finite_array",
    "finite_scalar",
]


class InputGrader:
    """Grade what is usable; set the rest aside with its reason.

    The partial-input contract: when anything usable is fed, the detector
    runs and returns a result; it raises only when nothing usable is fed.
    Everything that could not be graded -- an unknown key, an empty section,
    a non-finite or out-of-range value -- is recorded here with the reason,
    surfaced in the result under ``set_aside``, and the verdict is marked
    ``partial`` so it can never read as a full all-clear.  (The former
    behaviour refused the whole record on the first problem, so one bad
    tide reading rejected a whole surge series.)
    """

    def __init__(self) -> None:
        """Start with nothing set aside."""
        self.set_aside: list[dict[str, Any]] = []

    @property
    def partial(self) -> bool:
        """Whether anything was set aside (the verdict must say so)."""
        return bool(self.set_aside)

    def discard(self, where: str, reason: str, value: Any = None) -> None:
        """Record one unusable piece of input and why it was set aside.

        Args:
            where: Dotted location (``"rainfall_data.intensity_mm_hr"``).
            reason: Why it could not be graded, in the caller's words.
            value: The offending value; its repr is truncated to 80 chars.
        """
        entry: dict[str, Any] = {"where": where, "reason": reason}
        if value is not None:
            text = repr(value)
            entry["value"] = text if len(text) <= 80 else text[:77] + "..."
        self.set_aside.append(entry)

    def usable_scalar(
        self,
        value: Any,
        where: str,
        *,
        unit: str,
        lo: float | None = None,
        hi: float | None = None,
    ) -> float | None:
        """``finite_scalar`` that sets an unusable value aside instead of raising.

        Args:
            value: The observation (``None`` means unmeasured: no entry).
            where: Dotted location recorded with any set-aside.
            unit: Unit the detector expects.
            lo: Inclusive physical lower bound, if any.
            hi: Inclusive physical upper bound, if any.

        Returns:
            The value as ``float``, or ``None`` when unmeasured or set aside.
        """
        if value is None:
            return None
        try:
            return finite_scalar(value, where, unit=unit, lo=lo, hi=hi)
        except InvalidObservationError as exc:
            self.discard(where, str(exc), value)
            return None

    def grade_sections(
        self,
        data: dict[str, Any],
        schema: dict[str, dict[str, tuple[str, float, float]]],
    ) -> dict[str, dict[str, float]]:
        """Grade a sectioned record against its input schema.

        Unknown sections and keys, non-mapping sections, and unusable values
        are set aside with reasons; every usable value is returned.  The
        caller decides afterwards whether anything usable remains
        (:meth:`require_usable`).

        Args:
            data: The record (section name -> mapping of measurements).
            schema: section -> key -> (unit, lo, hi).

        Returns:
            section -> key -> graded float, holding only usable values.
        """
        graded: dict[str, dict[str, float]] = {}
        for section, values in data.items():
            if section not in schema:
                self.discard(section, f"unknown section; accepted: {sorted(schema)}")
                continue
            if not isinstance(values, dict):
                self.discard(section, f"must be a mapping, got {type(values).__name__}", values)
                continue
            fields = schema[section]
            usable: dict[str, float] = {}
            for key, value in values.items():
                if key not in fields:
                    self.discard(
                        f"{section}.{key}",
                        f"unknown key; accepted: {sorted(fields)}",
                    )
                    continue
                unit, lo, hi = fields[key]
                number = self.usable_scalar(value, f"{section}.{key}", unit=unit, lo=lo, hi=hi)
                if number is not None:
                    usable[key] = number
            if usable:
                graded[section] = usable
            elif values:
                self.discard(section, "no usable measurement in this section")
            else:
                self.discard(section, "empty section carries no measurement")
        return graded

    def grade_section(
        self,
        section: str,
        values: Any,
        *,
        scalars: dict[str, tuple[str, float | None, float | None]],
        others: frozenset[str] = frozenset(),
        complete: tuple[frozenset[str], ...] = (),
    ) -> dict[str, Any] | None:
        """One section's usable fields, or ``None`` when no measurement remains.

        Scalars are graded value by value; ``others`` (arrays, labels) pass
        through for the section's analyzer to check.  ``complete`` lists the
        key sets that each make a measurement; when given, a section holding
        none of them is set aside (an absent key must never read as a quiet
        default).

        Args:
            section: The section's name.
            values: The supplied section.
            scalars: key -> (unit, lo, hi) for the scalar fields.
            others: Accepted non-scalar keys.
            complete: Key sets of which at least one must be fully present.

        Returns:
            The usable fields, or ``None`` when the section was set aside.
        """
        if not isinstance(values, dict):
            self.discard(section, f"must be a mapping, got {type(values).__name__}", values)
            return None
        accepted = set(scalars) | set(others)
        usable: dict[str, Any] = {}
        for key, value in values.items():
            where = f"{section}.{key}"
            if key not in accepted:
                self.discard(where, f"unknown key; accepted: {sorted(accepted)}")
            elif key in scalars:
                unit, lo, hi = scalars[key]
                number = self.usable_scalar(value, where, unit=unit, lo=lo, hi=hi)
                if number is not None:
                    usable[key] = number
            elif value is not None:
                usable[key] = value
        if not values:
            self.discard(section, "empty section carries no measurement")
            return None
        if complete and not any(keys <= set(usable) for keys in complete):
            self.discard(
                section,
                f"carries no measurement; supply one of {[sorted(k) for k in complete]}",
            )
            return None
        if not usable:
            self.discard(section, "no usable measurement in this section")
            return None
        return usable

    def attempt(self, where: str, analyze: Callable[[Any], Any], value: Any) -> Any:
        """Run one stream's analyzer; a stream it refuses is set aside, not fatal.

        Returns:
            The analyzer's result, or ``None`` when it refused the stream
            (``InvalidObservationError`` / ``InputModalityError``).
        """
        try:
            return analyze(value)
        except (InvalidObservationError, InputModalityError) as exc:
            self.discard(where, str(exc))
            return None

    def require_usable(self, usable: Any, context: str) -> None:
        """Raise only when nothing usable was fed.

        Args:
            usable: Whatever the caller graded; falsy means nothing usable.
            context: What the detector needed, for the error message.

        Raises:
            InvalidObservationError: When nothing usable remains, naming
                every set-aside piece and its reason.
        """
        if usable:
            return
        details = "; ".join(f"{e['where']}: {e['reason']}" for e in self.set_aside[:8])
        raise InvalidObservationError(
            f"nothing usable was fed: {context}" + (f" (set aside: {details})" if details else "")
        )

    def result_fields(self) -> dict[str, Any]:
        """``{"partial": True, "set_aside": [...]}`` when anything was set aside."""
        if not self.set_aside:
            return {}
        return {"partial": True, "set_aside": list(self.set_aside)}


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
