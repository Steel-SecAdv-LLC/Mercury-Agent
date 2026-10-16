# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Continuous-wave radar I/Q front end: imbalance correction and demodulation.

A CW (or Six-Port) radar's baseband I/Q trace an ellipse rather than a circle:
DC offsets shift it, gain imbalance stretches it and phase imbalance tilts it.
Arctangent demodulation of the raw channels then distorts the phase. The
ellipse is fitted by the numerically stable direct least-squares method of
Halir and Flusser (1998), mapped onto the unit circle, and the unwrapped
angle is converted to displacement with ``d = phi * wavelength / (4 pi)``.

Any affine map that takes the fitted ellipse onto the unit circle differs
from the true imbalance inverse by a rotation or reflection, so the
demodulated phase equals the true phase up to a constant offset and sign;
rates and waveforms are unaffected.

References:
    Halir, R., Flusser, J. (1998). Numerically stable direct least squares
    fitting of ellipses. Proc. WSCG 1998, 125-132.

    Singh, A., Gao, X., Yavari, E., et al. (2013). Data-based quadrature
    imbalance compensation for a CW Doppler radar system. IEEE Trans.
    Microw. Theory Tech. 61(4), 1718-1724.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: Points used for the fit; longer recordings are subsampled evenly.
MAX_FIT_POINTS = 50_000


@dataclass(frozen=True)
class IQEllipse:
    """A fitted I/Q ellipse.

    Attributes:
        center: Ellipse centre ``(i0, q0)`` (the DC offsets).
        axes: Semi-axis lengths along the columns of ``rotation``.
        rotation: Orthonormal ``2x2`` matrix whose columns are the axes.
        residual: RMS of ``|w| - 1`` over the fitted points after mapping
            onto the unit circle (0 for an exact ellipse).
        arc_rad: Angular span the fitted points cover on the circle; an arc
            much shorter than ``pi`` leaves the ellipse poorly constrained.
    """

    center: tuple[float, float]
    axes: tuple[float, float]
    rotation: FloatArray
    residual: float
    arc_rad: float

    def to_circle(self, i: ArrayLike, q: ArrayLike) -> FloatArray:
        """Map I/Q samples onto the unit circle, shape ``(T, 2)``."""
        points = np.column_stack([np.asarray(i, float).ravel(), np.asarray(q, float).ravel()])
        local = (points - np.asarray(self.center)) @ self.rotation
        return np.asarray(local / np.asarray(self.axes), dtype=np.float64)


def _channels(i: ArrayLike, q: ArrayLike) -> tuple[FloatArray, FloatArray]:
    if np.iscomplexobj(i) or np.iscomplexobj(q):
        raise TypeError("pass the real I and Q channels separately")
    i_arr = np.asarray(i, dtype=np.float64).ravel()
    q_arr = np.asarray(q, dtype=np.float64).ravel()
    if i_arr.shape != q_arr.shape:
        raise ValueError(f"I and Q differ in length: {i_arr.size} vs {q_arr.size}")
    if i_arr.size < 6:
        raise ValueError("an ellipse fit needs at least 6 samples")
    if not (np.isfinite(i_arr).all() and np.isfinite(q_arr).all()):
        raise ValueError("I/Q contain non-finite samples")
    return i_arr, q_arr


def fit_iq_ellipse(i: ArrayLike, q: ArrayLike) -> IQEllipse:
    """Fit the ellipse traced by I/Q samples (Halir-Flusser direct least squares).

    Args:
        i: In-phase channel, shape ``(T,)``.
        q: Quadrature channel, shape ``(T,)``.

    Returns:
        The fitted ellipse.

    Raises:
        ValueError: On mismatched, short or non-finite input, or when the
            samples admit no ellipse (e.g. a straight line or a point).
        TypeError: On complex input.
    """
    i_arr, q_arr = _channels(i, q)
    step = max(1, math.ceil(i_arr.size / MAX_FIT_POINTS))
    x_raw, y_raw = i_arr[::step], q_arr[::step]
    # Normalising first keeps the scatter matrices well conditioned.
    mx, my = float(x_raw.mean()), float(y_raw.mean())
    scale = float(np.sqrt(np.mean((x_raw - mx) ** 2 + (y_raw - my) ** 2)))
    if not scale > 0.0:
        raise ValueError("I/Q are constant; no ellipse to fit")
    x, y = (x_raw - mx) / scale, (y_raw - my) / scale

    d1 = np.column_stack([x * x, x * y, y * y])
    d2 = np.column_stack([x, y, np.ones_like(x)])
    s1, s2, s3 = d1.T @ d1, d1.T @ d2, d2.T @ d2
    if np.linalg.cond(s3) > 1e9:
        # Collinear samples leave [x, y, 1] linearly dependent.
        raise ValueError("I/Q samples are degenerate (collinear)")
    try:
        t = -np.linalg.solve(s3, s2.T)
    except np.linalg.LinAlgError as exc:
        raise ValueError("I/Q samples are degenerate (collinear)") from exc
    m = s1 + s2 @ t
    m = np.array([m[2] / 2.0, -m[1], m[0] / 2.0])
    _, vectors = np.linalg.eig(m)
    vectors = np.real(vectors)
    condition = 4.0 * vectors[0] * vectors[2] - vectors[1] ** 2
    admissible = np.flatnonzero(condition > 0.0)
    if admissible.size != 1:
        raise ValueError("I/Q samples admit no ellipse")
    a1 = vectors[:, admissible[0]]
    a, b, c = a1
    d, e, f = t @ a1

    quad = np.array([[a, b / 2.0], [b / 2.0, c]])
    centre_n = np.linalg.solve(2.0 * quad, -np.array([d, e]))
    f_c = float(f + 0.5 * (d * centre_n[0] + e * centre_n[1]))
    eigvals, rotation = np.linalg.eigh(quad)
    if not (eigvals > 0).all() and not (eigvals < 0).all():
        raise ValueError("I/Q samples fit a hyperbola, not an ellipse")
    axes_sq = -f_c / eigvals
    if not (axes_sq > 0).all():
        raise ValueError("I/Q samples admit no real ellipse")
    axes_n = np.sqrt(axes_sq)

    ellipse = IQEllipse(
        center=(mx + scale * float(centre_n[0]), my + scale * float(centre_n[1])),
        axes=(scale * float(axes_n[0]), scale * float(axes_n[1])),
        rotation=np.asarray(rotation, dtype=np.float64),
        residual=0.0,
        arc_rad=0.0,
    )
    circle = ellipse.to_circle(x_raw, y_raw)
    radius = np.hypot(circle[:, 0], circle[:, 1])
    angle = np.sort(np.arctan2(circle[:, 1], circle[:, 0]))
    gaps = np.diff(np.concatenate([angle, angle[:1] + 2.0 * np.pi]))
    return IQEllipse(
        center=ellipse.center,
        axes=ellipse.axes,
        rotation=ellipse.rotation,
        residual=float(np.sqrt(np.mean((radius - 1.0) ** 2))),
        arc_rad=float(2.0 * np.pi - gaps.max()),
    )


def demodulate_iq(
    i: ArrayLike,
    q: ArrayLike,
    *,
    wavelength_m: float,
    ellipse: IQEllipse | None = None,
) -> FloatArray:
    """Return radial displacement (m) from CW radar I/Q.

    Args:
        i: In-phase channel, shape ``(T,)``.
        q: Quadrature channel, shape ``(T,)``.
        wavelength_m: Carrier wavelength (m), e.g. ``299792458 / 24e9``.
        ellipse: A previously fitted ellipse (fitted on ``i``/``q`` when
            ``None``).

    Returns:
        Displacement relative to the first sample, shape ``(T,)``; the sign
        convention is arbitrary (see the module docstring).

    Raises:
        ValueError: On a non-positive wavelength or an unfittable input.
        TypeError: On complex input.
    """
    if not (math.isfinite(wavelength_m) and wavelength_m > 0.0):
        raise ValueError(f"wavelength_m must be positive, got {wavelength_m}")
    i_arr, q_arr = _channels(i, q)
    fitted = ellipse if ellipse is not None else fit_iq_ellipse(i_arr, q_arr)
    circle = fitted.to_circle(i_arr, q_arr)
    phase = np.unwrap(np.arctan2(circle[:, 1], circle[:, 0]))
    return np.asarray((phase - phase[0]) * wavelength_m / (4.0 * np.pi), dtype=np.float64)
