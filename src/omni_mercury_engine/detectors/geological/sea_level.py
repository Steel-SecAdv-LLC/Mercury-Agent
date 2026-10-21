# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sea-level record preprocessing shared by tsunami detection and training.

A tsunami is read from the residual of a sea-level or bottom-pressure record
after the tide is removed. Without that step a pure 0.5 m tide scored as a
tsunami at confidence 1.0, because the tide itself was taken as the wave
height and its smooth segments as the noise floor. This module is the single
definition of that residual, used by
:class:`~omni_mercury_engine.detectors.geological.disaster_detectors.TsunamiDetector`
and by the ``tsunami_dart`` training pipeline, so the detector and the model
always see the same input. Pure NumPy, deterministic.

Tide model, by record span:

* 12 h or longer: ordinary least squares on [constant, linear, cos/sin of the
  M2, S2, K1, O1 constituents], the training pipeline's detiding. A 24 h
  window at the DART standard 15 min rate gets exactly the training design.
* 30 min to 12 h: a cubic trend. The constituents are not separable over less
  than one semidiurnal cycle, and the DART onboard tide predictor is likewise
  a cubic over the past 3 h (Mofjeld, NOAA/PMEL tsunami detection algorithm,
  https://nctr.pmel.noaa.gov/tda_documentation.html).
* Under 30 min: refused. The tsunami band reaches 60 min and more (IOC: 5-60
  min typical; Mofjeld: 2-90 min), so a shorter record cannot separate a wave
  from the tide.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from omni_mercury_engine.detectors._hazard_input import InvalidObservationError

__all__ = [
    "HARMONIC_MIN_SPAN_H",
    "MIN_RECORD_SPAN_MIN",
    "SENSOR_RESOLUTION_M",
    "TIDE_PERIODS_H",
    "detide",
    "quietest_segment_sigma",
    "remove_isolated_spikes",
]

#: Tidal constituent periods (hours): M2, S2, K1, O1.
TIDE_PERIODS_H = (12.4206012, 12.0, 23.9344697, 25.8193417)

#: Record span from which the harmonic tide model is used.
HARMONIC_MIN_SPAN_H = 12.0

#: Shortest record span the detector accepts.
MIN_RECORD_SPAN_MIN = 30.0

#: Noise-floor lower bound: 1 mm, above the sub-millimetre resolution of DART
#: bottom-pressure recorders (https://www.ndbc.noaa.gov/dart/bottom.shtml), so
#: a quantised or interpolated record cannot drive the z-score to infinity.
SENSOR_RESOLUTION_M = 1e-3


def _design(n: int, sample_period_s: float) -> np.ndarray[Any, Any]:
    """Tide-model design matrix for ``n`` equally spaced samples."""
    t = np.arange(n, dtype=np.float64) * sample_period_s
    span_h = (n - 1) * sample_period_s / 3600.0
    scaled = (t - t.mean()) / t.std()
    if span_h >= HARMONIC_MIN_SPAN_H:
        columns = [np.ones_like(t), scaled]
        for period_h in TIDE_PERIODS_H:
            omega = 2.0 * np.pi / (period_h * 3600.0)
            columns.append(np.cos(omega * t))
            columns.append(np.sin(omega * t))
    else:
        columns = [np.ones_like(t), scaled, scaled**2, scaled**3]
    return np.stack(columns, axis=1)


def detide(
    values: np.ndarray[Any, Any], sample_period_s: float
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Split a sea-level record into its tide model and the residual.

    Args:
        values: Finite water levels (m), one equally spaced record.
        sample_period_s: Sampling period, seconds.

    Returns:
        ``(residual, tide)`` in metres, ``float64``, each the record's shape.

    Raises:
        InvalidObservationError: If the record spans less than
            :data:`MIN_RECORD_SPAN_MIN` minutes.
    """
    record = np.asarray(values, dtype=np.float64)
    span_min = (record.size - 1) * sample_period_s / 60.0
    if record.size < 8 or span_min < MIN_RECORD_SPAN_MIN:
        raise InvalidObservationError(
            f"sea-level record spans {span_min:.1f} min ({record.size} samples at "
            f"{sample_period_s:g} s); at least {MIN_RECORD_SPAN_MIN:.0f} min are needed to "
            "separate a tsunami from the tide"
        )
    design = _design(record.size, sample_period_s)
    coef, *_ = np.linalg.lstsq(design, record, rcond=None)
    tide = design @ coef
    return record - tide, tide


def remove_isolated_spikes(
    residual: np.ndarray[Any, Any],
    sigma: float,
    sample_period_s: float,
    *,
    spike_z: float = 5.0,
    support_z: float = 2.5,
) -> tuple[np.ndarray[Any, Any], int]:
    """Replace single-sample glitches with the neighbouring mean.

    A sample is a glitch when it exceeds ``spike_z`` noise units while no other
    sample within one hour either side departs from that hour's median by
    more than ``support_z``. A tsunami, even one sampled every 15 minutes, puts
    its crest and trough within the hour of each other, so it is never
    removed; a lone 3 m telemetry spike (which read MAJOR, "EVACUATE coastal
    areas immediately", before) is. Support is measured against the local
    median because a least-squares tide fit spreads part of a spike into a
    smooth offset around it, which would otherwise count as support. Refit the
    tide after removing a glitch (the detector does).

    Args:
        residual: Detided residual (m).
        sigma: Noise scale (m).
        sample_period_s: Sampling period, seconds.
        spike_z: Glitch threshold in noise units.
        support_z: Level another nearby sample must exceed to make the
            excursion real rather than isolated.

    Returns:
        ``(cleaned, n_removed)``.
    """
    cleaned = residual.copy()
    big = np.flatnonzero(np.abs(residual) > spike_z * sigma)
    if big.size == 0:
        return cleaned, 0
    window = max(2, int(np.ceil(3600.0 / sample_period_s)))
    removed = 0
    for i in (int(j) for j in big):
        lo, hi = max(0, i - window), min(residual.size, i + window + 1)
        neighbours = np.delete(residual[lo:hi], i - lo)
        local = float(np.median(neighbours))
        if not np.any(np.abs(neighbours - local) > support_z * sigma):
            cleaned[i] = local
            removed += 1
    return cleaned, removed


def quietest_segment_sigma(residual: np.ndarray[Any, Any]) -> float:
    """Noise scale of the residual from its quietest segment.

    The residual is cut into up to eight segments of at least 64 samples and
    the smallest robust scale (1.4826 x MAD) is taken, so a long-lasting wave
    does not inflate its own noise floor. On the detided residual this is a
    noise estimate; on a raw record it was the flank of the tide. The result
    is floored at :data:`SENSOR_RESOLUTION_M`.

    Args:
        residual: Detided residual (m).

    Returns:
        Noise scale (m).
    """
    n_segments = max(1, min(8, residual.size // 64))
    scales = []
    for segment in np.array_split(residual, n_segments):
        mad = float(np.median(np.abs(segment - np.median(segment))))
        if mad > 0.0:
            scales.append(1.4826 * mad)
    sigma = min(scales) if scales else float(np.std(residual))
    return max(sigma, SENSOR_RESOLUTION_M)
