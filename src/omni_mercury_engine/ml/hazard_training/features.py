# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical feature specifications shared by training and inference.

The trained checkpoints only mean anything if the detector builds the *same*
feature vectors at inference that the pipeline built at training time, so the
builders live here (a numpy-only leaf) and are imported by both sides.

``GEOMAG_FEATURE_SPEC_V1`` (32 dims, matching
``GeomagneticStormPredictor(input_dim=32)``):

======  ==========================================================
 index  meaning
======  ==========================================================
     0  solar wind speed / 1000 (km/s)
     1  IMF Bz GSM (nT)
     2  IMF By GSM (nT)
     3  transverse IMF B_T = hypot(By, Bz) (nT)
     4  IMF field magnitude |B| (nT)
     5  IMF clock angle theta_c = atan2(|By|, Bz) (rad)
     6  sin^3(theta_c / 2) coupling factor
     7  rectified southward field B_s = max(-Bz, 0) (nT)
     8  proton density (cm^-3)
     9  log10(proton temperature (K))
    10  flow (dynamic) pressure (nPa)
    11  rectified motional E-field v * B_s * 1e-3 (mV/m)
    12  Boyle polar-cap potential (kV)
    13  log10(Boyle potential)
    14  Newell coupling v^{4/3} B_T^{2/3} sin^{8/3}(theta_c/2) * 1e-3
    15  v * B_T * 1e-3
    16  signed Bz magnitude Bz*|Bz|/10
    17  presence flag: By observed
    18  presence flag: |B| observed
    19  presence flag: density observed
    20  presence flag: temperature observed
 21-31  reserved (0.0) -- the architecture fixes input_dim=32; spec v1
        defines 21 informative dimensions
======  ==========================================================

Dims 0 and 1 intentionally coincide with the legacy 2-field zero-padding the
detector used before this spec existed (``[v/1000, bz, 0, ...]``), so old
call sites remain interpretable as "spec v1 with everything else missing".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

GEOMAG_FEATURE_SPEC_VERSION = "geomag-v1"
GEOMAG_FEATURE_DIM = 32

GEOMAG_FEATURE_NAMES: tuple[str, ...] = (
    "v_km_s_over_1000",
    "bz_gsm_nt",
    "by_gsm_nt",
    "bt_nt",
    "b_magnitude_nt",
    "clock_angle_rad",
    "sin3_half_clock",
    "bs_south_nt",
    "density_p_cm3",
    "log10_temperature_k",
    "flow_pressure_npa",
    "rectified_efield_mv_m",
    "boyle_potential_kv",
    "log10_boyle_potential",
    "newell_coupling_1e3",
    "v_bt_1e3",
    "bz_signed_sq_over_10",
    "flag_by",
    "flag_bmag",
    "flag_density",
    "flag_temperature",
) + tuple(f"reserved_{i}" for i in range(21, 32))

#: Fields the builder reads from ``magnetosphere_data`` (all optional except
#: speed and Bz, which every caller of the physics path already supplies).
GEOMAG_INPUT_FIELDS: tuple[str, ...] = (
    "solar_wind_speed_km_s",
    "bz_imf_nt",
    "by_imf_nt",
    "imf_magnitude_nt",
    "proton_density_p_cm3",
    "proton_temperature_k",
    "flow_pressure_npa",
)

#: Default fill values when optional fields are missing and no trained fill
#: statistics are available. These are long-run solar wind medians from the
#: OMNI2 archive (documented, fixed constants -- not fabricated per-sample).
GEOMAG_DEFAULT_FILL: dict[str, float] = {
    "by_imf_nt": 0.0,
    "imf_magnitude_nt": 5.0,
    "proton_density_p_cm3": 5.0,
    "proton_temperature_k": 8.0e4,
    "flow_pressure_npa": 1.7,
}


def _as_float(raw: object) -> float:
    """Coerce a loosely-typed observation field to ``float``.

    ``fields`` is typed ``Mapping[str, object]`` at the parse boundary, so a
    raw value arrives as ``object``; the geomag spec requires numeric inputs.
    This accepts Python/NumPy numbers and numeric strings and raises a clear
    ``TypeError`` for anything else, rather than letting ``float`` surface its
    opaque "float() argument must be..." message.
    """
    if isinstance(raw, (str, int, float, np.integer, np.floating)):
        return float(raw)
    raise TypeError(f"expected a numeric geomag field, got {type(raw).__name__}")


def _get(fields: Mapping[str, object], key: str, fill: Mapping[str, float]) -> tuple[float, bool]:
    """Fetch a numeric field, returning (value, was_observed)."""
    raw = fields.get(key)
    if raw is None:
        return float(fill.get(key, GEOMAG_DEFAULT_FILL.get(key, 0.0))), False
    value = _as_float(raw)
    if not np.isfinite(value):
        return float(fill.get(key, GEOMAG_DEFAULT_FILL.get(key, 0.0))), False
    return value, True


def build_geomag_feature_vector(
    fields: Mapping[str, object], fill: Mapping[str, float] | None = None
) -> np.ndarray:
    """Build the canonical 32-dim geomagnetic feature vector (spec v1).

    Args:
        fields: Raw observation dict; must contain finite
            ``solar_wind_speed_km_s`` and ``bz_imf_nt``. Optional fields per
            :data:`GEOMAG_INPUT_FIELDS` are filled (with a presence flag set
            to 0) when missing.
        fill: Optional fill values for missing optional fields -- shipped
            checkpoints carry the medians of their training years here so
            inference-time gaps are filled consistently with training.

    Returns:
        ``float32`` array of shape ``(32,)``.

    Raises:
        ValueError: If speed or Bz is missing/non-finite -- there is no
            transparent fill for the two primary drivers.
    """
    fill = fill or {}
    v_raw = fields.get("solar_wind_speed_km_s")
    bz_raw = fields.get("bz_imf_nt")
    if v_raw is None or bz_raw is None:
        raise ValueError(
            "geomag feature spec v1 requires 'solar_wind_speed_km_s' and 'bz_imf_nt'; "
            f"got keys {sorted(fields.keys())}"
        )
    v = _as_float(v_raw)
    bz = _as_float(bz_raw)
    if not (np.isfinite(v) and np.isfinite(bz)):
        raise ValueError(f"non-finite solar wind speed ({v}) or Bz ({bz})")

    by, has_by = _get(fields, "by_imf_nt", fill)
    bmag, has_bmag = _get(fields, "imf_magnitude_nt", fill)
    density, has_density = _get(fields, "proton_density_p_cm3", fill)
    temperature, has_temperature = _get(fields, "proton_temperature_k", fill)
    pressure_raw = fields.get("flow_pressure_npa")

    bt = float(np.hypot(by, bz))
    clock = float(np.arctan2(abs(by), bz))
    sin3 = float(np.sin(clock / 2.0) ** 3)
    bs = max(-bz, 0.0)
    pressure_val = _as_float(pressure_raw) if pressure_raw is not None else None
    if pressure_val is not None and np.isfinite(pressure_val):
        pressure = pressure_val
    else:
        # Standard dynamic-pressure formula (proton-only): 2e-6 * n * v^2 nPa.
        pressure = 2.0e-6 * density * v**2
    efield = v * bs * 1.0e-3
    boyle = 1.0e-4 * v**2 + 11.7 * bt * sin3
    newell = (
        (max(v, 0.0) ** (4.0 / 3.0)) * (bt ** (2.0 / 3.0)) * (np.sin(clock / 2.0) ** (8.0 / 3.0))
    )

    vec = np.zeros(GEOMAG_FEATURE_DIM, dtype=np.float32)
    vec[0] = v / 1000.0
    vec[1] = bz
    vec[2] = by
    vec[3] = bt
    vec[4] = bmag
    vec[5] = clock
    vec[6] = sin3
    vec[7] = bs
    vec[8] = density
    vec[9] = float(np.log10(max(temperature, 1.0)))
    vec[10] = pressure
    vec[11] = efield
    vec[12] = boyle
    vec[13] = float(np.log10(max(boyle, 1e-9)))
    vec[14] = float(newell) * 1.0e-3
    vec[15] = v * bt * 1.0e-3
    vec[16] = bz * abs(bz) / 10.0
    vec[17] = float(has_by)
    vec[18] = float(has_bmag)
    vec[19] = float(has_density)
    vec[20] = float(has_temperature)
    return vec
