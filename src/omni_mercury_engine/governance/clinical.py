"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""Clinical governance scalars (metric-only) under the abstention contract.

⚠️ NOT FOR CLINICAL USE.  These scalars reproduce published scoring tables for
*reporting and audit* only.  They are descriptive (metric-only), never gate any
decision, and must not be used for diagnosis, triage, or treatment.

Families implemented here:

* **SOFA** (Sequential Organ Failure Assessment; Vincent et al., Intensive Care Med
  1996).  The six organ sub-scores are obtained by reusing the in-tree
  :class:`SOFACalculator`; each is exposed only when its required input is present,
  and the total abstains unless all six inputs are present (a partial SOFA total is
  undefined).
* **NEWS2** (Royal College of Physicians National Early Warning Score 2, 2017, SpO2
  Scale 1).  Aggregate of seven physiological parameters, hand-implemented from the
  published table and abstaining when any parameter is missing.

The existing :class:`SOFACalculator` silently defaults missing inputs (e.g. platelets
-> 200); this layer deliberately checks input *presence* first and abstains instead,
so a missing measurement is never reported as a healthy organ.
"""

from omni_mercury_engine.governance.contract import GovernanceScalar, available, unavailable

_SOFA_FAMILY = "sofa"
_EWS_FAMILY = "ews"
_SOFA_ORGAN_MAX = 4.0  # each organ sub-score is 0..4
_SOFA_TOTAL_MAX = 24.0  # six organs x 4

# Required input key(s) per SOFA organ system (mirrors SOFACalculator's reads).
_SOFA_INPUTS: dict[str, tuple[str, ...]] = {
    "respiration": ("pao2_fio2_ratio",),
    "coagulation": ("platelets_k_ul",),
    "liver": ("bilirubin_mg_dl",),
    "cardiovascular": ("mean_arterial_pressure",),
    "cns": ("gcs_score",),
    "renal": ("creatinine_mg_dl",),
}

_NEWS2_PARAMS: tuple[str, ...] = (
    "respiratory_rate_bpm",
    "spo2_pct",
    "on_supplemental_o2",
    "systolic_bp_mmhg",
    "pulse_bpm",
    "consciousness",
    "temperature_c",
)


def sofa_scalars(patient_data: dict[str, object]) -> list[GovernanceScalar]:
    """Expose the six SOFA organ sub-scores plus the total as metric-only scalars.

    Each sub-score is produced only when its required input key is present in
    ``patient_data``; otherwise it abstains.  The total abstains unless every organ
    input is present.  Points are computed by the in-tree :class:`SOFACalculator`, so
    this layer reuses validated arithmetic rather than re-deriving it.

    Args:
        patient_data: Mapping of clinical inputs (e.g. ``platelets_k_ul``, ``gcs_score``).

    Returns:
        One :class:`GovernanceScalar` per organ system, plus the SOFA total.
    """
    try:
        from omni_mercury_engine.medical.critical_care.sepsis_detector import SOFACalculator
    except ImportError as exc:  # detector stack (torch) absent -> abstain cleanly
        reason = f"SOFACalculator unavailable ({exc.__class__.__name__})"
        names = [f"omni_sofa_{organ}" for organ in _SOFA_INPUTS] + ["omni_sofa_total"]
        return [unavailable(name, family=_SOFA_FAMILY, reason=reason) for name in names]

    sofa = SOFACalculator().calculate_sofa(patient_data)

    scalars: list[GovernanceScalar] = []
    all_present = True
    for organ, required in _SOFA_INPUTS.items():
        name = f"omni_sofa_{organ}"
        if all(key in patient_data and patient_data[key] is not None for key in required):
            points = int(sofa[organ])
            scalars.append(
                available(
                    name,
                    points / _SOFA_ORGAN_MAX,
                    family=_SOFA_FAMILY,
                    reason=f"SOFA {organ} = {points}/4",
                    provenance={"organ": organ, "points": points, "inputs": list(required)},
                )
            )
        else:
            all_present = False
            scalars.append(
                unavailable(
                    name,
                    family=_SOFA_FAMILY,
                    reason=f"missing input(s) {required} for SOFA {organ}",
                )
            )

    if all_present:
        total = int(sofa["sofa_score"])
        scalars.append(
            available(
                "omni_sofa_total",
                total / _SOFA_TOTAL_MAX,
                family=_SOFA_FAMILY,
                reason=f"SOFA total = {total}/24",
                provenance={"points": total},
            )
        )
    else:
        scalars.append(
            unavailable(
                "omni_sofa_total",
                family=_SOFA_FAMILY,
                reason="SOFA total undefined: not all six organ inputs present",
            )
        )
    return scalars


def _as_number(value: object) -> float | None:
    """Coerce a numeric measurement to float; reject bool and non-numbers."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _news2_respiration(rr: float) -> int:
    if rr <= 8:
        return 3
    if rr <= 11:
        return 1
    if rr <= 20:
        return 0
    if rr <= 24:
        return 2
    return 3


def _news2_spo2_scale1(spo2: float) -> int:
    if spo2 >= 96:
        return 0
    if spo2 >= 94:
        return 1
    if spo2 >= 92:
        return 2
    return 3


def _news2_systolic(sbp: float) -> int:
    if sbp <= 90:
        return 3
    if sbp <= 100:
        return 2
    if sbp <= 110:
        return 1
    if sbp <= 219:
        return 0
    return 3


def _news2_pulse(pulse: float) -> int:
    if pulse <= 40:
        return 3
    if pulse <= 50:
        return 1
    if pulse <= 90:
        return 0
    if pulse <= 110:
        return 1
    if pulse <= 130:
        return 2
    return 3


def _news2_temperature(temp: float) -> int:
    if temp <= 35.0:
        return 3
    if temp <= 36.0:
        return 1
    if temp <= 38.0:
        return 0
    if temp <= 39.0:
        return 1
    return 2


def news2_scalar(vitals: dict[str, object]) -> GovernanceScalar:
    """Compute the NEWS2 aggregate (RCP 2017, SpO2 Scale 1) as a metric-only scalar.

    Required ``vitals`` keys: ``respiratory_rate_bpm``, ``spo2_pct``,
    ``on_supplemental_o2`` (bool), ``systolic_bp_mmhg``, ``pulse_bpm``,
    ``consciousness`` (``"A"`` for alert, else one of C/V/P/U), and ``temperature_c``.
    Any missing or non-numeric parameter abstains -- a partial NEWS2 is not a NEWS2.

    Args:
        vitals: Mapping of the seven NEWS2 physiological parameters.

    Returns:
        A scalar equal to ``aggregate / 20`` when all parameters are present, else an
        abstention.  The aggregate and per-parameter points are recorded in provenance.
    """
    missing = [key for key in _NEWS2_PARAMS if key not in vitals or vitals[key] is None]
    if missing:
        return unavailable(
            "omni_ews_news2",
            family=_EWS_FAMILY,
            reason=f"missing NEWS2 parameter(s): {missing}",
        )

    rr = _as_number(vitals["respiratory_rate_bpm"])
    spo2 = _as_number(vitals["spo2_pct"])
    sbp = _as_number(vitals["systolic_bp_mmhg"])
    pulse = _as_number(vitals["pulse_bpm"])
    temp = _as_number(vitals["temperature_c"])
    if None in (rr, spo2, sbp, pulse, temp):
        return unavailable(
            "omni_ews_news2",
            family=_EWS_FAMILY,
            reason="NEWS2 parameter(s) not numeric",
        )
    assert rr is not None and spo2 is not None and sbp is not None
    assert pulse is not None and temp is not None

    on_o2 = bool(vitals["on_supplemental_o2"])
    alert = str(vitals["consciousness"]).strip().upper() == "A"

    points = {
        "respiration": _news2_respiration(rr),
        "spo2": _news2_spo2_scale1(spo2),
        "supplemental_o2": 2 if on_o2 else 0,
        "systolic_bp": _news2_systolic(sbp),
        "pulse": _news2_pulse(pulse),
        "consciousness": 0 if alert else 3,
        "temperature": _news2_temperature(temp),
    }
    aggregate = sum(points.values())
    return available(
        "omni_ews_news2",
        aggregate / 20.0,
        family=_EWS_FAMILY,
        reason=f"NEWS2 aggregate = {aggregate}/20 (Scale 1)",
        provenance={"aggregate": aggregate, "points": points, "scale": 1},
    )
