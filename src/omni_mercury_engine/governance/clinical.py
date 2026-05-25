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

"""Clinical governance scalars (metric-only) under the three-state abstention contract.

⚠️ NOT FOR CLINICAL USE.  These scalars reproduce published scoring tables/formulae for
*reporting and audit* only.  They are descriptive (metric-only), never gate any decision,
and must not be used for diagnosis, triage, or treatment.

All four families here vet **UNAVAILABLE-capable**: their inputs are members of the
engine's clinical observable surface (``patient_data`` mappings / vital-sign channel; see
``core/domain_feature_extractors.py:311-318`` and the in-tree ``SOFACalculator``), so each
scalar is GROUNDED when its inputs are present and UNAVAILABLE (kept, fires later) when a
required input is absent -- never a placeholder, never a healthy default.

Families:

* **SOFA** (Vincent et al., Intensive Care Med 1996) -- six organ sub-scores + total, via
  the in-tree :class:`SOFACalculator`.
* **NEWS2** (Royal College of Physicians 2017, SpO2 Scale 1) -- seven-parameter aggregate.
* **MEWS** (Subbe et al., QJM 2001) -- five-parameter aggregate; the point table is
  reproduced verbatim below so the published-table claim is reviewable in-source.
* **MELD-Na** (Kim et al., NEJM 2008; OPTN/UNOS policy) -- hepatic allocation score; its
  INR/sodium inputs are not yet consumed elsewhere in-engine, so it abstains (UNAVAILABLE)
  until they flow through the same clinical channel SOFA's labs already use.

The in-tree :class:`SOFACalculator` silently defaults missing inputs (e.g. platelets ->
200); this layer deliberately checks input *presence* first and abstains instead, so a
missing measurement is never reported as a healthy organ.
"""

import math

from omni_mercury_engine.governance.contract import GovernanceScalar, grounded, unavailable

_SOFA_FAMILY = "sofa"
_EWS_FAMILY = "ews"
_MEWS_FAMILY = "mews"
_MELD_FAMILY = "meld"
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

_MEWS_PARAMS: tuple[str, ...] = (
    "systolic_bp_mmhg",
    "pulse_bpm",
    "respiratory_rate_bpm",
    "temperature_c",
    "avpu",
)

_MELD_PARAMS: tuple[str, ...] = (
    "bilirubin_mg_dl",
    "creatinine_mg_dl",
    "inr",
    "sodium_meq_l",
)


def sofa_scalars(patient_data: dict[str, object]) -> list[GovernanceScalar]:
    """Expose the six SOFA organ sub-scores plus the total as metric-only scalars.

    Each sub-score is GROUNDED only when its required input key is present in
    ``patient_data``; otherwise it abstains UNAVAILABLE.  The total abstains unless every
    organ input is present.  Points are computed by the in-tree :class:`SOFACalculator`, so
    this layer reuses validated arithmetic rather than re-deriving it.

    Args:
        patient_data: Mapping of clinical inputs (e.g. ``platelets_k_ul``, ``gcs_score``).

    Returns:
        One :class:`GovernanceScalar` per organ system, plus the SOFA total.
    """
    try:
        from omni_mercury_engine.medical.critical_care.sepsis_detector import SOFACalculator
    except ImportError as exc:  # detector stack (torch) absent -> UNAVAILABLE, not a value
        reason = f"SOFACalculator unavailable ({exc.__class__.__name__}); signal real, deferred"
        names = [f"omni_sofa_{organ}" for organ in _SOFA_INPUTS] + ["omni_sofa_total"]
        return [
            unavailable(
                name, family=_SOFA_FAMILY, reason=reason, missing_inputs=("SOFACalculator",)
            )
            for name in names
        ]

    sofa = SOFACalculator().calculate_sofa(patient_data)

    scalars: list[GovernanceScalar] = []
    all_present = True
    for organ, required in _SOFA_INPUTS.items():
        name = f"omni_sofa_{organ}"
        present = [key for key in required if key in patient_data and patient_data[key] is not None]
        if len(present) == len(required):
            points = int(sofa[organ])
            scalars.append(
                grounded(
                    name,
                    points / _SOFA_ORGAN_MAX,
                    family=_SOFA_FAMILY,
                    reason=f"SOFA {organ} = {points}/4",
                    provenance={"organ": organ, "points": points, "inputs": list(required)},
                )
            )
        else:
            all_present = False
            missing = tuple(k for k in required if k not in present)
            scalars.append(
                unavailable(
                    name,
                    family=_SOFA_FAMILY,
                    reason=f"missing input(s) {missing} for SOFA {organ}",
                    missing_inputs=missing,
                )
            )

    if all_present:
        total = int(sofa["sofa_score"])
        scalars.append(
            grounded(
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
                missing_inputs=tuple(
                    k
                    for ks in _SOFA_INPUTS.values()
                    for k in ks
                    if k not in patient_data or patient_data[k] is None
                ),
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
    Any missing or non-numeric parameter abstains UNAVAILABLE -- a partial NEWS2 is not a
    NEWS2.

    Args:
        vitals: Mapping of the seven NEWS2 physiological parameters.

    Returns:
        A scalar equal to ``aggregate / 20`` when all parameters are present, else an
        abstention.  The aggregate and per-parameter points are recorded in provenance.
    """
    missing = tuple(key for key in _NEWS2_PARAMS if key not in vitals or vitals[key] is None)
    if missing:
        return unavailable(
            "omni_ews_news2",
            family=_EWS_FAMILY,
            reason=f"missing NEWS2 parameter(s): {list(missing)}",
            missing_inputs=missing,
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
            missing_inputs=tuple(
                p
                for p, v in (
                    ("respiratory_rate_bpm", rr),
                    ("spo2_pct", spo2),
                    ("systolic_bp_mmhg", sbp),
                    ("pulse_bpm", pulse),
                    ("temperature_c", temp),
                )
                if v is None
            ),
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
    return grounded(
        "omni_ews_news2",
        aggregate / 20.0,
        family=_EWS_FAMILY,
        reason=f"NEWS2 aggregate = {aggregate}/20 (Scale 1)",
        provenance={"aggregate": aggregate, "points": points, "scale": 1},
    )


# ----------------------------------------------------------------------------------------
# MEWS -- Subbe et al., QJM 2001;94:521-526.  Point table reproduced verbatim (so the
# published-table claim is checkable in-source); max aggregate = 3+3+3+2+3 = 14.
#
#   score        3        2        1        0        1        2        3
#   SBP(mmHg)   <=70    71-80    81-100  101-199            >=200
#   HR(bpm)             <=40     41-50   51-100  101-110  111-129    >=130
#   RR(/min)            <9               9-14    15-20    21-29      >=30
#   Temp(degC)          <35              35-38.4          >=38.5
#   AVPU                                 Alert    Voice    Pain     Unresponsive
# ----------------------------------------------------------------------------------------
_MEWS_AGG_MAX = 14.0


def _mews_systolic(sbp: float) -> int:
    if sbp <= 70:
        return 3
    if sbp <= 80:
        return 2
    if sbp <= 100:
        return 1
    if sbp <= 199:
        return 0
    return 2


def _mews_pulse(pulse: float) -> int:
    if pulse <= 40:
        return 2
    if pulse <= 50:
        return 1
    if pulse <= 100:
        return 0
    if pulse <= 110:
        return 1
    if pulse <= 129:
        return 2
    return 3


def _mews_respiration(rr: float) -> int:
    if rr < 9:
        return 2
    if rr <= 14:
        return 0
    if rr <= 20:
        return 1
    if rr <= 29:
        return 2
    return 3


def _mews_temperature(temp: float) -> int:
    if temp < 35.0:
        return 2
    if temp <= 38.4:
        return 0
    return 2


_MEWS_AVPU_POINTS: dict[str, int] = {"A": 0, "V": 1, "P": 2, "U": 3}


def mews_scalar(vitals: dict[str, object]) -> GovernanceScalar:
    """Compute the MEWS aggregate (Subbe et al. 2001) as a metric-only scalar.

    Required ``vitals`` keys: ``systolic_bp_mmhg``, ``pulse_bpm``,
    ``respiratory_rate_bpm``, ``temperature_c`` (numeric), and ``avpu``
    (one of ``A``/``V``/``P``/``U``).  These are the same vital-sign channel NEWS2 reads,
    so the family is UNAVAILABLE-capable; any missing/invalid parameter abstains.

    Args:
        vitals: Mapping of the five MEWS parameters.

    Returns:
        A scalar equal to ``aggregate / 14`` when all parameters are present and valid,
        else an UNAVAILABLE abstention recording the missing parameters.
    """
    missing = tuple(key for key in _MEWS_PARAMS if key not in vitals or vitals[key] is None)
    if missing:
        return unavailable(
            "omni_ews_mews",
            family=_MEWS_FAMILY,
            reason=f"missing MEWS parameter(s): {list(missing)}",
            missing_inputs=missing,
        )

    sbp = _as_number(vitals["systolic_bp_mmhg"])
    pulse = _as_number(vitals["pulse_bpm"])
    rr = _as_number(vitals["respiratory_rate_bpm"])
    temp = _as_number(vitals["temperature_c"])
    avpu = str(vitals["avpu"]).strip().upper()
    bad = tuple(
        p
        for p, v in (
            ("systolic_bp_mmhg", sbp),
            ("pulse_bpm", pulse),
            ("respiratory_rate_bpm", rr),
            ("temperature_c", temp),
        )
        if v is None
    )
    if bad or avpu not in _MEWS_AVPU_POINTS:
        return unavailable(
            "omni_ews_mews",
            family=_MEWS_FAMILY,
            reason="MEWS parameter(s) not numeric or AVPU not in A/V/P/U",
            missing_inputs=bad + (() if avpu in _MEWS_AVPU_POINTS else ("avpu",)),
        )
    assert sbp is not None and pulse is not None and rr is not None and temp is not None

    points = {
        "systolic_bp": _mews_systolic(sbp),
        "pulse": _mews_pulse(pulse),
        "respiration": _mews_respiration(rr),
        "temperature": _mews_temperature(temp),
        "avpu": _MEWS_AVPU_POINTS[avpu],
    }
    aggregate = sum(points.values())
    return grounded(
        "omni_ews_mews",
        aggregate / _MEWS_AGG_MAX,
        family=_MEWS_FAMILY,
        reason=f"MEWS aggregate = {aggregate}/14",
        provenance={"aggregate": aggregate, "points": points},
    )


# ----------------------------------------------------------------------------------------
# MELD-Na -- Kim et al., NEJM 2008;359:1018-1026; OPTN/UNOS allocation policy.
#   MELD(i) = round(10 * [0.957*ln(Cr) + 0.378*ln(bili) + 1.120*ln(INR) + 0.643])
#   labs floored at 1.0; Cr capped at 4.0 (also if dialysis >=2x/week)
#   if MELD(i) > 11:  MELD-Na = MELD(i) + 1.32*(137-Na) - [0.033*MELD(i)*(137-Na)]
#   Na clamped to [125, 137]; final rounded and bounded to [6, 40].
# ----------------------------------------------------------------------------------------
_MELD_MAX = 40.0


def meld_na_scalar(labs: dict[str, object]) -> GovernanceScalar:
    """Compute the MELD-Na score (OPTN policy) as a metric-only scalar in ``[0, 1]``.

    Required ``labs`` keys: ``bilirubin_mg_dl``, ``creatinine_mg_dl``, ``inr``,
    ``sodium_meq_l``; optional ``dialysis_twice_in_week`` (bool).  Two of these
    (bilirubin, creatinine) already flow through the SOFA clinical channel; INR and sodium
    are ordinary members of the same ``patient_data`` channel but are not yet consumed by
    another engine component, so this family most often abstains UNAVAILABLE today.

    Args:
        labs: Mapping of the four MELD-Na labs (+ optional dialysis flag).

    Returns:
        A scalar equal to ``meld_na / 40`` when all four labs are present and numeric, else
        an UNAVAILABLE abstention recording the missing labs.
    """
    missing = tuple(key for key in _MELD_PARAMS if key not in labs or labs[key] is None)
    if missing:
        return unavailable(
            "omni_meld_na",
            family=_MELD_FAMILY,
            reason=f"missing MELD-Na lab(s): {list(missing)}",
            missing_inputs=missing,
        )

    bili = _as_number(labs["bilirubin_mg_dl"])
    creat = _as_number(labs["creatinine_mg_dl"])
    inr = _as_number(labs["inr"])
    sodium = _as_number(labs["sodium_meq_l"])
    bad = tuple(
        p
        for p, v in (
            ("bilirubin_mg_dl", bili),
            ("creatinine_mg_dl", creat),
            ("inr", inr),
            ("sodium_meq_l", sodium),
        )
        if v is None
    )
    if bad:
        return unavailable(
            "omni_meld_na",
            family=_MELD_FAMILY,
            reason="MELD-Na lab(s) not numeric",
            missing_inputs=bad,
        )
    assert bili is not None and creat is not None and inr is not None and sodium is not None

    # OPTN clamps.
    bili_c = max(1.0, bili)
    inr_c = max(1.0, inr)
    creat_c = max(1.0, creat)
    if bool(labs.get("dialysis_twice_in_week")):
        creat_c = 4.0
    creat_c = min(4.0, creat_c)

    raw = 0.957 * math.log(creat_c) + 0.378 * math.log(bili_c) + 1.120 * math.log(inr_c) + 0.643
    meld_i = round(raw * 10.0)

    if meld_i > 11:
        sodium_c = max(125.0, min(137.0, sodium))
        meld_na = meld_i + 1.32 * (137.0 - sodium_c) - (0.033 * meld_i * (137.0 - sodium_c))
    else:
        meld_na = float(meld_i)

    final = max(6, min(40, round(meld_na)))
    return grounded(
        "omni_meld_na",
        final / _MELD_MAX,
        family=_MELD_FAMILY,
        reason=f"MELD-Na = {final} (MELD {meld_i})",
        provenance={"meld": meld_i, "meld_na": final},
    )
