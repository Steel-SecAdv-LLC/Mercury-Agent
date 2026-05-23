"""Rule-vs-citation pin harness.

Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

Task 6 (PR 2 refinement): every FDA / ADA / NWS / ASA threshold that is
cited in a docstring must equal the module-level constant the citation
points to.  This is a lint, not a semantics check: it catches regressions
where someone flips ``<`` to ``<=`` or rounds ``70.0`` to ``70`` while the
docstring still cites the original number.  Failures surface the citation
URL alongside the failing assertion so a reviewer can verify against the
source document.

Each row is rendered as a single line in the test session output so the
live pin-table is visible during ``pytest -v``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import pytest

# Both ``omni_mercury_engine.medical.anesthesiology_predictor`` and
# ``omni_mercury_engine.medical.endocrinology_detector`` import
# ``torch`` at module level; skip cleanly at collection time when
# torch is not installed so the rest of the suite is still
# discoverable in CI images without the optional ``ml`` extra.
pytest.importorskip("torch")

from omni_mercury_engine.compliance import osha_anomaly
from omni_mercury_engine.medical import anesthesiology_predictor, endocrinology_detector

if TYPE_CHECKING:
    from collections.abc import Callable

FDA_AFREZZA_LABEL_URL: Final[str] = (
    "https://www.accessdata.fda.gov/drugsatfda_docs/label/2014/022472lbl.pdf"
)
FDA_INSULIN_LABEL_URL: Final[str] = (
    "https://www.fda.gov/drugs/postmarket-drug-safety-information-patients-and-providers/insulin"
)
FDA_GLP1_BLACK_BOX_URL: Final[str] = (
    "https://www.fda.gov/safety/medical-product-safety-information/"
    "glp-1-receptor-agonists-drug-safety-communications"
)
ADA_STANDARDS_OF_CARE_URL: Final[str] = "https://diabetesjournals.org/care/issue/47/Supplement_1"
AARC_GUIDANCE_URL: Final[str] = (
    "https://www.aarc.org/wp-content/uploads/2014/08/clinical_practice_guideline.pdf"
)
ASA_GUIDANCE_URL: Final[str] = (
    "https://www.asahq.org/standards-and-practice-parameters/"
    "standards-for-basic-anesthetic-monitoring"
)
NWS_HEAT_INDEX_URL: Final[str] = "https://www.wpc.ncep.noaa.gov/html/heatindex.shtml"


@dataclass(frozen=True)
class _ClinicalRulePin:
    module: str
    rule_name: str
    citation_url: str
    asserted_constant_name: str
    asserted_operator: str
    asserted_value: object


_OPERATORS: Final[dict[str, Callable[[Any, Any], bool]]] = {
    "==": operator.eq,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "contains": lambda lhs, rhs: rhs in lhs,
}


CLINICAL_RULE_PINS: Final[tuple[_ClinicalRulePin, ...]] = (
    _ClinicalRulePin(
        "endocrinology_detector",
        "Afrezza_FEV1_contraindication",
        FDA_AFREZZA_LABEL_URL,
        "InhaledInsulinMonitor.FEV1_THRESHOLD",
        "==",
        70.0,
    ),
    _ClinicalRulePin(
        "endocrinology_detector",
        "SmartPen_dose_stack_window_hours",
        ADA_STANDARDS_OF_CARE_URL,
        "SmartInsulinPenMonitor.DOSE_STACK_WINDOW_HOURS",
        "==",
        2.0,
    ),
    _ClinicalRulePin(
        "endocrinology_detector",
        "SmartPen_max_bolus_units",
        FDA_INSULIN_LABEL_URL,
        "SmartInsulinPenMonitor.MAX_BOLUS_UNITS",
        "==",
        15.0,
    ),
    _ClinicalRulePin(
        "endocrinology_detector",
        "SmartPen_max_daily_insulin_units",
        ADA_STANDARDS_OF_CARE_URL,
        "SmartInsulinPenMonitor.MAX_DAILY_INSULIN_UNITS",
        "==",
        50.0,
    ),
    _ClinicalRulePin(
        "endocrinology_detector",
        "Inhaled_max_dose_units",
        FDA_AFREZZA_LABEL_URL,
        "InhaledInsulinMonitor.MAX_DOSE_UNITS",
        "==",
        12,
    ),
    _ClinicalRulePin(
        "endocrinology_detector",
        "Inhaled_min_technique_score",
        AARC_GUIDANCE_URL,
        "InhaledInsulinMonitor.MIN_TECHNIQUE_SCORE",
        "==",
        0.7,
    ),
    _ClinicalRulePin(
        "endocrinology_detector",
        "GLP1_pancreatitis_substring",
        FDA_GLP1_BLACK_BOX_URL,
        "(substring check in side_effects)",
        "contains",
        "pancreatitis",
    ),
    _ClinicalRulePin(
        "anesthesiology_predictor",
        "Hypoxemia_threshold_spo2_pct",
        ASA_GUIDANCE_URL,
        "HemodynamicMonitor.spo2_threshold",
        "==",
        92.0,
    ),
    _ClinicalRulePin(
        "osha_anomaly",
        "Heat_index_severity_floor_F",
        NWS_HEAT_INDEX_URL,
        "(_detect_agriculture_hazards heat-index gate)",
        "==",
        103.0,
    ),
)


_MODULES: Final[dict[str, Any]] = {
    "endocrinology_detector": endocrinology_detector,
    "anesthesiology_predictor": anesthesiology_predictor,
    "osha_anomaly": osha_anomaly,
}


def _resolve_pin_value(pin: _ClinicalRulePin) -> Any:
    """Resolve the actual runtime value referenced by ``asserted_constant_name``.

    Plain dotted lookups (``Class.ATTR``) are resolved via ``getattr``.
    The two special-cased rules whose names are wrapped in parentheses
    are resolved by inspecting the relevant source.
    """
    import inspect

    module = _MODULES[pin.module]
    name = pin.asserted_constant_name

    if name == "(substring check in side_effects)":
        src = inspect.getsource(endocrinology_detector.GLP1TherapyMonitor.monitor_glp1_therapy)
        return src

    if name == "(_detect_agriculture_hazards heat-index gate)":
        src = inspect.getsource(module.OSHAComplianceDetector._detect_agriculture_hazards)
        marker = "heat_index > "
        idx = src.find(marker)
        if idx == -1:
            raise AssertionError(
                f"Could not locate '{marker}' in _detect_agriculture_hazards "
                f"(citation: {pin.citation_url})"
            )
        rest = src[idx + len(marker) :]
        number_str = ""
        for ch in rest:
            if ch.isdigit() or ch == ".":
                number_str += ch
            else:
                break
        if not number_str:
            raise AssertionError(
                f"Could not parse heat-index threshold after marker '{marker}' "
                f"(citation: {pin.citation_url})"
            )
        return float(number_str)

    head, *attrs = name.split(".")
    obj: Any = getattr(module, head)
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


@pytest.mark.parametrize("pin", CLINICAL_RULE_PINS, ids=lambda p: f"{p.module}::{p.rule_name}")
def test_clinical_rule_pin(pin: _ClinicalRulePin) -> None:
    """Each cited threshold equals the module-level constant the citation points to."""
    op = _OPERATORS[pin.asserted_operator]

    if pin.asserted_constant_name == "(substring check in side_effects)":
        # Special case: assert the substring is actually used by the rule.
        consts = _resolve_pin_value(pin)
        assert pin.asserted_value in consts, (
            f"GLP-1 pancreatitis substring check missing.\n"
            f"  module    : {pin.module}\n"
            f"  rule      : {pin.rule_name}\n"
            f"  citation  : {pin.citation_url}\n"
            f"  required  : {pin.asserted_value!r}\n"
            f"  observed  : substring not found in monitor_glp1_therapy()."
        )
        return

    actual = _resolve_pin_value(pin)
    assert op(actual, pin.asserted_value), (
        f"Clinical rule pin drift detected.\n"
        f"  module      : {pin.module}\n"
        f"  rule        : {pin.rule_name}\n"
        f"  constant    : {pin.asserted_constant_name}\n"
        f"  citation    : {pin.citation_url}\n"
        f"  required    : {pin.asserted_operator} {pin.asserted_value!r}\n"
        f"  observed    : {actual!r}"
    )


def test_clinical_rule_pin_table_is_complete(capsys: pytest.CaptureFixture[str]) -> None:
    """Print the rule-vs-citation pin table so it is visible during ``pytest -v``."""
    lines = ["clinical_rule_pins:"]
    for pin in CLINICAL_RULE_PINS:
        lines.append(
            f"  {pin.module:>26s} :: {pin.rule_name:<40s} "
            f"{pin.asserted_constant_name:<58s} {pin.asserted_operator:>8s} "
            f"{pin.asserted_value!r:<14} [{pin.citation_url}]"
        )
    rendered = "\n".join(lines)
    print(rendered)
    captured = capsys.readouterr()
    assert "clinical_rule_pins:" in captured.out
    assert len(CLINICAL_RULE_PINS) == 9
