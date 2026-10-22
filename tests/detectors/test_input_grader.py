# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The partial-input contract's shared machinery.

If anything usable is fed, a detector runs and returns a result; it raises
only when nothing usable is fed.  Whatever could not be graded is set aside
with its reason and the verdict is marked partial, never a silent all-clear.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.detectors._hazard_input import InputGrader, InvalidObservationError

SCHEMA = {
    "rainfall_data": {
        "intensity_mm_hr": ("mm/h", 0.0, 1000.0),
        "duration_hours": ("h", 0.0, 2160.0),
    },
    "seismic_data": {"pga_g": ("g", 0.0, 10.0)},
}


class TestInputGrader:
    def test_fully_usable_record_is_not_partial(self) -> None:
        grader = InputGrader()
        graded = grader.grade_sections(
            {"rainfall_data": {"intensity_mm_hr": 5.0, "duration_hours": 2.0}}, SCHEMA
        )
        assert graded == {"rainfall_data": {"intensity_mm_hr": 5.0, "duration_hours": 2.0}}
        assert grader.partial is False
        assert grader.result_fields() == {}

    def test_misspelt_key_is_set_aside_not_refused(self) -> None:
        grader = InputGrader()
        graded = grader.grade_sections(
            {"rainfall_data": {"intensityy_mm_hr": 5.0, "duration_hours": 2.0}}, SCHEMA
        )
        assert graded == {"rainfall_data": {"duration_hours": 2.0}}
        assert grader.partial is True
        (entry,) = grader.set_aside
        assert entry["where"] == "rainfall_data.intensityy_mm_hr"
        assert "unknown key" in entry["reason"]

    def test_unknown_section_is_set_aside(self) -> None:
        grader = InputGrader()
        graded = grader.grade_sections(
            {"rainfal_data": {"intensity_mm_hr": 5.0}, "seismic_data": {"pga_g": 0.2}}, SCHEMA
        )
        assert graded == {"seismic_data": {"pga_g": 0.2}}
        assert grader.set_aside[0]["where"] == "rainfal_data"

    def test_one_bad_value_does_not_reject_the_rest(self) -> None:
        grader = InputGrader()
        graded = grader.grade_sections(
            {
                "rainfall_data": {"intensity_mm_hr": float("nan"), "duration_hours": 2.0},
                "seismic_data": {"pga_g": 0.3},
            },
            SCHEMA,
        )
        assert graded == {"rainfall_data": {"duration_hours": 2.0}, "seismic_data": {"pga_g": 0.3}}
        assert grader.partial is True
        assert any("finite" in e["reason"] for e in grader.set_aside)

    def test_out_of_range_value_is_set_aside_with_unit_hint(self) -> None:
        grader = InputGrader()
        grader.grade_sections({"seismic_data": {"pga_g": 500.0}}, SCHEMA)
        value_entry, section_entry = grader.set_aside
        assert "physical range" in value_entry["reason"]
        assert section_entry["reason"] == "no usable measurement in this section"

    def test_empty_section_is_set_aside(self) -> None:
        grader = InputGrader()
        graded = grader.grade_sections(
            {"rainfall_data": {}, "seismic_data": {"pga_g": 0.1}}, SCHEMA
        )
        assert graded == {"seismic_data": {"pga_g": 0.1}}
        assert grader.set_aside[0]["reason"] == "empty section carries no measurement"

    def test_nothing_usable_raises_with_the_reasons(self) -> None:
        grader = InputGrader()
        graded = grader.grade_sections({"rainfall_data": {"x": 1.0}}, SCHEMA)
        with pytest.raises(InvalidObservationError, match=r"nothing usable.*unknown key"):
            grader.require_usable(graded, "at least one monitoring section")

    def test_result_fields_mark_the_verdict_partial(self) -> None:
        grader = InputGrader()
        grader.discard("a.b", "test reason", value="x" * 200)
        fields = grader.result_fields()
        assert fields["partial"] is True
        assert fields["set_aside"][0]["where"] == "a.b"
        assert len(fields["set_aside"][0]["value"]) == 80
