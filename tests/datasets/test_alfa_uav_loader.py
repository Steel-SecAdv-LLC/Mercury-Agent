# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ALFA UAV loader parsers: topic CSVs, zero-order hold, ground-truth labels.

Fixtures mirror the real ALFA schema exactly (``%time`` in nanoseconds,
``field.*`` columns, failure_status codes published only while the fault is
active) so the unit contract matches the recorded corpus the loader parses.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.datasets.uav_flight import (
    _TOPIC_COLUMNS,
    ALFAUAVLoader,
    build_sequence,
    parse_topic_csv,
    wrap_angle_deg,
    zoh_resample,
)

_NS = 1_000_000_000


def _csv(header: list[str], rows: list[list[object]]) -> str:
    lines = [",".join(header)]
    lines += [",".join(str(v) for v in row) for row in rows]
    return "\n".join(lines) + "\n"


def _two_col_topic(t0_ns: int, n: int, step_ns: int, value: float = 1.0) -> str:
    header = ["%time", "field.commanded", "field.measured"]
    rows: list[list[object]] = [[t0_ns + i * step_ns, value, value + 0.5] for i in range(n)]
    return _csv(header, rows)


class TestParseTopicCSV:
    def test_parses_times_and_selected_columns(self) -> None:
        text = _csv(
            ["%time", "field.header.seq", "field.commanded", "field.measured"],
            [[5 * _NS, 1, 16.0, 16.8], [6 * _NS, 2, 17.0, 17.1]],
        )
        times, values = parse_topic_csv(text, ("field.commanded", "field.measured"))
        np.testing.assert_allclose(times, [5.0, 6.0])
        np.testing.assert_allclose(values, [[16.0, 16.8], [17.0, 17.1]])

    def test_sorts_out_of_order_rows(self) -> None:
        text = _csv(
            ["%time", "field.data"],
            [[6 * _NS, 2.0], [5 * _NS, 1.0]],
        )
        times, values = parse_topic_csv(text, ("field.data",))
        np.testing.assert_allclose(times, [5.0, 6.0])
        np.testing.assert_allclose(values.ravel(), [1.0, 2.0])

    def test_drops_non_numeric_and_non_finite_rows(self) -> None:
        text = _csv(
            ["%time", "field.data"],
            [[5 * _NS, 1.0], [6 * _NS, "nan"], [7 * _NS, ""], [8 * _NS, 2.0]],
        )
        times, _ = parse_topic_csv(text, ("field.data",))
        np.testing.assert_allclose(times, [5.0, 8.0])

    def test_missing_column_raises(self) -> None:
        with pytest.raises(ValueError, match="missing a requested column"):
            parse_topic_csv(_csv(["%time", "field.data"], []), ("field.absent",))

    def test_wrong_first_column_raises(self) -> None:
        with pytest.raises(ValueError, match="%time"):
            parse_topic_csv(_csv(["time", "field.data"], []), ("field.data",))


class TestZohResample:
    def test_holds_last_sample_within_staleness(self) -> None:
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([[10.0], [20.0], [30.0]])
        grid = np.array([0.5, 1.9, 2.4])
        held, valid = zoh_resample(times, values, grid, max_staleness_s=1.0)
        assert valid.all()
        np.testing.assert_allclose(held.ravel(), [10.0, 20.0, 30.0])

    def test_stale_and_pre_first_ticks_invalid(self) -> None:
        times = np.array([1.0])
        values = np.array([[10.0]])
        grid = np.array([0.5, 1.2, 5.0])
        _, valid = zoh_resample(times, values, grid, max_staleness_s=1.0)
        assert valid.tolist() == [False, True, False]


class TestWrapAngle:
    def test_wraps_into_half_open_interval(self) -> None:
        np.testing.assert_allclose(
            wrap_angle_deg(np.array([0.0, 190.0, -190.0, 180.0, 360.0])),
            [0.0, -170.0, 170.0, 180.0, 0.0],
        )


def _full_topic_texts(t0_ns: int, n: int, step_ns: int) -> dict[str, str]:
    """All required topics with plausible, in-sync samples."""
    texts: dict[str, str] = {}
    for topic, columns in _TOPIC_COLUMNS.items():
        header = ["%time", *columns]
        rows: list[list[object]] = [
            [t0_ns + i * step_ns, *(float(10 + j) for j in range(len(columns)))] for i in range(n)
        ]
        texts[topic] = _csv(header, rows)
    return texts


class TestBuildSequence:
    def test_labels_start_at_first_failure_status_stamp(self) -> None:
        t0 = 100 * _NS
        topics = _full_topic_texts(t0, n=120, step_ns=_NS // 4)
        inj_ns = t0 + 15 * _NS
        failure = _csv(
            ["%time", "field.data"],
            [[inj_ns + i * _NS // 4, 1] for i in range(20)],
        )
        seq = build_sequence("f1", topics, {"engines": failure}, sample_hz=4.0, max_staleness_s=1.5)
        assert seq is not None
        assert seq.fault_surfaces == ("engines",)
        inj_s = inj_ns / _NS
        np.testing.assert_array_equal(seq.labels, (seq.times_s >= inj_s).astype(int))
        assert 0 < seq.labels.sum() < len(seq.labels)

    def test_zero_status_rows_do_not_mark_faults(self) -> None:
        t0 = 100 * _NS
        topics = _full_topic_texts(t0, n=80, step_ns=_NS // 4)
        failure = _csv(["%time", "field.data"], [[t0 + 5 * _NS, 0]])
        seq = build_sequence("f2", topics, {"rudder": failure}, sample_hz=4.0, max_staleness_s=1.5)
        assert seq is not None
        assert seq.fault_surfaces == ()
        assert seq.labels.sum() == 0

    def test_nonzero_code_two_counts_as_fault(self) -> None:
        """ALFA encodes the zero-deflection aileron variant as code 2."""
        t0 = 100 * _NS
        topics = _full_topic_texts(t0, n=80, step_ns=_NS // 4)
        failure = _csv(["%time", "field.data"], [[t0 + 10 * _NS, 2]])
        seq = build_sequence("f3", topics, {"aileron": failure}, sample_hz=4.0, max_staleness_s=1.5)
        assert seq is not None
        assert seq.fault_surfaces == ("aileron",)
        assert seq.labels.sum() > 0

    def test_missing_topic_returns_none(self) -> None:
        topics = _full_topic_texts(100 * _NS, n=40, step_ns=_NS // 4)
        topics.pop("mavros-vfr_hud")
        assert build_sequence("f4", topics, {}, sample_hz=4.0, max_staleness_s=1.5) is None

    def test_feature_width_matches_loader_contract(self) -> None:
        topics = _full_topic_texts(100 * _NS, n=40, step_ns=_NS // 4)
        seq = build_sequence("f5", topics, {}, sample_hz=4.0, max_staleness_s=1.5)
        assert seq is not None
        assert seq.features.shape[1] == len(ALFAUAVLoader.FEATURE_NAMES)
        assert np.isfinite(seq.features).all()
