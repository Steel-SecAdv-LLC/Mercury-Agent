# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the detector-tier config + explicit NaN/Inf policy.

Covers :class:`DetectionConfig` resolution (defaults / env / config file /
overrides precedence) and the four NaN policies (``neutral`` / ``impute`` /
``flag`` / ``raise``) applied by :func:`apply_nan_policy` and
:func:`guard_finite_scalar`, plus property-based invariants on the guards.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from omni_mercury_engine.detectors.detection_config import (
    DEFAULT_MAX_MAGNITUDE,
    DEFAULT_RIDGE_FACTOR,
    DetectionConfig,
    NaNPolicy,
    NonFinitePolicyError,
    apply_nan_policy,
    guard_finite_scalar,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestNaNPolicyEnum:
    def test_coerce_member_and_string(self) -> None:
        assert NaNPolicy.coerce("neutral") is NaNPolicy.NEUTRAL
        assert NaNPolicy.coerce("IMPUTE") is NaNPolicy.IMPUTE
        assert NaNPolicy.coerce(NaNPolicy.RAISE) is NaNPolicy.RAISE

    def test_coerce_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown NaN policy"):
            NaNPolicy.coerce("propagate")


class TestConfigResolution:
    def test_defaults(self) -> None:
        cfg = DetectionConfig()
        assert cfg.nan_policy is NaNPolicy.NEUTRAL
        assert cfg.max_magnitude == DEFAULT_MAX_MAGNITUDE
        assert cfg.ridge_factor == DEFAULT_RIDGE_FACTOR
        assert cfg.ensemble_calibration == "rank"
        assert cfg.ensemble_warmup is None

    def test_invalid_fields_raise(self) -> None:
        with pytest.raises(ValueError):
            DetectionConfig(max_magnitude=-1.0)
        with pytest.raises(ValueError):
            DetectionConfig(max_magnitude=float("inf"))
        with pytest.raises(ValueError):
            DetectionConfig(ridge_factor=-1.0)
        with pytest.raises(ValueError):
            DetectionConfig(ensemble_calibration="bogus")
        with pytest.raises(ValueError):
            DetectionConfig(ensemble_warmup=1.5)  # fraction out of (0, 1]
        with pytest.raises(ValueError):
            DetectionConfig(ensemble_warmup=1)  # count < 2
        with pytest.raises(ValueError):
            DetectionConfig(ensemble_warmup=True)  # bool rejected

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMNI_DETECTOR_NAN_POLICY", "impute")
        monkeypatch.setenv("OMNI_DETECTOR_MAX_MAGNITUDE", "1000.0")
        monkeypatch.setenv("OMNI_DETECTOR_RIDGE_FACTOR", "1e-3")
        monkeypatch.setenv("OMNI_ENSEMBLE_CALIBRATION", "isotonic")
        monkeypatch.setenv("OMNI_ENSEMBLE_WARMUP", "256")
        cfg = DetectionConfig.resolve()
        assert cfg.nan_policy is NaNPolicy.IMPUTE
        assert cfg.max_magnitude == 1000.0
        assert cfg.ridge_factor == 1e-3
        assert cfg.ensemble_calibration == "isotonic"
        assert cfg.ensemble_warmup == 256

    def test_env_warmup_fraction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMNI_ENSEMBLE_WARMUP", "0.25")
        assert DetectionConfig.resolve().ensemble_warmup == 0.25

    def test_bad_env_float_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMNI_DETECTOR_MAX_MAGNITUDE", "not-a-number")
        assert DetectionConfig.resolve().max_magnitude == DEFAULT_MAX_MAGNITUDE

    def test_overrides_win_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMNI_DETECTOR_NAN_POLICY", "impute")
        cfg = DetectionConfig.resolve({"nan_policy": "flag", "ridge_factor": 5e-4})
        assert cfg.nan_policy is NaNPolicy.FLAG
        assert cfg.ridge_factor == 5e-4

    def test_config_file_layering(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "detection.json"
        path.write_text(json.dumps({"nan_policy": "raise", "max_magnitude": 42.0}))
        # File sets the values; env then overrides just the policy.
        monkeypatch.setenv("OMNI_DETECTOR_NAN_POLICY", "impute")
        cfg = DetectionConfig.resolve(config_path=str(path))
        assert cfg.nan_policy is NaNPolicy.IMPUTE  # env wins over file
        assert cfg.max_magnitude == 42.0  # file wins over default

    def test_config_file_detection_section(self, tmp_path: Path) -> None:
        path = tmp_path / "detection.yaml"
        path.write_text("detection:\n  nan_policy: flag\n  ridge_factor: 0.01\n")
        cfg = DetectionConfig.from_file(str(path))
        assert cfg.nan_policy is NaNPolicy.FLAG
        assert cfg.ridge_factor == 0.01

    def test_invalid_warmup_type_raises(self) -> None:
        with pytest.raises(ValueError, match="ensemble_warmup must be"):
            DetectionConfig(ensemble_warmup="lots")  # type: ignore[arg-type]

    def test_bad_warmup_env_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMNI_ENSEMBLE_WARMUP", "not-a-number")
        assert DetectionConfig.resolve().ensemble_warmup is None

    def test_missing_config_file_uses_defaults(self, tmp_path: Path) -> None:
        cfg = DetectionConfig.from_file(str(tmp_path / "does-not-exist.yaml"))
        assert cfg == DetectionConfig()

    def test_config_file_non_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="must contain a mapping"):
            DetectionConfig.from_file(str(path))

    def test_merge_no_recognised_keys_is_noop(self) -> None:
        cfg = DetectionConfig()
        assert cfg.merge({"unrelated": 1}) is cfg
        assert cfg.merge(None) is cfg

    def test_to_dict_roundtrips(self) -> None:
        # ``nan_policy`` accepts a raw string (coerced in ``__post_init__``); the
        # ``type: ignore`` documents that we are intentionally exercising that
        # string path, mirroring ``test_invalid_warmup_type_raises`` above.
        cfg = DetectionConfig(nan_policy="impute", ensemble_warmup=100)  # type: ignore[arg-type]
        d = cfg.to_dict()
        assert d["nan_policy"] == "impute"
        assert d["ensemble_warmup"] == 100
        assert json.loads(json.dumps(d))["nan_policy"] == "impute"


class TestApplyNaNPolicy:
    def _arr(self) -> np.ndarray:
        return np.array([1.0, np.nan, np.inf, -np.inf, 2.0])

    def test_neutral_replaces_and_clamps(self) -> None:
        out, flags = apply_nan_policy(
            self._arr(), policy="neutral", detector="d", field="input", max_magnitude=1e6
        )
        assert np.all(np.isfinite(out))
        assert out[1] == 0.0  # NaN -> neutral 0.0
        assert out[2] == 1e6 and out[3] == -1e6  # inf -> +-max_magnitude
        assert flags.tolist() == [False, True, True, True, False]

    def test_impute_uses_median(self) -> None:
        out, _ = apply_nan_policy(self._arr(), policy="impute", detector="d", max_magnitude=1e6)
        # median of finite {1, 2} = 1.5
        assert out[1] == pytest.approx(1.5)
        assert out[2] == 1e6 and out[3] == -1e6

    def test_impute_all_nonfinite_falls_back_to_neutral(self) -> None:
        out, _ = apply_nan_policy(
            np.array([np.nan, np.nan]),
            policy="impute",
            detector="d",
            neutral_value=0.0,
            max_magnitude=1e6,
        )
        assert np.all(out == 0.0)

    def test_flag_returns_mask(self) -> None:
        out, flags = apply_nan_policy(self._arr(), policy="flag", detector="d", max_magnitude=1e6)
        assert flags.sum() == 3
        assert np.all(np.isfinite(out))

    def test_raise_policy(self) -> None:
        with pytest.raises(NonFinitePolicyError, match="non-finite"):
            apply_nan_policy(self._arr(), policy="raise", detector="d")

    def test_clean_input_only_clamps(self) -> None:
        out, flags = apply_nan_policy(
            np.array([1.0, 5e9, -5e9]), policy="neutral", detector="d", max_magnitude=1e6
        )
        assert not flags.any()
        assert out[1] == 1e6 and out[2] == -1e6  # huge finite clamped to the regime

    def test_2d_shape_preserved(self) -> None:
        arr = np.array([[1.0, np.nan], [np.inf, 2.0]])
        out, flags = apply_nan_policy(arr, policy="neutral", detector="d", max_magnitude=1e6)
        assert out.shape == (2, 2)
        assert flags.shape == (2, 2)
        assert np.all(np.isfinite(out))


class TestGuardFiniteScalar:
    def test_finite_passthrough_with_clamp(self) -> None:
        assert guard_finite_scalar(3.0, detector="d", field="z_q", max_magnitude=1e6) == 3.0
        assert guard_finite_scalar(5e9, detector="d", field="z_q", max_magnitude=1e6) == 1e6

    def test_nan_to_neutral(self) -> None:
        assert guard_finite_scalar(np.nan, detector="d", field="gamma", max_magnitude=1e6) == 0.0

    def test_inf_to_magnitude(self) -> None:
        assert guard_finite_scalar(np.inf, detector="d", field="z_q", max_magnitude=1e6) == 1e6
        assert guard_finite_scalar(-np.inf, detector="d", field="z_q", max_magnitude=1e6) == -1e6

    def test_raise_policy(self) -> None:
        with pytest.raises(NonFinitePolicyError):
            guard_finite_scalar(np.nan, policy="raise", detector="d", field="z_q")


# ---------------------------------------------------------------------------
# Property-based invariants (Hypothesis)
# ---------------------------------------------------------------------------
hyp = pytest.importorskip("hypothesis")
from hypothesis import (
    given,
    settings,
    strategies as st,
)
from hypothesis.extra import numpy as hnp

_any_float = st.floats(allow_nan=True, allow_infinity=True, width=64)


class TestGuardProperties:
    @settings(max_examples=200, deadline=None)
    @given(
        hnp.arrays(
            dtype=np.float64,
            shape=hnp.array_shapes(min_dims=1, max_dims=2, max_side=20),
            elements=_any_float,
        ),
        st.sampled_from(["neutral", "impute", "flag"]),
    )
    def test_output_always_finite_and_bounded(self, arr: np.ndarray, policy: str) -> None:
        """For any array and non-raising policy, output is finite and within the regime."""
        max_mag = 1e6
        out, flags = apply_nan_policy(arr, policy=policy, detector="p", max_magnitude=max_mag)
        assert np.all(np.isfinite(out)), "output must be finite"
        assert np.all(np.abs(out) <= max_mag + 1e-6), "output must be within +-max_magnitude"
        assert out.shape == arr.shape
        assert flags.shape == arr.shape
        # flags mark exactly the non-finite input positions
        assert np.array_equal(flags, ~np.isfinite(arr))

    @settings(max_examples=200, deadline=None)
    @given(_any_float)
    def test_scalar_guard_always_finite_bounded(self, value: float) -> None:
        for policy in ("neutral", "impute", "flag"):
            out = guard_finite_scalar(value, policy=policy, detector="p", max_magnitude=1e6)
            assert np.isfinite(out)
            assert abs(out) <= 1e6 + 1e-6

    @settings(max_examples=100, deadline=None)
    @given(
        hnp.arrays(
            dtype=np.float64,
            shape=st.integers(1, 30),
            elements=st.floats(allow_nan=True, allow_infinity=True),
        )
    )
    def test_raise_iff_nonfinite_present(self, arr: np.ndarray) -> None:
        has_nonfinite = bool((~np.isfinite(arr)).any())
        if has_nonfinite:
            with pytest.raises(NonFinitePolicyError):
                apply_nan_policy(arr, policy="raise", detector="p")
        else:
            out, _ = apply_nan_policy(arr, policy="raise", detector="p", max_magnitude=1e300)
            assert np.all(np.isfinite(out))
