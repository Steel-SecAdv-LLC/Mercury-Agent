# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the hazard checkpoint-training pipeline (T5).

Covers the registry audit, the merit gate, temporal-split anti-leakage,
the shipped solar-storm checkpoint (real OMNI2-trained artifact with
provenance), and the detector's default-load path. No network: the shipped
artifact and its sidecar are committed; pipeline-stage tests run against
them, not against live archives (the weekly network lane re-runs fetch).
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

torch = pytest.importorskip("torch")

from omni_mercury_engine.ml.hazard_training.common import (
    EvaluationOutcome,
    HazardDataUnavailableError,
    TemporalSplit,
)
from omni_mercury_engine.ml.hazard_training.registry import (
    HOOK_REGISTRY,
    get_hook,
    run_stage,
)
from omni_mercury_engine.models.checkpoint_paths import (
    load_shipped_checkpoint,
    shipped_checkpoint_path,
)


class TestRegistryAudit:
    def test_all_eleven_hooks_present(self) -> None:
        assert len(HOOK_REGISTRY) == 11

    def test_every_hook_has_category_and_requirement(self) -> None:
        for entry in HOOK_REGISTRY.values():
            assert entry.category in ("a", "b", "c")
            assert len(entry.data_requirement) > 40, entry.name
            assert entry.detector and entry.architecture

    def test_category_a_hooks_have_pipelines(self) -> None:
        for entry in HOOK_REGISTRY.values():
            if entry.category == "a":
                assert entry.pipeline_module is not None
                assert entry.checkpoint_name is not None
                entry.load_pipeline()  # imports for real

    def test_category_bc_hooks_fail_loud_with_requirement(self) -> None:
        from omni_mercury_engine.ml.hazard_training.common import PipelineContext

        for entry in HOOK_REGISTRY.values():
            if entry.category in ("b", "c"):
                with pytest.raises(HazardDataUnavailableError) as exc:
                    run_stage(entry.name, "train", PipelineContext())
                # The loud error must carry the actionable requirement.
                assert entry.data_requirement[:40] in str(exc.value)

    def test_unknown_hook_and_stage_rejected(self) -> None:
        from omni_mercury_engine.ml.hazard_training.common import PipelineContext

        with pytest.raises(KeyError, match="valid hooks"):
            get_hook("nope")
        with pytest.raises(ValueError, match="valid stages"):
            run_stage("solar_storm", "deploy", PipelineContext())


class TestTemporalSplit:
    def test_ordering_enforced(self) -> None:
        with pytest.raises(ValueError, match="temporal split violated"):
            TemporalSplit(train_years=(2020,), val_years=(2019,), test_years=(2021,))

    def test_masks_are_disjoint(self) -> None:
        split = TemporalSplit(train_years=(2005, 2006), val_years=(2007,), test_years=(2008,))
        years = np.array([2005, 2006, 2007, 2008, 2008])
        train, val, test = split.masks(years)
        assert not np.any(train & val) and not np.any(val & test) and not np.any(train & test)
        assert int(test.sum()) == 2


class TestMeritGate:
    def _outcome(self, learned: float, physics: float) -> EvaluationOutcome:
        return EvaluationOutcome(
            hook="solar_storm_geomag",
            primary_metric="kp_mae",
            higher_is_better=False,
            learned={"kp_mae": learned},
            physics={"kp_mae": physics},
            n_test_samples=100,
            test_years=(2022,),
        )

    def test_lower_is_better_direction(self) -> None:
        assert self._outcome(0.5, 1.0).learned_beats_physics is True
        assert self._outcome(1.0, 0.5).learned_beats_physics is False

    def test_non_finite_never_ships(self) -> None:
        assert self._outcome(float("nan"), 1.0).learned_beats_physics is False

    def _constrained_outcome(
        self, learned_recall: float, physics_recall: float
    ) -> EvaluationOutcome:
        return EvaluationOutcome(
            hook="solar_storm_geomag",
            primary_metric="kp_mae",
            higher_is_better=False,
            learned={"kp_mae": 0.5, "storm_recall_op": learned_recall},
            physics={"kp_mae": 1.0, "storm_recall_op": physics_recall},
            n_test_samples=100,
            test_years=(2022,),
            constraints=[
                {
                    "metric": "storm_recall_op",
                    "higher_is_better": True,
                    "description": "recall must not regress",
                }
            ],
        )

    def test_failed_constraint_refuses_despite_primary_win(self) -> None:
        """A primary-metric win must not ship over an operational regression.

        This is the exact failure mode of the first shipped solar-storm
        checkpoint: Kp MAE won while storm recall halved. The constraint
        layer exists so that can never ship silently again.
        """
        outcome = self._constrained_outcome(learned_recall=0.30, physics_recall=0.57)
        assert outcome.primary_metric_wins is True
        assert outcome.learned_beats_physics is False
        failed = outcome.failed_constraints
        assert len(failed) == 1 and failed[0]["metric"] == "storm_recall_op"

    def test_constraint_parity_is_allowed(self) -> None:
        """Constraints demand non-regression (>=/<=), not a strict win."""
        outcome = self._constrained_outcome(learned_recall=0.57, physics_recall=0.57)
        assert outcome.learned_beats_physics is True

    def test_non_finite_constraint_refuses(self) -> None:
        """An unmeasurable constraint must refuse, never pass silently."""
        outcome = self._constrained_outcome(learned_recall=float("nan"), physics_recall=0.57)
        assert outcome.learned_beats_physics is False

    def test_ship_checkpoint_refuses_on_failed_constraint(self, tmp_path: Path) -> None:
        from omni_mercury_engine.ml.hazard_training.common import (
            MeritGateError,
            ship_checkpoint,
        )

        out_dir = tmp_path / "shipped"
        with pytest.raises(MeritGateError, match="secondary non-regression"):
            ship_checkpoint(
                hook="solar_storm",
                checkpoint_name="solar_storm_geomag",
                data_dir=tmp_path,
                outcome=self._constrained_outcome(0.30, 0.57),
                data_sources=[],
                seed=0,
                out_dir=out_dir,
            )
        assert not out_dir.exists() or not any(out_dir.iterdir()), "refusal must not write files"

    def test_ship_checkpoint_refuses_when_physics_wins(self, tmp_path: Path) -> None:
        """The central safety property: a losing model must never ship.

        Exercises :func:`ship_checkpoint` directly (not just the boolean),
        proving the refusal raises before any file is written.
        """
        from omni_mercury_engine.ml.hazard_training.common import (
            MeritGateError,
            ship_checkpoint,
        )

        out_dir = tmp_path / "shipped"
        with pytest.raises(MeritGateError, match="MERIT GATE REFUSED"):
            ship_checkpoint(
                hook="solar_storm",
                checkpoint_name="solar_storm_geomag",
                data_dir=tmp_path,
                outcome=self._outcome(learned=1.0, physics=0.5),
                data_sources=[],
                seed=0,
                out_dir=out_dir,
            )
        assert not out_dir.exists() or not any(out_dir.iterdir()), "refusal must not write files"


class TestShippedSolarStormCheckpoint:
    """The committed artifact is real, provenanced, and loadable by the hook."""

    def test_artifact_and_sidecar_exist(self) -> None:
        path = shipped_checkpoint_path("solar_storm_geomag")
        assert path.exists(), "shipped checkpoint missing"
        sidecar = path.with_suffix(".provenance.json")
        assert sidecar.exists(), "provenance sidecar missing"

    def test_tampered_checkpoint_refuses_to_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A .pt that no longer matches its sidecar's pinned sha256 must not load."""
        import shutil

        from omni_mercury_engine.models import checkpoint_paths as cp

        src = shipped_checkpoint_path("solar_storm_geomag")
        fake_dir = tmp_path / "checkpoints"
        fake_dir.mkdir()
        tampered = fake_dir / src.name
        shutil.copy(src, tampered)
        shutil.copy(src.with_suffix(".provenance.json"), fake_dir / f"{src.stem}.provenance.json")
        with tampered.open("r+b") as fh:  # flip one byte past the header
            fh.seek(128)
            byte = fh.read(1)
            fh.seek(128)
            fh.write(bytes([byte[0] ^ 0xFF]))
        monkeypatch.setattr(cp, "checkpoints_dir", lambda: fake_dir)
        with pytest.raises(RuntimeError, match="does not match its provenance"):
            cp.load_shipped_checkpoint("solar_storm_geomag")

    def test_provenance_is_complete_and_merit_gated(self) -> None:
        payload, provenance = load_shipped_checkpoint("solar_storm_geomag")
        assert provenance is not None
        assert provenance["checkpoint_sha256"]
        sources = provenance["data_sources"]
        assert len(sources) >= 20, "20 training years of OMNI2 + cross-checks"
        assert all(src["sha256"] for src in sources)
        evaluation = provenance["evaluation"]
        assert evaluation["learned_beats_physics"] is True
        assert evaluation["learned"]["kp_mae"] < evaluation["physics"]["kp_mae"]
        assert evaluation["n_test_samples"] > 10000

    def test_dual_rule_operating_point_drives_storm_onset(self) -> None:
        """The classifier head must raise storm onset at the ratified threshold.

        With an operating point injected, a case whose regressed Kp stays
        below 5 but whose storm probability crosses tau must emit the G1
        onset level ("minor") with the dual-threshold method; without an
        operating point the same case must stay at the Kp-derived level.
        The kp_index itself must be identical in both configurations — the
        operating point changes the ALERT decision, never the estimate.
        """
        from omni_mercury_engine.space.solar_storm_detector import SolarStormDetector

        detector = SolarStormDetector(enable_flare_detection=False, enable_cme_tracking=False)
        detector.load_neural_weights()  # shipped default
        case = {
            "solar_wind_speed_km_s": 520.0,
            "bz_imf_nt": -7.5,
            "by_imf_nt": 3.0,
        }
        base = detector._predict_geomagnetic_storm(dict(case))
        # Pick tau just below the emitted confidence so the classifier rule
        # fires deterministically for this real-shaped input.
        tau = max(min(float(base["confidence"]) - 1e-6, 1.0 - 1e-9), 1e-9)
        detector._operating_point = {"storm_prob_threshold": tau}
        dual = detector._predict_geomagnetic_storm(dict(case))
        assert dual["kp_index"] == base["kp_index"], "estimate must not change"
        if base["storm_level"] == "none":
            assert dual["storm_level"] == "minor"
            assert dual["method"] == "neural_dual_threshold"
            assert dual["operating_point_triggered"] is True
        else:  # regressed Kp already at storm level: dual rule must not fire
            assert dual["storm_level"] == base["storm_level"]

    def test_operating_point_threshold_validated_on_load(self, tmp_path: Path) -> None:
        """A checkpoint carrying a nonsensical tau must refuse to load."""
        import torch as _torch

        from omni_mercury_engine.models.checkpoint_paths import shipped_checkpoint_path
        from omni_mercury_engine.space.solar_storm_detector import SolarStormDetector

        payload = _torch.load(
            shipped_checkpoint_path("solar_storm_geomag"),
            map_location="cpu",
            weights_only=True,
        )
        payload["operating_point"] = {"storm_prob_threshold": 1.5}
        bad = tmp_path / "bad_op.pt"
        _torch.save(payload, bad)
        detector = SolarStormDetector(enable_flare_detection=False, enable_cme_tracking=False)
        with pytest.raises(ValueError, match=r"not a\s+probability"):
            detector.load_neural_weights(str(bad))

    @pytest.mark.parametrize(
        "bad_op",
        [
            {"threshold": 0.5},  # missing 'storm_prob_threshold' key
            [0.5],  # not a mapping
            "0.5",  # not a mapping
            0.5,  # not a mapping
            {"storm_prob_threshold": "high"},  # unparseable threshold
        ],
    )
    def test_malformed_operating_point_raises_value_error(
        self, tmp_path: Path, bad_op: object
    ) -> None:
        """A missing key or non-mapping operating point raises a clear
        ValueError, never a raw KeyError/TypeError leaking from ``float(op[...])``.
        """
        import torch as _torch

        from omni_mercury_engine.models.checkpoint_paths import shipped_checkpoint_path
        from omni_mercury_engine.space.solar_storm_detector import SolarStormDetector

        payload = _torch.load(
            shipped_checkpoint_path("solar_storm_geomag"), map_location="cpu", weights_only=True
        )
        payload["operating_point"] = bad_op
        bad = tmp_path / "malformed_op.pt"
        _torch.save(payload, bad)
        detector = SolarStormDetector(enable_flare_detection=False, enable_cme_tracking=False)
        with pytest.raises(ValueError):
            detector.load_neural_weights(str(bad))

    def test_load_replaces_operating_point_and_feature_stats_wholesale(
        self, tmp_path: Path
    ) -> None:
        """A second load whose checkpoint omits the operating point / feature
        stats must reset them, not inherit the previous checkpoint's (a stale
        onset rule or a train/serve standardization mismatch across loads).
        """
        import torch as _torch

        from omni_mercury_engine.models.checkpoint_paths import shipped_checkpoint_path
        from omni_mercury_engine.space.solar_storm_detector import SolarStormDetector

        payload = _torch.load(
            shipped_checkpoint_path("solar_storm_geomag"), map_location="cpu", weights_only=True
        )
        detector = SolarStormDetector(enable_flare_detection=False, enable_cme_tracking=False)
        detector.load_neural_weights()  # shipped default: sets op point + feature stats
        assert detector._operating_point is not None
        assert detector._feature_mean is not None and detector._feature_std is not None
        assert detector._feature_spec is not None

        stripped = {k: v for k, v in payload.items()}
        for absent in ("operating_point", "feature_mean", "feature_std", "feature_spec"):
            stripped.pop(absent, None)
        path = tmp_path / "no_op_no_stats.pt"
        _torch.save(stripped, path)
        detector.load_neural_weights(str(path))
        assert detector._operating_point is None, "stale operating point must be cleared"
        assert detector._feature_mean is None and detector._feature_std is None
        assert detector._feature_spec is None

    def test_select_operating_point_refuses_saturated_head(self) -> None:
        """A storm head that saturates to a constant yields no threshold strictly
        inside (0, 1); selection must refuse loudly rather than record a boundary
        tau the detector's loader would later reject.
        """
        from omni_mercury_engine.ml.hazard_training.solar_storm import (
            GeomagDataset,
            _select_operating_point,
        )

        n = 40
        storm = (np.arange(n) % 2).astype(np.float32)  # mixed classes
        ds = GeomagDataset(
            features=np.zeros((n, 3), dtype=np.float32),
            kp=np.zeros(n, dtype=np.float32),
            storm=storm,
            years=np.zeros(n, dtype=np.int32),
            raw_fields=[{} for _ in range(n)],
            feature_fill={},
            feature_mean=np.zeros(3, dtype=np.float32),
            feature_std=np.ones(3, dtype=np.float32),
        )

        class _SaturatedHead:
            """Emits storm_prob == 0.0 for every row (a degenerate head)."""

            def eval(self) -> _SaturatedHead:
                return self

            def __call__(self, x: Any) -> tuple[Any, Any]:
                rows = int(x.shape[0])
                return torch.zeros(rows, 1), torch.zeros(rows, 1)

        with pytest.raises(RuntimeError, match=r"strictly inside \(0, 1\)"):
            _select_operating_point(
                _SaturatedHead(),
                ds,
                x_val=torch.zeros(n, 3),
                val_mask=np.ones(n, dtype=bool),
            )

    def test_detector_default_load_uses_shipped_checkpoint(self) -> None:
        from omni_mercury_engine.space.solar_storm_detector import SolarStormDetector

        detector = SolarStormDetector()
        detector.load_neural_weights()  # no path -> shipped default
        assert detector._neural_trained is True
        assert detector._feature_spec == "geomag-v1"
        # The learned path must produce a finite Kp from real-shaped inputs.
        result = detector.predict_solar_storm(
            {
                "magnetosphere_data": {
                    "solar_wind_speed_km_s": 650.0,
                    "bz_imf_nt": -12.0,
                    "density_p_cc": 8.0,
                }
            }
        )
        assert result.kp_index is not None and np.isfinite(float(result.kp_index))
        # The detector's real G-scale vocabulary is the GeostormScale enum
        # VALUES (a previous revision asserted "G1".."G5", which the detector
        # never emits -- that check could only ever match "none").
        assert result.geomagnetic_storm_level in (
            "none",
            "minor",
            "moderate",
            "strong",
            "severe",
            "extreme",
        )
        # Strong southward-Bz driving must predict elevated (near-storm) Kp.
        assert float(result.kp_index) > 4.0

    def test_corrupt_checkpoint_fails_loud(self, tmp_path: Path) -> None:
        from omni_mercury_engine.space.solar_storm_detector import SolarStormDetector

        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"not a checkpoint")
        detector = SolarStormDetector()
        # torch.load(weights_only=True) refuses garbage with UnpicklingError
        # (zip-container damage surfaces as RuntimeError instead).
        with pytest.raises((pickle.UnpicklingError, RuntimeError)):
            detector.load_neural_weights(str(bad))
        assert detector._neural_trained is False
