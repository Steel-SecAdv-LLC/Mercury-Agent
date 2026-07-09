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

from pathlib import Path

import numpy as np
import pytest

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


class TestShippedSolarStormCheckpoint:
    """The committed artifact is real, provenanced, and loadable by the hook."""

    def test_artifact_and_sidecar_exist(self) -> None:
        path = shipped_checkpoint_path("solar_storm_geomag")
        assert path.exists(), "shipped checkpoint missing"
        sidecar = path.with_suffix("").with_suffix(".provenance.json")
        assert Path(str(path)[: -len(".pt")] + ".provenance.json").exists()

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
        assert result.geomagnetic_storm_level in ("none", "G1", "G2", "G3", "G4", "G5")
        # Strong southward-Bz driving must predict elevated (near-storm) Kp.
        assert float(result.kp_index) > 4.0

    def test_corrupt_checkpoint_fails_loud(self, tmp_path: Path) -> None:
        from omni_mercury_engine.space.solar_storm_detector import SolarStormDetector

        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"not a checkpoint")
        detector = SolarStormDetector()
        with pytest.raises(Exception):  # torch raises UnpicklingError/RuntimeError
            detector.load_neural_weights(str(bad))
        assert detector._neural_trained is False
