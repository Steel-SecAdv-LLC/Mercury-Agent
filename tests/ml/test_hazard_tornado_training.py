# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the tornado_radar hazard-training pipeline (NEXRAD + SPC).

All tests are offline; they run against committed REAL-data fixtures in
``tests/fixtures/hazard_training/tornado/``:

* ``moore_ef5_ktlx_sector.npz`` -- a cropped block (121 rays x 144 gates)
  of the lowest VEL sweep of NEXRAD volume
  ``2013/05/20/KTLX/KTLX20130520_195527_V06.gz`` (Unidata Level-II mirror;
  sha256 pinned inside the fixture's ``provenance`` JSON) around the
  2013-05-20 Moore, OK EF5 tornado (SPC report om=451537, 13:56 CST =
  19:56 UTC; scan time 19:55:27 UTC), plus the exact ``(61, 64)`` sector
  the pipeline extracts from the FULL sweep.
* ``spc_rows.csv`` -- five rows copied verbatim from the real SPC WCM file
  ``1950-2023_actual_tornadoes.csv`` (Moore 2013 EF5; three 2022 EF1 rows
  whose evening CST times roll over to the next UTC day; one 2011 EF4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.ml.hazard_training import tornado_radar as tr
from omni_mercury_engine.ml.hazard_training.common import TemporalSplit
from omni_mercury_engine.models.checkpoint_paths import shipped_checkpoint_path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "hazard_training" / "tornado"
SECTOR_NPZ = FIXTURES / "moore_ef5_ktlx_sector.npz"
SPC_CSV = FIXTURES / "spc_rows.csv"


@pytest.fixture(scope="module")
def moore():
    """Load the committed Moore EF5 sector fixture (see module docstring)."""
    with np.load(SECTOR_NPZ) as z:
        return {k: z[k] for k in z.files}


class TestSpcTimeConversion:
    """SPC WCM times are CST (tz=3); UTC = CST + 6 h."""

    def test_cst_to_utc_plus_six(self) -> None:
        utc = tr.spc_time_to_utc("2013-05-20", "13:56:00")
        assert utc.isoformat() == "2013-05-20T19:56:00+00:00"

    def test_utc_day_rollover(self) -> None:
        """Evening CST reports belong to the NEXT UTC day (radar key prefix)."""
        utc = tr.spc_time_to_utc("2022-10-24", "19:46:00")
        assert utc.isoformat() == "2022-10-25T01:46:00+00:00"

    def test_load_spc_reports_on_real_rows(self) -> None:
        reports, stats = tr.load_spc_reports(
            SPC_CSV, years=(2011, 2013, 2022, 2023), min_mag=1, full_track_only=True
        )
        assert stats["excluded_non_cst_tz"] == 0
        by_om = {r["om"]: r for r in reports}
        # Moore EF5: same UTC day.
        assert by_om[451537]["utc"].isoformat() == "2013-05-20T19:56:00+00:00"
        assert by_om[451537]["mag"] == 5
        # 2022-10-24 19:46 CST rolled into 2022-10-25 UTC.
        assert by_om[621942]["utc"].isoformat() == "2022-10-25T01:46:00+00:00"
        assert by_om[621942]["yr"] == 2022  # UTC year unchanged by this rollover

    def test_non_cst_rows_are_excluded_and_counted(self, tmp_path: Path) -> None:
        """A tz code other than 3 (CST) must be dropped, never mis-shifted."""
        import pandas as pd

        df = pd.read_csv(SPC_CSV)
        df.loc[df["om"] == 621942, "tz"] = 9  # GMT code, per SPC WCM notes
        tampered = tmp_path / "spc_tampered.csv"
        df.to_csv(tampered, index=False)
        reports, stats = tr.load_spc_reports(
            tampered, years=(2011, 2013, 2022, 2023), min_mag=1, full_track_only=True
        )
        assert stats["excluded_non_cst_tz"] == 1
        assert 621942 not in {r["om"] for r in reports}


class TestSectorGeometry:
    """Extraction geometry against the committed real decoded sweep block."""

    def test_extraction_reproduces_committed_sector(self, moore: dict[str, Any]) -> None:
        window = tr.extract_sector(
            moore["az_deg"],
            moore["vel"],
            float(moore["first_gate_km"]),
            float(moore["gate_width_km"]),
            float(moore["bearing_deg"]),
            float(moore["range_km"]),
        )
        assert window is not None
        assert window.shape == (tr.SECTOR_RAYS, tr.SECTOR_GATES)
        assert window.dtype == np.float32
        # Bit-identical to the sector cut from the FULL sweep at fixture
        # generation time (NaN positions included).
        np.testing.assert_array_equal(window, moore["expected_sector"])

    def test_geometry_recomputed_from_coordinates(self, moore: dict[str, Any]) -> None:
        """Bearing/range stored in the fixture come from the site/report coords."""
        bearing = tr.initial_bearing_deg(
            float(moore["radar_lat"]),
            float(moore["radar_lon"]),
            float(moore["report_lat"]),
            float(moore["report_lon"]),
        )
        dist = tr.haversine_km(
            float(moore["radar_lat"]),
            float(moore["radar_lon"]),
            float(moore["report_lat"]),
            float(moore["report_lon"]),
        )
        assert bearing == pytest.approx(float(moore["bearing_deg"]), abs=1e-9)
        assert dist == pytest.approx(float(moore["range_km"]), abs=1e-9)
        # KTLX -> Moore: ~32 km to the west; sanity against public geography.
        assert 25.0 < dist < 40.0

    def test_out_of_coverage_returns_none_not_padding(self, moore: dict[str, Any]) -> None:
        """Ranges outside the sweep's gate coverage must refuse, not zero-pad."""
        for bad_range in (1.0, 5000.0):
            assert (
                tr.extract_sector(
                    moore["az_deg"],
                    moore["vel"],
                    float(moore["first_gate_km"]),
                    float(moore["gate_width_km"]),
                    float(moore["bearing_deg"]),
                    bad_range,
                )
                is None
            )

    def test_fixture_is_real_velocity_data(self, moore: dict[str, Any]) -> None:
        """Raw archive m/s: Nyquist-bounded magnitudes and censored (NaN) gates."""
        sector = moore["expected_sector"]
        finite = sector[np.isfinite(sector)]
        assert finite.size > 0.5 * sector.size
        assert float(np.max(np.abs(finite))) < 100.0
        assert np.isnan(sector).any(), "fixture must exercise NaN handling"


class TestNaNHandling:
    """NaN gates must map to 0.0 exactly as the deployed detector does."""

    def test_couplet_v_rot_nan_parity(self, moore: dict[str, Any]) -> None:
        sector = moore["expected_sector"]
        zero_filled = np.where(np.isfinite(sector), sector, 0.0).astype(np.float32)
        assert np.isfinite(zero_filled).all()
        assert tr.couplet_v_rot(sector) == tr.couplet_v_rot(zero_filled)
        assert tr.couplet_v_rot(sector) == pytest.approx(float(moore["expected_v_rot"]))

    def test_detector_physics_parity_via_public_api(self, moore: dict[str, Any]) -> None:
        """couplet_v_rot mirrors the detector's deployed physics byte-for-byte."""
        from omni_mercury_engine.detectors.geological.tornado_detector import TornadoDetector

        det = TornadoDetector(
            enable_radar=True,
            enable_atmospheric=False,
            enable_pressure=False,
            enable_resonance=False,
            enable_recursion=False,
            enable_refactoring=False,
        )
        sector = moore["expected_sector"]
        zero_filled = np.where(np.isfinite(sector), sector, 0.0).astype(np.float32)
        result = det.predict_tornado({"radar_sequence": zero_filled})
        assert result.rotation_velocity_ms == pytest.approx(tr.couplet_v_rot(sector))
        # The Moore fixture scan is a real sub-threshold couplet for the
        # per-ray-median physics: rotational velocity under the 15 m/s
        # operational threshold, so no mesocyclone flag on this path.
        assert result.mesocyclone_detected is (result.rotation_velocity_ms >= 15.0)


class TestTemporalSplitConfig:
    """Split-by-year enforcement for the tornado pipeline."""

    def test_split_years(self) -> None:
        assert tr.SPLIT.train_years == tuple(range(2011, 2020))
        assert tr.SPLIT.val_years == (2020, 2021)
        assert tr.SPLIT.test_years == (2022, 2023)

    def test_masks_disjoint_over_sample_years(self) -> None:
        years = np.array([2011, 2015, 2019, 2020, 2021, 2022, 2023, 2023])
        train, val, test = tr.SPLIT.masks(years)
        assert not np.any(train & val) and not np.any(val & test) and not np.any(train & test)
        assert int(train.sum()) == 3 and int(val.sum()) == 2 and int(test.sum()) == 3

    def test_interleaved_years_refused(self) -> None:
        with pytest.raises(ValueError, match="temporal split violated"):
            TemporalSplit(train_years=(2011, 2022), val_years=(2020,), test_years=(2023,))


_SHIPPED = shipped_checkpoint_path("tornado_nexrad")


@pytest.mark.skipif(not _SHIPPED.exists(), reason="tornado_nexrad checkpoint not shipped")
class TestShippedTornadoCheckpoint:
    """Differential physics-vs-shipped behavior through the public API."""

    def _detector(self):
        from omni_mercury_engine.detectors.geological.tornado_detector import TornadoDetector

        return TornadoDetector(
            enable_radar=True,
            enable_atmospheric=False,
            enable_pressure=False,
            enable_resonance=False,
            enable_recursion=False,
            enable_refactoring=False,
        )

    def test_default_load_uses_shipped_checkpoint(self) -> None:
        det = self._detector()
        det.load_neural_weights()  # no path -> shipped default
        assert det._neural_trained is True

    def test_payload_contract(self) -> None:
        payload = torch.load(_SHIPPED, map_location="cpu", weights_only=True)
        assert payload["feature_spec"] == "tornado-nexrad-v1"
        assert payload["gates"] == 64
        assert payload["sector_rays"] == 61
        assert payload["units"] == "m/s"
        assert "radar_analyzer" in payload

    def test_provenance_is_merit_gated(self) -> None:
        sidecar = _SHIPPED.with_suffix(".provenance.json")
        assert sidecar.exists()
        provenance = json.loads(sidecar.read_text())
        evaluation = provenance["evaluation"]
        assert evaluation["learned_beats_physics"] is True
        assert evaluation["learned"]["auc"] > evaluation["physics"]["auc"]
        assert all(src["sha256"] for src in provenance["data_sources"])

    def test_differential_physics_vs_learned(self, moore: dict[str, Any]) -> None:
        """Identical real input through both deployed paths."""
        sector = moore["expected_sector"]
        window = np.where(np.isfinite(sector), sector, 0.0).astype(np.float32)
        case = {"radar_sequence": window}

        physics = self._detector().predict_tornado(case)
        learned_det = self._detector()
        learned_det.load_neural_weights()
        learned = learned_det.predict_tornado(case)

        # Physics emits the couplet observable exactly.
        assert physics.rotation_velocity_ms == pytest.approx(tr.couplet_v_rot(sector))
        # The learned path emits its own finite, non-negative rotation
        # estimate (the head is ReLU * 50) -- not the physics number.
        assert np.isfinite(learned.rotation_velocity_ms)
        assert learned.rotation_velocity_ms >= 0.0
        assert isinstance(learned.mesocyclone_detected, bool)
