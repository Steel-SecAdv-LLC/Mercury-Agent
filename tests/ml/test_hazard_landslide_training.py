# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline tests for the landslide_stability training pipeline.

Everything here runs against committed REAL fixtures (verbatim CHIRPS v2.0
series of one Kerala grid cell + a verbatim NASA GLC AGOL page excerpt --
see ``tests/fixtures/hazard_training/landslide/PROVENANCE.json``); no
network. Covered: content-based GLC field introspection, the feature
builder's no-lookahead property, climatology-percentile determinism against
the fixed 1981-2006 era, temporal-split enforcement, and (when a checkpoint
has been shipped through the merit gate) the physics-vs-learned differential
through the public detector API.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.ml.hazard_training import landslide_stability as ls
from omni_mercury_engine.models.checkpoint_paths import shipped_checkpoint_path

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "hazard_training" / "landslide"


@pytest.fixture(scope="module")
def cell_fixture() -> dict[str, np.ndarray]:
    """Real CHIRPS daily series for one cell (climatology era + 2018)."""
    with np.load(FIXTURES / "chirps_cell_fixture.npz") as npz:
        return {key: npz[key] for key in npz.files}


@pytest.fixture(scope="module")
def agol_rows() -> list[dict[str, object]]:
    """Verbatim attribute rows from the committed AGOL page excerpt."""
    raw = json.loads((FIXTURES / "glc_agol_excerpt.json").read_text())
    return [f["attributes"] for f in raw["features"]]


class TestGlcIntrospection:
    """Field names are located by content, and failures are loud."""

    def test_columns_resolved_on_real_excerpt(self, agol_rows: list[dict[str, object]]) -> None:
        cols = ls.introspect_glc_columns(agol_rows)
        # The AGOL mirror's real (DBF-truncated) names -- introspection must
        # find them without them being hardcoded anywhere in the pipeline.
        assert cols["date"] == "event_date"
        assert cols["latitude"] == "latitude"
        assert cols["longitude"] == "longitude"
        assert cols["trigger"] == "landslide1"
        assert cols["category"] == "landslide_"
        assert cols["accuracy"] == "location_a"

    def test_normalization_yields_clean_events(self, agol_rows: list[dict[str, object]]) -> None:
        events = ls.normalize_glc_rows(agol_rows)
        assert events, "real excerpt must normalize to at least one event"
        for event in events:
            assert abs(event.lat) <= 90 and abs(event.lon) <= 180
            assert event.trigger == event.trigger.strip().lower()
            assert 1900 <= event.date.year <= 2035

    def test_missing_trigger_column_fails_loud(self, agol_rows: list[dict[str, object]]) -> None:
        crippled = [{k: v for k, v in row.items() if k != "landslide1"} for row in agol_rows]
        with pytest.raises(RuntimeError, match="trigger column"):
            ls.introspect_glc_columns(crippled)

    def test_small_integers_never_parse_as_dates(self) -> None:
        """Fatality counts / IDs must not masquerade as 1970 epoch dates."""
        assert ls._parse_event_date(0) is None
        assert ls._parse_event_date(37) is None
        assert ls._parse_event_date(1534464000000) == dt.date(2018, 8, 17)
        assert ls._parse_event_date(20180817) == dt.date(2018, 8, 17)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("exact", 0.0),
            ("known exactly", 0.0),
            ("10km", 10.0),
            ("known within 5 km", 5.0),
            ("unknown", None),
            ("", None),
            (None, None),
        ],
    )
    def test_accuracy_parsing(self, raw: str | None, expected: float | None) -> None:
        assert ls.parse_accuracy_km(raw) == expected


class TestFeatureBuilderNoLookahead:
    """Features may use days <= event day only -- never anything after."""

    def test_future_perturbation_changes_nothing(self, cell_fixture: dict[str, np.ndarray]) -> None:
        series = cell_fixture["era2018_mm"].astype(np.float64)
        event_row = int(dt.date(2018, 8, 17).toordinal() - int(cell_fixture["era2018_dates"][0]))
        clean = ls.compute_rain_quantities(series, event_row)
        perturbed = series.copy()
        perturbed[event_row + 1 :] = 12345.0  # rewrite the entire future
        after = ls.compute_rain_quantities(perturbed, event_row)
        for key, value in clean.items():
            assert after[key] == pytest.approx(value, abs=0.0), key

    def test_pre_window_perturbation_changes_nothing(
        self, cell_fixture: dict[str, np.ndarray]
    ) -> None:
        series = cell_fixture["era2018_mm"].astype(np.float64)
        event_row = 200
        clean = ls.compute_rain_quantities(series, event_row)
        perturbed = series.copy()
        perturbed[: event_row - ls.ANTECEDENT_DAYS] = 999.0
        after = ls.compute_rain_quantities(perturbed, event_row)
        for key, value in clean.items():
            assert after[key] == pytest.approx(value, abs=0.0), key

    def test_in_window_perturbation_is_visible(self, cell_fixture: dict[str, np.ndarray]) -> None:
        series = cell_fixture["era2018_mm"].astype(np.float64)
        perturbed = series.copy()
        perturbed[200 - 45] += 500.0  # inside the 60-day window
        assert ls.compute_rain_quantities(perturbed, 200)["sum60"] > (
            ls.compute_rain_quantities(series, 200)["sum60"]
        )

    def test_insufficient_history_raises(self, cell_fixture: dict[str, np.ndarray]) -> None:
        with pytest.raises(ValueError, match="antecedent days"):
            ls.compute_rain_quantities(cell_fixture["era2018_mm"], ls.ANTECEDENT_DAYS - 1)

    def test_vector_spec_shape_and_reserved_dims(self, cell_fixture: dict[str, np.ndarray]) -> None:
        assert len(ls.FEATURE_NAMES) == ls.FEATURE_DIM == 64
        series = cell_fixture["era2018_mm"].astype(np.float64)
        tables = ls.climatology_tables(cell_fixture["clim_mm"])
        assert tables is not None
        quantities = ls.compute_rain_quantities(series, 228)
        percentiles = {
            key: ls.percentile_of(tables[key], quantities[key]) for key in ls.RAIN_QUANTITY_KEYS
        }
        vec = ls.build_feature_vector(
            percentiles=percentiles, quantities=quantities, lat=12.4, day_of_year=229
        )
        assert vec.shape == (64,) and vec.dtype == np.float32
        # Site/geotechnical dims are reserved: this corpus has no slope
        # observations, so they must be zero with presence flags zero.
        assert np.all(vec[26:64] == 0.0)
        assert vec[24] == 1.0  # complete real rain window
        assert vec[25] == 1.0  # climatology present
        assert np.all((vec[9:18] >= 0.0) & (vec[9:18] <= 1.0))
        # The canonical vector must feed the detector architecture directly.
        from omni_mercury_engine.detectors.geological.landslide import SlopeStabilityModel

        model = SlopeStabilityModel(input_dim=ls.FEATURE_DIM).eval()
        with torch.no_grad():
            prob, logits = model(torch.from_numpy(vec).unsqueeze(0))
        assert prob.shape == (1, 1) and logits.shape == (1, 6)


class TestClimatologyPercentiles:
    """Percentiles vs the fixed 1981-2006 era are deterministic and sane."""

    def test_deterministic_on_real_cell(self, cell_fixture: dict[str, np.ndarray]) -> None:
        tables_a = ls.climatology_tables(cell_fixture["clim_mm"])
        tables_b = ls.climatology_tables(cell_fixture["clim_mm"])
        assert tables_a is not None and tables_b is not None
        for key in ls.RAIN_QUANTITY_KEYS:
            np.testing.assert_array_equal(tables_a[key], tables_b[key])
            assert np.all(np.diff(tables_a[key]) >= 0.0)  # sorted

    def test_percentile_bounds_and_monotonicity(self, cell_fixture: dict[str, np.ndarray]) -> None:
        tables = ls.climatology_tables(cell_fixture["clim_mm"])
        assert tables is not None
        sum7 = tables["sum7"]
        n = sum7.size
        assert ls.percentile_of(sum7, float(sum7[0]) - 1.0) == 0.0
        assert ls.percentile_of(sum7, float(sum7[-1]) + 1.0) == 1.0
        assert ls.percentile_of(sum7, float(sum7[-1])) >= 1.0 - 1.0 / n
        queries = np.linspace(float(sum7[0]), float(sum7[-1]), 25)
        pcts = [ls.percentile_of(sum7, float(q)) for q in queries]
        assert all(b >= a for a, b in itertools.pairwise(pcts))
        assert np.isnan(ls.percentile_of(sum7, float("nan")))

    def test_degenerate_climatology_is_refused(self) -> None:
        """A too-short/ocean series yields None, never fabricated tables."""
        assert ls.climatology_tables(np.full(200, np.nan)) is None


class TestSplitEnforcement:
    """Temporal split by year with a strictly pre-train climatology era."""

    def test_split_years(self) -> None:
        assert ls.SPLIT.train_years == tuple(range(2007, 2016))
        assert ls.SPLIT.val_years == (2016, 2017)
        assert ls.SPLIT.test_years == tuple(range(2018, 2025))
        assert max(ls.SPLIT.train_years) < min(ls.SPLIT.val_years)
        assert max(ls.SPLIT.val_years) < min(ls.SPLIT.test_years)

    def test_masks_partition_every_sample_year(self) -> None:
        years = np.arange(ls.MIN_EVENT_YEAR, ls.MAX_EVENT_YEAR + 1)
        train, val, test = ls.SPLIT.masks(years)
        combined = train.astype(int) + val.astype(int) + test.astype(int)
        assert np.all(combined == 1), "each sample year must fall in exactly one split"

    def test_climatology_era_is_strictly_pre_train(self) -> None:
        assert max(ls.CLIMATOLOGY_YEARS) < min(ls.SPLIT.train_years)
        assert ls.CLIMATOLOGY_ERA == "1981-2006"


_SHIPPED = shipped_checkpoint_path(ls.CHECKPOINT_NAME)


@pytest.mark.skipif(not _SHIPPED.exists(), reason="landslide_coolr checkpoint not shipped")
class TestDifferentialPhysicsVsShipped:
    """Physics fallback vs shipped weights, through the public API."""

    def _case(self, cell_fixture: dict[str, np.ndarray]) -> tuple[dict[str, object], np.ndarray]:
        """Build the 2018-08-17 Kerala case from the real fixture series."""
        event = json.loads((FIXTURES / "glc_event_kerala_2018.json").read_text())
        series = cell_fixture["era2018_mm"].astype(np.float64)
        event_row = int(dt.date(2018, 8, 17).toordinal() - int(cell_fixture["era2018_dates"][0]))
        quantities = ls.compute_rain_quantities(series, event_row)
        tables = ls.climatology_tables(cell_fixture["clim_mm"])
        assert tables is not None
        percentiles = {
            key: ls.percentile_of(tables[key], quantities[key]) for key in ls.RAIN_QUANTITY_KEYS
        }
        vec = ls.build_feature_vector(
            quantities, percentiles, lat=float(event["latitude"]), day_of_year=229
        )
        rainfall_data = {
            "intensity_mm_hr": quantities["day0"] / 24.0,
            "duration_hours": 24.0,
            "antecedent_7day_mm": quantities["antecedent_7day_mm"],
        }
        return {"rainfall_data": rainfall_data}, vec

    def test_payload_spec(self) -> None:
        from omni_mercury_engine.models.checkpoint_paths import load_shipped_checkpoint

        payload, provenance = load_shipped_checkpoint(ls.CHECKPOINT_NAME)
        assert payload["feature_spec"] == "landslide-coolr-v1"
        assert list(payload["feature_names"]) == list(ls.FEATURE_NAMES)
        assert len(payload["feature_mean"]) == 64 and len(payload["feature_std"]) == 64
        assert payload["climatology_era"] == "1981-2006"
        assert provenance is not None
        assert provenance["evaluation"]["learned_beats_physics"] is True

    def test_learned_differs_from_physics_on_real_case(
        self, cell_fixture: dict[str, np.ndarray]
    ) -> None:
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        case, vec = self._case(cell_fixture)
        physics = LandslideDetector(enable_ml_ensemble=False, enable_recursion=False)
        learned = LandslideDetector(enable_ml_ensemble=False, enable_recursion=False)
        learned.load_neural_weights()  # no path -> shipped default
        assert learned._neural_trained is True

        physics_out = physics.predict_landslide(dict(case))
        learned_out = learned.predict_landslide({**case, "slope_features": vec})
        # Physics abstains on rainfall-only inputs (no slope/saturation/
        # displacement observations exist in this corpus): probability 0.
        assert physics_out.slope_failure_probability == 0.0
        prob = learned_out.slope_failure_probability
        assert 0.0 <= prob <= 1.0 and np.isfinite(prob)
        assert prob != physics_out.slope_failure_probability
        assert learned_out.landslide_type in ls.LANDSLIDE_TYPE_LABELS
        # Peak-monsoon Kerala 2018 (the real flood disaster) must score
        # above the same cell in the dry season, from the same real data.
        dry_row = int(dt.date(2018, 1, 20).toordinal() - int(cell_fixture["era2018_dates"][0]))
        dry_q = ls.compute_rain_quantities(cell_fixture["era2018_mm"].astype(np.float64), dry_row)
        tables = ls.climatology_tables(cell_fixture["clim_mm"])
        assert tables is not None
        dry_pct = {k: ls.percentile_of(tables[k], dry_q[k]) for k in ls.RAIN_QUANTITY_KEYS}
        dry_vec = ls.build_feature_vector(dry_q, dry_pct, lat=12.4, day_of_year=20)
        dry_out = learned.predict_landslide({"slope_features": dry_vec})
        assert prob > dry_out.slope_failure_probability

    def test_corrupt_checkpoint_fails_loud(self, tmp_path: Path) -> None:
        import pickle

        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"not a checkpoint")
        detector = LandslideDetector(enable_ml_ensemble=False, enable_recursion=False)
        with pytest.raises((pickle.UnpicklingError, RuntimeError)):
            detector.load_neural_weights(str(bad))
        assert detector._neural_trained is False
