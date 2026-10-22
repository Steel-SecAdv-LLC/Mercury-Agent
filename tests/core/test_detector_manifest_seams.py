# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Manifest seam declarations are measured truth, and both bridges fail closed.

Regression guarded here: 55 of 101 manifest entries raised ``TypeError`` from
``enable_detector`` (listed but never enableable), and 19 registered detectors
had neither ``extract_features`` nor ``predict``, so every registry extraction
reported ``success=True`` with no features and no scores -- a silent no-op in
the fusion path.  Each entry now declares the seams it was *driven through*
(``seams``), the payload it was driven with (``modality``), and the bridges
refuse what is not declared instead of wrapping it in theater.

The matrix below re-drives every declared seam whose payload is cheap to
build (series, matrices, physical rows, graphs, trajectories, records).
Image/video/tensor detectors assert the contract and the typed refusal of a
wrong-modality payload here; their happy paths are driven with real images
and frames by their own suites (``tests/detectors/test_visual_*``,
``tests/test_distillation.py``, ...).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.detector_registry import (
    DETECTOR_MANIFEST,
    MANIFEST_MODALITIES,
    MANIFEST_SEAMS,
    DetectorCategory,
    DetectorManifestEntry,
    DetectorRegistry,
)

ENTRIES: dict[str, DetectorManifestEntry] = {e.name: e for e in DETECTOR_MANIFEST}

#: Physically valid values for the row adapters' declared columns; anything
#: not named here passes its adapter's validation at 1.0.
COLUMN_VALUES: dict[str, float] = {
    "cape_j_kg": 1500.0,
    "cin_j_kg": -50.0,
    "srh_m2_s2": 150.0,
    "bulk_shear_kt": 35.0,
    "lcl_m": 900.0,
    "central_pressure_mb": 980.0,
    "environmental_pressure_mb": 1010.0,
    "sst_celsius": 28.0,
    "flux_long_wm2": 1e-6,
    "solar_wind_speed_km_s": 450.0,
    "bz_imf_nt": -3.0,
    "bx_nt": 20000.0,
    "by_nt": 100.0,
    # solar-storm snapshot (every optional column measured)
    "by_imf_nt": 2.0,
    "imf_magnitude_nt": 5.0,
    "proton_density_p_cm3": 5.0,
    "proton_temperature_k": 1e5,
    "flow_pressure_npa": 2.0,
    "proton_flux_pfu": 0.5,
    # CME cone kinematics
    "speed_km_s": 900.0,
    "latitude_deg": 0.0,
    "longitude_deg": 0.0,
    "half_angle_deg": 30.0,
    "time_21_5_epoch_s": 1.7e9,
}

#: Modalities whose payloads are cheap to synthesize; these seams are
#: re-driven end to end on every run.
DRIVEN_MODALITIES = frozenset(
    {
        "series_1d",
        "matrix_2d",
        "rows",
        "trajectory",
        "graph",
        "record",
        "windows_3d",
        "image_batch",
        "window",
        "event",
        "text",
    }
)

#: Seam-suite budgets for detectors that train: the seam is driven with real
#: images, the detection quality is measured in
#: ``tests/detectors/test_visual_real_images.py`` at realistic budgets.
FAST_CONFIG: dict[str, dict[str, Any]] = {
    "patchcore": {"image_size": (64, 64)},
    "padim": {"image_size": (64, 64)},
    "stfpm": {"image_size": (64, 64), "num_epochs": 1},
    "reverse_distillation": {"image_size": (64, 64), "num_epochs": 1},
    "cflow": {"image_size": (64, 64), "num_epochs": 1},
    "dual_student": {"num_epochs": 1, "device": "cpu"},
}

RECORD_PAYLOADS: dict[str, Any] = {
    "intelligence_fusion": {
        "SIGINT": {
            "confidence": 0.8,
            "observations": np.linspace(-1.0, 1.0, 16),
        },
        "OSINT": {
            "confidence": 0.6,
            "observations": np.linspace(1.0, -1.0, 16),
        },
    },
    "medical_abms": {
        "vitals": {
            "heart_rate": 88.0,
            "spo2": 97.0,
            "resp_rate": 16.0,
            "temp_c": 37.1,
            "sbp": 118.0,
        },
        "labs": {"lactate": 1.2, "wbc": 7.0},
    },
}


def _construct(entry: DetectorManifestEntry) -> Any:
    module = importlib.import_module(entry.module_path)
    cls = getattr(module, entry.class_name)
    return cls(FAST_CONFIG[entry.name]) if entry.name in FAST_CONFIG else cls()


def _real_image_batch(n: int = 8, size: int = 64) -> np.ndarray[Any, Any]:
    """Real CC0 brick-wall crops, channel-first, from tests/fixtures/visual."""
    from PIL import Image

    path = Path(__file__).resolve().parent.parent / "fixtures" / "visual" / "brick.png"
    with Image.open(path) as image:
        brick = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    rng = np.random.default_rng(3)
    ys, xs = rng.integers(0, brick.shape[0] - size, size=(2, n))
    crops = np.stack([brick[y : y + size, x : x + size] for y, x in zip(ys, xs, strict=True)])
    return np.ascontiguousarray(crops.transpose(0, 3, 1, 2))


def _minutes(n: int, start: float = 1.7e9, step: float = 60.0) -> np.ndarray[Any, Any]:
    return start + step * np.arange(n)


def _sep_window() -> np.ndarray[Any, Any]:
    flux = [0.5] * 4 + [15.0, 40.0, 120.0, 300.0]
    return np.column_stack([_minutes(len(flux), step=300.0), flux, [np.nan] * len(flux)])


def _gic_window() -> np.ndarray[Any, Any]:
    ramp = np.r_[np.zeros(30), 120.0 * np.arange(1, 31)]
    return np.column_stack([_minutes(60), 2e4 + ramp, 100.0 + 0.5 * ramp])


def _scintillation_record() -> dict[str, Any]:
    rng = np.random.default_rng(1)
    return {
        "intensity": rng.exponential(1.0, 3000),
        "phase_rad": rng.normal(0.0, 0.5, 3000),
        "sample_rate_hz": 50.0,
    }


def _cascade_record() -> dict[str, Any]:
    return {
        "now": "2024-05-10T23:00:00Z",
        "flares": [
            {
                "flrID": "2024-05-09T08:45:00-FLR-001",
                "classType": "X2.2",
                "beginTime": "2024-05-09T08:45Z",
                "sourceLocation": "S17W45",
            }
        ],
        "kp_series": [["2024-05-10T21:00:00Z", 7.33]],
    }


#: Payload builders for entries whose modality carries an entry-specific
#: schema (records, windows, events, text).  A driven entry with neither a
#: builder here nor a generic modality payload fails the suite, so a seam
#: can never be declared without being driven.
ENTRY_PAYLOADS: dict[str, Any] = {
    "sep_storm": _sep_window,
    "gic": _gic_window,
    "ionospheric_scintillation": _scintillation_record,
    "solar_gic_cascade": _cascade_record,
}


def _payload(entry: DetectorManifestEntry) -> Any:
    """The declared modality's example payload for ``entry``."""
    if entry.name in ENTRY_PAYLOADS:
        return ENTRY_PAYLOADS[entry.name]()
    rng = np.random.default_rng(7)
    if entry.modality == "series_1d":
        if entry.name == "solar_flare":  # physical flux: positive W/m^2
            return np.abs(rng.lognormal(-14.0, 1.0, 200))
        if entry.name == "earthquake":  # one STA + LTA window at 100 Hz needs 550 samples
            return rng.normal(size=1200)
        return rng.normal(size=200)
    if entry.modality == "matrix_2d":
        return rng.normal(size=(60, 8))
    if entry.modality == "rows":
        module = importlib.import_module(entry.module_path)
        cls = getattr(module, entry.class_name)
        row = [COLUMN_VALUES.get(c, 1.0) for c in cls.COLUMNS]
        return np.array([row] * 3)
    if entry.modality == "trajectory":
        lat = 37.0 + np.cumsum(rng.normal(0.0, 1e-4, 60))
        lon = -122.0 + np.cumsum(rng.normal(0.0, 1e-4, 60))
        return np.stack([lat, lon, np.arange(60) * 60.0], axis=1)
    if entry.modality == "graph":
        networkx = pytest.importorskip("networkx")
        return networkx.karate_club_graph()
    if entry.modality == "record":
        return RECORD_PAYLOADS[entry.name]
    if entry.modality == "image_batch":
        return _real_image_batch()
    if entry.modality == "windows_3d":
        # Default MultivariateTSDetector config: window_size=100, num_features=10.
        return rng.normal(size=(6, 100, 10))
    raise AssertionError(f"no example payload for modality {entry.modality!r}")


def _registry_harvest(detector: Any, payload: Any) -> tuple[Any, Any]:
    """Features and scores exactly as ``DetectorRegistry`` harvests them."""
    features = None
    if hasattr(detector, "extract_features"):
        features = detector.extract_features(payload)
    scores = None
    if hasattr(detector, "predict"):
        result = detector.predict(payload)
        if isinstance(result, dict):
            scores = result.get("anomaly_scores", result.get("scores"))
    return features, scores


class _NoInterface:
    """A detector-shaped object with neither registry method."""


class _NoHarvest:
    """Implements the registry contract but never produces output."""

    def extract_features(self, data: Any) -> None:
        return None


class TestBridgesFailClosed:
    """The two bridges refuse what they cannot serve, loudly."""

    def test_register_refuses_detector_with_no_registry_methods(self) -> None:
        reg = DetectorRegistry(auto_discover=False)
        with pytest.raises(TypeError, match="neither 'extract_features' nor 'predict'"):
            reg.register("useless", _NoInterface(), DetectorCategory.BASE)
        assert reg.list_all() == []

    def test_extraction_with_no_harvest_is_a_failure(self) -> None:
        """A detector that yields nothing must not report success."""
        reg = DetectorRegistry(auto_discover=False)
        reg.register("hollow", _NoHarvest(), DetectorCategory.BASE)
        result = reg.extract_features("hollow", np.zeros(8))
        assert result.success is False
        assert result.error is not None and "no features and no scores" in result.error
        info = reg.get("hollow")
        assert info is not None and info.error_count == 1

    def test_enable_detector_refuses_library_only_entry(self) -> None:
        from omni_mercury_engine.engine import OmniMercuryEngine

        eng = OmniMercuryEngine(mode="fusion", auto_load_checkpoint=False)
        with pytest.raises(ValueError, match="does not serve the engine seam"):
            eng.enable_detector("volcanic")

    def test_enable_detector_names_the_covering_adapter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.core import detector_registry
        from omni_mercury_engine.engine import OmniMercuryEngine

        covered = DetectorManifestEntry(
            "library_only_probe",
            "omni_mercury_engine.space.gic_detector",
            "GICDetector",
            DetectorCategory.SPACE,
            "library-only entry with a covering adapter",
            covered_by="gic_minute_series",
        )
        monkeypatch.setattr(
            detector_registry, "DETECTOR_MANIFEST", [*detector_registry.DETECTOR_MANIFEST, covered]
        )
        eng = OmniMercuryEngine(mode="fusion", auto_load_checkpoint=False)
        with pytest.raises(ValueError, match="gic_minute_series"):
            eng.enable_detector("library_only_probe")

    def test_auto_discover_registers_only_the_registry_seam(self) -> None:
        reg = DetectorRegistry(auto_discover=False)
        count = reg.auto_discover_detectors()
        registered = set(reg.list_all())
        declared = {e.name for e in DETECTOR_MANIFEST if "registry" in e.seams}
        undeclared = registered - declared
        assert not undeclared, f"registered without a registry seam: {sorted(undeclared)}"
        assert count == len(registered)
        # No library-only entry is ever registered (19 used to be, as silent
        # no-ops); entries now served by adapters register as real detectors.
        library_only = {e.name for e in DETECTOR_MANIFEST if not e.seams}
        assert not registered & library_only

    def test_modality_filter_selects_matching_detectors(self) -> None:
        reg = DetectorRegistry(auto_discover=False)
        reg.register("rows_det", _NoHarvest(), DetectorCategory.BASE, modality="rows")
        reg.register("series_det", _NoHarvest(), DetectorCategory.BASE, modality="series_1d")
        results = reg.extract_all_features(np.zeros(8), parallel=False, modality="series_1d")
        assert set(results) == {"series_det"}


class TestManifestDeclarations:
    """The declarations themselves stay coherent."""

    def test_every_entry_declares_known_seams_and_modality(self) -> None:
        for entry in DETECTOR_MANIFEST:
            assert set(entry.seams) <= MANIFEST_SEAMS
            assert entry.modality in MANIFEST_MODALITIES

    def test_covered_by_points_at_an_entry_serving_a_seam(self) -> None:
        for entry in DETECTOR_MANIFEST:
            if entry.covered_by is None:
                continue
            target = ENTRIES.get(entry.covered_by)
            assert (
                target is not None
            ), f"{entry.name}: covered_by {entry.covered_by!r} not in manifest"
            assert target.seams, f"{entry.name}: covering entry {target.name!r} serves no seam"

    def test_unknown_seam_or_modality_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown seams"):
            DetectorManifestEntry("x", "m", "C", DetectorCategory.BASE, "d", seams=("webhook",))
        with pytest.raises(ValueError, match="unknown modality"):
            DetectorManifestEntry("x", "m", "C", DetectorCategory.BASE, "d", modality="vibes")


def _driven(seam: str) -> list[str]:
    return sorted(
        e.name
        for e in DETECTOR_MANIFEST
        if seam in e.seams and e.modality in DRIVEN_MODALITIES and not e.external
    )


class TestSeamTruth:
    """Every declared seam is re-driven with its declared modality."""

    @pytest.mark.parametrize("name", _driven("registry"))
    def test_registry_seam_yields_output(self, name: str) -> None:
        entry = ENTRIES[name]
        detector = _construct(entry)
        payload = _payload(entry)
        if entry.fit_required:
            detector.fit(payload)
        features, scores = _registry_harvest(detector, payload)
        assert (
            features is not None or scores is not None
        ), f"{name} declares the registry seam but harvested nothing"

    @pytest.mark.parametrize("name", _driven("engine"))
    def test_engine_seam_fit_detect(self, name: str) -> None:
        entry = ENTRIES[name]
        detector = _construct(entry)
        assert isinstance(
            detector, BaseDetector
        ), f"{name} declares the engine seam but is not a BaseDetector"
        payload = _payload(entry)
        result = detector.fit(payload).detect(payload)
        assert isinstance(result, dict)
        assert "is_anomaly" in result or "anomaly_score" in result

    @pytest.mark.parametrize(
        "name",
        sorted(
            e.name
            for e in DETECTOR_MANIFEST
            if "engine" in e.seams and e.modality not in DRIVEN_MODALITIES
        ),
    )
    def test_undriven_engine_entries_hold_the_contract(self, name: str) -> None:
        """Video entries: contract present, wrong modality refused.

        Their seams need vision-language weights; they are driven end to end
        on the real pinned models by ``tests/detectors/test_vlm_real_models.py``.
        """
        entry = ENTRIES[name]
        detector = _construct(entry)
        assert isinstance(detector, BaseDetector)
        with pytest.raises(Exception, match=r"expects|shape|channels|frames"):
            detector.fit(np.zeros(16)).detect(np.zeros(16))

    @pytest.mark.parametrize("name", sorted(e.name for e in DETECTOR_MANIFEST if not e.seams))
    def test_library_entries_are_refused_by_name(self, name: str) -> None:
        """Library-only entries are unreachable through the engine bridge."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        eng = OmniMercuryEngine(mode="fusion", auto_load_checkpoint=False)
        with pytest.raises(ValueError, match="engine seam"):
            eng.enable_detector(name)
