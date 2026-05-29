"""
Tests for label-provenance de-leaking (Issue #6: de-leak the loaders at source).

Datasets whose anomaly labels were manufactured by thresholding a
detector-like score/feature must be declared ``label_source="statistical"`` at
source and excluded from the headline supervised AUC, with genuinely-labelled
sets (ground-truth / expert-annotated) reported as the headline.

Mercury Agent - Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GNU GPL v3
"""

from __future__ import annotations

import importlib

import pytest

from omni_mercury_engine.datasets.metadata import (
    GENUINE_LABEL_SOURCES,
    MANUFACTURED_LABEL_SOURCES,
    is_supervised_eval_safe,
)

# (module, class) -> expected LABEL_SOURCE
MANUFACTURED_LOADERS = [
    ("environmental", "USGSEarthquakeLoader"),
    ("environmental", "NOAAWeatherLoader"),
    ("environmental", "WildfireDataLoader"),
    ("environmental", "USGSGeochemistryLoader"),
    ("ocean", "NOAABuoyLoader"),
    ("noaa_storm", "NOAAStormEventsLoader"),
    ("noaa_gsod", "NOAAGSODLoader"),
    ("noaa_erddap", "NOAAERDDAPLoader"),
    ("epa_air", "EPAAirQualityLoader"),
    ("disaster", "FEMADisasterLoader"),
    ("disaster", "FEMAHazardMitigationLoader"),
    ("space", "NASAExoplanetLoader"),
    ("space", "SolarDynamicsLoader"),
    ("space", "SETILoader"),
    ("security", "ThreatIntelLoader"),
    ("ucr_archive", "MSDSLoader"),
]

GENUINE_LOADERS = [
    ("security", "NSLKDDLoader"),
    ("timeseries", "SMDLoader"),
    ("industrial", "BATADALLoader"),
    ("mitbih", "MITBIHLoader"),
    ("adbench", "ADBenchLoader"),
    ("adrepository", "ADRepositoryLoader"),
]


def _load_class(module: str, cls_name: str):
    try:
        mod = importlib.import_module(f"omni_mercury_engine.datasets.{module}")
    except Exception as exc:  # optional dependency missing
        pytest.skip(f"module {module} unimportable: {exc}")
    return getattr(mod, cls_name)


class TestProvenanceHelper:
    def test_genuine_sources_eval_safe(self) -> None:
        for src in GENUINE_LABEL_SOURCES:
            assert is_supervised_eval_safe(src)

    def test_manufactured_and_none_not_eval_safe(self) -> None:
        for src in MANUFACTURED_LABEL_SOURCES:
            assert not is_supervised_eval_safe(src)
        assert not is_supervised_eval_safe("none")
        assert not is_supervised_eval_safe("unknown")


class TestLoaderProvenanceAtSource:
    @pytest.mark.parametrize("module,cls_name", MANUFACTURED_LOADERS)
    def test_manufactured_loaders_flagged(self, module: str, cls_name: str) -> None:
        cls = _load_class(module, cls_name)
        assert cls.LABEL_SOURCE == "statistical", (
            f"{cls_name} manufactures labels by thresholding; must declare "
            f"LABEL_SOURCE='statistical', got {cls.LABEL_SOURCE!r}"
        )
        assert not is_supervised_eval_safe(cls.LABEL_SOURCE)

    @pytest.mark.parametrize("module,cls_name", GENUINE_LOADERS)
    def test_genuine_loaders_eval_safe(self, module: str, cls_name: str) -> None:
        cls = _load_class(module, cls_name)
        assert is_supervised_eval_safe(
            cls.LABEL_SOURCE
        ), f"{cls_name} carries genuine labels; LABEL_SOURCE={cls.LABEL_SOURCE!r}"


class TestBenchmarkExclusion:
    """The headline benchmark must resolve provenance from loader classes and
    exclude manufactured-label domain datasets."""

    def test_domain_datasets_provenance_resolves(self) -> None:
        bench = importlib.import_module("benchmarks.mercury_benchmark")
        manufactured_names = set()
        genuine_names = set()
        for name, _cat, cls_name, module, _kwargs in bench.DOMAIN_DATASETS:
            cls = _load_class(module, cls_name)
            src = getattr(cls, "LABEL_SOURCE", "ground_truth")
            (manufactured_names if not is_supervised_eval_safe(src) else genuine_names).add(name)

        # Known circular-label domain datasets must be excluded from headline.
        for expected in (
            "USGS_Earthquake",
            "NOAA_Weather",
            "Wildfire",
            "NOAA_Buoy",
            "EPA_AirQuality",
            "FEMA_Disaster",
            "ThreatIntel",
            "MSDS",
        ):
            assert expected in manufactured_names

        # Established benchmarks with real labels remain in the headline.
        for expected in ("NSL-KDD", "SMD", "BATADAL", "MIT-BIH"):
            assert expected in genuine_names
