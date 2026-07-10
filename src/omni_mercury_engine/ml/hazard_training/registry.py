# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Registry of the eleven ``load_neural_weights()`` hooks and their pipelines.

Each entry answers, per hook: what real labeled data would train it, whether
that data is obtainable from this environment, and — when it is — which
pipeline module runs. Hooks whose real data is NOT obtainable here fail loud
with the full data requirement (:class:`HazardDataUnavailableError`); that is
the honest terminal state, not a stub: the audit is the deliverable that
prevents anyone from quietly training these on synthetic data.

Categories (see ``docs/HAZARD_CHECKPOINT_TRAINING.md``):

* ``a`` — real labeled data fetchable from this environment; pipeline runs.
* ``b`` — real data exists but needs archives/credentials absent here; the
  entry documents the exact source so an operator can run the same stages.
* ``c`` — no real labeled corpus exists for the architecture's input
  contract; training it would require fabricating data, which this codebase
  forbids. The physics fallback is the permanent honest default.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.ml.hazard_training.common import HazardDataUnavailableError

if TYPE_CHECKING:
    from omni_mercury_engine.ml.hazard_training.common import (
        EvaluationOutcome,
        PipelineContext,
    )


@dataclass(frozen=True)
class HookEntry:
    """One ``load_neural_weights()`` hook in the training registry.

    Attributes:
        name: Registry key (also the CLI ``--hook`` value).
        detector: Dotted path of the detector class owning the hook.
        architecture: The network the checkpoint must populate.
        category: ``"a"`` (trainable here), ``"b"`` (real data elsewhere),
            or ``"c"`` (no real corpus exists).
        data_requirement: Human-actionable statement of the real data the
            hook trains on — for ``b``/``c`` this is the error message.
        pipeline_module: Module implementing fetch/build/train/evaluate/ship
            (category ``a`` only).
        checkpoint_name: Shipped-checkpoint basename (category ``a`` only).
    """

    name: str
    detector: str
    architecture: str
    category: str
    data_requirement: str
    pipeline_module: str | None = None
    checkpoint_name: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def load_pipeline(self) -> Any:
        """Import and return the pipeline module, or fail loud.

        Raises:
            HazardDataUnavailableError: For category ``b``/``c`` hooks —
                the message carries the full data requirement.
        """
        if self.pipeline_module is None:
            raise HazardDataUnavailableError(
                f"hook '{self.name}' (category {self.category}) has no runnable "
                f"pipeline in this environment. Data requirement: "
                f"{self.data_requirement}"
            )
        return importlib.import_module(self.pipeline_module)


HOOK_REGISTRY: dict[str, HookEntry] = {
    "solar_storm": HookEntry(
        name="solar_storm",
        detector="space.solar_storm_detector.SolarStormDetector",
        architecture="GeomagneticStormPredictor (32-feature MLP; Kp regression + storm head)",
        category="a",
        data_requirement=(
            "NASA SPDF OMNI2 hourly solar-wind/IMF archive paired with observed "
            "planetary Kp (spdf.gsfc.nasa.gov, public), cross-checked against the "
            "GFZ Potsdam definitive Kp service."
        ),
        pipeline_module="omni_mercury_engine.ml.hazard_training.solar_storm",
        checkpoint_name="solar_storm_geomag",
    ),
    "earthquake_precursor": HookEntry(
        name="earthquake_precursor",
        detector="space.disaster_precursor_detector.DisasterPrecursorDetector",
        architecture="EarthquakePrecursorAnalyzer(128)",
        category="b",
        data_requirement=(
            "USGS FDSN event catalog (earthquake.usgs.gov, public — reachable "
            "here) reshaped into regional seismicity-sequence samples with "
            "did-M6+-follow-within-window labels. Trainable in principle from "
            "this environment; deliberately NOT shipped from this pass because "
            "an honest evaluation needs multi-decade regional feature "
            "engineering reviewed against the seismology literature — a "
            "half-reviewed earthquake forecaster is worse than the physics "
            "fallback that abstains. Run this hook's stages once the feature "
            "spec is reviewed; the catalog fetcher (features.py seams) is real."
        ),
    ),
    "seismic_wave": HookEntry(
        name="seismic_wave",
        detector="detectors.geological.disaster_detectors.EarthquakeDetector",
        architecture="SeismicWaveAnalyzer (spectrogram encoder)",
        category="b",
        data_requirement=(
            "Real labeled waveforms: EarthScope/IRIS FDSN dataselect "
            "(service.iris.edu) miniSEED windows around USGS-cataloged events "
            "plus noise windows. service.iris.edu is not reachable from this "
            "environment's proxy allowlist."
        ),
    ),
    "tsunami_waveform": HookEntry(
        name="tsunami_waveform",
        detector="detectors.geological.disaster_detectors.TsunamiDetector",
        architecture="WaveformFFTAnalyzer",
        category="b",
        data_requirement=(
            "NOAA NDBC/DART bottom-pressure records for tsunamigenic events "
            "(historical DART archives at ndbc.noaa.gov) with event windows "
            "labeled from the NOAA tsunami event database. ndbc.noaa.gov "
            "historical archives are not reachable from this environment."
        ),
    ),
    "hurricane_wind": HookEntry(
        name="hurricane_wind",
        detector="detectors.geological.hurricane_detector.HurricaneDetector",
        architecture="WindPatternAnalyzer (CNN+LSTM over wind fields)",
        category="a",
        data_requirement=(
            "ERA5 10 m u/v wind patch sequences streamed as raw zarr-2 chunk "
            "GETs from the public ARCO-ERA5 mirror (storage.googleapis.com, "
            "no CDS key required) labeled with IBTrACS v04r01 best-track "
            "positions, USA_WIND intensities, and USA_SSHS classes "
            "(www.ncei.noaa.gov, public)."
        ),
        pipeline_module="omni_mercury_engine.ml.hazard_training.hurricane_wind",
        checkpoint_name="hurricane_era5",
    ),
    "landslide_stability": HookEntry(
        name="landslide_stability",
        detector="detectors.geological.landslide.LandslideDetector",
        architecture="SlopeStabilityModel",
        category="b",
        data_requirement=(
            "NASA Global Landslide Catalog / COOLR event records joined with "
            "site geotechnical and antecedent-rainfall covariates; the joined "
            "covariates require GPM/soil archives not reachable here."
        ),
    ),
    "tornado_radar": HookEntry(
        name="tornado_radar",
        detector="detectors.geological.tornado_detector.TornadoDetector",
        architecture="DopplerRadarAnalyzer (LSTM+attention over velocity fields)",
        category="b",
        data_requirement=(
            "NEXRAD Level-II Doppler velocity volumes (AWS Open Data "
            "noaa-nexrad-level2) labeled with SPC tornado reports; the S3 "
            "bucket is not reachable from this environment."
        ),
    ),
    "volcanic_eruption": HookEntry(
        name="volcanic_eruption",
        detector="detectors.geological.volcanic.VolcanicEruptionDetector",
        architecture="EruptionForecastModel(128) + SeismicSwarmDetector LSTM",
        category="b",
        data_requirement=(
            "Observatory-grade multiparameter monitoring series (seismic RSAM, "
            "SO2 flux, deformation) for labeled eruptive/non-eruptive periods "
            "— e.g. USGS/AVO and INGV archives distributed per-volcano on "
            "request; no public bulk archive is reachable here."
        ),
    ),
    "wildfire_ignition": HookEntry(
        name="wildfire_ignition",
        detector="detectors.geological.wildfire.WildfireDetector",
        architecture="FireIgnitionDetector CNN (+optional WildfireCNN)",
        category="b",
        data_requirement=(
            "NASA FIRMS active-fire granules (firms.modaps.eosdis.nasa.gov — "
            "requires a MAP_KEY) as thermal rasters with confirmed-fire labels."
        ),
    ),
    "schumann_harmonics": HookEntry(
        name="schumann_harmonics",
        detector="space.schumann_resonance.SchumannResonanceDetector",
        architecture="1D CNN + LSTM(512) over harmonic spectra",
        category="b",
        data_requirement=(
            "Calibrated ELF station spectrograms (e.g. HeartMath GCMS or "
            "university ELF observatories) with anomaly annotations; no such "
            "labeled public corpus is downloadable here — the BGS ELF client "
            "in data_sources is explicitly simulated and must never be used "
            "as training data."
        ),
    ),
    "consciousness_field": HookEntry(
        name="consciousness_field",
        detector="models.parapsychology.ConsciousnessField",
        architecture="LSTM field analyzer",
        category="c",
        data_requirement=(
            "No real labeled corpus exists for this architecture's input "
            "contract; any training set would be fabricated. The hook remains "
            "for research checkpoints supplied by the operator; the shipped "
            "default stays the physics/statistics fallback permanently."
        ),
    ),
}


def get_hook(name: str) -> HookEntry:
    """Look up a hook entry by name.

    Raises:
        KeyError: With the list of valid names.
    """
    if name not in HOOK_REGISTRY:
        raise KeyError(f"unknown hook '{name}'; valid hooks: {sorted(HOOK_REGISTRY)}")
    return HOOK_REGISTRY[name]


def run_stage(name: str, stage: str, ctx: PipelineContext) -> Any:
    """Run one pipeline stage for a hook, honoring the category audit.

    Args:
        name: Hook registry key.
        stage: One of ``fetch``, ``build``, ``train``, ``evaluate``, ``ship``.

    Returns:
        The stage's return value (``evaluate`` returns an
        :class:`EvaluationOutcome`; ``ship`` returns the shipped paths).

    Raises:
        HazardDataUnavailableError: For category ``b``/``c`` hooks.
        ValueError: For an unknown stage name.
    """
    entry = get_hook(name)
    module = entry.load_pipeline()
    stage_fns = {
        "fetch": "fetch",
        "build": "build_dataset",
        "train": "train",
        "evaluate": "evaluate",
        "ship": "ship",
    }
    if stage not in stage_fns:
        raise ValueError(f"unknown stage '{stage}'; valid stages: {sorted(stage_fns)}")
    fn = getattr(module, stage_fns[stage])
    return fn(ctx)


def evaluate_hook(name: str, ctx: PipelineContext) -> EvaluationOutcome:
    """Convenience wrapper returning the typed evaluation outcome."""
    outcome: EvaluationOutcome = run_stage(name, "evaluate", ctx)
    return outcome
