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
        category="a",
        data_requirement=(
            "USGS ComCat FDSN event catalog (earthquake.usgs.gov, public — "
            "reachable here), California 1980-2024 M≥2.5, reshaped into "
            "0.5-degree (cell, epoch) samples labeled P(M≥5.0 within 30 d) "
            "per the binding feature-spec review "
            "(docs/research/EARTHQUAKE_PRECURSOR_LITERATURE_REVIEW.md). The "
            "trained head is a catalog-statistical seismicity-rate forecast "
            "— never EM/Schumann precursor detection, never per-event "
            "magnitude/time prediction. The merit gate compares against a "
            "Reasenberg-Jones/ETAS-lite clustering baseline (not bare "
            "Poisson) with auc/brier/reliability_ece non-regression "
            "constraints; 'clustering baseline wins, not shipped' is a valid "
            "recorded outcome — and is the CURRENT one (2026-07-10, "
            "seismicity-catalog-v2 stacked-RJ candidate): the learned model "
            "wins held-out log-loss (0.00414 vs 0.00629, bootstrap 95% CI "
            "excludes zero) but its ranking AUC (0.8895) stays below the RJ "
            "baseline (0.8975), so the gate refused the ship and the "
            "abstaining physics fallback stays in charge. Honest record: "
            "artifacts/hazard_training/earthquake_precursor.eval.json (committed)."
        ),
        pipeline_module="omni_mercury_engine.ml.hazard_training.earthquake_precursor",
        checkpoint_name="earthquake_precursor_ca",
    ),
    "seismic_wave": HookEntry(
        name="seismic_wave",
        detector="detectors.geological.disaster_detectors.EarthquakeDetector",
        architecture="SeismicWaveAnalyzer (spectrogram CNN; eq/magnitude/P/S heads)",
        category="a",
        data_requirement=(
            "STEAD -- the STanford EArthquake Dataset (Mousavi et al. 2019, "
            "CC-BY-4.0): 1.27M labeled 60 s 100 Hz traces served by the public "
            "SeisBench mirror (seisbench.gfz.de). metadata.csv is downloaded "
            "whole and sha256-pinned; the balanced Z-component subset is "
            "streamed out of the 91 GB waveforms.hdf5 via HTTP Range requests "
            "(never downloaded whole). The merit gate compares against the "
            "detector's STA/LTA + band-resonance physics fallback through the "
            "public predict_earthquake API with deployed-rule recall/FAR "
            "non-regression constraints; 'physics wins, not shipped' is a "
            "valid recorded outcome."
        ),
        pipeline_module="omni_mercury_engine.ml.hazard_training.seismic_wave",
        checkpoint_name="seismic_stead",
    ),
    "tsunami_waveform": HookEntry(
        name="tsunami_waveform",
        detector="detectors.geological.disaster_detectors.TsunamiDetector",
        architecture="WaveformFFTAnalyzer (Conv1d+LSTM over detided DART windows)",
        category="a",
        data_requirement=(
            "NOAA NDBC DART historical bottom-pressure archives "
            "(www.ndbc.noaa.gov/data/historical/dart/, public) as detided "
            "24 h windows, labeled with deep-ocean BPR arrival times and "
            "measured amplitudes from the NCEI HazEL tsunami event/runup "
            "database (www.ngdc.noaa.gov/hazel/, public), station ids "
            "cross-checked against the NDBC station table. The merit gate "
            "adds deployed-rule recall/FAR, event recall, and wave-height "
            "MAE non-regression constraints; 'physics wins, not shipped' is "
            "a valid recorded outcome."
        ),
        pipeline_module="omni_mercury_engine.ml.hazard_training.tsunami_waveform",
        checkpoint_name="tsunami_dart",
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
        category="a",
        data_requirement=(
            "NEXRAD Level-II Doppler velocity volumes from the public Unidata "
            "mirror (unidata-nexrad-level2.s3.amazonaws.com, anonymous, "
            "per-scan objects; the AWS noaa-nexrad-level2 bucket itself is "
            "not allowlisted here) labeled with SPC WCM tornado reports "
            "(www.spc.noaa.gov), SPC wind/hail reports for storm-day hard "
            "negatives, and the NCEI HOMR WSR-88D site table for "
            "range/azimuth geometry."
        ),
        pipeline_module="omni_mercury_engine.ml.hazard_training.tornado_radar",
        checkpoint_name="tornado_nexrad",
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
        category="a",
        data_requirement=(
            "NASA FIRMS VIIRS-SNPP science-quality active-fire country CSVs "
            "(firms.modaps.eosdis.nasa.gov/data/country/, keyless, US "
            "2012-2024) rasterized to a 0.04-degree daily California grid: "
            "32x32 detection-derived patches (brightness-temperature / FRP / "
            "count channels from days <= t only) labeled with next-day "
            "confirmed fire activity in the center 2x2 cells. The archive is "
            "a census of satellite-confirmed fires, so absence means 'no "
            "confirmed fire', never a fabricated background class; the "
            "physics persistence baseline must be beaten on held-out years "
            "with recall/false-alarm/Brier non-regression at the deployed "
            "alert decision."
        ),
        pipeline_module="omni_mercury_engine.ml.hazard_training.wildfire_ignition",
        checkpoint_name="wildfire_firms",
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
