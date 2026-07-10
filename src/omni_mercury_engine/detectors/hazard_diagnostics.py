# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hazard detector diagnostics payload.

The hazard detectors (earthquake, tsunami, meteor, wildfire, tornado, hurricane,
volcanic, landslide, Schumann) all compute rich intermediate arrays -- spectrograms,
STA/LTA ratio series, FFT power spectra, Doppler velocity fields, thermal hotspot
masks, attention series -- that historically were reduced to scalars and discarded.
:class:`HazardDiagnostics` is the single, uniform container those detectors use to
*persist* the arrays they genuinely computed, opt-in via each detector's
``keep_diagnostics`` constructor flag (default ``False``: no memory overhead, and the
result's ``diagnostics`` field stays ``None`` -- absent, never an empty fake).

Array vocabulary (every array is captured from the real compute path, never
fabricated; names are stable because the renderers in
:mod:`omni_mercury_engine.detectors.hazard_visuals` key on them):

- ``earthquake``: ``spectrogram_freqs_hz`` (F,), ``spectrogram_times_s`` (T,),
  ``spectrogram_norm`` (F, T) -- the log-normalized scipy spectrogram fed to the
  CNN -- and ``sta_lta_ratio`` (N,), the STA/LTA arrival-detection series.
  Context: ``sampling_rate_hz``, ``p_arrival_index``, ``s_arrival_index``.
- ``tsunami``: ``fft_freqs_hz`` and ``fft_power`` exactly as computed by the
  resonance scan (full two-sided FFT ordering). Context: ``sampling_rate_hz``,
  ``resonance_score``.
- ``meteor``: ``doppler_shift_profile`` (N-1,) -- the first-difference Doppler
  profile of the radar series whose mean drives the velocity estimate.
- ``wildfire``: ``thermal_image_k`` (H, W) channel-max brightness temperature,
  ``hotspot_mask`` (H, W) bool (> ``hotspot_threshold_k``), ``ignition_pixels``
  (K, 2) int hotspot pixel (row, col) coordinates, ``ignition_centroids`` (C, 2)
  connected-component centroids, ``ignition_component_sizes`` (C,). Context:
  ``hotspot_threshold_k``, ``hotspot_count``, ``coordinate_space`` (``"pixel"``),
  and ``pixel_size_km`` when the caller supplied one.
- ``tornado``: ``doppler_velocity_field`` (T, G) -- the radar velocity field the
  LSTM consumed -- and ``radar_attention`` (T,), its attention weights. Context:
  ``couplet_row``, ``couplet_col``, ``couplet_shear`` (max adjacent-gate shear;
  the classic velocity-couplet signature located on the consumed field).
- ``hurricane``: ``wind_speed_field`` (H, W), and when u/v components were
  provided ``wind_u``, ``wind_v``, ``vorticity_field`` (dv/dx - du/dy via central
  finite differences). Context: ``max_wind_speed``, ``max_abs_vorticity``,
  ``mean_vorticity``, ``grid_spacing_m``. No storm-track cone is emitted anywhere:
  the track model was removed as uncomputed and remains so.
- ``volcanic``: ``seismic_attention`` (T,) -- the per-timestep swarm attention
  series -- and ``hmm_state_belief`` (5,). Context: ``swarm_probability``,
  ``hmm_state_names``, ``hmm_state``.
- ``landslide``: ``failure_type_probs`` (6,) -- the softmax distribution the
  argmax previously discarded. Context: ``failure_type_labels``,
  ``failure_probability``. The landslide detector computes NO zonal/geographic
  output (its evacuation zones are string labels), so no coordinates exist here.
- ``schumann``: ``frequencies_hz`` and ``power_spectrum`` -- the one-sided,
  max-normalized ELF power spectrum. Context: ``sampling_rate_hz``,
  ``fundamental_freq_hz``, ``fundamental_power``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

KNOWN_HAZARDS: tuple[str, ...] = (
    "earthquake",
    "tsunami",
    "meteor",
    "wildfire",
    "tornado",
    "hurricane",
    "volcanic",
    "landslide",
    "schumann",
)

_NPZ_HAZARD_KEY = "__hazard__"
_NPZ_CONTEXT_KEY = "__context__"


@dataclass
class HazardDiagnostics:
    """Intermediate arrays a hazard detector genuinely computed for one prediction.

    Attributes:
        hazard: Which detector produced this payload (one of :data:`KNOWN_HAZARDS`).
        arrays: Named numpy arrays captured from the compute path (see the module
            docstring for the per-hazard vocabulary).
        context: JSON-serializable scalars that situate the arrays (sampling rates,
            thresholds, detected indices). Values are numbers, strings, booleans,
            ``None``, or flat lists thereof.
    """

    hazard: str
    arrays: dict[str, np.ndarray[Any, Any]]
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the hazard name and array payload eagerly (fail loud)."""
        if self.hazard not in KNOWN_HAZARDS:
            raise ValueError(
                f"unknown hazard {self.hazard!r}; expected one of {sorted(KNOWN_HAZARDS)}"
            )
        if not self.arrays:
            raise ValueError(
                "HazardDiagnostics requires at least one captured array; an empty "
                "payload would be an empty fake, not a diagnostic"
            )
        for name, arr in self.arrays.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"diagnostics array {name!r} must be a numpy array")

    # -- serialization ------------------------------------------------------

    def to_jsonable(self) -> dict[str, Any]:
        """Return a pure-JSON representation (arrays become nested lists).

        Returns:
            ``{"hazard": ..., "arrays": {...}, "context": {...}}`` suitable for
            HTTP/MCP transport and :meth:`from_jsonable` round-trips.
        """
        return {
            "hazard": self.hazard,
            "arrays": {name: arr.tolist() for name, arr in self.arrays.items()},
            "context": dict(self.context),
        }

    @classmethod
    def from_jsonable(cls, payload: dict[str, Any]) -> HazardDiagnostics:
        """Rebuild a payload produced by :meth:`to_jsonable`.

        Args:
            payload: A dict with ``hazard`` (str), ``arrays`` (dict of nested
                numeric/boolean lists), and optional ``context`` (dict).

        Returns:
            The reconstructed :class:`HazardDiagnostics`.

        Raises:
            ValueError: If the payload is structurally invalid or an array is
                not numeric/boolean.
        """
        if not isinstance(payload, dict):
            raise ValueError("diagnostics payload must be an object")
        hazard = payload.get("hazard")
        if not isinstance(hazard, str):
            raise ValueError("diagnostics payload requires a string 'hazard'")
        raw_arrays = payload.get("arrays")
        if not isinstance(raw_arrays, dict) or not raw_arrays:
            raise ValueError("diagnostics payload requires a non-empty 'arrays' object")
        arrays: dict[str, np.ndarray[Any, Any]] = {}
        for name, value in raw_arrays.items():
            try:
                arr = np.asarray(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"diagnostics array {name!r} is not array-like: {exc}") from exc
            if arr.dtype == object or arr.dtype.kind not in "bifu":
                raise ValueError(
                    f"diagnostics array {name!r} must be numeric or boolean, "
                    f"got dtype {arr.dtype}"
                )
            arrays[name] = arr
        context = payload.get("context") or {}
        if not isinstance(context, dict):
            raise ValueError("diagnostics 'context' must be an object")
        return cls(hazard=hazard, arrays=arrays, context=dict(context))

    def to_npz(self, path: str | Path) -> None:
        """Persist the payload to a compressed ``.npz`` file.

        Args:
            path: Destination file path.
        """
        members: dict[str, np.ndarray[Any, Any]] = {
            **self.arrays,
            _NPZ_HAZARD_KEY: np.array(self.hazard),
            _NPZ_CONTEXT_KEY: np.array(json.dumps(self.context)),
        }
        # numpy's stub folds **kwds into allow_pickle's type; the call is valid.
        np.savez_compressed(Path(path), **members)  # type: ignore[arg-type]

    @classmethod
    def from_npz(cls, path: str | Path) -> HazardDiagnostics:
        """Load a payload previously written by :meth:`to_npz`.

        Args:
            path: Source ``.npz`` file path.

        Returns:
            The reconstructed :class:`HazardDiagnostics`.

        Raises:
            ValueError: If the file lacks the hazard/context markers.
        """
        with np.load(Path(path), allow_pickle=False) as data:
            names = set(data.files)
            if _NPZ_HAZARD_KEY not in names or _NPZ_CONTEXT_KEY not in names:
                raise ValueError(
                    f"{path} is not a HazardDiagnostics archive (missing "
                    f"{_NPZ_HAZARD_KEY}/{_NPZ_CONTEXT_KEY} markers)"
                )
            hazard = str(data[_NPZ_HAZARD_KEY])
            context = json.loads(str(data[_NPZ_CONTEXT_KEY]))
            arrays = {
                name: np.array(data[name])
                for name in data.files
                if name not in (_NPZ_HAZARD_KEY, _NPZ_CONTEXT_KEY)
            }
        return cls(hazard=hazard, arrays=arrays, context=context)


PRIMARY_INPUT_ARRAY: dict[str, str] = {
    "earthquake": "series",
    "tsunami": "series",
    "schumann": "series",
    "meteor": "radar_series",
    "wildfire": "thermal_image",
    "tornado": "radar_sequence",
    "volcanic": "seismic_sequence",
    "landslide": "slope_features",
    "hurricane": "wind_speed",
}


def _series_input(arrays: dict[str, np.ndarray[Any, Any]], name: str) -> np.ndarray[Any, Any]:
    """Fetch and ravel a required 1-D input array, failing loud when absent."""
    if name not in arrays:
        raise ValueError(f"this hazard requires input array {name!r}; got {sorted(arrays)}")
    return np.asarray(arrays[name], dtype=float).ravel()


def run_hazard_detector(
    hazard: str,
    arrays: dict[str, np.ndarray[Any, Any]],
    params: dict[str, Any] | None = None,
) -> HazardDiagnostics:
    """Run one hazard detector with diagnostics enabled and return its payload.

    This is the shared runtime behind the ``hazard-viz`` CLI, the
    ``POST /api/v1/hazard/visualize`` HTTP route, and the
    ``mercury_hazard_visualize`` MCP tool: it builds the named detector with
    ``keep_diagnostics=True``, feeds it the caller's input arrays, and returns
    the captured :class:`HazardDiagnostics`.

    Expected input arrays per hazard (see :data:`PRIMARY_INPUT_ARRAY`):

    - ``earthquake``/``tsunami``/``schumann``: ``series`` (1-D waveform);
      ``params['sampling_rate_hz']`` optional.
    - ``meteor``: ``radar_series`` (1-D radar returns). NASA CNEOS lookups are
      disabled here so the run is hermetic.
    - ``wildfire``: ``thermal_image`` shaped ``(3, H, W)`` -- the ignition CNN
      consumes 3-channel thermal imagery; ``params['pixel_size_km']`` optional.
    - ``tornado``: ``radar_sequence`` shaped ``(sweeps, 64)`` (the Doppler
      analyzer's fixed gate count).
    - ``hurricane``: ``wind_u`` + ``wind_v`` (2-D components,
      ``params['grid_spacing_m']`` optional) or ``wind_speed`` (2-D, no
      vorticity derivable).
    - ``volcanic``: ``seismic_sequence`` shaped ``(timesteps, 32)``.
    - ``landslide``: ``slope_features`` (64 features).

    Args:
        hazard: One of :data:`KNOWN_HAZARDS`.
        arrays: Named input arrays for the detector.
        params: Optional scalar parameters (see above).

    Returns:
        The diagnostics payload the detector captured.

    Raises:
        ValueError: On an unknown hazard or invalid input arrays.
        ImportError: When the detector's ML stack (torch) is unavailable.
    """
    params = params or {}
    if hazard not in KNOWN_HAZARDS:
        raise ValueError(f"unknown hazard {hazard!r}; expected one of {sorted(KNOWN_HAZARDS)}")

    diagnostics: HazardDiagnostics | None
    if hazard == "earthquake":
        from omni_mercury_engine.detectors.geological.disaster_detectors import (
            EarthquakeDetector,
        )

        series = _series_input(arrays, "series")
        detector = EarthquakeDetector(
            sampling_rate=float(params.get("sampling_rate_hz", 100.0)),
            keep_diagnostics=True,
        )
        diagnostics = detector.predict_earthquake(series).diagnostics
    elif hazard == "tsunami":
        from omni_mercury_engine.detectors.geological.disaster_detectors import TsunamiDetector

        series = _series_input(arrays, "series")
        detector_t = TsunamiDetector(
            sampling_rate=float(params.get("sampling_rate_hz", 1.0)),
            keep_diagnostics=True,
        )
        diagnostics = detector_t.predict_tsunami(series.astype(np.float32)).diagnostics
    elif hazard == "schumann":
        from omni_mercury_engine.space.schumann_resonance import SchumannResonanceDetector

        series = _series_input(arrays, "series")
        detector_s = SchumannResonanceDetector(
            sampling_rate=float(params.get("sampling_rate_hz", 100.0)),
            keep_diagnostics=True,
        )
        diagnostics = detector_s.detect_resonance_anomaly(series).diagnostics
    elif hazard == "meteor":
        from omni_mercury_engine.detectors.geological.disaster_detectors import MeteorDetector

        radar = _series_input(arrays, "radar_series")
        if len(radar) < 2:
            raise ValueError("meteor 'radar_series' needs at least 2 samples for a profile")
        detector_m = MeteorDetector(use_nasa_data=False, keep_diagnostics=True)
        diagnostics = detector_m.predict_meteor(radar_data=radar).diagnostics
    elif hazard == "wildfire":
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        if "thermal_image" not in arrays:
            raise ValueError(f"wildfire requires input array 'thermal_image'; got {sorted(arrays)}")
        thermal = np.asarray(arrays["thermal_image"], dtype=float)
        if thermal.ndim != 3 or thermal.shape[0] != 3:
            raise ValueError(
                "wildfire 'thermal_image' must be shaped (3, H, W): the ignition CNN "
                f"consumes 3-channel thermal imagery, got shape {thermal.shape}"
            )
        wildfire_data: dict[str, Any] = {"thermal_image": thermal}
        if "pixel_size_km" in params:
            wildfire_data["pixel_size_km"] = float(params["pixel_size_km"])
        detector_w = WildfireDetector(keep_diagnostics=True)
        diagnostics = detector_w.predict_wildfire(wildfire_data).diagnostics
    elif hazard == "tornado":
        from omni_mercury_engine.detectors.geological.tornado_detector import TornadoDetector

        if "radar_sequence" not in arrays:
            raise ValueError(f"tornado requires input array 'radar_sequence'; got {sorted(arrays)}")
        radar_seq = np.asarray(arrays["radar_sequence"], dtype=float)
        if radar_seq.ndim != 2 or radar_seq.shape[1] != 64:
            raise ValueError(
                "tornado 'radar_sequence' must be shaped (sweeps, 64) -- the Doppler "
                f"analyzer's fixed gate count -- got shape {radar_seq.shape}"
            )
        detector_to = TornadoDetector(keep_diagnostics=True)
        diagnostics = detector_to.predict_tornado({"radar_sequence": radar_seq}).diagnostics
    elif hazard == "hurricane":
        from omni_mercury_engine.detectors.geological.hurricane_detector import (
            HurricaneDetector,
        )

        wind_field: Any
        if "wind_u" in arrays and "wind_v" in arrays:
            wind_field = {
                "u": np.asarray(arrays["wind_u"], dtype=float),
                "v": np.asarray(arrays["wind_v"], dtype=float),
            }
            if "grid_spacing_m" in params:
                wind_field["grid_spacing_m"] = float(params["grid_spacing_m"])
        elif "wind_speed" in arrays:
            wind_field = np.asarray(arrays["wind_speed"], dtype=float)
        else:
            raise ValueError(
                "hurricane requires 'wind_u'+'wind_v' component arrays or a 'wind_speed' "
                f"field; got {sorted(arrays)}"
            )
        detector_h = HurricaneDetector(keep_diagnostics=True)
        diagnostics = detector_h.predict_hurricane({"wind_field": wind_field}).diagnostics
    elif hazard == "volcanic":
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

        if "seismic_sequence" not in arrays:
            raise ValueError(
                f"volcanic requires input array 'seismic_sequence'; got {sorted(arrays)}"
            )
        seismic = np.asarray(arrays["seismic_sequence"], dtype=float)
        if seismic.ndim != 2 or seismic.shape[1] != 32:
            raise ValueError(
                "volcanic 'seismic_sequence' must be shaped (timesteps, 32) -- the swarm "
                f"detector's feature width -- got shape {seismic.shape}"
            )
        detector_v = VolcanicEruptionDetector(keep_diagnostics=True)
        diagnostics = detector_v.predict_eruption(
            {"seismic_sequence": seismic.astype(np.float32)}
        ).diagnostics
    else:  # landslide
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        features = _series_input(arrays, "slope_features")
        if features.shape != (64,):
            raise ValueError(
                "landslide 'slope_features' must have 64 features (the stability "
                f"model's input width), got shape {features.shape}"
            )
        detector_l = LandslideDetector(enable_ml_ensemble=False, keep_diagnostics=True)
        diagnostics = detector_l.predict_landslide(
            {"slope_features": features.astype(np.float32)}
        ).diagnostics

    if diagnostics is None:
        raise RuntimeError(
            f"the {hazard} detector produced no diagnostics for this input; this is a "
            "bug in the capture path, not a rendering fallback"
        )
    return diagnostics


__all__ = ["KNOWN_HAZARDS", "PRIMARY_INPUT_ARRAY", "HazardDiagnostics", "run_hazard_detector"]
