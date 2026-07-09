# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render hazard detector diagnostics into artifacts (PNG / GeoJSON).

Pure functions over :class:`~omni_mercury_engine.detectors.hazard_diagnostics.HazardDiagnostics`
payloads. Every pixel rendered here comes from an array a detector genuinely
computed -- nothing is synthesized to fill a panel.

PNG rendering is deterministic: fixed figure size, DPI, colormaps, no
timestamps in the image or its metadata, and matplotlib's object-oriented API
(no pyplot global state). Rendering the same payload twice yields identical
bytes.

matplotlib is an optional dependency (the ``[benchmark]`` extra); importing
this module is cheap and the matplotlib import is gated inside the render
functions with a clear error naming the extra.

GeoJSON: only the wildfire detector produces a real spatial output (hotspot
pixel locations from its thermal mask). RFC 7946 mandates WGS84 lon/lat, and
the detector has no geotransform, so :func:`build_hazard_geojson` requires the
caller to supply one -- it refuses to invent coordinates. The flood and
landslide detectors compute NO zonal output at all (their zone fields are
string labels; see ``flood_detector.py`` / ``landslide.py``), so requesting
GeoJSON for them fails loudly with that code-level reason.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.detectors.hazard_diagnostics import HazardDiagnostics

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Deterministic rendering constants (never derived from wall-clock state).
_FIGSIZE = (8.0, 5.0)
_DPI = 100
_SPECTROGRAM_CMAP = "viridis"
_DOPPLER_CMAP = "RdBu_r"
_THERMAL_CMAP = "inferno"
_WIND_CMAP = "viridis"
_VORTICITY_CMAP = "RdBu_r"
_PNG_METADATA = {"Software": "mercury-agent hazard-visuals"}

_MATPLOTLIB_HINT = (
    "matplotlib is required for hazard PNG rendering. Install it with "
    "'pip install matplotlib' (it ships with the mercury-agent[benchmark] extra)."
)


def _new_figure() -> Figure:
    """Create a deterministic Agg-backed figure (gated matplotlib import).

    Returns:
        A fresh matplotlib figure bound to the Agg canvas.

    Raises:
        ImportError: If matplotlib is not installed, naming the extra.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ImportError(_MATPLOTLIB_HINT) from exc

    fig = Figure(figsize=_FIGSIZE, dpi=_DPI)
    FigureCanvasAgg(fig)
    return fig


def _finish(fig: Figure) -> bytes:
    """Serialize a figure to deterministic PNG bytes."""
    fig.set_layout_engine("tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, metadata=dict(_PNG_METADATA))
    return buf.getvalue()


def _coerce(diagnostics: HazardDiagnostics | dict[str, Any]) -> HazardDiagnostics:
    """Accept either a payload object or its ``to_jsonable`` dict form."""
    if isinstance(diagnostics, HazardDiagnostics):
        return diagnostics
    return HazardDiagnostics.from_jsonable(diagnostics)


def _require(diag: HazardDiagnostics, expected_hazard: str, *names: str) -> None:
    """Fail loud when a payload lacks the arrays a renderer needs."""
    if diag.hazard != expected_hazard:
        raise ValueError(
            f"this renderer draws {expected_hazard!r} diagnostics, got {diag.hazard!r}"
        )
    missing = [n for n in names if n not in diag.arrays]
    if missing:
        raise ValueError(
            f"{expected_hazard} diagnostics payload is missing arrays {missing}; "
            f"present: {sorted(diag.arrays)}"
        )


def render_spectrogram(diagnostics: HazardDiagnostics | dict[str, Any]) -> bytes:
    """Render the earthquake spectrogram (+ STA/LTA panel when present) as PNG.

    Args:
        diagnostics: An ``earthquake`` payload with ``spectrogram_freqs_hz``,
            ``spectrogram_times_s``, ``spectrogram_norm`` and optionally
            ``sta_lta_ratio`` (with P/S arrival indices in the context).

    Returns:
        Deterministic PNG bytes.

    Raises:
        ValueError: On a wrong-hazard or incomplete payload.
        ImportError: If matplotlib is unavailable.
    """
    diag = _coerce(diagnostics)
    _require(diag, "earthquake", "spectrogram_freqs_hz", "spectrogram_times_s", "spectrogram_norm")

    f = diag.arrays["spectrogram_freqs_hz"]
    t = diag.arrays["spectrogram_times_s"]
    sxx = diag.arrays["spectrogram_norm"]
    sta_lta = diag.arrays.get("sta_lta_ratio")

    fig = _new_figure()
    if sta_lta is not None:
        ax_spec, ax_sta = fig.subplots(2, 1, height_ratios=[2, 1])
    else:
        ax_spec = fig.subplots(1, 1)
        ax_sta = None

    mesh = ax_spec.pcolormesh(t, f, sxx, cmap=_SPECTROGRAM_CMAP, shading="auto")
    fig.colorbar(mesh, ax=ax_spec, label="normalized log power")
    ax_spec.set_xlabel("time (s)")
    ax_spec.set_ylabel("frequency (Hz)")
    ax_spec.set_title("Seismic spectrogram (normalized log power)")

    if ax_sta is not None and sta_lta is not None:
        fs = float(diag.context.get("sampling_rate_hz", 1.0))
        times = np.arange(len(sta_lta)) / fs
        ax_sta.plot(times, sta_lta, color="#333333", linewidth=0.8)
        for key, color, label in (
            ("p_arrival_index", "#d62728", "P arrival"),
            ("s_arrival_index", "#1f77b4", "S arrival"),
        ):
            idx = diag.context.get(key)
            if idx is not None:
                ax_sta.axvline(idx / fs, color=color, linestyle="--", label=label)
        ax_sta.set_xlabel("time (s)")
        ax_sta.set_ylabel("STA/LTA")
        if diag.context.get("p_arrival_index") is not None or (
            diag.context.get("s_arrival_index") is not None
        ):
            ax_sta.legend(loc="upper right", fontsize=8)
    return _finish(fig)


def render_doppler_field(diagnostics: HazardDiagnostics | dict[str, Any]) -> bytes:
    """Render the tornado Doppler velocity field with the located couplet as PNG.

    Args:
        diagnostics: A ``tornado`` payload with ``doppler_velocity_field`` and
            couplet coordinates in the context (marked only when present).

    Returns:
        Deterministic PNG bytes.

    Raises:
        ValueError: On a wrong-hazard or incomplete payload.
        ImportError: If matplotlib is unavailable.
    """
    diag = _coerce(diagnostics)
    _require(diag, "tornado", "doppler_velocity_field")
    field = diag.arrays["doppler_velocity_field"]

    fig = _new_figure()
    ax = fig.subplots(1, 1)
    vmax = float(np.max(np.abs(field))) or 1.0
    mesh = ax.pcolormesh(field, cmap=_DOPPLER_CMAP, vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(mesh, ax=ax, label="radial velocity")
    ax.set_xlabel("range gate")
    ax.set_ylabel("sweep")
    ax.set_title("Doppler radial velocity field")

    row = diag.context.get("couplet_row")
    col = diag.context.get("couplet_col")
    if row is not None and col is not None:
        # +0.5 centers the marker on the pcolormesh cell.
        ax.plot(
            [col + 0.5, col + 1.5],
            [row + 0.5, row + 0.5],
            marker="o",
            color="#000000",
            markersize=6,
            linewidth=1.5,
        )
        ax.annotate(
            "velocity couplet",
            xy=(col + 1.0, row + 0.5),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    return _finish(fig)


def render_thermal_map(diagnostics: HazardDiagnostics | dict[str, Any]) -> bytes:
    """Render the wildfire thermal map with hotspot-mask contours as PNG.

    Args:
        diagnostics: A ``wildfire`` payload with ``thermal_image_k`` and
            ``hotspot_mask`` (ignition centroids are marked when present).

    Returns:
        Deterministic PNG bytes.

    Raises:
        ValueError: On a wrong-hazard or incomplete payload.
        ImportError: If matplotlib is unavailable.
    """
    diag = _coerce(diagnostics)
    _require(diag, "wildfire", "thermal_image_k", "hotspot_mask")
    thermal = diag.arrays["thermal_image_k"]
    mask = diag.arrays["hotspot_mask"].astype(bool)

    fig = _new_figure()
    ax = fig.subplots(1, 1)
    image = ax.imshow(thermal, cmap=_THERMAL_CMAP, origin="upper", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="brightness temperature (K)")
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors="#00e5ff", linewidths=1.2)
    centroids = diag.arrays.get("ignition_centroids")
    if centroids is not None and len(centroids):
        ax.scatter(
            centroids[:, 1],
            centroids[:, 0],
            marker="x",
            s=60,
            color="#00e5ff",
            label="ignition centroid (pixel space)",
        )
        ax.legend(loc="upper right", fontsize=8)
    threshold = diag.context.get("hotspot_threshold_k", "?")
    ax.set_title(f"Thermal map with hotspot mask (> {threshold} K)")
    ax.set_xlabel("pixel column")
    ax.set_ylabel("pixel row")
    return _finish(fig)


def render_wind_field(diagnostics: HazardDiagnostics | dict[str, Any]) -> bytes:
    """Render the hurricane wind-speed field (and vorticity when present) as PNG.

    No storm-track cone is drawn: the track model was removed as uncomputed and
    this renderer draws only the fields the detector derived.

    Args:
        diagnostics: A ``hurricane`` payload with ``wind_speed_field`` and
            optionally ``wind_u``/``wind_v``/``vorticity_field``.

    Returns:
        Deterministic PNG bytes.

    Raises:
        ValueError: On a wrong-hazard or incomplete payload.
        ImportError: If matplotlib is unavailable.
    """
    diag = _coerce(diagnostics)
    _require(diag, "hurricane", "wind_speed_field")
    speed = diag.arrays["wind_speed_field"]
    vorticity = diag.arrays.get("vorticity_field")

    fig = _new_figure()
    if vorticity is not None:
        ax_speed, ax_vort = fig.subplots(1, 2)
    else:
        ax_speed = fig.subplots(1, 1)
        ax_vort = None

    mesh = ax_speed.pcolormesh(speed, cmap=_WIND_CMAP, shading="auto")
    fig.colorbar(mesh, ax=ax_speed, label="wind speed")
    ax_speed.set_title("Wind speed field")
    ax_speed.set_xlabel("x")
    ax_speed.set_ylabel("y")

    u = diag.arrays.get("wind_u")
    v = diag.arrays.get("wind_v")
    if u is not None and v is not None:
        # Down-sample the quiver deterministically so arrows stay legible.
        step = max(1, max(u.shape) // 16)
        yy, xx = np.mgrid[0 : u.shape[0] : step, 0 : u.shape[1] : step]
        ax_speed.quiver(
            xx + 0.5,
            yy + 0.5,
            u[::step, ::step],
            v[::step, ::step],
            color="#ffffff",
            width=0.003,
            scale_units="xy",
        )

    if ax_vort is not None and vorticity is not None:
        vmax = float(np.max(np.abs(vorticity))) or 1.0
        mesh_v = ax_vort.pcolormesh(
            vorticity, cmap=_VORTICITY_CMAP, vmin=-vmax, vmax=vmax, shading="auto"
        )
        fig.colorbar(mesh_v, ax=ax_vort, label="relative vorticity (1/s)")
        ax_vort.set_title("Vorticity (dv/dx - du/dy)")
        ax_vort.set_xlabel("x")
    return _finish(fig)


def render_power_spectrum(diagnostics: HazardDiagnostics | dict[str, Any]) -> bytes:
    """Render a 1-D diagnostic series/spectrum (tsunami, schumann, meteor) as PNG.

    - ``tsunami``: FFT power vs frequency (positive-frequency half of the
      two-sided spectrum, exactly as captured).
    - ``schumann``: one-sided harmonic power spectrum with the canonical
      Schumann harmonics marked.
    - ``meteor``: the Doppler shift profile vs sample index.

    Args:
        diagnostics: A payload from one of the hazards above.

    Returns:
        Deterministic PNG bytes.

    Raises:
        ValueError: On an unsupported hazard or incomplete payload.
        ImportError: If matplotlib is unavailable.
    """
    diag = _coerce(diagnostics)
    fig = _new_figure()
    ax = fig.subplots(1, 1)

    if diag.hazard == "tsunami":
        _require(diag, "tsunami", "fft_freqs_hz", "fft_power")
        freqs = diag.arrays["fft_freqs_hz"]
        power = diag.arrays["fft_power"]
        positive = freqs > 0
        ax.semilogy(freqs[positive], power[positive] + 1e-30, color="#1f77b4", linewidth=0.9)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("power")
        ax.set_title("Oceanic waveform FFT power spectrum")
    elif diag.hazard == "schumann":
        _require(diag, "schumann", "frequencies_hz", "power_spectrum")
        freqs = diag.arrays["frequencies_hz"]
        power = diag.arrays["power_spectrum"]
        band = freqs <= 45.0
        ax.plot(freqs[band], power[band], color="#1f77b4", linewidth=0.9)
        for harmonic in diag.context.get("schumann_harmonics_hz", []):
            ax.axvline(float(harmonic), color="#d62728", linestyle=":", linewidth=0.8)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("normalized power")
        ax.set_title("Schumann harmonic power spectrum (harmonics dotted)")
    elif diag.hazard == "meteor":
        _require(diag, "meteor", "doppler_shift_profile")
        profile = diag.arrays["doppler_shift_profile"]
        ax.plot(np.arange(len(profile)), profile, color="#1f77b4", linewidth=0.9)
        ax.axhline(float(np.mean(profile)), color="#d62728", linestyle="--", linewidth=0.8)
        ax.set_xlabel("sample")
        ax.set_ylabel("Doppler shift (first difference)")
        ax.set_title("Radar Doppler shift profile (mean dashed)")
    else:
        raise ValueError(
            f"render_power_spectrum draws tsunami/schumann/meteor payloads, got {diag.hazard!r}"
        )
    return _finish(fig)


def render_score_series(diagnostics: HazardDiagnostics | dict[str, Any]) -> bytes:
    """Render volcanic attention / landslide type-probability diagnostics as PNG.

    - ``volcanic``: the per-timestep seismic swarm attention series, plus the
      HMM state-belief bars when present.
    - ``landslide``: the failure-type probability distribution as bars.

    Args:
        diagnostics: A payload from one of the hazards above.

    Returns:
        Deterministic PNG bytes.

    Raises:
        ValueError: On an unsupported hazard or incomplete payload.
        ImportError: If matplotlib is unavailable.
    """
    diag = _coerce(diagnostics)
    fig = _new_figure()

    if diag.hazard == "volcanic":
        _require(diag, "volcanic", "seismic_attention")
        attention = diag.arrays["seismic_attention"]
        belief = diag.arrays.get("hmm_state_belief")
        if belief is not None:
            ax_att, ax_belief = fig.subplots(1, 2, width_ratios=[2, 1])
        else:
            ax_att = fig.subplots(1, 1)
            ax_belief = None
        ax_att.plot(np.arange(len(attention)), attention, color="#1f77b4", linewidth=1.0)
        ax_att.set_xlabel("timestep")
        ax_att.set_ylabel("attention weight")
        ax_att.set_title("Seismic swarm attention series")
        if ax_belief is not None and belief is not None:
            names = diag.context.get("hmm_state_names") or [str(i) for i in range(len(belief))]
            ax_belief.bar(np.arange(len(belief)), belief, color="#1f77b4")
            ax_belief.set_xticks(np.arange(len(belief)))
            ax_belief.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax_belief.set_ylabel("belief")
            ax_belief.set_title("HMM state belief")
    elif diag.hazard == "landslide":
        _require(diag, "landslide", "failure_type_probs")
        probs = diag.arrays["failure_type_probs"]
        labels = diag.context.get("failure_type_labels") or [str(i) for i in range(len(probs))]
        ax = fig.subplots(1, 1)
        ax.bar(np.arange(len(probs)), probs, color="#1f77b4")
        ax.set_xticks(np.arange(len(probs)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("probability")
        ax.set_title("Slope failure type distribution")
    else:
        raise ValueError(
            f"render_score_series draws volcanic/landslide payloads, got {diag.hazard!r}"
        )
    return _finish(fig)


def render_hazard_png(diagnostics: HazardDiagnostics | dict[str, Any]) -> bytes:
    """Render any hazard diagnostics payload to PNG via the per-hazard renderer.

    Args:
        diagnostics: Any :class:`HazardDiagnostics` payload (or its dict form).

    Returns:
        Deterministic PNG bytes.

    Raises:
        ValueError: On an unknown hazard or incomplete payload.
        ImportError: If matplotlib is unavailable.
    """
    diag = _coerce(diagnostics)
    renderers = {
        "earthquake": render_spectrogram,
        "tornado": render_doppler_field,
        "wildfire": render_thermal_map,
        "hurricane": render_wind_field,
        "tsunami": render_power_spectrum,
        "schumann": render_power_spectrum,
        "meteor": render_power_spectrum,
        "volcanic": render_score_series,
        "landslide": render_score_series,
    }
    renderer = renderers.get(diag.hazard)
    if renderer is None:
        raise ValueError(f"no PNG renderer exists for hazard {diag.hazard!r}")
    return renderer(diag)


def build_hazard_geojson(
    diagnostics: HazardDiagnostics | dict[str, Any],
    geotransform: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build an RFC 7946 FeatureCollection from real spatial detector output.

    Only the wildfire detector produces a genuine spatial output: hotspot
    pixel locations from its thermal mask. Each connected hotspot region
    becomes one Point feature at its centroid. Feature properties record the
    provenance exactly: ``pixel_row``/``pixel_col`` come from the connected-
    component centroid on the ``hotspot_mask`` array, ``component_pixels``
    from the component size, ``hotspot_threshold_k`` from the detector's
    threshold, and ``component_area_km2`` from ``component_pixels *
    pixel_size_km**2`` only when the detector was given a ground resolution.

    RFC 7946 mandates WGS84 lon/lat, and the detector works in pixel space
    with no georeferencing, so the caller MUST supply the mapping; this
    function refuses to fabricate coordinates.

    The flood and landslide detectors compute no zonal output at all --
    ``FloodPredictionResult`` zone fields are strings (evacuation route /
    shelter labels in ``flood_detector.py``) and ``LandslidePredictionResult``
    evacuation zones are string labels (``landslide.py``) -- so requesting
    GeoJSON for them raises with that reason.

    Args:
        diagnostics: A ``wildfire`` :class:`HazardDiagnostics` payload.
        geotransform: Pixel->WGS84 affine mapping with keys ``origin_lon``,
            ``origin_lat`` (the lon/lat of pixel (row=0, col=0) center),
            ``deg_per_pixel_lon`` and ``deg_per_pixel_lat`` (typically negative
            when row index grows southward).

    Returns:
        An RFC 7946 ``FeatureCollection`` dict.

    Raises:
        ValueError: For non-wildfire hazards (with the code-level reason) or a
            missing/incomplete geotransform.
    """
    diag = _coerce(diagnostics)
    if diag.hazard in ("flood", "landslide"):
        raise ValueError(
            f"the {diag.hazard} detector computes no zonal/geographic output -- its zone "
            "fields are string labels (see flood_detector.py / landslide.py) -- so there "
            "is nothing real to map as GeoJSON"
        )
    if diag.hazard != "wildfire":
        raise ValueError(
            f"GeoJSON is only derivable from wildfire diagnostics (hotspot pixel "
            f"locations); the {diag.hazard} detector emits no coordinates"
        )
    _require(diag, "wildfire", "hotspot_mask", "ignition_centroids", "ignition_component_sizes")

    if geotransform is None:
        raise ValueError(
            "RFC 7946 GeoJSON requires WGS84 lon/lat, but wildfire ignition locations are "
            "pixel-space (row, col) coordinates with no geotransform; supply one with "
            "origin_lon/origin_lat/deg_per_pixel_lon/deg_per_pixel_lat rather than have "
            "coordinates fabricated"
        )
    required_keys = ("origin_lon", "origin_lat", "deg_per_pixel_lon", "deg_per_pixel_lat")
    missing = [k for k in required_keys if k not in geotransform]
    if missing:
        raise ValueError(f"geotransform is missing keys {missing}; required: {required_keys}")

    origin_lon = float(geotransform["origin_lon"])
    origin_lat = float(geotransform["origin_lat"])
    dlon = float(geotransform["deg_per_pixel_lon"])
    dlat = float(geotransform["deg_per_pixel_lat"])

    centroids = diag.arrays["ignition_centroids"]
    sizes = diag.arrays["ignition_component_sizes"]
    threshold_k = diag.context.get("hotspot_threshold_k")
    pixel_size_km = diag.context.get("pixel_size_km")

    features: list[dict[str, Any]] = []
    for (row, col), size in zip(centroids, sizes):
        lon = origin_lon + float(col) * dlon
        lat = origin_lat + float(row) * dlat
        properties: dict[str, Any] = {
            "source": "wildfire_ignition_hotspot",
            "provenance": "connected-component centroid of the >threshold thermal mask",
            "pixel_row": float(row),
            "pixel_col": float(col),
            "component_pixels": int(size),
            "hotspot_threshold_k": threshold_k,
        }
        if pixel_size_km is not None:
            properties["component_area_km2"] = float(int(size) * float(pixel_size_km) ** 2)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "features": features}


__all__ = [
    "build_hazard_geojson",
    "render_doppler_field",
    "render_hazard_png",
    "render_power_spectrum",
    "render_score_series",
    "render_spectrogram",
    "render_thermal_map",
    "render_wind_field",
]
