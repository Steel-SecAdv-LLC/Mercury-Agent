# Hazard Visualization & Diagnostics (T3)

Applies to Mercury Agent v2.1.x.

Hazard detectors can persist the intermediate arrays they previously
discarded and render them as deterministic artifacts. Everything drawn comes
from **actual detector intermediate data** — no visualization path fabricates
arrays, and paths that cannot produce a claimed artifact refuse loudly
instead of drawing something invented.

## Capturing diagnostics

Diagnostics capture is **opt-in** (default off, zero cost on existing paths):

```python
from omni_mercury_engine.detectors.geological.disaster_detectors import EarthquakeDetector

detector = EarthquakeDetector(keep_diagnostics=True)
result = detector.predict_earthquake(waveform)
diag = result.diagnostics   # HazardDiagnostics or None (honestly absent)
```

`HazardDiagnostics` (`detectors/hazard_diagnostics.py`) carries the named
numpy arrays per hazard, round-trips through JSON and NPZ, and validates its
hazard vocabulary (`KNOWN_HAZARDS`).

Per-hazard intermediates:

| Hazard | Arrays captured | Rendered as |
|---|---|---|
| earthquake | spectrogram (freqs/times/magnitude), STA/LTA series | 2-D spectrogram PNG (+ STA/LTA panel) |
| tsunami | wave FFT spectrum | 1-D spectrum PNG |
| tornado | Doppler velocity field | field PNG |
| hurricane | wind / vorticity fields | field PNG |
| wildfire | channel-max thermal map, hotspot mask, ignition centroids | thermal PNG + RFC 7946 GeoJSON |
| meteor | Doppler shift profile | 1-D series PNG |
| schumann | harmonic power spectrum | 1-D spectrum PNG |
| volcanic / landslide | score series | 1-D series PNG |

**Honest scope notes**

* **Schumann/ELF**: the detector computes a **1-D harmonic power spectrum**
  (Welch PSD over the record), not a time–frequency spectrogram — no
  windowed time axis exists on its compute path, so none is drawn. A
  time-frequency ELF spectrogram would require streaming instrument data the
  BGS client only receives in caller-supplied instrument mode.
* **Hurricane track cones are NOT rendered**: no track model exists in the
  codebase (`hurricane_detector.py` removed its fabricated track fields in
  the honesty wave), so only the real computed wind/vorticity fields are
  drawn.
* **Flood/landslide zones**: `FloodPredictionResult`/`LandslidePredictionResult`
  zone fields are evacuation-route/shelter **string labels**, not geometry.
  Requesting GeoJSON for landslide raises with that reason; flood emits no
  diagnostics payload at all. Only wildfire produces genuine spatial output
  (hotspot pixels), and its GeoJSON requires a caller-supplied
  pixel→WGS84 geotransform — coordinates are never fabricated.

## Rendering

`detectors/hazard_visuals.py` renders deterministic PNGs (Agg backend,
`savefig` with fixed dpi/metadata — byte-stable for identical input) and
RFC 7946 GeoJSON:

```python
from omni_mercury_engine.detectors.hazard_visuals import (
    render_hazard_png, build_hazard_geojson,
)

png_bytes = render_hazard_png(diag)
feature_collection = build_hazard_geojson(
    wildfire_diag,
    geotransform={"origin_lon": -120.0, "origin_lat": 40.0,
                  "deg_per_pixel_lon": 0.01, "deg_per_pixel_lat": -0.01},
)
```

## Surfaces

* **CLI**: `mercury-agent hazard-viz --input diag.npz --out out.png`
  (also `--detector <name> --data <input>` to run a detector with
  `keep_diagnostics=True` and render in one step).
* **HTTP**: `POST /api/v1/hazard/visualize` — accepts a JSON diagnostics
  payload, returns base64 PNG or GeoJSON; bad hazards / missing arrays /
  missing geotransform are HTTP 400 with the code-level reason.
* **MCP**: `mercury_hazard_visualize` tool (same payload contract).
* **Dashboard**: `gui/visualization_dashboard.py` panels read real artifact
  metadata (hazard, array names, shapes) from the diagnostics payloads.

## Tests

`tests/detectors/test_hazard_diagnostics.py` (capture: arrays match the
compute path, off-by-default, honestly-absent cases),
`tests/detectors/test_hazard_visuals.py` (PNG magic bytes + determinism,
RFC 7946 structure, refusal paths), `tests/test_cli_hazard_viz.py`,
`tests/api/*`, `tests/test_mcp_hazard_visualize.py`,
`tests/test_hazard_dashboard.py`.
