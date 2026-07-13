# Mercury-Agent Data Sources

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-13.

Last verified: 2026-07-11 (loader catalog refresh; the per-dataset
tables below still derive from the legacy 2026-02-15 sweep — the
51-success `mercury_benchmark.py` regression-gate baseline — preserved
here as the auditable starting point). The canonical public
benchmarking figure is the committed `mercury_benchmark_results.json`
run documented in the README "Latest Benchmark Results" section:
**66 successful / 75 attempted**, Mean AUC **0.8251**, Median
**0.8747**, Mean Oracle F1 **0.5998** (2026-06-21, commit a7a194b).
The 11 watch-listed loaders (SMAP, MSL, CICIDS-2017, MIT-BIH, UCR,
SWaT, WADI, USGS Geochemistry, NOAA StormEvents, NOAA ERDDAP, FEMA
HazardMitigation) are tracked by a two-lane reachability harness
rather than counted as silent benchmark drops (9 failed in the
committed run; NOAA StormEvents and NOAA ERDDAP recovered). See `docs/BENCHMARKS.md`.

## Successfully Loading (legacy 2026-02-15 sweep: 51 datasets)

### ADBench Tabular (47 datasets)

Source: `https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/{index}_{name}.npz`
Auth: None
Loader: `ADBenchLoader` in `src/omni_mercury_engine/datasets/adbench.py`
Status: All 47 datasets download and evaluate successfully

| Index | Dataset | n_total | Anomaly Ratio |
|-------|---------|---------|---------------|
| 1 | ALOI | 49,534 | 0.030 |
| 2 | annthyroid | 7,200 | 0.074 |
| 3 | backdoor | 95,329 | 0.024 |
| 4 | breastw | 683 | 0.350 |
| 5 | campaign | 41,188 | 0.113 |
| 6 | cardio | 1,831 | 0.096 |
| 7 | Cardiotocography | 2,114 | 0.220 |
| 8 | celeba | 202,599 | 0.022 |
| 9 | census | 299,285 | 0.062 |
| 10 | cover | 286,048 | 0.010 |
| 11 | donors | 619,326 | 0.059 |
| 12 | fault | 1,941 | 0.347 |
| 13 | fraud | 284,807 | 0.002 |
| 14 | glass | 214 | 0.042 |
| 15 | Hepatitis | 80 | 0.163 |
| 16 | http | 567,498 | 0.004 |
| 17 | InternetAds | 1,966 | 0.187 |
| 18 | Ionosphere | 351 | 0.359 |
| 19 | landsat | 6,435 | 0.207 |
| 20 | letter | 1,600 | 0.063 |
| 21 | Lymphography | 148 | 0.041 |
| 22 | magic.gamma | 19,020 | 0.352 |
| 23 | mammography | 11,183 | 0.023 |
| 24 | mnist | 7,603 | 0.092 |
| 25 | musk | 3,062 | 0.032 |
| 26 | optdigits | 5,216 | 0.029 |
| 27 | PageBlocks | 5,393 | 0.095 |
| 28 | pendigits | 6,870 | 0.023 |
| 29 | Pima | 768 | 0.349 |
| 30 | satellite | 6,435 | 0.316 |
| 31 | satimage-2 | 5,803 | 0.012 |
| 32 | shuttle | 49,097 | 0.072 |
| 33 | skin | 245,057 | 0.208 |
| 34 | smtp | 95,156 | 0.000 |
| 35 | SpamBase | 4,207 | 0.399 |
| 36 | speech | 3,686 | 0.017 |
| 37 | Stamps | 340 | 0.091 |
| 38 | thyroid | 3,772 | 0.025 |
| 39 | vertebral | 240 | 0.125 |
| 40 | vowels | 1,456 | 0.034 |
| 41 | Waveform | 3,443 | 0.029 |
| 42 | WBC | 223 | 0.045 |
| 43 | WDBC | 367 | 0.027 |
| 44 | Wilt | 4,819 | 0.053 |
| 45 | wine | 129 | 0.078 |
| 46 | WPBC | 198 | 0.237 |
| 47 | yeast | 1,484 | 0.342 |

### NSL-KDD

Source:
- Train: `https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt`
- Test: `https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt`

Auth: None
Loader: `NSLKDDLoader` in `src/omni_mercury_engine/datasets/security.py`
Status: Downloads and evaluates. n_total=148,517, anomaly_ratio=0.481

### BATADAL

Source:
- Train: `https://www.batadal.net/data/BATADAL_dataset03.csv`
- Test: `https://www.batadal.net/data/BATADAL_dataset04.csv`

Auth: None
Loader: `BATADALLoader` in `src/omni_mercury_engine/datasets/industrial.py`
Status: Downloads and evaluates. n_total=12,938, anomaly_ratio=0.017
Note: ATT_FLAG values: 1=attack, 0=normal, -999=concealment (mapped to normal). Column names stripped of whitespace before concat.

### SMD (Server Machine Dataset)

Source: `https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset/{split}/{machine}.txt`
Auth: None
Loader: `SMDLoader` in `src/omni_mercury_engine/datasets/timeseries.py`
Status: Downloads and evaluates. 28 machine subsets. n_total=75,876, anomaly_ratio=0.053

### NAB (Numenta Anomaly Benchmark)

Source:
- Data: `https://raw.githubusercontent.com/numenta/NAB/master/data/`
- Labels: `https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json`

Auth: None
Loader: `NABLoader` in `src/omni_mercury_engine/datasets/timeseries.py`
Status: Downloads and evaluates. Uses realKnownCause category. n_total=69,561, anomaly_ratio=0.095

### USGS Earthquake

Source: `https://earthquake.usgs.gov/fdsnws/event/1/` (GeoJSON API)
Auth: None
Loader: `USGSEarthquakeLoader` in `src/omni_mercury_engine/datasets/environmental.py`
Status: Live API. Not included in mercury_benchmark.py suite (API-based, not static dataset).

### FEMA Disaster Declarations

Source: `https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries`
Auth: None (public domain, US Government)
Loader: `FEMADisasterLoader` in `src/omni_mercury_engine/datasets/disaster.py`
Status: Live API. Not included in mercury_benchmark.py suite (API-based, not static dataset).

### USGS Geochemistry (NURE-HSSR Stream Sediment)

Source: `https://mrdata.usgs.gov/nure/sediment/nuresed-csv.zip`
Auth: None (public domain, US Government)
Loader: `USGSGeochemistryLoader` in `src/omni_mercury_engine/datasets/environmental.py`
Status: Bulk CSV. 397,609 stream-sediment samples across the continental US (1973-1984 NURE-HSSR program). The loader materialises only the eleven columns its `FEATURE_NAMES` schema exposes (lat/lon + EPA-screening metals + Fe/Ca/pH) and applies the standard USGS half-threshold convention for below-detection-limit values. Anomaly labels follow the EPA Regional Screening Levels for soil contamination.
Note: Listed below in the unreachable-11 watch list — the harness exists to catch upstream-provider outages on top of loader-code regressions, so this loader stays on the watch list even after gaining a real downloader.

> **v1.7.0 label-polarity fix.** Prior to v1.7.0 the FEMA Disaster
> loader handed the anomaly detector inverted labels — historical
> FEMA records make "DR + multi-program" the majority class, which
> drove benchmark AUC below 0.5 and earned the loader a
> "known-broken" note in the CHANGELOG reproducibility footnote.
> `FEMADisasterLoader._select_anomaly_polarity` now enforces the
> minority-as-anomaly convention used elsewhere in Mercury and
> exposes `loader.labels_inverted` for downstream reporters.
> Locked by `tests/datasets/test_disaster.py::TestFEMAInvertedScoresCorrection`.

## Unavailable — Credential-Gated (not counted in benchmarks)

### SMAP / MSL (NASA Spacecraft Telemetry)

Source: OmniAnomaly mirror at `https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/data/`
Labels: `https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv`
Issue: GitHub raw URLs do not serve .npy files from this mirror.
To access: Download from https://github.com/khundman/telemanom or the Kaggle dataset. Place files in the path expected by the loader.
Loader: `SMAPMSLLoader` in `src/omni_mercury_engine/datasets/timeseries.py`

### CICIDS-2017

Source (attempted, in priority order):
1. Hugging Face: `bvk/CICIDS-2017`
2. Distrinet: `https://intrusion-detection.distrinet-research.be/Dataset/dataset.zip`
3. CIC Official: `https://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip`

Issue: All known mirrors are dead or require institutional access.
To access: Request from https://www.unb.ca/cic/datasets/ids-2017.html or place `kaggle.json` in `~/.kaggle/`.
Loader: `CICIDSLoader` in `src/omni_mercury_engine/datasets/security.py`

### MIT-BIH Arrhythmia

Source: PhysioNet via `wfdb` library
Issue: Requires `wfdb` Python library (not in default requirements).
To access: `pip install wfdb`, then loader works automatically. Add `wfdb` to `pyproject.toml [project.optional-dependencies.medical]`.
Loader: `MITBIHLoader` in `src/omni_mercury_engine/datasets/mitbih.py`

### SWaT / WADI (Secure Water Treatment)

Source: iTrust, Singapore University of Technology and Design
Issue: Credential-gated, requires institutional agreement.
To access: Request from https://itrust.sutd.edu.sg/itrust-labs-datasets/
Loader: `SWaTLoader`, `WADILoader` in `src/omni_mercury_engine/datasets/industrial.py`

### MIMIC-III

Source: `https://physionet.org/content/mimiciii/1.4/`
Issue: Requires credentialed access and CITI training.
To access: https://mimic.physionet.org/gettingstarted/access/
Loader: `MIMICLoader` in `src/omni_mercury_engine/datasets/medical.py`

---

## Real-World Domain API Reference

This section catalogs all real-world data sources used by Mercury-Agent's domain-specific
anomaly detectors. Every benchmark result is from real data with a verifiable source.
No synthetic data is used in any benchmark.

### Domain: Earthquake

| Field | Value |
|-------|-------|
| **Loader** | `EarthquakeLoader` |
| **API** | USGS Earthquake Hazards Program |
| **Real-time URL** | `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson` |
| **Historical URL** | `https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson` |
| **API Key** | Not required |
| **License** | Public Domain (US Government) |
| **Ground Truth Events** | Turkey-Syria 2023 (M7.8), Noto 2024 (M7.5), Tohoku 2011 (M9.1) |

### Domain: Tsunami

| Field | Value |
|-------|-------|
| **Loader** | `TsunamiLoader` |
| **API** | NOAA National Data Buoy Center (NDBC) — DART buoys |
| **Real-time URL** | `https://www.ndbc.noaa.gov/data/realtime2/{station_id}.dart` |
| **API Key** | Not required |
| **License** | Public Domain (US Government) |
| **Ground Truth Events** | Tohoku 2011, Chile 2010, Tonga 2022 |

### Domain: Hurricane / Cyclone

| Field | Value |
|-------|-------|
| **Loader** | `HurricaneLoader` |
| **API** | NOAA International Best Track Archive (IBTrACS) |
| **API Key** | Not required |
| **License** | Public Domain (US Government) |
| **Anomaly Target** | Rapid intensification (ΔV >= 30kt in 24h) |

### Domain: Tornado

| Field | Value |
|-------|-------|
| **Loader** | `TornadoLoader` |
| **API** | NOAA Storm Prediction Center (SPC) |
| **API Key** | Not required |
| **License** | Public Domain (US Government) |

### Domain: Flood

| Field | Value |
|-------|-------|
| **Loader** | `FloodLoader` |
| **API** | USGS Water Services + OpenFEMA |
| **API Key** | Not required |
| **License** | Public Domain (US Government) |

### Domain: Wildfire

| Field | Value |
|-------|-------|
| **Loader** | `WildfireLoader` |
| **API** | NASA FIRMS (Fire Information for Resource Management System) |
| **API Key** | **Required** (free) — Register at https://firms.modaps.eosdis.nasa.gov/api/map_key/ |
| **Env Variable** | `NASA_FIRMS_MAP_KEY` |
| **License** | Public Domain (US Government / NASA) |

### Domain: Volcanic

| Field | Value |
|-------|-------|
| **Loader** | `VolcanicLoader` |
| **API** | USGS Volcano Hazards Program |
| **API Key** | Not required |
| **License** | Public Domain (US Government) |

### Domain: Landslide

| Field | Value |
|-------|-------|
| **Loader** | `LandslideLoader` |
| **API** | NASA Global Landslide Catalog (COOLR) |
| **API Key** | Not required |
| **License** | Public Domain (US Government / NASA) |

### Domain: Sepsis / Critical Care

| Field | Value |
|-------|-------|
| **Loader** | `SepsisLoader` |
| **API** | PhysioNet/CinC Challenge 2019 |
| **API Key** | Not required (open challenge dataset) |
| **License** | PhysioNet Credentialed Health Data License |

### Domain: Pandemic

| Field | Value |
|-------|-------|
| **Loader** | `PandemicLoader` |
| **API** | Our World in Data + WHO GHO |
| **API Key** | Not required |
| **License** | Creative Commons (OWID), Public Domain (WHO) |

### Domain: Financial Crisis

| Field | Value |
|-------|-------|
| **Loader** | `FinancialLoader` |
| **API** | FRED (Federal Reserve Economic Data) |
| **API Key** | **Required** (free) — Register at https://fred.stlouisfed.org/docs/api/api_key.html |
| **Env Variable** | `FRED_API_KEY` |
| **License** | Public Domain (US Government) |

### Domain: EMP / Energy Grid

| Field | Value |
|-------|-------|
| **Loader** | `EnergyLoader` |
| **API** | NOAA Space Weather Prediction Center + EIA |
| **EIA Key** | Optional (free) — Register at https://www.eia.gov/opendata/register.php |
| **Env Variable** | `EIA_API_KEY` (optional) |

### Domain: Marine Biodiversity

| Field | Value |
|-------|-------|
| **Loader** | `MarineLoader` |
| **API** | OBIS (Ocean Biodiversity Information System) |
| **API Key** | Not required |
| **License** | Open Access |

### Domain: Network Security

| Field | Value |
|-------|-------|
| **Loader** | `NetworkSecurityLoader` |
| **Datasets** | NSL-KDD, CICIDS2017, BATADAL |
| **API Key** | Not required |
| **License** | Research use |

### Domain: FEMA Cross-Domain

| Field | Value |
|-------|-------|
| **Loader** | `FEMALoader` |
| **API** | OpenFEMA |
| **API Key** | Not required |
| **License** | Public Domain (US Government) |

### Domain: Drought

| Field | Value |
|-------|-------|
| **Loader** | `DroughtLoader` |
| **API** | NOAA NCEI Global Summary of the Month (GSOM) |
| **Source URL** | `https://www.ncei.noaa.gov/data/gsom/access/` |
| **API Key** | Not required |
| **License** | Public Domain (US Government / NOAA) |

### Domain: Hail

| Field | Value |
|-------|-------|
| **Loader** | `HailLoader` |
| **API** | NOAA Storm Prediction Center (SPC) severe-weather archive |
| **Archive URL** | `https://www.spc.noaa.gov/wcm/data/1955-2023_hail.csv.zip` |
| **Daily-reports URL** | `https://www.spc.noaa.gov/climo/reports/today_filtered.csv` |
| **API Key** | Not required |
| **License** | Public Domain (US Government / NOAA) |

### Domain: Heatwave

| Field | Value |
|-------|-------|
| **Loader** | `HeatwaveLoader` |
| **API** | NOAA NCEI Global Summary of the Day (GSOD) |
| **Source URL** | `https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/` |
| **API Key** | Not required |
| **License** | Public Domain (US Government / NOAA) |

### Domain: Meteor / Fireball

| Field | Value |
|-------|-------|
| **Loader** | `MeteorLoader` |
| **API** | NASA/JPL CNEOS Fireball archive + NASA NeoWs close-approach feed |
| **Fireball URL** | `https://ssd-api.jpl.nasa.gov/fireball.api` |
| **NeoWs URL** | `https://api.nasa.gov/neo/rest/v1/feed` |
| **API Key** | Optional (free) — NeoWs accepts `DEMO_KEY`; supply a registered `NASA_API_KEY` to lift rate limits |
| **Env Variable** | `NASA_API_KEY` (optional) |
| **License** | Public Domain (US Government / NASA) |

### Domain: Space Weather / Geomagnetic Storm

| Field | Value |
|-------|-------|
| **Loader** | `SpaceWeatherLoader` |
| **API** | USGS Geomagnetism web service + NASA DONKI GST |
| **USGS URL** | `https://geomag.usgs.gov/ws/data/` |
| **DONKI URL** | `https://api.nasa.gov/DONKI/GST` |
| **API Key** | Optional (free) — DONKI accepts `DEMO_KEY`; supply a registered `NASA_API_KEY` to lift rate limits |
| **Env Variable** | `NASA_API_KEY` (optional) |
| **License** | Public Domain (US Government / NASA / USGS) |

### API Key Summary

| API | Key Required | Env Variable |
|-----|-------------|--------------|
| USGS / NOAA / FEMA / WHO | No | — |
| NASA FIRMS | **Yes** (free) | `NASA_FIRMS_MAP_KEY` |
| FRED | **Yes** (free) | `FRED_API_KEY` |
| EIA | Optional (free) | `EIA_API_KEY` |
| NASA NeoWs / DONKI | Optional (free; `DEMO_KEY` fallback) | `NASA_API_KEY` |

All API keys are stored in environment variables, never in code. See `.env.example` for the complete list.

---

## Operating the SafeHTTP gate

Every loader that reaches the public internet flows through
`omni_mercury_engine.security.safe_http.SafeHTTPClient`, which is
Mercury's single SSRF / DNS-rebinding defence layer.  Two operational
behaviours are worth pinning down for operators because they are
intentional but counter-intuitive on first encounter.

### DNS resolution fails closed for `user_configured=True` URLs

When a loader (or any caller) hands the SafeHTTP gate a URL that came
from operator configuration — anything where `user_configured=True`
is passed to `SafeHTTPClient.validate_url` — and DNS resolution of
the host fails for **any** reason, the gate raises
`omni_mercury_engine.security.safe_http.UnsafeURLError` rather than
treating the failure as a benign network blip and falling through.

This is **intentional**.  A DNS-rebinding attack works by first
giving the validator a public-IP A record, then flipping the same
hostname to a private IP for the actual HTTP request.  Failing
closed on resolution errors is the only way to prevent a timing
window where the attacker can race the validator.  See:

* `src/omni_mercury_engine/security/safe_http.py:138-155` —
  the `getaddrinfo` failure branch.
* `tests/security/test_safe_http.py::test_unresolvable_host_rejected`
  (line 166) — the regression test that locks the "DNS failure must
  NOT be classified as non-fatal" contract (patches
  `socket.getaddrinfo`, asserts `UnsafeURLError` match="did not resolve").

**Operator symptoms.**  Any of these usually mean DNS-fails-closed
fired, not that Mercury is broken:

* `UnsafeURLError: DNS resolution failed for <host>: <reason>`
* A dataset loader that worked yesterday now fails at
  `download()` with the above, even though `curl <host>` from the
  same box succeeds.
* CI is green but a production deployment cannot reach an
  on-premises mirror whose host is only resolvable via the
  internal resolver.

**Remediation, in order of preference.**

1.  **Verify the host is actually resolvable from the Mercury
    process's resolver.**  In containers this commonly means the
    pod is missing `dnsConfig.searches` or the internal stub
    resolver, not a Mercury bug.  Fix the resolver and the loader
    starts working immediately.
2.  **If you are intentionally pointing Mercury at a private
    mirror,** use a code path that explicitly passes
    `allow_private=True` to `SafeHTTPClient` together with
    `user_configured=True`.  The dataset loaders do not currently
    expose a repository-wide `allow_private` preprocessing key, so
    adding a new private-mirror endpoint requires plumbing that flag
    in the specific loader instead of relying on ignored config.
    Document this opt-in alongside the deployment so the next
    operator knows the gate has been relaxed for that one endpoint.
3.  **Place the dataset on disk and use the loader's
    `local_path` preprocessing key.**  CICIDS-2017 (see above) is
    the reference implementation — `_load_from_local_path` skips
    the network entirely and only the on-disk parsing path runs.
    Several other loaders accept the same key; consult the loader
    source for its exact preprocessing schema.

Do **not** try to work around this by toggling `allow_untrusted=True`
on `SafeHTTPClient` — the kwarg was removed in v1.7.0 (PR #210) and
attempting to construct the client with it raises `TypeError`.  See
`docs/MIGRATION-1.6-to-1.7.md` §1 for the migration guide.

### Reachability harness for the historically-unreachable 11

The 11 datasets listed in `CHANGELOG.md`'s reproducibility footnote
(SMAP, MSL, CICIDS-2017, MIT-BIH, UCR, SWaT, WADI, USGS
Geochemistry, NOAA StormEvents, NOAA ERDDAP, FEMA
HazardMitigation) each have a two-lane reachability harness so the
loaders do not silently bitrot when an upstream provider goes away:

* `tests/datasets/test_unreachable_loaders_offline.py` — runs in
  every CI lane.  Constructs each loader, exercises the
  metadata contract, and asserts that a simulated upstream
  outage produces a loud `DataSourceUnavailableError` /
  `ConnectionError` rather than a silent `False` return.
* `tests/datasets/test_unreachable_loaders_network.py` — marked
  `@pytest.mark.network`, deselected by default, run nightly via
  `.github/workflows/dataset-reachability.yml`.  Calls the real
  `download()` and accepts either successful retrieval (asserting
  non-empty features) or a loud upstream-unavailable exception.

If you add or remove a loader from the unreachable-11 set, update
**all three** of: this section, the CHANGELOG footnote, and the
`_UNREACHABLE_LOADERS` table in
`tests/datasets/test_unreachable_loaders_offline.py`.  The harness
includes a coverage-drift assertion (`test_harness_covers_eleven_loaders`)
that fails the build if these get out of sync.

## Live hazard ingestion clients (T1e)

Every hazard detector's optional live path goes through one uniform,
provenance-checked seam: `data_sources/live_ingestion.py`
(`fetch_live_datapoints`). The seam **fails loud** (`LiveDataError`) on any
fetch failure, refuses sources that label their points
`metadata["simulated"] = True` unless the caller passes
`allow_simulated=True` (`SimulatedDataError`), and stamps
`data_provenance` (`"live"` / `"simulated"`) plus `live_context` onto every
detector result. Transport-level failures (DNS/timeout/5xx/breaker/429)
carry `unreachable=True` end-to-end (`FetchResult.unreachable`,
`LiveDataError.unreachable`) so consumers can distinguish "service down"
from "service drifted".

| Client | Module | Upstream | Wired detector(s) | Recorded fixture |
|---|---|---|---|---|
| `USGSEarthquakeSource` | `data_sources/earth_science.py` | USGS FDSN event service | `EarthquakeDetector.detect_live`, `TsunamiDetector` | `usgs_earthquakes.json` |
| `USGSVolcanoSource` | `data_sources/earth_science.py` | USGS HANS (real API; the simulated `US_VOLCANOES` table was removed, DEPRECATION §6.9) | `VolcanicEruptionDetector.detect_live` | `hans_monitored.json`, `hans_elevated.json` |
| `NWSWeatherAlertsSource` | `data_sources/earth_science.py` | NWS CAP active alerts | `TornadoDetector.detect_live`, `FloodDetector.detect_live` | `nws_alerts_tornado.json`, `nws_alerts_flood.json` |
| `NOAANWPSSource` | `data_sources/earth_science.py` | NOAA NWPS v1 river gauges (requires a `bbox`) | `FloodDetector.detect_live` | `nwps_gauges.json` |
| `NOAACOOPSSource` | `data_sources/earth_science.py` | NOAA CO-OPS tides & currents | coastal water-level context | `coops_water_level.json` |
| `NOAASWPCSource` | `data_sources/space_weather.py` | NOAA SWPC products (Kp, X-ray, solar wind) | `SolarFlareDetector.detect_live`, `SolarStormDetector.predict_live` | `swpc_kp.json`, `swpc_xray.json`, `swpc_solar_wind.json` |
| `NASADONKISource` | `data_sources/space_weather.py` | NASA DONKI FLR/GST events | `SolarFlareDetector.detect_live` corroboration context | `donki_flr.json`, `donki_gst.json` |
| `NASANeoWsSource` | `data_sources/space_weather.py` | NASA NeoWs close approaches | `MeteorDetector.predict_meteor` | `neows_feed.json` |
| `JPLFireballSource` | `data_sources/jpl_ssd.py` | JPL CNEOS fireball API | `MeteorDetector.predict_meteor` | `jpl_fireball.json` |
| `JPLSentrySource` | `data_sources/jpl_ssd.py` | JPL Sentry impact risk | `MeteorDetector.predict_meteor` | `jpl_sentry.json` |
| `BGSELFStationSource` | `data_sources/geomagnetic.py` | BGS ELF (instrument mode: caller-supplied raw samples; simulated mode is labelled and refused without opt-in) | `SchumannResonanceDetector` | n/a (instrument/simulated modes) |
| `HeartMathGCMSSource` | `data_sources/geomagnetic.py` | none (no public machine-readable API — emits *labelled placeholder* spectra only; see class HONESTY CONTRACT) | none (refused by the seam without `allow_simulated=True`) | n/a |

Test lanes:

* **Offline (every CI run):** `tests/test_live_wiring_sources.py`,
  `tests/test_live_wiring_space.py`,
  `tests/detectors/test_live_wiring_geological.py` replay the recorded
  fixtures in `tests/fixtures/live_wiring/` (captured from the real APIs;
  capture dates in each test module docstring).
* **Network (weekly `network-tests.yml` lane / on demand):**
  `MERCURY_NETWORK_TESTS=1 pytest tests/test_live_wiring_network.py -m network`.
  Skips **only** on genuine transport-level unreachability; any error from a
  reachable service (schema/endpoint drift) FAILS the lane.
