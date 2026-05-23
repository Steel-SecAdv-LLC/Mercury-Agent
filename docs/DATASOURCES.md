# Mercury-Agent Data Sources

Last verified: 2026-05-19 (loader catalog refresh; benchmark snapshots
in the per-dataset tables below still derive from the 2026-02-15
sweep — that sweep is the legacy 51/55 `mercury_benchmark.py` baseline
the README headline improves on, and is preserved here as the
auditable starting point). The canonical public benchmarking figure
is the **65 reproducible / 65 attempted** set documented in the
README "Latest Benchmark Results" section (Mean AUC 0.8464, Mean
Oracle F1 0.6441, run timestamp 2026-05-14T22:14:04 UTC). The earlier
**64/75** snapshot from 2026-03-04 (Mean AUC 0.8285, Mean Oracle F1
0.6370) is the public headline that this loader catalog originally
verified against; the 11 historically-unreachable loaders (SMAP, MSL,
CICIDS-2017, MIT-BIH, UCR, SWaT, WADI, USGS Geochemistry, NOAA
StormEvents, NOAA ERDDAP, FEMA HazardMitigation) are now tracked by
a two-lane reachability harness rather than counted as silent
benchmark drops. The two earlier views are not the same measured
baseline; see `docs/BENCHMARKS.md` for the full reconciliation.

## Successfully Loading (51 datasets)

### ADBench Tabular (47 datasets)

Source: `https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/{index}_{name}.npz`
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
3. CIC Official: `http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip`

Issue: All known mirrors are dead or require institutional access.
To access: Request from https://www.unb.ca/cic/datasets/ids-2017.html or place `kaggle.json` in `~/.kaggle/`.
Loader: `CICIDSLoader` in `src/omni_mercury_engine/datasets/security.py`

### MIT-BIH Arrhythmia

Source: PhysioNet via `wfdb` library
Issue: Requires `wfdb` Python library (not in default requirements).
To access: `pip install wfdb`, then loader works automatically. Add `wfdb` to `pyproject.toml [project.optional-dependencies.medical]`.
Loader: `MITBIHLoader` in `src/omni_mercury_engine/datasets/medical.py`

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

### API Key Summary

| API | Key Required | Env Variable |
|-----|-------------|--------------|
| USGS / NOAA / FEMA / WHO | No | — |
| NASA FIRMS | **Yes** (free) | `NASA_FIRMS_MAP_KEY` |
| FRED | **Yes** (free) | `FRED_API_KEY` |
| EIA | Optional (free) | `EIA_API_KEY` |

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
* `tests/loaders/test_base_loader.py:99` — the regression test
  that locks the "DNS failure must NOT be classified as
  non-fatal" contract.

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
