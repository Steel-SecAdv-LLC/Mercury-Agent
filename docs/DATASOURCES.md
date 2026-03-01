# Mercury-Agent Data Sources

Last verified: 2026-02-15
Benchmark run: 51/55 datasets loaded successfully

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

## Credentialed Datasets

### MIMIC-III (PhysioNet)
Benchmark: `benchmarks/credentialed_benchmarks.py`
Access: https://physionet.org/content/mimiciii/
Steps:
1. Complete PhysioNet credentialing (CITI training required)
2. Download `NOTEEVENTS.csv.gz` and place in `~/.cache/mercury_agent/mimic/`
3. Run: `python benchmarks/credentialed_benchmarks.py`

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
