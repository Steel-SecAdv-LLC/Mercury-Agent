# Mercury Agent - Active Dataset Catalog

Applies to Mercury Agent **v1.8.x**. Last updated: 2026-05-20.

Every dataset in the benchmark pipeline. Datasets are loaded at runtime;
API-sourced datasets may be unavailable if endpoints are down. The benchmark
records `api_unavailable` and continues.

> **v1.7 reachability harness.** The 11 historically-unreachable
> loaders (SMAP, MSL, CICIDS-2017, MIT-BIH, UCR, SWaT, WADI, USGS
> Geochemistry, NOAA StormEvents, NOAA ERDDAP, FEMA HazardMitigation)
> are now covered by a two-lane harness — an always-on offline lane
> (`tests/datasets/test_unreachable_loaders_offline.py`) plus a
> nightly network lane (`tests/datasets/test_unreachable_loaders_network.py`
> + `.github/workflows/dataset-reachability.yml`, 04:17 UTC) — so an
> upstream provider outage surfaces as a failed nightly run rather
> than as a benchmark silently dropping a dataset.
>
> The full data-sources catalog (loader paths, API URLs, auth
> requirements, license terms, ground-truth events) is in
> [`../docs/DATASOURCES.md`](../docs/DATASOURCES.md).

## ADBench (47 datasets)

| # | Name | Domain | Source | Auth | Samples | Features | Anomaly Ratio | License |
|---|------|--------|--------|------|---------|----------|---------------|---------|
| 1-47 | ADBench-01 .. ADBench-47 | Tabular anomaly detection | [ADBench](https://github.com/Minqi824/ADBench) | None | ~100-50,000 | 2-274 | 1-35% | MIT |

ADBench is a standardised benchmark suite containing 47 real-world tabular
anomaly detection datasets spanning healthcare, image, NLP, and other
domains.

## Domain Datasets

### Environmental

| Name | Class | Module | Source | Auth | Expected Samples | Features | Anomaly Ratio | Citation / License |
|------|-------|--------|--------|------|-----------------|----------|---------------|-------------------|
| USGS_Earthquake | USGSEarthquakeLoader | environmental | [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/) | None | ~1,000-10,000 | 5-10 | ~5-15% | Public Domain (USGS) |
| NOAA_Weather | NOAAWeatherLoader | environmental | [NOAA Climate Data Online](https://www.ncdc.noaa.gov/cdo-web/) | None | ~500-5,000 | 8-15 | ~10-20% | Public Domain (NOAA) |
| Wildfire | WildfireDataLoader | environmental | [MODIS/FIRMS](https://firms.modaps.eosdis.nasa.gov/) | None | ~1,000-10,000 | 6-12 | ~5-10% | NASA Open Data |
| USGS_Geochemistry | USGSGeochemistryLoader | environmental | [USGS NURE-HSSR Stream Sediment](https://mrdata.usgs.gov/nure/sediment/nuresed-csv.zip) | None | 397,609 | 11 | ~5-15% | Public Domain (USGS) |

### Ocean

| Name | Class | Module | Source | Auth | Expected Samples | Features | Anomaly Ratio | Citation / License |
|------|-------|--------|--------|------|-----------------|----------|---------------|-------------------|
| NOAA_Buoy | NOAABuoyLoader | ocean | [NOAA NDBC](https://www.ndbc.noaa.gov/) | None | ~1,000-10,000 | 8-15 | ~5-10% | Public Domain (NOAA) |

### Climate

| Name | Class | Module | Source | Auth | Expected Samples | Features | Anomaly Ratio | Citation / License |
|------|-------|--------|--------|------|-----------------|----------|---------------|-------------------|
| NOAA_StormEvents | NOAAStormEventsLoader | noaa_storm | [NOAA Storm Events DB](https://www.ncdc.noaa.gov/stormevents/) | None | ~1,000-50,000 | 5-15 | ~5-20% | Public Domain (NOAA) |
| NOAA_GSOD | NOAAGSODLoader | noaa_gsod | [NOAA GSOD](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00516) | None | ~1,000-10,000 | 10-20 | ~5-10% | Public Domain (NOAA) |
| NOAA_ERDDAP | NOAAERDDAPLoader | noaa_erddap | [NOAA ERDDAP](https://coastwatch.pfeg.noaa.gov/erddap/) | None | ~500-5,000 | 5-15 | ~5-10% | Public Domain (NOAA) |

### Air Quality

| Name | Class | Module | Source | Auth | Expected Samples | Features | Anomaly Ratio | Citation / License |
|------|-------|--------|--------|------|-----------------|----------|---------------|-------------------|
| EPA_AirQuality | EPAAirQualityLoader | epa_air | [EPA AQS](https://aqs.epa.gov/aqsweb/documents/data_api.html) | None | ~1,000-10,000 | 5-10 | ~5-15% | Public Domain (EPA) |

### Disaster

| Name | Class | Module | Source | Auth | Expected Samples | Features | Anomaly Ratio | Citation / License |
|------|-------|--------|--------|------|-----------------|----------|---------------|-------------------|
| FEMA_Disaster | FEMADisasterLoader | disaster | [FEMA OpenFEMA](https://www.fema.gov/about/openfema/data-sets) | None | ~1,000-50,000 | 5-15 | ~5-20% | Public Domain (FEMA) |
| FEMA_HazardMitigation | FEMAHazardMitigationLoader | disaster | [FEMA HMA](https://www.fema.gov/grants/mitigation) | None | ~500-5,000 | 5-10 | ~5-10% | Public Domain (FEMA) |

### Space

| Name | Class | Module | Source | Auth | Expected Samples | Features | Anomaly Ratio | Citation / License |
|------|-------|--------|--------|------|-----------------|----------|---------------|-------------------|
| NASA_Exoplanet | NASAExoplanetLoader | space | [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) | None | ~1,000-5,000 | 10-30 | ~5-15% | NASA Open Data |
| SolarDynamics | SolarDynamicsLoader | space | [NOAA SWPC](https://www.swpc.noaa.gov/) | None | ~500-5,000 | 5-15 | ~5-10% | Public Domain (NOAA) |

### Academic / Archive

| Name | Class | Module | Source | Auth | Expected Samples | Features | Anomaly Ratio | Citation / License |
|------|-------|--------|--------|------|-----------------|----------|---------------|-------------------|
| UCR | UCRLoader | ucr_archive | [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) | None | ~100-10,000 | 1-100 | Varies | CC BY 4.0 |
| CWRU_Bearing | CWRUBearingLoader | ucr_archive | [CWRU Bearing Data](https://engineering.case.edu/bearingdatacenter/download-data-file) | None | ~1,000-10,000 | 1-4 | ~10-30% | Public |
| MSDS | MSDSLoader | ucr_archive | [Multi-Source Data Stream](https://archive.ics.uci.edu/) | None | ~500-5,000 | 5-20 | ~5-15% | CC BY 4.0 |

### Security

| Name | Class | Module | Source | Auth | Expected Samples | Features | Anomaly Ratio | Citation / License |
|------|-------|--------|--------|------|-----------------|----------|---------------|-------------------|
| NSL-KDD | NSLKDDLoader | security | [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) | None | ~125,000 | 41 | ~48% | Public |
| CICIDS-2017 | CICIDSLoader | security | [CIC IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html) | None | ~2.8M | 78 | ~19% | Public |
| ThreatIntel | ThreatIntelLoader | security | Various OSINT feeds | None | ~500-5,000 | 5-20 | ~10-20% | OSINT |

### General

| Name | Class | Module | Source | Auth | Expected Samples | Features | Anomaly Ratio | Citation / License |
|------|-------|--------|--------|------|-----------------|----------|---------------|-------------------|
| ADRepository | ADRepositoryLoader | adrepository | [Anomaly Detection Repository](https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets) | None | ~100-10,000 | 2-100 | 1-35% | Various |

### Timeseries

| Name | Class | Module | Source | Auth | Expected Samples | Features | Anomaly Ratio | Citation / License |
|------|-------|--------|--------|------|-----------------|----------|---------------|-------------------|
| SMD | SMDLoader | timeseries | [Server Machine Dataset](https://github.com/NetManAIOps/OmniAnomaly) | None | ~25,000 | 38 | ~4% | MIT |
| NAB | NABLoader | timeseries | [NAB](https://github.com/numenta/NAB) | None | ~22,000 | 1 | ~10% | AGPL-3.0 |
| SMAP | SMAPMSLLoader | timeseries | [NASA SMAP](https://github.com/khundman/telemanom) | None | ~135,000 | 25 | ~12% | Apache-2.0 |
| MSL | SMAPMSLLoader | timeseries | [NASA MSL](https://github.com/khundman/telemanom) | None | ~58,000 | 55 | ~10% | Apache-2.0 |

### Industrial

| Name | Class | Module | Source | Auth | Expected Samples | Features | Anomaly Ratio | Citation / License |
|------|-------|--------|--------|------|-----------------|----------|---------------|-------------------|
| BATADAL | BATADALLoader | industrial | [BATADAL](https://www.batadal.net/) | None | ~9,000 | 43 | ~7% | CC BY-SA 4.0 |
| SWaT | SWaTLoader | industrial | [SWaT](https://itrust.sutd.edu.sg/itrust-labs_datasets/) | Download required | ~495,000 | 51 | ~12% | Academic use |
| WADI | WADILoader | industrial | [WADI](https://itrust.sutd.edu.sg/itrust-labs_datasets/) | Download required | ~172,000 | 123 | ~6% | Academic use |

### Medical

| Name | Class | Module | Source | Auth | Expected Samples | Features | Anomaly Ratio | Citation / License |
|------|-------|--------|--------|------|-----------------|----------|---------------|-------------------|
| MIT-BIH | MITBIHLoader | mitbih | [PhysioNet MIT-BIH](https://physionet.org/content/mitdb/1.0.0/) | None | ~109,000 | 187 | ~2.5% | ODC-BY |

## Total Active Datasets

| Category | Count |
|----------|-------|
| ADBench | 47 |
| Environmental | 4 |
| Ocean | 1 |
| Climate | 3 |
| Air Quality | 1 |
| Disaster | 2 |
| Space | 2 |
| Academic | 3 |
| Security | 3 |
| General | 1 |
| Timeseries | 4 |
| Industrial | 3 |
| Medical | 1 |
| **Total** | **75** |
