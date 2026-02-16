# Mercury-Agent Data Sources Reference

Mercury Agent †
Copyright (C) 2025 Steel Security Advisors LLC

This document catalogs all real-world data sources used by Mercury-Agent's
domain-specific anomaly detectors. Every benchmark result is from real data
with a verifiable source. No synthetic data is used in any benchmark.

---

## Domain 1: Earthquake

| Field | Value |
|-------|-------|
| **Loader** | `EarthquakeLoader` |
| **API** | USGS Earthquake Hazards Program |
| **Real-time URL** | `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson` |
| **Historical URL** | `https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson` |
| **API Key** | Not required |
| **License** | Public Domain (US Government) |
| **Format** | GeoJSON |
| **Ground Truth Events** | Turkey-Syria 2023 (M7.8), Noto 2024 (M7.5), Tohoku 2011 (M9.1), Haiti 2010 (M7.0), Nepal 2015 (M7.8) |

## Domain 2: Tsunami

| Field | Value |
|-------|-------|
| **Loader** | `TsunamiLoader` |
| **API** | NOAA National Data Buoy Center (NDBC) — DART buoys |
| **Real-time URL** | `https://www.ndbc.noaa.gov/data/realtime2/{station_id}.dart` |
| **API Key** | Not required |
| **License** | Public Domain (US Government) |
| **Format** | Whitespace-delimited text |
| **Key Stations** | 46402, 46407, 46410, 51407, 32412, 21413, 21418, 52402 |
| **Ground Truth Events** | Tohoku 2011, Chile 2010, Tonga 2022 |

## Domain 3: Hurricane / Cyclone

| Field | Value |
|-------|-------|
| **Loader** | `HurricaneLoader` |
| **API** | NOAA International Best Track Archive (IBTrACS) |
| **URL** | `https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/` |
| **API Key** | Not required |
| **License** | Public Domain (US Government) |
| **Format** | CSV |
| **Ground Truth Events** | Katrina 2005, Harvey 2017, Maria 2017, Ian 2022, Helene 2024, Milton 2024 |
| **Anomaly Target** | Rapid intensification (ΔV ≥ 30kt in 24h) |

## Domain 4: Tornado

| Field | Value |
|-------|-------|
| **Loader** | `TornadoLoader` |
| **API** | NOAA Storm Prediction Center (SPC) |
| **Archive URL** | `https://www.spc.noaa.gov/wcm/data/1950-2023_actual_tornadoes.csv` |
| **Daily Reports** | `https://www.spc.noaa.gov/climo/reports/today.csv` |
| **API Key** | Not required |
| **License** | Public Domain (US Government) |
| **Format** | CSV |
| **Ground Truth Events** | 2011 Super Outbreak, 2013 Moore OK (EF5), 2024 Midwest outbreaks |

## Domain 5: Flood

| Field | Value |
|-------|-------|
| **Loader** | `FloodLoader` |
| **API** | USGS Water Services + OpenFEMA |
| **USGS URL** | `https://waterservices.usgs.gov/nwis/iv/?format=json` |
| **FEMA URL** | `https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries` |
| **API Key** | Not required |
| **License** | Public Domain (US Government) |
| **Format** | JSON |
| **Ground Truth Events** | Helene 2024 (Appalachia), Vermont 2023, European 2021 |

## Domain 6: Wildfire

| Field | Value |
|-------|-------|
| **Loader** | `WildfireLoader` |
| **API** | NASA FIRMS (Fire Information for Resource Management System) |
| **URL** | `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_NOAA20_NRT/{area}/{days}` |
| **API Key** | **Required** (free) — Register at https://firms.modaps.eosdis.nasa.gov/api/map_key/ |
| **Env Variable** | `NASA_FIRMS_MAP_KEY` |
| **License** | Public Domain (US Government / NASA) |
| **Format** | CSV |
| **Ground Truth Events** | Los Angeles 2025, Maui 2023, Australia 2020, US West Coast 2020 |

## Domain 7: Volcanic

| Field | Value |
|-------|-------|
| **Loader** | `VolcanicLoader` |
| **API** | USGS Volcano Hazards Program |
| **Alerts URL** | `https://volcanoes.usgs.gov/vsc/api/volcanoApi/alerts` |
| **Volcano List** | `https://volcanoes.usgs.gov/vsc/api/volcanoApi/volcanoList` |
| **API Key** | Not required |
| **License** | Public Domain (US Government) |
| **Format** | JSON |
| **Ground Truth Events** | Hunga Tonga 2022, Cumbre Vieja 2021, Kilauea 2018, Eyjafjallajökull 2010 |

## Domain 8: Landslide

| Field | Value |
|-------|-------|
| **Loader** | `LandslideLoader` |
| **API** | NASA Global Landslide Catalog (COOLR) |
| **URL** | `https://maps.nccs.nasa.gov/arcgis/rest/services/global_landslide_catalog/` |
| **API Key** | Not required |
| **License** | Public Domain (US Government / NASA) |
| **Format** | JSON (ArcGIS REST) |
| **Ground Truth Events** | Oso WA 2014, Sierra Leone 2017, Japan 2018 |

## Domain 9: Sepsis / Critical Care

| Field | Value |
|-------|-------|
| **Loader** | `SepsisLoader` |
| **API** | PhysioNet/CinC Challenge 2019 |
| **URL** | `https://physionet.org/content/challenge-2019/` |
| **API Key** | Not required (open challenge dataset) |
| **License** | PhysioNet Credentialed Health Data License |
| **Format** | PSV (pipe-separated values) |
| **Ground Truth** | SepsisLabel column (binary) |

## Domain 10: Pandemic

| Field | Value |
|-------|-------|
| **Loader** | `PandemicLoader` |
| **API** | Our World in Data + WHO GHO |
| **OWID URL** | `https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv` |
| **WHO URL** | `https://ghoapi.azureedge.net/api/` |
| **API Key** | Not required |
| **License** | Creative Commons (OWID), Public Domain (WHO) |
| **Format** | CSV (OWID), JSON (WHO) |
| **Ground Truth Events** | COVID-19 waves (USA, Italy, India), Ebola 2014, Mpox 2022 |

## Domain 11: Financial Crisis

| Field | Value |
|-------|-------|
| **Loader** | `FinancialLoader` |
| **API** | FRED (Federal Reserve Economic Data) |
| **URL** | `https://api.stlouisfed.org/fred/series/observations` |
| **API Key** | **Required** (free) — Register at https://fred.stlouisfed.org/docs/api/api_key.html |
| **Env Variable** | `FRED_API_KEY` |
| **License** | Public Domain (US Government) |
| **Key Series** | VIXCLS (VIX), T10Y2Y (yield curve), BAMLH0A0HYM2 (credit spreads) |
| **Ground Truth Events** | GFC 2008, COVID crash 2020, SVB 2023, Asian crisis 1997, Flash crash 2010 |

## Domain 12: EMP / Energy Grid

| Field | Value |
|-------|-------|
| **Loader** | `EnergyLoader` |
| **API** | NOAA Space Weather Prediction Center + EIA |
| **SWPC URL** | `https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json` |
| **EIA URL** | `https://api.eia.gov/v2/` (optional) |
| **SWPC Key** | Not required |
| **EIA Key** | Optional (free) — Register at https://www.eia.gov/opendata/register.php |
| **Env Variable** | `EIA_API_KEY` (optional) |
| **Format** | JSON |
| **Ground Truth Events** | Quebec blackout 1989, Halloween storms 2003, Texas 2021, Bastille Day 2000 |

## Domain 13: Marine Biodiversity

| Field | Value |
|-------|-------|
| **Loader** | `MarineLoader` |
| **API** | OBIS (Ocean Biodiversity Information System) |
| **URL** | `https://api.obis.org/v3/occurrence` |
| **API Key** | Not required |
| **License** | Open Access |
| **Format** | JSON |
| **Ground Truth Events** | GBR bleaching 2016, GBR bleaching 2020, Marine heatwave 2023 |

## Domain 14: Network Security

| Field | Value |
|-------|-------|
| **Loader** | `NetworkSecurityLoader` |
| **Datasets** | NSL-KDD, CICIDS2017, BATADAL |
| **CICIDS URL** | `https://www.unb.ca/cic/datasets/ids-2017.html` |
| **API Key** | Not required |
| **License** | Research use |
| **Current Performance** | NSL-KDD AUC 0.972 via core detector |

## Domain 15: FEMA Cross-Domain

| Field | Value |
|-------|-------|
| **Loader** | `FEMALoader` |
| **API** | OpenFEMA |
| **URL** | `https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries` |
| **API Key** | Not required |
| **License** | Public Domain (US Government) |
| **Format** | JSON (OData-style) |
| **Max Records** | 10,000 per call (with pagination) |
| **Use Case** | Cross-domain validation — correlate FEMA declarations with domain detectors |

---

## API Key Summary

| API | Key Required | How to Get | Env Variable |
|-----|-------------|------------|--------------|
| USGS Earthquake | No | Free, open | — |
| NOAA NDBC/DART | No | Free, open | — |
| IBTrACS | No | CSV download | — |
| NOAA SPC | No | CSV download | — |
| NOAA AHPS/USGS Water | No | Free, open | — |
| NASA FIRMS | **Yes** (free) | https://firms.modaps.eosdis.nasa.gov/api/map_key/ | `NASA_FIRMS_MAP_KEY` |
| USGS Volcano | No | Free, open | — |
| NASA COOLR | No | Free, open | — |
| PhysioNet Challenge | No | Open access | — |
| WHO GHO / OWID | No | Free, open | — |
| FRED | **Yes** (free) | https://fred.stlouisfed.org/docs/api/api_key.html | `FRED_API_KEY` |
| EIA | Optional (free) | https://www.eia.gov/opendata/register.php | `EIA_API_KEY` |
| OBIS | No | Free, open | — |
| OpenFEMA | No | Free, open | — |

All API keys are stored in environment variables, never in code.
See `.env.example` for the complete list.

---

*Generated for Steel Security Advisors LLC — Mercury-Agent †*
*Every URL in this document has been verified against official API documentation.*
