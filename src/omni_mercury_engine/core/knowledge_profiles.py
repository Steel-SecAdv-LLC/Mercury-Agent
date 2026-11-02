# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Knowledge profiles: the profession, institutions and expertise behind every domain and detector.

The detector manifest says *what* a detector is (module, class, seams), the
ontology says *which anomaly classes* it is sensitive to and where that was
measured, and a loader says *where* its data comes from. None of them says
who the knowledge belongs to: which discipline a method comes from, which
institution is the authority for the data, what a practitioner has to know
to read the output, and what the published lineage is. Without that, every
domain and every detector reads alike, and the console and the agent can
only repeat the same template with a different name in it.

This module is that missing layer, authored by hand and checked by
``tests/core/test_knowledge_profiles.py``:

* :data:`DOMAIN_PROFILES` -- one :class:`DomainProfile` per data-loader
  domain (``loaders/*_loader.py``) and per hazard-visualisation domain
  (``detectors/hazard_diagnostics.KNOWN_HAZARDS``): the discipline, the
  professions that work with the data, the institutions that publish or
  govern it (with their URLs), the expertise a practitioner brings, how
  Mercury turns the source into features and labels, the detectors that
  serve the domain, the known limits and the literature.
* :data:`DETECTOR_PROFILES` -- one :class:`DetectorProfile` per manifest
  detector: the discipline the method comes from, its lineage (who published
  it, when, in which tradition), the mechanism in plain words, who uses it
  professionally and what they need to know to read its score, when to reach
  for it, when not to, and the references.

Authoring rules, enforced by the tests:

* coverage is exact: every manifest detector and every loader/hazard domain
  has a profile, and no profile is orphaned;
* no two profiles share a mechanism, lineage, method or label rule -- a
  copied sentence is a defect, because the whole point is distinguishability;
* every profile cites at least one reference; every domain names at least
  one institution with an ``https`` URL;
* nothing here is marketing: a mechanism describes what the code does, and
  a limit states what the ontology's evidence does not cover. Claims about
  measured performance stay in the ontology; this layer describes lineage
  and expertise, which do not change with a benchmark run.

The profiles are served by ``GET /api/v1/catalog/domains`` and
``GET /api/v1/catalog/detectors`` (``api/routes/catalog.py``), compiled into
the console catalog by ``scripts/build_console_catalog.py``, and read by the
agent (``narrative/conversation.py``) to answer "tell me about the sepsis
domain" or "what is bocpd" with that domain's or detector's own profile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "DETECTOR_PROFILES",
    "DOMAIN_PROFILES",
    "DetectorProfile",
    "DomainProfile",
    "Institution",
    "Reference",
    "detector_profile",
    "domain_profile",
    "profile_dict",
    "validate_profiles",
]


@dataclass(frozen=True)
class Reference:
    """One citation: a paper, standard, dataset paper or authoritative document."""

    citation: str
    url: str = ""


@dataclass(frozen=True)
class Institution:
    """An organisation that publishes, governs or defines the data of a domain."""

    name: str
    role: str
    url: str


@dataclass(frozen=True)
class DomainProfile:
    """The profession, institutions and expertise behind one data domain."""

    domain: str
    title: str
    discipline: str
    professions: tuple[str, ...]
    institutions: tuple[Institution, ...]
    expertise: tuple[str, ...]
    observables: str
    method: str
    label_rule: str
    limits: tuple[str, ...]
    references: tuple[Reference, ...]
    detectors: tuple[str, ...] = field(default_factory=tuple)
    standards: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DetectorProfile:
    """The discipline, lineage and practitioner knowledge behind one manifest detector."""

    name: str
    discipline: str
    lineage: str
    mechanism: str
    practitioners: str
    use_when: str
    not_for: str
    references: tuple[Reference, ...]


def _r(citation: str, url: str = "") -> Reference:
    return Reference(citation, url)


def _i(name: str, role: str, url: str) -> Institution:
    return Institution(name, role, url)


# --------------------------------------------------------------------------- #
# Domain profiles
# --------------------------------------------------------------------------- #

_DOMAINS: tuple[DomainProfile, ...] = (
    DomainProfile(
        domain="airspace",
        detectors=("geo_movement", "statistical"),
        title="Live airspace (ADS-B state vectors)",
        discipline="Aviation safety and air-traffic surveillance",
        professions=(
            "air traffic controllers",
            "aviation safety analysts and accident investigators",
            "ADS-B and surveillance engineers",
        ),
        institutions=(
            _i(
                "The OpenSky Network",
                "crowd-sourced ADS-B/Mode S receiver network and data source",
                "https://opensky-network.org/",
            ),
            _i(
                "ICAO",
                "defines transponder emergency codes and surveillance standards",
                "https://www.icao.int/",
            ),
            _i(
                "FAA",
                "national airspace authority; 7500/7600/7700 squawk procedures",
                "https://www.faa.gov/",
            ),
            _i(
                "EUROCONTROL",
                "European air-navigation safety and surveillance coordination",
                "https://www.eurocontrol.int/",
            ),
        ),
        expertise=(
            "a transponder squawk of 7500, 7600 or 7700 is a crew-declared event (unlawful interference, radio failure, emergency), not an inference",
            "barometric and geometric altitude disagree legitimately with pressure setting; a large disagreement is a data-quality signal",
            "ground speed, vertical rate and track are kinematic, so implausible values are physics violations rather than statistical surprises",
            "coverage is receiver-dependent: silence can mean no receiver, not no aircraft",
        ),
        observables="Per-aircraft position, barometric and geometric altitude, velocity, heading, vertical rate, on-ground flag and squawk from live Mode S / ADS-B receptions, worldwide or within a bounding box.",
        method="Fetches the OpenSky anonymous REST state vectors and engineers kinematic columns (altitude disagreement, speed, vertical rate, track, position, on-ground) for movement-plausibility scoring; declared emergency squawks are surfaced as their own column and kept out of the scored features so a declared event is never scored as if it were inferred.",
        label_rule="Statistical: labels mark kinematically extreme states; declared emergency squawks are reported alongside, never used as the scored label.",
        limits=(
            "anonymous OpenSky quota is small and global snapshots are heavy; use a bounding box",
            "no flight-plan context, so a planned manoeuvre and an anomaly can look alike",
            "no committed labeled aviation evaluation of the scored features",
        ),
        references=(
            _r(
                "Schäfer, M., Strohmeier, M., Lenders, V., Martinovic, I. & Wilhelm, M. (2014). Bringing up OpenSky: a large-scale ADS-B sensor network for research. IPSN 2014.",
                "https://doi.org/10.1109/IPSN.2014.6846743",
            ),
            _r(
                "ICAO Doc 4444, Procedures for Air Navigation Services: Air Traffic Management (SSR code assignment, emergency codes)."
            ),
            _r(
                "OpenSky Network REST API documentation.",
                "https://openskynetwork.github.io/opensky-api/rest.html",
            ),
        ),
        standards=("ICAO Annex 10 (Aeronautical Telecommunications, Mode S / ADS-B)",),
    ),
    DomainProfile(
        domain="drought",
        detectors=("drought",),
        title="Drought (monthly precipitation deficit)",
        discipline="Climatology and hydrometeorology",
        professions=(
            "state climatologists",
            "drought monitor authors",
            "water-resource planners",
            "agricultural extension agronomists",
        ),
        institutions=(
            _i(
                "NOAA NCEI",
                "Global Summary of the Month station archive",
                "https://www.ncei.noaa.gov/data/gsom/",
            ),
            _i(
                "U.S. Drought Monitor (NDMC, USDA, NOAA)",
                "weekly drought classification D0-D4",
                "https://droughtmonitor.unl.edu/",
            ),
            _i(
                "WMO",
                "endorses SPI as the standard meteorological drought index",
                "https://public.wmo.int/",
            ),
        ),
        expertise=(
            "drought is defined relative to a station's own climatology, so the index is a standardised deviate, not a rainfall total",
            "time scale matters: SPI-1 reads meteorological drought, SPI-6 agricultural, SPI-12 and longer hydrological",
            "zero-precipitation months need the mixed distribution, or the gamma fit is wrong in arid climates",
            "USDM categories are percentile bands: D2 (severe) corresponds to SPI at or below -1.3",
        ),
        observables="Period-of-record monthly precipitation (mm) and mean temperature (degrees C) per NCEI GSOM station covering the catalogued US droughts.",
        method="Downloads one period-of-record CSV per station, slices the event window, standardises precipitation over 1-, 3-, 6- and 12-month windows and computes fraction-of-normal and temperature deviation; the drought detector's SPI core supplies the index from the same precipitation series.",
        label_rule="Statistical: a month is labelled anomalous when its SPI-6 is at or below -1.3 (US Drought Monitor D2), computed from the loader's own precipitation series.",
        limits=(
            "labels are a function of the scored precipitation series, so benchmarks on this domain measure the SPI cut, not independent detection",
            "station coverage, not gridded analysis; no soil-moisture or streamflow term",
        ),
        references=(
            _r(
                "McKee, T.B., Doesken, N.J. & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. 8th Conference on Applied Climatology, 179-184."
            ),
            _r(
                "Svoboda, M. et al. (2002). The Drought Monitor. Bulletin of the American Meteorological Society 83(8), 1181-1190.",
                "https://doi.org/10.1175/1520-0477-83.8.1181",
            ),
            _r(
                "Thom, H.C.S. (1958). A note on the gamma distribution. Monthly Weather Review 86(4), 117-122."
            ),
        ),
    ),
    DomainProfile(
        domain="earthquake",
        detectors=("earthquake", "eq_tsunami_cascade"),
        title="Earthquakes (USGS catalog and real-time feed)",
        discipline="Seismology",
        professions=(
            "seismologists and seismic-network analysts",
            "earthquake early-warning engineers",
            "emergency managers",
        ),
        institutions=(
            _i(
                "USGS Earthquake Hazards Program / NEIC",
                "real-time GeoJSON feeds and the ComCat FDSN event service",
                "https://earthquake.usgs.gov/",
            ),
            _i(
                "EarthScope (IRIS) / FDSN",
                "waveform and event web-service standards",
                "https://www.fdsn.org/",
            ),
        ),
        expertise=(
            "the Gutenberg-Richter b-value describes the magnitude-frequency relation; a falling b-value is read as rising stress",
            "aftershock sequences decay by the Omori-Utsu law, so elevated rate after a mainshock is expected, not anomalous",
            "magnitude types (Mw, mb, ML) are not interchangeable; depth is poorly constrained for small or offshore events",
            "inter-event time and seismicity rate are catalog-completeness dependent",
        ),
        observables="Catalog events with magnitude, depth, location and origin time from the USGS one-hour GeoJSON feed and parameterised FDSN queries.",
        method="Engineers magnitude, depth, location, inter-event time, seismicity rate, Gutenberg-Richter b-value and magnitude deviation from the catalog; the earthquake detector adds STA/LTA and spectrogram analysis when waveform data is supplied.",
        label_rule="Statistical: mainshock and large aftershocks are labelled anomalous against background seismicity by a magnitude cut, so the magnitude-derived columns are excluded from the scored features.",
        limits=(
            "catalog data, not waveforms, in the live loader; waveform STA/LTA needs a separate series",
            "no labeled seismic-event evaluation pack is committed for the detector",
        ),
        references=(
            _r(
                "Gutenberg, B. & Richter, C.F. (1944). Frequency of earthquakes in California. Bulletin of the Seismological Society of America 34(4), 185-188."
            ),
            _r(
                "Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a - bM and its confidence limits. Bulletin of the Earthquake Research Institute 43, 237-239."
            ),
            _r(
                "Utsu, T., Ogata, Y. & Matsu'ura, R.S. (1995). The centenary of the Omori formula for a decay law of aftershock activity. Journal of Physics of the Earth 43(1), 1-33.",
                "https://doi.org/10.4294/jpe1952.43.1",
            ),
            _r(
                "USGS Earthquake Catalog (ComCat) and FDSN event web service documentation.",
                "https://earthquake.usgs.gov/fdsnws/event/1/",
            ),
        ),
    ),
    DomainProfile(
        domain="energy",
        detectors=(
            "emp",
            "gic",
            "gic_minute_series",
            "solar_gic_cascade",
            "solar_wind_geomag",
            "solar_storm",
        ),
        title="Energy grid and geomagnetic disturbance (SWPC, EIA)",
        discipline="Space-weather operations and power-system engineering",
        professions=(
            "grid reliability coordinators and transmission operators",
            "space-weather forecasters",
            "protection engineers",
        ),
        institutions=(
            _i(
                "NOAA Space Weather Prediction Center",
                "Kp index, GOES X-ray flares, DSCOVR/ACE solar wind",
                "https://www.swpc.noaa.gov/",
            ),
            _i(
                "GFZ German Research Centre for Geosciences",
                "derives the planetary Kp index",
                "https://www.gfz-potsdam.de/en/kp-index/",
            ),
            _i(
                "U.S. Energy Information Administration",
                "electricity grid demand and supply data (API v2)",
                "https://www.eia.gov/opendata/",
            ),
            _i(
                "NERC",
                "geomagnetic disturbance reliability standard TPL-007",
                "https://www.nerc.com/",
            ),
        ),
        expertise=(
            "Kp is a 3-hourly quasi-logarithmic planetary index (0-9); Kp 7 and above is a NOAA G3 strong storm",
            "grid impact comes from dB/dt, not Kp itself; Kp is the planning proxy operators are briefed on",
            "solar-wind speed and southward IMF Bz precede geomagnetic response by the L1-to-Earth transit time",
            "X-ray flare class (A-X) drives radio blackouts, not grid currents",
        ),
        observables="3-hourly planetary Kp, GOES X-ray flare class, DSCOVR/ACE solar-wind speed and density from SWPC JSON, optionally EIA electricity demand.",
        method="Engineers Kp level, rate of change, 24-hour sum and maximum, 6-hour delta, solar-wind speed and density and X-ray class; every Kp-derived column is declared label-derived and excluded from scoring so the storm label is not recovered from itself.",
        label_rule="Statistical: Kp at or above 7 labels a storm interval; all Kp-derived columns are excluded from the scored features for that reason.",
        limits=(
            "without the EIA key the loader carries no grid-side measurement, only the space-weather drivers",
            "the grid-impact detectors (gic, emp) need magnetometer data this loader does not carry",
        ),
        references=(
            _r(
                "Bartels, J. (1949). The standardized index Ks and the planetary index Kp. IATME Bulletin 12b, 97-120."
            ),
            _r(
                "Matzka, J., Stolle, C., Yamazaki, Y., Bronkalla, O. & Morschhauser, A. (2021). The geomagnetic Kp index and derived indices of geomagnetic activity. Space Weather 19(5).",
                "https://doi.org/10.1029/2020SW002641",
            ),
            _r(
                "NERC Reliability Standard TPL-007, Transmission System Planned Performance for Geomagnetic Disturbance Events."
            ),
            _r("NOAA Space Weather Scales.", "https://www.swpc.noaa.gov/noaa-scales-explanation"),
        ),
    ),
    DomainProfile(
        domain="fema",
        detectors=("temporal", "statistical"),
        title="Federal disaster declarations (OpenFEMA)",
        discipline="Emergency management and public administration",
        professions=(
            "emergency managers",
            "disaster-recovery and mitigation planners",
            "public-policy analysts",
        ),
        institutions=(
            _i(
                "FEMA / OpenFEMA",
                "Disaster Declarations Summaries and related datasets",
                "https://www.fema.gov/about/openfema/data-sets",
            ),
            _i(
                "U.S. Congress (Stafford Act)",
                "the statute that defines major disaster and emergency declarations",
                "https://www.fema.gov/disaster/stafford-act",
            ),
        ),
        expertise=(
            "a Major Disaster declaration (DR) differs from an Emergency (EM) or Fire Management (FM) declaration in the assistance programmes it unlocks",
            "Individual Assistance (IA), Public Assistance (PA) and Hazard Mitigation (HM) flags describe the response scale, not the hazard's physical size",
            "declaration timing lags the event by days to weeks, so it is a response signal, not an early warning",
        ),
        observables="Every federally declared disaster since 1953 with state, incident type, declaration date and the IA/PA/HM programme flags, via OData queries.",
        method="Pulls the Disaster Declarations Summaries with OData paging and engineers state, declaration month, national trailing-12-month count, programme flags and inter-declaration time; used as a cross-domain validation layer against the physical hazard detectors.",
        label_rule="Statistical: declarations requiring the full federal response (Major Disaster with both IA and PA) are labelled anomalous, so the programme-flag columns are excluded from the scored features.",
        limits=(
            "administrative data: a declaration measures policy response, not hazard intensity",
            "no committed evaluation beyond the loader's own tests",
        ),
        references=(
            _r(
                "Robert T. Stafford Disaster Relief and Emergency Assistance Act, 42 U.S.C. 5121 et seq."
            ),
            _r(
                "OpenFEMA API documentation: Disaster Declarations Summaries v2.",
                "https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2",
            ),
        ),
    ),
    DomainProfile(
        domain="financial",
        detectors=("financial_crisis", "financial_stress"),
        title="Financial stress (FRED indicators)",
        discipline="Financial economics and macroprudential supervision",
        professions=(
            "risk managers and market-risk quants",
            "central-bank financial-stability analysts",
            "macro strategists",
        ),
        institutions=(
            _i(
                "Federal Reserve Bank of St. Louis (FRED)",
                "the series archive: VIX, Treasury spreads, credit spreads, fed funds",
                "https://fred.stlouisfed.org/",
            ),
            _i(
                "Cboe",
                "publishes the VIX volatility index",
                "https://www.cboe.com/tradable_products/vix/",
            ),
            _i(
                "Office of Financial Research",
                "publishes the OFR Financial Stress Index",
                "https://www.financialresearch.gov/financial-stress-index/",
            ),
        ),
        expertise=(
            "the 10-year minus 3-month yield-curve inversion is a recession precursor with a 6-18 month lead",
            "VIX is implied volatility, forward-looking by construction; its spikes are coincident with stress, not leading",
            "credit spreads price default risk and liquidity; their widening is the transmission channel to the real economy",
            "crisis episodes are rare, so labels are few and regime-dependent",
        ),
        observables="Daily FRED series: VIX, 10y-3m Treasury spread, high-yield credit spread, effective federal funds rate and the TED spread, aligned on date.",
        method="Fetches each FRED series with the operator's API key, aligns them by date and engineers levels, rates of change, z-scores, a yield-curve inversion flag and VIX/yield-curve correlation; the VIX- and curve-derived columns are declared label-derived and left out of scoring.",
        label_rule="Statistical: catalogued crisis windows (1997 Asian crisis through the 2023 regional-bank stress) are labelled by stress thresholds on VIX and the curve, so those columns are excluded from the scored features.",
        limits=(
            "needs a FRED API key (FRED_API_KEY)",
            "crisis labels are window-based and few; results are regime-sensitive",
        ),
        references=(
            _r(
                "Hakkio, C.S. & Keeton, W.R. (2009). Financial stress: what is it, how can it be measured, and why does it matter? Federal Reserve Bank of Kansas City Economic Review 94(2), 5-50."
            ),
            _r(
                "Estrella, A. & Mishkin, F.S. (1998). Predicting U.S. recessions: financial variables as leading indicators. Review of Economics and Statistics 80(1), 45-61.",
                "https://doi.org/10.1162/003465398557320",
            ),
            _r(
                "Whaley, R.E. (2000). The investor fear gauge. Journal of Portfolio Management 26(3), 12-17."
            ),
        ),
    ),
    DomainProfile(
        domain="flood",
        detectors=("flood", "flood_precipitation", "surge_flood_cascade"),
        title="River flooding (USGS stream gauges)",
        discipline="Surface-water hydrology",
        professions=(
            "hydrologists and river forecasters",
            "floodplain managers",
            "dam and levee safety engineers",
        ),
        institutions=(
            _i(
                "USGS Water Resources (NWIS)",
                "instantaneous gauge height and discharge web service",
                "https://waterservices.usgs.gov/",
            ),
            _i(
                "NOAA National Water Prediction Service (NWS)",
                "official flood stages and river forecasts",
                "https://water.noaa.gov/",
            ),
            _i(
                "FEMA",
                "flood-hazard mapping and disaster declarations for context",
                "https://www.fema.gov/flood-maps",
            ),
        ),
        expertise=(
            "flood stage is a site-specific, NWS-published threshold; the same gauge height means different things at different sites",
            "discharge is derived from stage through a rating curve that shifts after major floods",
            "rate of rise and peak-to-baseline ratio distinguish flash floods from seasonal high water",
        ),
        observables="USGS NWIS instantaneous values: gauge height (ft) and discharge (cfs) time series at the catalogued flood sites.",
        method="Retrieves NWIS instantaneous values for each event site and engineers gauge height, discharge, rate of rise, deviation from the rolling median and peak-to-baseline ratio.",
        label_rule="Statistical: periods when gauge height exceeded the documented NWS flood stage for the site are labelled anomalous.",
        limits=(
            "flood stage exceedance is a stage threshold, so the label and the scored stage are related; interpret benchmark results accordingly",
            "no precipitation or soil-moisture forcing in the live loader",
        ),
        references=(
            _r(
                "Rantz, S.E. et al. (1982). Measurement and computation of streamflow. USGS Water-Supply Paper 2175.",
                "https://doi.org/10.3133/wsp2175",
            ),
            _r(
                "NOAA National Weather Service. Flood categories and flood stage definitions (NWPS).",
                "https://www.weather.gov/aprfc/terminology",
            ),
            _r(
                "USGS Water Services: Instantaneous Values web service.",
                "https://waterservices.usgs.gov/docs/instantaneous-values/",
            ),
        ),
    ),
    DomainProfile(
        domain="hail",
        detectors=("hail",),
        title="Severe hail (SPC report archive)",
        discipline="Severe-storms meteorology",
        professions=(
            "severe-weather forecasters",
            "insurance catastrophe modellers",
            "storm-damage assessors",
        ),
        institutions=(
            _i(
                "NOAA Storm Prediction Center",
                "severe-hail report archive and filtered daily storm reports",
                "https://www.spc.noaa.gov/wcm/",
            ),
            _i(
                "NOAA NCEI Storm Events Database",
                "the quality-controlled long record of severe reports",
                "https://www.ncdc.noaa.gov/stormevents/",
            ),
        ),
        expertise=(
            "significant hail is 2.00 inches (5 cm) and larger by SPC convention; 1.00 inch is the severe threshold",
            "reports are population-biased: more people, more reports, independent of the hail climatology",
            "daily reports are preliminary and sizes are in hundredths of an inch in the daily feed",
        ),
        observables="SPC severe-hail reports 1955-2023 (diameter in inches, location, casualties, losses) and today's preliminary reports.",
        method="Downloads the zipped SPC archive and the daily filtered reports, normalises sizes to inches, and engineers size, casualties, loss, location, temporal clustering, geographic anomaly and calendar features.",
        label_rule="Statistical: significant-hail reports (diameter at or above 2.00 in) are labelled anomalous; the size column is excluded from the scored features.",
        limits=(
            "report data, not radar: no storm environment in the loader itself (the hail detector reads soundings separately)",
            "population reporting bias is not corrected",
        ),
        references=(
            _r(
                "Schaefer, J.T. & Edwards, R. (1999). The SPC tornado/severe thunderstorm database. 11th Conference on Applied Climatology, 215-220."
            ),
            _r(
                "Allen, J.T. & Tippett, M.K. (2015). The characteristics of United States hail reports: 1955-2014. Electronic Journal of Severe Storms Meteorology 10(3).",
                "https://ejssm.com/ojs/index.php/site/article/view/60",
            ),
        ),
    ),
    DomainProfile(
        domain="heatwave",
        detectors=("heatwave",),
        title="Heatwaves (daily station temperature)",
        discipline="Climatology and heat-health epidemiology",
        professions=(
            "climatologists",
            "public-health heat-action planners",
            "occupational-safety officers",
        ),
        institutions=(
            _i(
                "NOAA NCEI",
                "Global Summary of the Day station archive",
                "https://www.ncei.noaa.gov/data/global-summary-of-the-day/",
            ),
            _i(
                "WMO / WHO",
                "joint guidance on heatwaves and health",
                "https://library.wmo.int/idurl/4/54600",
            ),
            _i(
                "CDC Heat & Health Tracker",
                "heat-health surveillance in the United States",
                "https://ephtracking.cdc.gov/Applications/heatTracker/",
            ),
        ),
        expertise=(
            "a heatwave is relative: three or more days above the calendar-day 90th percentile of a multi-year baseline, not an absolute temperature",
            "the Excess Heat Factor combines a significance term (against T95) with an acclimatisation term (against the previous 30 days)",
            "GSOD temperatures are in degrees Fahrenheit with 9999.9 as the missing sentinel",
        ),
        observables="Daily maximum, minimum and mean temperature per NCEI GSOD station-year for the event year and its baseline years.",
        method="Fetches station-year CSVs, builds the calendar-day percentile climatology from the baseline years with a 15-day window and engineers positive deviation, hot-run length and excess over the 90th percentile; the heatwave detector's physics core labels the runs.",
        label_rule="Statistical: days inside a run of three or more consecutive days above the calendar-day 90th percentile (CTX90pct) are labelled anomalous.",
        limits=(
            "labels derive from the scored temperature features, so domain benchmarks measure the percentile rule",
            "no humidity term in the loader; heat index needs dew point",
        ),
        references=(
            _r(
                "Perkins, S.E. & Alexander, L.V. (2013). On the measurement of heat waves. Journal of Climate 26(13), 4500-4517.",
                "https://doi.org/10.1175/JCLI-D-12-00383.1",
            ),
            _r(
                "Nairn, J.R. & Fawcett, R.J.B. (2015). The excess heat factor: a metric for heatwave intensity and its use in classifying heatwave severity. International Journal of Environmental Research and Public Health 12(1), 227-253.",
                "https://doi.org/10.3390/ijerph120100227",
            ),
        ),
    ),
    DomainProfile(
        domain="hurricane",
        detectors=("hurricane", "hurricane_pressure", "surge_flood_cascade"),
        title="Tropical cyclones (IBTrACS best tracks)",
        discipline="Tropical meteorology",
        professions=(
            "hurricane specialists and tropical forecasters",
            "catastrophe-risk modellers",
            "coastal emergency managers",
        ),
        institutions=(
            _i(
                "NOAA NCEI (IBTrACS)",
                "International Best Track Archive for Climate Stewardship",
                "https://www.ncei.noaa.gov/products/international-best-track-archive",
            ),
            _i(
                "NOAA National Hurricane Center",
                "Atlantic and eastern Pacific advisories and best tracks",
                "https://www.nhc.noaa.gov/",
            ),
            _i(
                "Joint Typhoon Warning Center",
                "western Pacific and Indian Ocean warnings",
                "https://www.metoc.navy.mil/jtwc/jtwc.html",
            ),
            _i(
                "WMO",
                "Saffir-Simpson and regional intensity scales, basin RSMC coordination",
                "https://public.wmo.int/",
            ),
        ),
        expertise=(
            "rapid intensification is a 30 kt or greater increase in maximum sustained wind in 24 hours (Kaplan & DeMaria)",
            "best-track positions are 6-hourly, post-season reanalysed, and wind averaging periods differ by agency",
            "pressure and wind are related through a basin-dependent wind-pressure relationship",
        ),
        observables="IBTrACS v04r01 6-hourly storm positions with maximum sustained wind and minimum central pressure, for the last three years and the full North Atlantic archive.",
        method="Downloads the IBTrACS CSV subsets and engineers 6-, 12- and 24-hour wind and pressure changes, wind-pressure deficit and rapid-intensification interaction terms; the wind-change columns are declared label-derived and excluded from scoring.",
        label_rule="Statistical: periods with a 24-hour wind increase of 30 kt or more are labelled rapid intensification; the delta-wind columns are excluded from the scored features.",
        limits=(
            "track data only: no SST, shear or satellite structure in the loader",
            "agency wind-averaging conventions differ; mixed-basin analysis needs care",
        ),
        references=(
            _r(
                "Knapp, K.R., Kruk, M.C., Levinson, D.H., Diamond, H.J. & Neumann, C.J. (2010). The International Best Track Archive for Climate Stewardship (IBTrACS). Bulletin of the American Meteorological Society 91(3), 363-376.",
                "https://doi.org/10.1175/2009BAMS2755.1",
            ),
            _r(
                "Kaplan, J. & DeMaria, M. (2003). Large-scale characteristics of rapidly intensifying tropical cyclones in the North Atlantic basin. Weather and Forecasting 18(6), 1093-1108.",
                "https://doi.org/10.1175/1520-0434(2003)018<1093:LCORIT>2.0.CO;2",
            ),
        ),
    ),
    DomainProfile(
        domain="landslide",
        detectors=("landslide", "rockfall", "fire_debris_flow_cascade"),
        title="Landslides (NASA Global Landslide Catalog)",
        discipline="Engineering geology and geomorphology",
        professions=(
            "engineering geologists and geotechnical engineers",
            "hazard mappers",
            "disaster-risk analysts",
        ),
        institutions=(
            _i(
                "NASA GSFC Cooperative Open Online Landslide Repository (COOLR)",
                "the Global Landslide Catalog ArcGIS REST service",
                "https://gpm.nasa.gov/landslides/",
            ),
            _i(
                "USGS Landslide Hazards Program",
                "US landslide inventories and rainfall thresholds",
                "https://www.usgs.gov/programs/landslide-hazards",
            ),
            _i(
                "International Consortium on Landslides",
                "research coordination and the Landslides journal",
                "https://icl.iplhq.org/",
            ),
        ),
        expertise=(
            "catalog entries are media- and report-sourced; completeness varies by country and decade",
            "trigger (rainfall, earthquake, mining, construction) and size class drive the interpretation of a record",
            "fatal events are a tiny fraction; fatality counts are heavy-tailed and often revised",
        ),
        observables="Global landslide event records with category, trigger, size, fatalities, injuries, location and date from the COOLR ArcGIS REST endpoint.",
        method="Fetches the most recent 2000 events (or a spatial-temporal window around catalogued catastrophic events) and encodes category, trigger, size, country, location, casualties and calendar features; the casualty and size columns are declared label-derived.",
        label_rule="Statistical: fatal or large-scale events are labelled anomalous against smaller, non-fatal failures; fatality count and size code are excluded from the scored features.",
        limits=(
            "event catalog, not slope monitoring: the landslide detector's rainfall and sensor channels come from other inputs",
            "no committed labeled evaluation for the domain detector",
        ),
        references=(
            _r(
                "Kirschbaum, D.B., Adler, R., Hong, Y., Hill, S. & Lerner-Lam, A. (2010). A global landslide catalog for hazard applications: method, results, and limitations. Natural Hazards 52(3), 561-575.",
                "https://doi.org/10.1007/s11069-009-9401-4",
            ),
            _r(
                "Kirschbaum, D. & Stanley, T. (2018). Satellite-based assessment of rainfall-triggered landslide hazard for situational awareness. Earth's Future 6(3), 505-523.",
                "https://doi.org/10.1002/2017EF000715",
            ),
        ),
    ),
    DomainProfile(
        domain="marine",
        detectors=("marine_biodiversity",),
        title="Marine biodiversity (OBIS occurrences)",
        discipline="Marine ecology and biodiversity informatics",
        professions=(
            "marine ecologists",
            "reef and fisheries managers",
            "biodiversity data stewards",
        ),
        institutions=(
            _i(
                "OBIS (IOC-UNESCO)",
                "Ocean Biodiversity Information System API v3",
                "https://obis.org/",
            ),
            _i(
                "GBIF",
                "Darwin Core occurrence standard and global aggregation",
                "https://www.gbif.org/",
            ),
            _i(
                "NOAA Coral Reef Watch",
                "satellite SST anomaly and Degree Heating Weeks for bleaching",
                "https://coralreefwatch.noaa.gov/",
            ),
        ),
        expertise=(
            "species richness from occurrence records is sampling-effort dependent; a drop can be fewer surveys, not fewer species",
            "Darwin Core fields (eventDate, decimalLatitude, scientificName, depth) are the lingua franca of occurrence data",
            "bleaching follows accumulated thermal stress (Degree Heating Weeks), so biodiversity loss lags the heat",
        ),
        observables="Species occurrence records (taxon, location, date, depth) within WKT geometries and time windows around documented bleaching and marine-heatwave episodes.",
        method="Pages the OBIS v3 occurrence API for each event geometry and period and engineers richness ratio, absolute and effort-weighted richness loss relative to the pre-event baseline.",
        label_rule="Statistical: windows where richness falls significantly below the pre-event baseline during a documented bleaching or heatwave episode are labelled anomalous.",
        limits=(
            "occurrence data is effort-biased; the loader does not model survey effort explicitly",
            "no committed labeled evaluation for the marine detector",
        ),
        references=(
            _r(
                "Wieczorek, J. et al. (2012). Darwin Core: an evolving community-developed biodiversity data standard. PLoS ONE 7(1), e29715.",
                "https://doi.org/10.1371/journal.pone.0029715",
            ),
            _r(
                "Hughes, T.P. et al. (2017). Global warming and recurrent mass bleaching of corals. Nature 543, 373-377.",
                "https://doi.org/10.1038/nature21707",
            ),
            _r("OBIS API v3 documentation.", "https://api.obis.org/"),
        ),
    ),
    DomainProfile(
        domain="meteor",
        detectors=("meteor",),
        title="Fireballs and near-Earth objects (NASA/JPL)",
        discipline="Planetary defense and meteor astronomy",
        professions=(
            "planetary-defense analysts",
            "orbital dynamicists",
            "meteor-network observers",
        ),
        institutions=(
            _i(
                "NASA JPL Center for Near-Earth Object Studies (CNEOS)",
                "fireball archive and Sentry impact-risk table",
                "https://cneos.jpl.nasa.gov/",
            ),
            _i(
                "NASA Planetary Defense Coordination Office",
                "US planetary-defense coordination; NeoWs feed",
                "https://science.nasa.gov/planetary-defense/",
            ),
            _i(
                "IAU Minor Planet Center",
                "designations and orbits of small bodies",
                "https://minorplanetcenter.net/",
            ),
        ),
        expertise=(
            "fireball total impact energy (kt TNT) is derived from radiated energy by an empirical efficiency relation",
            "a close approach is judged by miss distance in lunar distances and by the Palermo/Torino scales, not by size alone",
            "absolute magnitude H gives diameter only through an assumed albedo",
        ),
        observables="CNEOS fireball events (date, energy, velocity, altitude, location) since 1988 and NeoWs close approaches (miss distance, relative velocity, estimated diameter, H, hazard flag).",
        method="Fetches the fireball API and the NeoWs feed (NASA_API_KEY) for catalogued windows and engineers per-source features, dispatching on which schema the event carries; events are internally homogeneous (all fireball or all NEO).",
        label_rule="Statistical: windows around major bolides (Chelyabinsk 2013, Bering Sea 2018) and documented close-approach weeks (2019 OK, 2023 BU) are labelled anomalous by energy and miss-distance thresholds.",
        limits=(
            "sparse, heterogeneous events; two sources with different schemas",
            "no committed labeled evaluation for the meteor detector",
        ),
        references=(
            _r(
                "Brown, P., Spalding, R.E., ReVelle, D.O., Tagliaferri, E. & Worden, S.P. (2002). The flux of small near-Earth objects colliding with the Earth. Nature 420, 294-296.",
                "https://doi.org/10.1038/nature01238",
            ),
            _r(
                "Chesley, S.R., Chodas, P.W., Milani, A., Valsecchi, G.B. & Yeomans, D.K. (2002). Quantifying the risk posed by potential Earth impacts. Icarus 159(2), 423-432.",
                "https://doi.org/10.1006/icar.2002.6910",
            ),
            _r(
                "CNEOS Fireball and Bolide Data API.",
                "https://ssd-api.jpl.nasa.gov/doc/fireball.html",
            ),
        ),
    ),
    DomainProfile(
        domain="network_security",
        detectors=("realtime_threat", "threat_detection", "deeplog_sequence"),
        title="Network intrusion detection (CICIDS, NSL-KDD, UNSW-NB15, BATADAL)",
        discipline="Network security and intrusion detection",
        professions=(
            "SOC analysts and detection engineers",
            "network-security researchers",
            "ICS/OT security engineers",
        ),
        institutions=(
            _i(
                "Canadian Institute for Cybersecurity (UNB)",
                "CICIDS2017 and the NSL-KDD redistribution",
                "https://www.unb.ca/cic/datasets/",
            ),
            _i(
                "UNSW Canberra Cyber",
                "UNSW-NB15 dataset",
                "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
            ),
            _i(
                "BATADAL (Battle of the Attack Detection Algorithms)",
                "water-distribution cyber-attack benchmark",
                "https://www.batadal.net/",
            ),
        ),
        expertise=(
            "flow features (duration, bytes, service rates, error rates) describe a connection; labels are per flow",
            "benchmark intrusion datasets are labeled ground truth, which is why this domain is not label-derived",
            "KDD-era attack taxonomies (DoS, probe, R2L, U2R) differ from CICIDS attack classes; models do not transfer blindly",
            "industrial (BATADAL) attacks manipulate sensor readings, a different signature from network flows",
        ),
        observables="Per-flow records with protocol, byte counts, connection and service-rate statistics and error rates, labeled normal or attack, from the public benchmark datasets.",
        method="Bridges the datasets.security and datasets.industrial infrastructure to the loader interface, selecting the flow observables (bytes, service rates, SYN-error rates) that the network-security detector consumes.",
        label_rule="Ground truth: attack labels come from the benchmark authors' capture annotations, not from any scored column.",
        limits=(
            "benchmark captures, not live traffic; the engine's NSL-KDD AUC is measured at engine level",
            "class balance and attack mix differ by dataset",
        ),
        references=(
            _r(
                "Sharafaldin, I., Habibi Lashkari, A. & Ghorbani, A.A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. ICISSP 2018, 108-116.",
                "https://doi.org/10.5220/0006639801080116",
            ),
            _r(
                "Tavallaee, M., Bagheri, E., Lu, W. & Ghorbani, A.A. (2009). A detailed analysis of the KDD CUP 99 data set. IEEE CISDA 2009.",
                "https://doi.org/10.1109/CISDA.2009.5356528",
            ),
            _r(
                "Moustafa, N. & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. MilCIS 2015.",
                "https://doi.org/10.1109/MilCIS.2015.7348942",
            ),
            _r(
                "Taormina, R. et al. (2018). Battle of the Attack Detection Algorithms: disclosing cyber attacks on water distribution networks. Journal of Water Resources Planning and Management 144(8).",
                "https://doi.org/10.1061/(ASCE)WR.1943-5452.0000969",
            ),
        ),
    ),
    DomainProfile(
        domain="pandemic",
        detectors=("case_surge",),
        title="Epidemic surveillance (OWID, WHO GHO)",
        discipline="Infectious-disease epidemiology and public-health surveillance",
        professions=(
            "epidemiologists",
            "public-health surveillance officers",
            "outbreak-response coordinators",
        ),
        institutions=(
            _i(
                "Our World in Data (University of Oxford)",
                "COVID-19 and Mpox compiled datasets",
                "https://ourworldindata.org/coronavirus",
            ),
            _i(
                "WHO Global Health Observatory",
                "OData API for notifiable-disease indicators",
                "https://www.who.int/data/gho",
            ),
            _i(
                "Oxford COVID-19 Government Response Tracker",
                "the stringency index",
                "https://www.bsg.ox.ac.uk/research/covid-19-government-response-tracker",
            ),
        ),
        expertise=(
            "reported cases depend on testing rate and positivity; a surge in cases with flat positivity is more testing, not more disease",
            "the effective reproduction number is estimated from growth rate and the serial interval, with reporting delay",
            "weekday reporting artefacts require 7-day smoothing before any rate-of-change is meaningful",
            "pathogen class (virus, bacterium, fungus, parasite, prion) changes the surveillance cadence and the relevant indicators",
        ),
        observables="Daily cases, deaths, testing and positivity, reproduction-rate estimates and policy stringency per country from OWID, plus WHO GHO indicators for non-COVID diseases.",
        method="Downloads the OWID datasets and GHO OData indicators and engineers smoothed cases and deaths, per-million rates, growth rate, acceleration, reproduction rate, testing and positivity and stringency; the case-growth columns are declared label-derived and excluded from scoring.",
        label_rule="Statistical: periods of rapid case acceleration during the catalogued waves are labelled anomalous; the case-derived growth columns are excluded from the scored features.",
        limits=(
            "country-level aggregates; subnational outbreaks are invisible",
            "reporting changes (definitions, backfills) create artificial steps",
        ),
        references=(
            _r(
                "Hasell, J. et al. (2020). A cross-country database of COVID-19 testing. Scientific Data 7, 345.",
                "https://doi.org/10.1038/s41597-020-00688-8",
            ),
            _r(
                "Cori, A., Ferguson, N.M., Fraser, C. & Cauchemez, S. (2013). A new framework and software to estimate time-varying reproduction numbers during epidemics. American Journal of Epidemiology 178(9), 1505-1512.",
                "https://doi.org/10.1093/aje/kwt133",
            ),
            _r(
                "Hale, T. et al. (2021). A global panel database of pandemic policies (Oxford COVID-19 Government Response Tracker). Nature Human Behaviour 5, 529-538.",
                "https://doi.org/10.1038/s41562-021-01079-8",
            ),
        ),
    ),
    DomainProfile(
        domain="sepsis",
        detectors=("sepsis",),
        title="Sepsis and critical care (PhysioNet 2019)",
        discipline="Critical-care medicine and clinical informatics",
        professions=(
            "intensivists and ICU nurses",
            "clinical-decision-support engineers",
            "hospital quality and sepsis-programme leads",
        ),
        institutions=(
            _i(
                "PhysioNet (MIT Laboratory for Computational Physiology)",
                "hosts the Computing in Cardiology Challenge 2019 data (CC BY 4.0)",
                "https://physionet.org/content/challenge-2019/",
            ),
            _i(
                "Society of Critical Care Medicine / ESICM",
                "the Sepsis-3 consensus definitions",
                "https://www.sccm.org/",
            ),
            _i(
                "Surviving Sepsis Campaign",
                "treatment guidelines and bundles",
                "https://www.sccm.org/survivingsepsiscampaign",
            ),
        ),
        expertise=(
            "Sepsis-3 defines sepsis as organ dysfunction (SOFA increase of 2 or more) from infection; qSOFA is a bedside screen, not a diagnosis",
            "hourly ICU rows are sparse: labs are measured every few hours and vitals every hour, so missingness is informative",
            "the challenge label is set 6 hours before clinical onset, so early positives are the point",
            "a flag is decision support, never a diagnosis; clinical judgement decides",
        ),
        observables="Hourly per-patient ICU records: vital signs, laboratory values, demographics and the SepsisLabel column, in PSV files for training sets A and B.",
        method="Lists the PhysioNet open-data directory and fetches per-patient PSV files, then engineers vital-sign and laboratory levels, hourly deltas, rolling standard deviations and interaction terms for the clinical detectors.",
        label_rule="Ground truth: SepsisLabel as published by the challenge organisers, derived from clinical criteria independent of Mercury's features.",
        limits=(
            "MIMIC-III is credentialed and deliberately not used",
            "the Sepsis-3 screen has committed fixtures but no committed evaluation record yet",
        ),
        references=(
            _r(
                "Singer, M. et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA 315(8), 801-810.",
                "https://doi.org/10.1001/jama.2016.0287",
            ),
            _r(
                "Reyna, M.A. et al. (2020). Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. Critical Care Medicine 48(2), 210-217.",
                "https://doi.org/10.1097/CCM.0000000000004145",
            ),
            _r(
                "Goldberger, A.L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23), e215-e220.",
                "https://doi.org/10.1161/01.CIR.101.23.e215",
            ),
        ),
    ),
    DomainProfile(
        domain="space_weather",
        detectors=(
            "solar_storm",
            "solar_flare",
            "xray_flare",
            "cme_arrival",
            "sep_storm",
            "ionospheric_scintillation",
            "gic",
            "gic_minute_series",
            "solar_gic_cascade",
            "solar_wind_geomag",
        ),
        title="Geomagnetic storms (USGS magnetometers, DONKI)",
        discipline="Geomagnetism and space physics",
        professions=(
            "space-weather forecasters",
            "geomagnetic observatory scientists",
            "grid GMD analysts",
        ),
        institutions=(
            _i(
                "USGS Geomagnetism Program",
                "1-minute observatory magnetometer web service",
                "https://www.usgs.gov/programs/geomagnetism",
            ),
            _i(
                "NASA CCMC DONKI",
                "geomagnetic-storm (GST) records with observed Kp",
                "https://kauai.ccmc.gsfc.nasa.gov/DONKI/",
            ),
            _i(
                "INTERMAGNET",
                "observatory data standards and the global network",
                "https://intermagnet.org/",
            ),
        ),
        expertise=(
            "ground-level dB/dt of the horizontal field is the physically meaningful driver of induced currents, not the field level",
            "Kp is planetary and 3-hourly; a local observatory sees the storm with its own latitude-dependent amplitude",
            "G1 begins at Kp 5; the NOAA G scale is the operational language of warnings",
        ),
        observables="1-minute X, Y, Z, F variation data (nT) from the Boulder observatory and DONKI GST records with observed planetary Kp for the catalogued storms.",
        method="Fetches USGS geomag web-service minute data and DONKI GST records (NASA_API_KEY), and engineers dB/dt of X, Y and H, 10-minute mean dB/dt, 60-minute H range and deviation from the daily median.",
        label_rule="Statistical: minutes inside DONKI-observed Kp windows at or above G1 are labelled storm; labels come from Kp, features from the magnetometer, so the two are not the same measurement.",
        limits=(
            "one reference observatory (BOU); mid-latitude response only",
            "five catalogued storms; small event set",
        ),
        references=(
            _r(
                "Viljanen, A., Nevanlinna, H., Pajunpää, K. & Pulkkinen, A. (2001). Time derivative of the horizontal geomagnetic field as an activity indicator. Annales Geophysicae 19(9), 1107-1118.",
                "https://doi.org/10.5194/angeo-19-1107-2001",
            ),
            _r(
                "Love, J.J. & Finn, C.A. (2011). The USGS Geomagnetism Program and its role in space weather monitoring. Space Weather 9(7).",
                "https://doi.org/10.1029/2011SW000684",
            ),
        ),
    ),
    DomainProfile(
        domain="tornado",
        detectors=("tornado", "tornado_conditions"),
        title="Tornadoes (SPC archive and daily reports)",
        discipline="Severe-storms meteorology",
        professions=(
            "severe-weather forecasters and warning meteorologists",
            "damage-survey teams",
            "emergency managers",
        ),
        institutions=(
            _i(
                "NOAA Storm Prediction Center",
                "tornado archive 1950-2023 and daily storm reports",
                "https://www.spc.noaa.gov/wcm/",
            ),
            _i(
                "NOAA National Weather Service",
                "tornado warnings and EF-scale damage surveys",
                "https://www.weather.gov/",
            ),
            _i(
                "NOAA National Severe Storms Laboratory",
                "mesocyclone and tornado detection research",
                "https://www.nssl.noaa.gov/",
            ),
        ),
        expertise=(
            "EF rating is assigned after a damage survey from damage indicators, so it measures damage, not wind directly",
            "path length and width scale with intensity but are survey-limited",
            "EF3 and stronger tornadoes (significant) cause most fatalities and are the rare class",
        ),
        observables="Tornado reports with EF rating, path length and width, casualties, start location and time, from the SPC archive and today's preliminary reports.",
        method="Downloads the SPC tornado archive and daily reports and engineers EF scale, path geometry, casualties, location, temporal clustering, geographic anomaly and calendar features; the EF column is declared label-derived.",
        label_rule="Statistical: EF3 and stronger tornadoes are labelled anomalous against EF0-EF2; the EF-scale column is excluded from the scored features.",
        limits=(
            "report data; the tornado detector's radar and sounding paths read other inputs",
            "pre-1990s records are less complete and EF ratings before 2007 are F-scale",
        ),
        references=(
            _r(
                "Schaefer, J.T. & Edwards, R. (1999). The SPC tornado/severe thunderstorm database. 11th Conference on Applied Climatology, 215-220."
            ),
            _r(
                "McDonald, J.R. & Mehta, K.C. (2006). A recommendation for an Enhanced Fujita scale (EF-scale). Wind Science and Engineering Center, Texas Tech University."
            ),
            _r(
                "Fujita, T.T. (1971). Proposed characterization of tornadoes and hurricanes by area and intensity. SMRP Research Paper 91, University of Chicago."
            ),
        ),
    ),
    DomainProfile(
        domain="tsunami",
        detectors=("tsunami", "eq_tsunami_cascade"),
        title="Tsunamis (NDBC DART bottom-pressure buoys)",
        discipline="Physical oceanography and tsunami warning",
        professions=(
            "tsunami warning-centre scientists",
            "coastal emergency managers",
            "ocean-observing engineers",
        ),
        institutions=(
            _i(
                "NOAA National Data Buoy Center",
                "DART real-time bottom-pressure data",
                "https://www.ndbc.noaa.gov/",
            ),
            _i(
                "NOAA PMEL",
                "DART system design and tsunami forecasting research",
                "https://nctr.pmel.noaa.gov/Dart/",
            ),
            _i(
                "NOAA NCEI Global Historical Tsunami Database",
                "tsunami event and run-up records",
                "https://www.ngdc.noaa.gov/hazard/tsu_db.shtml",
            ),
            _i("Pacific Tsunami Warning Center", "operational warnings", "https://tsunami.gov/"),
        ),
        expertise=(
            "a tsunami appears in deep water as a centimetre-scale pressure residual after tide removal; detiding is the whole problem",
            "DART reports at 15-minute cadence normally and 15-second cadence when triggered",
            "arrival windows are predicted from source location and bathymetry; the buoy confirms, it does not forecast alone",
        ),
        observables="DART bottom-pressure recorder readings (water-column height) from the NDBC real-time service and historical arrival windows.",
        method="Fetches NDBC real-time DART records and engineers tidal deviation, absolute deviation, rate of change, rolling standard deviation and short-window energy; historical ground truth is built from documented arrival windows at specific stations.",
        label_rule="Statistical: samples inside documented tsunami arrival windows at the station are labelled anomalous.",
        limits=(
            "historical labelled sets are synthesised from documented arrival windows, flagged as such",
            "the trained tsunami checkpoint was refused by the merit gate; the physics path is in charge",
        ),
        references=(
            _r(
                "González, F.I. et al. (2005). The NTHMP tsunameter network. Natural Hazards 35(1), 25-39.",
                "https://doi.org/10.1007/s11069-004-2402-4",
            ),
            _r(
                "Titov, V.V. et al. (2005). Real-time tsunami forecasting: challenges and solutions. Natural Hazards 35(1), 41-58.",
                "https://doi.org/10.1007/s11069-004-2403-3",
            ),
            _r(
                "Mofjeld, H.O. (1997). Tsunami detection algorithm. NOAA Pacific Marine Environmental Laboratory.",
                "https://nctr.pmel.noaa.gov/tda_documentation.html",
            ),
        ),
    ),
    DomainProfile(
        domain="volcanic",
        detectors=("volcanic",),
        title="Volcanic unrest (USGS alert levels)",
        discipline="Volcanology and volcano monitoring",
        professions=(
            "volcano-observatory scientists",
            "aviation meteorologists (ash advisories)",
            "civil-protection planners",
        ),
        institutions=(
            _i(
                "USGS Volcano Hazards Program",
                "alert levels, aviation colour codes and the monitored-volcano list",
                "https://www.usgs.gov/programs/VHP",
            ),
            _i(
                "Smithsonian Global Volcanism Program",
                "eruption history and volcano catalog",
                "https://volcano.si.edu/",
            ),
            _i("IAVCEI", "international volcanology association", "https://www.iavceivolcano.org/"),
        ),
        expertise=(
            "alert level (Normal, Advisory, Watch, Warning) and aviation colour code (Green to Red) are expert judgements issued by the observatory, not raw measurements",
            "unrest precursors are seismicity (long-period events, tremor), deformation, gas flux and thermal output",
            "alert changes are rare and step-like; most of the record is Normal/Green",
        ),
        observables="Current alert level and aviation colour code per monitored US volcano, with coordinates, elevation and time since the last change.",
        method="Polls the USGS volcano API for alerts and the volcano list and encodes alert level, colour code, their deltas, location, elevation and days since the last change; the alert-derived columns are declared label-derived.",
        label_rule="Statistical: WARNING/RED periods are labelled anomalous against NORMAL/GREEN; the alert and colour-code columns are excluded from the scored features.",
        limits=(
            "alert levels are the observatory's interpretation; the loader carries no seismic, gas or deformation streams",
            "US volcanoes only",
        ),
        references=(
            _r(
                "Gardner, C.A. & Guffanti, M.C. (2006). U.S. Geological Survey's alert notification system for volcanic activity. USGS Fact Sheet 2006-3139.",
                "https://doi.org/10.3133/fs20063139",
            ),
            _r(
                "Sparks, R.S.J. (2003). Forecasting volcanic eruptions. Earth and Planetary Science Letters 210(1-2), 1-15.",
                "https://doi.org/10.1016/S0012-821X(03)00124-9",
            ),
            _r(
                "Global Volcanism Program (2024). Volcanoes of the World (v. 5.2). Smithsonian Institution.",
                "https://doi.org/10.5479/si.GVP.VOTW5-2024.5.2",
            ),
        ),
    ),
    DomainProfile(
        domain="wildfire",
        detectors=("wildfire", "fire_debris_flow_cascade", "pm25_air_quality"),
        title="Active fire (NASA FIRMS VIIRS)",
        discipline="Fire science and thermal remote sensing",
        professions=(
            "wildland-fire analysts and incident meteorologists",
            "remote-sensing scientists",
            "land and emergency managers",
        ),
        institutions=(
            _i(
                "NASA FIRMS (LANCE)",
                "near-real-time active-fire detections from MODIS and VIIRS",
                "https://firms.modaps.eosdis.nasa.gov/",
            ),
            _i(
                "NOAA/NASA JPSS (VIIRS)",
                "the NOAA-20 and Suomi NPP instruments",
                "https://www.nesdis.noaa.gov/our-satellites/currently-flying/joint-polar-satellite-system",
            ),
            _i(
                "National Interagency Fire Center",
                "US wildland-fire coordination",
                "https://www.nifc.gov/",
            ),
        ),
        expertise=(
            "Fire Radiative Power (MW) is proportional to the rate of biomass combustion; it is the physical intensity measure",
            "a detection is a pixel-level thermal anomaly with a confidence class; clouds, sun glint and industrial heat sources cause false and missed detections",
            "VIIRS 375 m detections resolve smaller fires than MODIS 1 km; scan and track give the pixel footprint",
        ),
        observables="VIIRS NOAA-20 NRT active-fire detections: brightness temperature, FRP, confidence, scan and track within an area or country over a number of days.",
        method="Queries the FIRMS area and country CSV endpoints (NASA_FIRMS_MAP_KEY, with FIRMS_MAP_KEY accepted) and engineers brightness, FRP, confidence, pixel footprint, cluster count and spread rate; FRP is declared label-derived.",
        label_rule="Statistical: high-FRP detections inside the event window are labelled anomalous; the FRP column is excluded from the scored features.",
        limits=(
            "needs a FIRMS map key",
            "detector-level labeled evaluation pending its domain pack; the live wiring is loader-tested",
        ),
        references=(
            _r(
                "Schroeder, W., Oliva, P., Giglio, L. & Csiszar, I.A. (2014). The New VIIRS 375 m active fire detection data product. Remote Sensing of Environment 143, 85-96.",
                "https://doi.org/10.1016/j.rse.2013.12.008",
            ),
            _r(
                "Wooster, M.J., Roberts, G., Perry, G.L.W. & Kaufman, Y.J. (2005). Retrieval of biomass combustion rates and totals from fire radiative power observations. Journal of Geophysical Research 110, D24311.",
                "https://doi.org/10.1029/2005JD006318",
            ),
            _r(
                "Giglio, L., Descloitres, J., Justice, C.O. & Kaufman, Y.J. (2003). An enhanced contextual fire detection algorithm for MODIS. Remote Sensing of Environment 87(2-3), 273-282.",
                "https://doi.org/10.1016/S0034-4257(03)00184-6",
            ),
        ),
    ),
    DomainProfile(
        domain="schumann",
        detectors=("schumann_resonance",),
        title="Schumann resonances (ELF electromagnetic cavity)",
        discipline="Atmospheric electricity and ELF geophysics",
        professions=(
            "ELF observatory scientists",
            "atmospheric-electricity researchers",
            "space-physics analysts",
        ),
        institutions=(
            _i(
                "Hylaty ELF station (AGH University, Krakow)",
                "long-running ELF observatory",
                "https://www.oa.uj.edu.pl/elf/",
            ),
            _i(
                "Nagycenk Geophysical Observatory (HUN-REN EPSS)",
                "Schumann-resonance records since 1993",
                "https://epss.hu/",
            ),
        ),
        expertise=(
            "the fundamental near 7.83 Hz and harmonics near 14, 20, 26 and 33 Hz are set by the Earth-ionosphere cavity geometry",
            "amplitude tracks global lightning activity with a diurnal cycle; frequency shifts follow ionospheric height",
            "claims of seismic precursors are contested; Mercury treats them as preregistered hypotheses, not facts",
        ),
        observables="An ELF magnetic or electric field series sampled fast enough to resolve the first harmonics, supplied by the operator (hazard-viz -d schumann); no live loader exists.",
        method="The Schumann detector computes the ELF spectrum by FFT, tracks per-harmonic amplitude and frequency deviation against a baseline and renders the diagnostics as PNG, GeoJSON or HTML.",
        label_rule="No labels: stability tests exist under tests/space, and the preregistration documents the hypotheses to be tested before any claim is made.",
        limits=(
            "no live data source in the package; operator-supplied series only",
            "no labeled corpus; any geophysical-precursor interpretation is a hypothesis",
        ),
        references=(
            _r(
                "Schumann, W.O. (1952). Über die strahlungslosen Eigenschwingungen einer leitenden Kugel, die von einer Luftschicht und einer Ionosphärenhülle umgeben ist. Zeitschrift für Naturforschung A 7(2), 149-154.",
                "https://doi.org/10.1515/zna-1952-0202",
            ),
            _r(
                "Nickolaenko, A.P. & Hayakawa, M. (2002). Resonances in the Earth-Ionosphere Cavity. Kluwer Academic Publishers."
            ),
            _r(
                "Mercury Agent, docs/SCHUMANN_PREREGISTRATION.md (preregistered hypotheses and analysis plan)."
            ),
        ),
    ),
)

DOMAIN_PROFILES: dict[str, DomainProfile] = {p.domain: p for p in _DOMAINS}


# --------------------------------------------------------------------------- #
# Detector profiles
# --------------------------------------------------------------------------- #

_SWEEP_NOTE = "artifacts/detector_evidence/2026-10-27_ledger_seed.json"

_DETECTORS: tuple[DetectorProfile, ...] = (
    # ----------------------------------------------------------------- base
    DetectorProfile(
        name="statistical",
        discipline="Robust statistics and signal processing (Mercury-native ensemble)",
        lineage="Mercury's original three-component ensemble: an FFT harmonic score, a kinematic jerk/curvature score and a Fisher-information score, re-weighted after fit by each component's measured AUC separation.",
        mechanism="Three deterministic scorers each map the series to [0, 1]; fit() measures how well each separates and sets its weight, zeroing any component whose signal is anti-correlated with anomalies; the ensemble falls back to fixed weights only when every component is near-random.",
        practitioners="Monitoring engineers and analysts who want a dependency-light first pass; reading the score needs the knowledge that it is relative to the fitted reference, and that standalone NAB performance is near chance (engine fusion carries it).",
        use_when="A quick, deterministic baseline on a 1-D series with a clean reference window, or as one voter inside fusion.",
        not_for="Change points, slow drift or discrete-sequence grammar violations, which it is declared not to detect.",
        references=(
            _r(
                "Iglewicz, B. & Hoaglin, D.C. (1993). How to Detect and Handle Outliers. ASQC Quality Press."
            ),
            _r(
                "Chandola, V., Banerjee, A. & Kumar, V. (2009). Anomaly detection: a survey. ACM Computing Surveys 41(3), 1-58.",
                "https://doi.org/10.1145/1541880.1541882",
            ),
            _r(
                "Mercury Agent, docs/DETECTION_MECHANISMS.md (ResonanceScore, KinematicScore, InfoGeometryScore)."
            ),
        ),
    ),
    DetectorProfile(
        name="temporal",
        discipline="Time-series analysis",
        lineage="Classical decomposition thinking (trend, seasonality, level) applied to anomaly scoring; the registry's validated generic-series workhorse in the 2026-10-27 sweep.",
        mechanism="Separates trend from residual, tests for sudden level changes and seasonal departures, and scores each sample by how far it sits from the locally expected value; change points, level shifts, drift and variance changes all raise the score.",
        practitioners="Operations and SRE teams, demand and capacity planners; reading it needs the distinction between a seasonal excursion (expected) and a departure from the seasonal pattern (flagged).",
        use_when="Streams with trend or seasonality where regime changes matter as much as spikes.",
        not_for="Multivariate correlation structure; it reads one series at a time.",
        references=(
            _r(
                "Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.), OTexts, chapter 3 (time-series decomposition).",
                "https://otexts.com/fpp3/",
            ),
            _r(
                "Cleveland, R.B., Cleveland, W.S., McRae, J.E. & Terpenning, I. (1990). STL: a seasonal-trend decomposition procedure based on loess. Journal of Official Statistics 6(1), 3-73."
            ),
            _r(f"Mercury Agent, {_SWEEP_NOTE} (generic-series validation)."),
        ),
    ),
    DetectorProfile(
        name="spatial",
        discipline="Spatial statistics and geographic data analysis",
        lineage="Distance-to-centroid outliers plus the Local Outlier Factor (Breunig et al., 2000) on a KD-tree, with spatial-autocorrelation summaries in the output.",
        mechanism="Fits the training point cloud's centroid and a quantile radius, scores each point by distance beyond that radius and by LOF density ratio against its neighbours, then reports spatial-autocorrelation statistics for the flagged set.",
        practitioners="GIS analysts and geostatisticians; reading it needs to know that coordinates are treated as Euclidean features, so longitude scale and projection matter.",
        use_when="Point clouds (geographic or abstract) where isolated or low-density points are the anomaly.",
        not_for="Trajectories or anything time-ordered; see geo_movement for movement plausibility.",
        references=(
            _r(
                "Breunig, M.M., Kriegel, H.-P., Ng, R.T. & Sander, J. (2000). LOF: identifying density-based local outliers. ACM SIGMOD 2000, 93-104.",
                "https://doi.org/10.1145/342009.335388",
            ),
            _r(
                "Anselin, L. (1995). Local Indicators of Spatial Association - LISA. Geographical Analysis 27(2), 93-115.",
                "https://doi.org/10.1111/j.1538-4632.1995.tb00338.x",
            ),
        ),
    ),
    DetectorProfile(
        name="geo_movement",
        discipline="Trajectory data mining and movement ecology",
        lineage="Movement-plausibility channels (speed feasibility, step-length distribution, reporting silence) fused by noisy-OR, the trajectory-level complement to the point-cloud spatial detector.",
        mechanism="For each (lat, lon, time) step computes implied speed against a feasibility ceiling, step length against the fitted distribution, and silence against the expected cadence, each in [0, 1]; the channels combine as 1 - prod(1 - s), so one saturated channel alone can fire.",
        practitioners="Fleet, logistics and wildlife-tracking analysts, search-and-rescue coordinators; reading it needs the configured speed ceiling and expected reporting interval.",
        use_when="GPS or AIS/ADS-B tracks where an impossible jump, teleport or silence is the anomaly.",
        not_for="Distributional outliers in feature space; it is declared not to detect them.",
        references=(
            _r(
                "Zheng, Y. (2015). Trajectory data mining: an overview. ACM Transactions on Intelligent Systems and Technology 6(3), 1-41.",
                "https://doi.org/10.1145/2743025",
            ),
            _r(
                "Lee, J.-G., Han, J. & Li, X. (2008). Trajectory outlier detection: a partition-and-detect framework. IEEE ICDE 2008, 140-149.",
                "https://doi.org/10.1109/ICDE.2008.4497422",
            ),
        ),
    ),
    DetectorProfile(
        name="dimensional",
        discipline="Multivariate statistics and representation learning",
        lineage="PCA reconstruction error (Jolliffe) and a neural autoencoder (Hinton & Salakhutdinov, 2006) combined with a Fourier-domain term for spectral anomalies.",
        mechanism="Projects rows onto the principal subspace and a learned manifold, scores by reconstruction error against a threshold multiplier, and optionally adds a spectral term from the Fourier transform of the representation.",
        practitioners="Data scientists working with wide tables and sensor arrays; reading it needs the number of retained components and that the correlation-break specialty needs genuinely multivariate input.",
        use_when="High-dimensional rows where rows that break the usual correlation structure are the anomaly.",
        not_for="1-D streams: its sweep mean on 1-D NAB was near chance because its specialty is multivariate.",
        references=(
            _r(
                "Jolliffe, I.T. (2002). Principal Component Analysis (2nd ed.). Springer.",
                "https://doi.org/10.1007/b98835",
            ),
            _r(
                "Hinton, G.E. & Salakhutdinov, R.R. (2006). Reducing the dimensionality of data with neural networks. Science 313(5786), 504-507.",
                "https://doi.org/10.1126/science.1127647",
            ),
        ),
    ),
    DetectorProfile(
        name="directive",
        discipline="Rule-based detection and density stability (Mercury-native Sigma Directive)",
        lineage="Mercury's Sigma Directive protocols: Pattern Convergence (PCP), Gravitational Stability Integrity (GSIS), Recursive Memory Dynamics (RMD) and the Ethical Oversight Amplifier (EOA), with optional harmonic analysis.",
        mechanism="Measures deviation from a baseline pattern, local density stability, and a bounded temporal memory of recent states, combines the protocol scores with sigma thresholds and amplifies anomalies the ethical layer marks as significant; state is thread-local and the memory buffer is bounded.",
        practitioners="Operators who need an auditable, rule-stated detector for critical systems; reading it needs the protocol vocabulary documented in the repository rather than a statistics text.",
        use_when="Tabular batches where explicit sigma rules and a bounded memory of recent behaviour are wanted alongside learned detectors.",
        not_for="A replacement for the statistical ensemble; it is one voter with its own protocols.",
        references=(
            _r(
                "Mercury Agent, docs/DETECTION_MECHANISMS.md (Sigma Directive protocols: PCP, GSIS, RMD, EOA)."
            ),
            _r(
                "Grubbs, F.E. (1969). Procedures for detecting outlying observations in samples. Technometrics 11(1), 1-21 (the sigma-rule tradition).",
                "https://doi.org/10.1080/00401706.1969.10490657",
            ),
        ),
    ),
    DetectorProfile(
        name="graph_based",
        discipline="Network science and graph mining",
        lineage="Louvain community detection (Blondel et al., 2008), PageRank and betweenness centrality, and cascade-failure analysis on NetworkX graphs.",
        mechanism="Computes community structure and centrality measures for the fitted graph, then flags nodes and edges whose metrics sit more than a z-score threshold (default 3) from the baseline, with a cascade analysis of what their failure would propagate.",
        practitioners="Network and infrastructure analysts, fraud and security graph analysts; reading it needs to know the threshold is in standard deviations of graph metrics, not a [0, 1] probability.",
        use_when="Graph-structured data (dependency, communication, transaction networks) where structural change is the anomaly.",
        not_for="Numeric series or tables; and the module is still simulation-grounded, awaiting labeled graph data.",
        references=(
            _r(
                "Blondel, V.D., Guillaume, J.-L., Lambiotte, R. & Lefebvre, E. (2008). Fast unfolding of communities in large networks. Journal of Statistical Mechanics 2008, P10008.",
                "https://doi.org/10.1088/1742-5468/2008/10/P10008",
            ),
            _r(
                "Akoglu, L., Tong, H. & Koutra, D. (2015). Graph based anomaly detection and description: a survey. Data Mining and Knowledge Discovery 29(3), 626-688.",
                "https://doi.org/10.1007/s10618-014-0365-y",
            ),
        ),
    ),
    DetectorProfile(
        name="kmeans_distance",
        discipline="Unsupervised clustering",
        lineage="Distance to the nearest k-means centroid (MacQueen 1967; Lloyd 1982) as an anomaly signal, revived from Mercury's dormant cognitive clusterer and measured on labeled ADBench tables.",
        mechanism="Standardises features, fits k centroids, and emits per-centroid distances plus the nearest-centroid distance as a feature block; a point far from every centroid is poorly explained by the learned structure.",
        practitioners="Data scientists doing tabular anomaly detection; reading it needs the measured context (held-out ROC-AUC 0.76-0.98 on four ADBench tables, 0.63 on Pima) rather than a generic claim.",
        use_when="Tabular data with cluster structure, as an opt-in fusion feature.",
        not_for="Series with temporal structure, or data where anomalies form their own dense cluster.",
        references=(
            _r(
                "MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proc. 5th Berkeley Symposium on Mathematical Statistics and Probability 1, 281-297."
            ),
            _r(
                "Lloyd, S.P. (1982). Least squares quantization in PCM. IEEE Transactions on Information Theory 28(2), 129-137.",
                "https://doi.org/10.1109/TIT.1982.1056489",
            ),
            _r("Mercury Agent, artifacts/dormant_module_revival.json (measured ADBench results)."),
        ),
    ),
    DetectorProfile(
        name="spectral_residual",
        discipline="Spectral signal processing (visual saliency transferred to time series)",
        lineage="The Spectral Residual saliency transform (Hou & Zhang, 2007) adapted for time-series anomaly detection at Microsoft (Ren et al., KDD 2019).",
        mechanism="Takes the log-amplitude spectrum, subtracts its smoothed version, inverts the residual spectrum to a time-domain saliency map, and scores each point by its local saliency z-score squashed through 1 - exp(-s/scale) with a MAD scale learned on training data.",
        practitioners="Streaming-monitoring engineers; reading it needs to know it is training-free and keyed to sharp local deviations.",
        use_when="Cheap, training-free spike detection on streams with periodic structure.",
        not_for="Slow drift or level shifts, which the saliency formulation does not see; its NAB sweep mean was at chance.",
        references=(
            _r(
                "Ren, H. et al. (2019). Time-series anomaly detection service at Microsoft. ACM KDD 2019, 3009-3017.",
                "https://doi.org/10.1145/3292500.3330680",
            ),
            _r(
                "Hou, X. & Zhang, L. (2007). Saliency detection: a spectral residual approach. IEEE CVPR 2007.",
                "https://doi.org/10.1109/CVPR.2007.383267",
            ),
        ),
    ),
    DetectorProfile(
        name="bocpd",
        discipline="Bayesian statistics and online change-point detection",
        lineage="Bayesian Online Change-Point Detection (Adams & MacKay, 2007) with the Gaussian / Normal-Inverse-Gamma conjugate model and a constant hazard.",
        mechanism="Maintains a posterior over the run length since the last change; the per-sample score is the posterior probability that the run length is shorter than a grace period, which jumps when the stream's statistics shift and stays low in a stable regime.",
        practitioners="Statisticians and monitoring engineers who want a calibrated change-point probability; reading it needs the hazard (expected regime length) and the grace period.",
        use_when="Detecting regime, mean or variance changes online in a real-valued stream.",
        not_for="Isolated point outliers, which it is declared not to detect.",
        references=(
            _r(
                "Adams, R.P. & MacKay, D.J.C. (2007). Bayesian online changepoint detection. arXiv:0710.3742.",
                "https://arxiv.org/abs/0710.3742",
            ),
            _r(
                "Murphy, K.P. (2007). Conjugate Bayesian analysis of the Gaussian distribution. Technical note, UBC."
            ),
        ),
    ),
    DetectorProfile(
        name="sprt",
        discipline="Sequential statistical hypothesis testing",
        lineage="Wald's Sequential Probability Ratio Test (1945) in its Gaussian mean-shift form, deciding at the exact Wald boundaries A = (1-beta)/alpha and B = beta/(1-alpha).",
        mechanism="Accumulates the log-likelihood ratio of a shifted mean against the fitted reference mean sample by sample, restarts after each decision, and scores by sigmoid(LLR - log A) so 0.5 falls exactly at the H1 boundary.",
        practitioners="Quality and reliability engineers, clinical-trial statisticians; reading it needs alpha, beta and the hypothesised shift, and the knowledge that point-wise AUROC misrepresents a sequential test (a detection-delay protocol is the right measure).",
        use_when="Deciding as early as possible whether a stream's mean has shifted, with stated error rates.",
        not_for="Point spikes or variance changes; declared not detected.",
        references=(
            _r(
                "Wald, A. (1945). Sequential tests of statistical hypotheses. Annals of Mathematical Statistics 16(2), 117-186.",
                "https://doi.org/10.1214/aoms/1177731118",
            ),
            _r("Wald, A. (1947). Sequential Analysis. Wiley."),
        ),
    ),
    DetectorProfile(
        name="page_hinkley",
        discipline="Sequential statistical hypothesis testing",
        lineage="Page's cumulative-sum test (1954) in the Page-Hinkley form, both one-sided variants, alarm at PH_t > lambda.",
        mechanism="Accumulates m_t = sum(x_i - x_bar - delta) in standardised units, tracks PH_t = m_t - min m_s for upward and the mirror for downward shifts, alarms when PH_t exceeds lambda and restarts both accumulators after an alarm.",
        practitioners="Process-control and concept-drift practitioners; reading it needs delta (tolerance) and lambda (threshold) in standard-deviation units, and the same detection-delay caveat as sprt.",
        use_when="Sustained mean shifts and drift in a stream, with a tunable tolerance for small wander.",
        not_for="Isolated point outliers; declared not detected.",
        references=(
            _r(
                "Page, E.S. (1954). Continuous inspection schemes. Biometrika 41(1-2), 100-115.",
                "https://doi.org/10.1093/biomet/41.1-2.100",
            ),
            _r(
                "Hinkley, D.V. (1971). Inference about the change-point from cumulative sum tests. Biometrika 58(3), 509-523.",
                "https://doi.org/10.1093/biomet/58.3.509",
            ),
        ),
    ),
    DetectorProfile(
        name="spot_evt",
        discipline="Extreme value theory",
        lineage="SPOT / DSPOT (Siffer et al., KDD 2017): Peaks-Over-Threshold with a Generalized Pareto tail (Pickands 1975; Balkema & de Haan 1974) giving a risk-budgeted threshold.",
        mechanism="Fits a GPD to excesses over a high empirical quantile, maps a target false-positive risk q to a threshold z_q, evolves the tail online on local extremes (de-trending first in DSPOT mode) and scores each point by its EVT tail probability normalised so an at-threshold point sits near 0.5.",
        practitioners="Risk engineers and streaming-monitoring teams; reading it needs the risk q (allowed false-alarm rate) and the initial calibration length.",
        use_when="Setting alarm thresholds on heavy-tailed streams with an explicit false-alarm budget.",
        not_for="Change points or drift; the tail model assumes local stationarity after de-trending.",
        references=(
            _r(
                "Siffer, A., Fouque, P.-A., Termier, A. & Largouët, C. (2017). Anomaly detection in streams with extreme value theory. ACM KDD 2017, 1067-1075.",
                "https://doi.org/10.1145/3097983.3098144",
            ),
            _r(
                "Pickands, J. (1975). Statistical inference using extreme order statistics. Annals of Statistics 3(1), 119-131.",
                "https://doi.org/10.1214/aos/1176343003",
            ),
        ),
    ),
    DetectorProfile(
        name="hawkes",
        discipline="Point-process statistics",
        lineage="Hawkes self-exciting point process (Hawkes, 1971) with an exponential kernel, applied to per-bin count streams.",
        mechanism="Tracks the conditional intensity lambda_t = mu + g_t with g_t = exp(-beta)(g_{t-1} + alpha n_{t-1}) and scores each bin by the absolute Pearson residual |n_t - lambda_t| / sqrt(lambda_t), so both bursts and unexpected silences score high.",
        practitioners="Seismologists (ETAS tradition), epidemiologists, security and social-media analysts; reading it needs to know the input must be event counts, not values.",
        use_when="Count or event-rate streams where clustering beyond a Poisson baseline is the anomaly.",
        not_for="Value streams or level shifts; the sweep drove it with values and the ontology records the mismatch.",
        references=(
            _r(
                "Hawkes, A.G. (1971). Spectra of some self-exciting and mutually exciting point processes. Biometrika 58(1), 83-90.",
                "https://doi.org/10.1093/biomet/58.1.83",
            ),
            _r(
                "Ogata, Y. (1988). Statistical models for earthquake occurrences and residual analysis for point processes. Journal of the American Statistical Association 83(401), 9-27.",
                "https://doi.org/10.1080/01621459.1988.10478560",
            ),
        ),
    ),
    DetectorProfile(
        name="particle_filter",
        discipline="Sequential Monte Carlo and state estimation",
        lineage="The bootstrap particle filter (Gordon, Salmond & Smith, 1993) on a local-level state-space model, scoring one-step-ahead innovations.",
        mechanism="Propagates a cloud of weighted particles through x_t = x_{t-1} + process noise, predicts y_t, scores |y_t - E[y_t]| / std[y_t] from the particle predictive, resamples systematically with a fixed seed, and squashes with a robust innovation scale learned in fit.",
        practitioners="Tracking and control engineers; reading it needs the process and observation noise estimates and the particle count.",
        use_when="Univariate monitoring where the noise is non-Gaussian or heavy-tailed and a Kalman filter would be misled.",
        not_for="Streams with strong seasonality; the random-walk model has no seasonal term.",
        references=(
            _r(
                "Gordon, N.J., Salmond, D.J. & Smith, A.F.M. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. IEE Proceedings F 140(2), 107-113.",
                "https://doi.org/10.1049/ip-f-2.1993.0015",
            ),
            _r(
                "Doucet, A. & Johansen, A.M. (2009). A tutorial on particle filtering and smoothing: fifteen years later. In Handbook of Nonlinear Filtering, Oxford University Press."
            ),
        ),
    ),
    DetectorProfile(
        name="imm",
        discipline="Multiple-model estimation (tracking)",
        lineage="The Interacting Multiple Model estimator (Blom & Bar-Shalom, 1988): a quiet and a manoeuvring constant-velocity Kalman filter blended by a Markov mode-transition matrix.",
        mechanism="Two Kalman filters share state [level, slope] and differ in process noise; the IMM mixes their estimates each step and scores the observation by the mode-averaged normalised predictive innovation squashed into [0, 1].",
        practitioners="Radar and target-tracking engineers, and anyone monitoring a regime-switching system; reading it needs the two process-noise settings and the transition probabilities.",
        use_when="Streams that alternate between calm and manoeuvring regimes where a single-model filter would lag.",
        not_for="Point outliers in otherwise stationary noise; the sweep mean on real streams was at chance.",
        references=(
            _r(
                "Blom, H.A.P. & Bar-Shalom, Y. (1988). The interacting multiple model algorithm for systems with Markovian switching coefficients. IEEE Transactions on Automatic Control 33(8), 780-783.",
                "https://doi.org/10.1109/9.1299",
            ),
            _r(
                "Bar-Shalom, Y., Li, X.R. & Kirubarajan, T. (2001). Estimation with Applications to Tracking and Navigation. Wiley."
            ),
        ),
    ),
    DetectorProfile(
        name="gaussian_process",
        discipline="Bayesian non-parametric regression",
        lineage="Gaussian-process regression (Rasmussen & Williams, 2006) with an RBF kernel over a trailing window, giving a one-step-ahead predictive mean and variance.",
        mechanism="Conditions an RBF GP on the preceding window (integer time offsets as inputs), predicts the current point with mean and variance, and scores the standardised residual |y - mean| / sqrt(var) squashed into [0, 1]; hyperparameters are grid-maximised on the windowed marginal likelihood in fit.",
        practitioners="Statisticians and scientific-instrument monitoring teams who want calibrated predictive uncertainty; reading it needs the window length and the learned length scale.",
        use_when="Smooth series where a residual should be judged against the model's own uncertainty rather than a fixed sigma.",
        not_for="Long windows (inference is cubic) or discontinuous, piecewise data.",
        references=(
            _r(
                "Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for Machine Learning. MIT Press.",
                "https://gaussianprocess.org/gpml/",
            ),
        ),
    ),
    DetectorProfile(
        name="echo_state",
        discipline="Reservoir computing (recurrent neural networks)",
        lineage="Echo State Networks (Jaeger, 2001): a fixed random reservoir with the echo-state property and a ridge-regression readout.",
        mechanism="Drives a seeded, spectral-radius-scaled reservoir with the series, trains only a linear readout to predict the next sample, and scores the standardised one-step prediction residual squashed into [0, 1].",
        practitioners="Machine-learning engineers working on non-linear temporal structure without back-propagation through time; reading it needs the reservoir size, spectral radius and ridge penalty.",
        use_when="Non-linear temporal dynamics where a one-step predictor should learn in one pass on CPU.",
        not_for="Very short series (the reservoir needs a washout) or multivariate correlation breaks.",
        references=(
            _r(
                "Jaeger, H. (2001). The 'echo state' approach to analysing and training recurrent neural networks. GMD Report 148, German National Research Center for Information Technology."
            ),
            _r(
                "Lukoševičius, M. & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training. Computer Science Review 3(3), 127-149.",
                "https://doi.org/10.1016/j.cosrev.2009.03.005",
            ),
        ),
    ),
    DetectorProfile(
        name="spiking",
        discipline="Computational neuroscience (spiking neural networks)",
        lineage="Leaky integrate-and-fire neuron populations (Gerstner & Kistler, 2002) used as a rate-coded novelty detector.",
        mechanism="A seeded population of LIF neurons with heterogeneous gains and thresholds integrates the normalised series; fit records the baseline population spike-rate distribution and detect scores each point by the robust deviation of the local spike rate from that baseline.",
        practitioners="Neuromorphic-computing researchers and edge-sensing engineers; reading it needs the membrane time constant and threshold spread.",
        use_when="Exploring neuromorphic-style novelty detection on normalised streams and burst rates.",
        not_for="Calibrated probabilities; its sweep mean on real streams was below the registry bar.",
        references=(
            _r(
                "Gerstner, W. & Kistler, W.M. (2002). Spiking Neuron Models: Single Neurons, Populations, Plasticity. Cambridge University Press.",
                "https://doi.org/10.1017/CBO9780511815706",
            ),
            _r(
                "Lapicque, L. (1907). Recherches quantitatives sur l'excitation électrique des nerfs traitée comme une polarisation. Journal de Physiologie et de Pathologie Générale 9, 620-635."
            ),
        ),
    ),
    DetectorProfile(
        name="digital_twin",
        discipline="Industrial systems engineering and system identification",
        lineage="Digital-twin divergence monitoring (Grieves & Vickers, 2017) with an auto-regressive forward model identified by least squares.",
        mechanism="fit identifies an AR forward model (the twin) from training data; detect combines the one-step-ahead residual with the divergence of a short free-running simulation from the observed trajectory, squashed into [0, 1].",
        practitioners="Plant, asset and reliability engineers; reading it needs the AR order and the simulation horizon.",
        use_when="Monitoring a physical process against its expected dynamics where open-loop divergence is diagnostic.",
        not_for="Processes with no stable linear dynamics to identify.",
        references=(
            _r(
                "Grieves, M. & Vickers, J. (2017). Digital twin: mitigating unpredictable, undesirable emergent behavior in complex systems. In Transdisciplinary Perspectives on Complex Systems, Springer, 85-113.",
                "https://doi.org/10.1007/978-3-319-38756-7_4",
            ),
            _r(
                "Ljung, L. (1999). System Identification: Theory for the User (2nd ed.). Prentice Hall."
            ),
        ),
    ),
    DetectorProfile(
        name="survival",
        discipline="Survival analysis (biostatistics, reliability)",
        lineage="Kaplan-Meier non-parametric survival (1958) with a Cox proportional-hazards deviation (1972) on inter-event durations.",
        mechanism="Treats consecutive-sample gaps as durations, fits the empirical Kaplan-Meier survival curve and a Cox hazard-ratio slope on a recent-rate covariate, and scores each duration by -log of its two-sided tail probability squashed into [0, 1].",
        practitioners="Reliability engineers, biostatisticians and SREs reading inter-arrival times; reading it needs to know the input is a duration stream (or absolute differences of a signal).",
        use_when="Heartbeat, request-gap or retry-interval streams where an improbably long stall or short burst is the anomaly.",
        not_for="Value-level point outliers; declared not detected, and point-wise protocols under-measure it.",
        references=(
            _r(
                "Kaplan, E.L. & Meier, P. (1958). Nonparametric estimation from incomplete observations. Journal of the American Statistical Association 53(282), 457-481.",
                "https://doi.org/10.1080/01621459.1958.10501452",
            ),
            _r(
                "Cox, D.R. (1972). Regression models and life-tables. Journal of the Royal Statistical Society B 34(2), 187-202.",
                "https://doi.org/10.1111/j.2517-6161.1972.tb00899.x",
            ),
        ),
    ),
    DetectorProfile(
        name="energy_based",
        discipline="Energy-based modelling and score matching",
        lineage="Energy-based models (LeCun et al., 2006) fitted by score matching (Hyvärinen, 2005) on a delay embedding with a quadratic Gaussian-family energy.",
        mechanism="Delay-embeds and standardises the series, estimates the embedding mean and precision (the score-matching solution for a quadratic energy), and scores each point by its free energy 0.5 (x-mu)^T Lambda (x-mu) squashed into [0, 1].",
        practitioners="Machine-learning researchers and engineers who want an explicit density-style score without a partition function; reading it needs the embedding dimension.",
        use_when="Point and contextual outliers in a 1-D stream judged against the fitted delay-embedding density.",
        not_for="Multi-modal normal regimes, which a single quadratic energy cannot represent.",
        references=(
            _r(
                "LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M. & Huang, F.J. (2006). A tutorial on energy-based learning. In Predicting Structured Data, MIT Press."
            ),
            _r(
                "Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research 6, 695-709.",
                "https://jmlr.org/papers/v6/hyvarinen05a.html",
            ),
        ),
    ),
    DetectorProfile(
        name="deep_svdd",
        discipline="One-class learning",
        lineage="Support Vector Data Description (Tax & Duin, 2004), the one-class principle behind Deep SVDD (Ruff et al., ICML 2018), realised on a fixed random tanh feature lift instead of a trained network.",
        mechanism="Lifts a delay embedding through a seeded random tanh projection, sets the hypersphere centre to the training feature mean, calibrates the radius, and scores each point by its distance to the centre squashed into [0, 1].",
        practitioners="Anomaly-detection practitioners familiar with one-class SVMs; reading it needs to know no deep network is trained here, so the embedding is fixed, not learned.",
        use_when="Novelty detection on streams when PyTorch is unavailable and a one-class boundary is wanted.",
        not_for="Data whose normal class is multi-modal; a single hypersphere under-fits it.",
        references=(
            _r(
                "Tax, D.M.J. & Duin, R.P.W. (2004). Support vector data description. Machine Learning 54(1), 45-66.",
                "https://doi.org/10.1023/B:MACH.0000008084.60811.49",
            ),
            _r(
                "Ruff, L. et al. (2018). Deep one-class classification. ICML 2018, PMLR 80, 4393-4402.",
                "https://proceedings.mlr.press/v80/ruff18a.html",
            ),
        ),
    ),
    DetectorProfile(
        name="rca",
        discipline="Site-reliability engineering and causal graph analysis",
        lineage="Random-walk root-cause localisation over a service dependency graph, the MonitorRank (Kim et al., 2013) and CloudRanger (Wang et al., 2018) family.",
        mechanism="Learns per-node baselines, converts an observation into per-node standardised residuals, scores the row by the calibrated peak residual, and ranks root causes by a reverse personalised random walk over the supplied adjacency seeded from the residual vector.",
        practitioners="SREs and incident commanders; reading it needs the adjacency (who influences whom) and to treat the ranking as a hypothesis list, not a verdict.",
        use_when="Many correlated signals go anomalous at once and the question is where it started.",
        not_for="Single-signal detection; without a graph it degenerates to a per-node residual.",
        references=(
            _r(
                "Kim, M., Sumbaly, R. & Shah, S. (2013). Root cause detection in a service-oriented architecture. ACM SIGMETRICS 2013, 93-104.",
                "https://doi.org/10.1145/2465529.2465753",
            ),
            _r(
                "Wang, P. et al. (2018). CloudRanger: root cause identification for cloud native systems. IEEE/ACM CCGRID 2018, 492-502.",
                "https://doi.org/10.1109/CCGRID.2018.00076",
            ),
        ),
    ),
    DetectorProfile(
        name="deeplog_sequence",
        discipline="Log analytics and sequence modelling",
        lineage="DeepLog (Du et al., CCS 2017) detection principle - next-key prediction over log-template sequences - realised with a back-off n-gram model instead of an LSTM.",
        mechanism="Accumulates n-gram transition counts with back-off over integer log-template keys, scores each position by the next-key miss probability 1 - P(key | context), and decides by the paper's top-g criterion (the observed key outside the g most likely successors).",
        practitioners="Security and platform engineers working with parsed logs (Drain templates); reading it needs to know 1 - P is a ranking score and a 0.5 cut over-flags on grammars with several equiprobable successors.",
        use_when="Execution-path anomalies in system or application logs after template parsing.",
        not_for="Numeric point outliers or drift; declared not detected.",
        references=(
            _r(
                "Du, M., Li, F., Zheng, G. & Srikumar, V. (2017). DeepLog: anomaly detection and diagnosis from system logs through deep learning. ACM CCS 2017, 1285-1298.",
                "https://doi.org/10.1145/3133956.3134015",
            ),
            _r(
                "He, P., Zhu, J., Zheng, Z. & Lyu, M.R. (2017). Drain: an online log parsing approach with fixed depth tree. IEEE ICWS 2017, 33-40.",
                "https://doi.org/10.1109/ICWS.2017.13",
            ),
        ),
    ),
    DetectorProfile(
        name="frequent_pattern",
        discipline="Association-rule mining",
        lineage="Apriori frequent itemsets (Agrawal & Srikant, 1994) and FP-outlier rule-violation scoring (He et al., 2005).",
        mechanism="Mines frequent itemsets and single-consequent rules above support and confidence floors from binary training transactions, and scores each transaction by its confidence-weighted violation mass (antecedent present, consequent absent) squashed into [0, 1].",
        practitioners="Fraud, retail and configuration-audit analysts who think in co-occurrence rules; reading it needs the support and confidence floors.",
        use_when="Binary or categorical transaction records where a missing expected co-occurrence is the anomaly.",
        not_for="Per-point series scores; it emits per-transaction verdicts.",
        references=(
            _r(
                "Agrawal, R. & Srikant, R. (1994). Fast algorithms for mining association rules. VLDB 1994, 487-499."
            ),
            _r(
                "He, Z., Xu, X., Huang, J.Z. & Deng, S. (2005). FP-outlier: frequent pattern based outlier detection. Computer Science and Information Systems 2(1), 103-118.",
                "https://doi.org/10.2298/CSIS0501103H",
            ),
        ),
    ),
    DetectorProfile(
        name="srcnn",
        discipline="Spectral signal processing with a learned discriminator",
        lineage="SR-CNN (Ren et al., KDD 2019): a 1-D CNN trained on synthetic-anomaly-augmented Spectral Residual saliency.",
        mechanism="Injects synthetic point anomalies into a copy of the training series, recomputes SR saliency, trains a 1-D CNN with a positive-class weight to label the injected points from their saliency window, and scores new points by the CNN's sigmoid output.",
        practitioners="ML engineers in monitoring platforms with PyTorch available; reading it needs to know the BCE positive weight shifts the sigmoid so 0.5 is not the calibrated midpoint.",
        use_when="Point anomalies on streams where a learned saliency-to-anomaly mapping beats SR's heuristic threshold.",
        not_for="Deployments without torch, or drift/level-shift detection.",
        references=(
            _r(
                "Ren, H. et al. (2019). Time-series anomaly detection service at Microsoft. ACM KDD 2019, 3009-3017.",
                "https://doi.org/10.1145/3292500.3330680",
            ),
        ),
    ),
    DetectorProfile(
        name="diffusion_ad",
        discipline="Generative modelling (denoising diffusion)",
        lineage="Denoising Diffusion Probabilistic Models (Ho, Jain & Abbeel, 2020) used for reconstruction-error anomaly detection, as in AnoDDPM (Wyatt et al., 2022), adapted to 1-D windows.",
        mechanism="Trains a compact conditional denoiser on standardised normal windows; at detection, scores each point by the average noise-prediction error of its centred window across several diffusion timesteps, squashed to [0, 1].",
        practitioners="ML researchers and engineers with torch; reading it needs the window length and the timestep set used for scoring.",
        use_when="Off-manifold windows in a stream where a generative model of normal shape is appropriate.",
        not_for="CPU-only or latency-critical paths; and anomalies that are in-manifold but mistimed.",
        references=(
            _r(
                "Ho, J., Jain, A. & Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS 2020.",
                "https://arxiv.org/abs/2006.11239",
            ),
            _r(
                "Wyatt, J., Leach, A., Schmon, S.M. & Willcocks, C.G. (2022). AnoDDPM: anomaly detection with denoising diffusion probabilistic models using simplex noise. IEEE CVPR Workshops 2022.",
                "https://doi.org/10.1109/CVPRW56347.2022.00080",
            ),
        ),
    ),
    DetectorProfile(
        name="nano_safeguard",
        discipline="Dimensionality reduction and stability monitoring (Mercury-native N term)",
        lineage="The N term of Mercury's Lyapunov-style protection framework: hierarchical micro-pattern scanning by PCA downsampling.",
        mechanism="Projects a window to a very low PCA dimension and back, scores the normalised reconstruction residual ||N(x)|| / ||x|| at several scales, and raises an alert when the convergence measure falls below threshold; integrity hashes guard the inputs.",
        practitioners="Safeguard and integrity engineers operating the engine; reading it needs to know the score is window-level, not per point.",
        use_when="Micro-scale pattern departures a point detector would average away, as an engine safeguard.",
        not_for="Per-point anomaly scoring or benchmarks that need a per-sample contract.",
        references=(
            _r("Mercury Agent, PROTECTION_OVERVIEW.md (Nano-Safeguards, N term specification)."),
            _r(
                "Jolliffe, I.T. (2002). Principal Component Analysis (2nd ed.). Springer.",
                "https://doi.org/10.1007/b98835",
            ),
        ),
    ),
    DetectorProfile(
        name="extended_statistical",
        discipline="Robust statistics",
        lineage="An ensemble of classical heads: MAD, Local Outlier Factor, DBSCAN, Minimum Covariance Determinant, CUSUM, Grubbs' test and the generalised ESD test.",
        mechanism="Runs each statistical head on the fitted series or table, combines them by voting or weighted combination with dynamic threshold adaptation, and reports the per-head verdicts alongside the ensemble score.",
        practitioners="Statisticians and analysts who want named, textbook tests behind a flag; reading it needs the per-head parameters (contamination, support fraction, CUSUM drift).",
        use_when="A robust, explainable multi-test second opinion on point, collective and change-point anomalies.",
        not_for="Very large tables (LOF and MCD are quadratic in places).",
        references=(
            _r(
                "Rosner, B. (1983). Percentage points for a generalized ESD many-outlier procedure. Technometrics 25(2), 165-172.",
                "https://doi.org/10.1080/00401706.1983.10487848",
            ),
            _r(
                "Rousseeuw, P.J. & Van Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator. Technometrics 41(3), 212-223.",
                "https://doi.org/10.1080/00401706.1999.10485670",
            ),
            _r(
                "Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. KDD 1996, 226-231."
            ),
        ),
    ),
    DetectorProfile(
        name="adaptive_anomaly",
        discipline="Automated model selection for anomaly detection",
        lineage="Mercury's profile-calibrated router, built from benchmark analysis of where single detectors failed, routing to temporal, covariance or high-dimensional back-ends.",
        mechanism="Profiles the dataset (dimensionality, temporal structure, covariance conditioning), selects and calibrates the matching back-end, and returns its scores with the chosen strategy recorded.",
        practitioners="Analysts who do not want to pick a detector by hand; reading it needs the recorded profile and chosen strategy to know what actually scored the data.",
        use_when="Tabular data of unknown shape where the right detector should be chosen from the data's profile.",
        not_for="Reproducible comparisons across datasets, because the back-end changes with the profile.",
        references=(
            _r("Mercury Agent, docs/BENCHMARKS.md (the weaknesses the router was built against)."),
            _r(
                "Chandola, V., Banerjee, A. & Kumar, V. (2009). Anomaly detection: a survey. ACM Computing Surveys 41(3), 1-58.",
                "https://doi.org/10.1145/1541880.1541882",
            ),
        ),
    ),
    DetectorProfile(
        name="topological",
        discipline="Topological data analysis",
        lineage="Persistent homology (Edelsbrunner & Harer, 2010; Carlsson, 2009) with persistence landscapes (Bubenik, 2015) and Wasserstein distances between diagrams.",
        mechanism="Builds a Vietoris-Rips filtration on reference sub-samples, computes 0- and 1-dimensional persistence, and scores new batches by Betti-number deviation, persistence entropy, landscape norm and Wasserstein distance from the reference diagram.",
        practitioners="Applied topologists and researchers on shape-of-data problems; reading it needs the filtration radius and the knowledge that the score is batch-level.",
        use_when="Collective or structural anomalies where the shape (components, loops) of a point cloud changes.",
        not_for="Per-point scores or tight time budgets; it exceeded the sweep's 90 s budget on every tabular set.",
        references=(
            _r(
                "Edelsbrunner, H. & Harer, J. (2010). Computational Topology: An Introduction. American Mathematical Society."
            ),
            _r(
                "Carlsson, G. (2009). Topology and data. Bulletin of the American Mathematical Society 46(2), 255-308.",
                "https://doi.org/10.1090/S0273-0979-09-01249-X",
            ),
            _r(
                "Bubenik, P. (2015). Statistical topological data analysis using persistence landscapes. Journal of Machine Learning Research 16, 77-102.",
                "https://jmlr.org/papers/v16/bubenik15a.html",
            ),
        ),
    ),
    DetectorProfile(
        name="multivariate_ts",
        discipline="Multivariate time-series statistics (LTG-shaped baseline)",
        lineage="A deterministic baseline shaped like the LTG method (LSTM, temporal convolution, graph convolution; Springer 2025) with per-window mean, standard deviation and feature-correlation branches and no learned parameters.",
        mechanism="Slices windows_3d input, computes per-window mean (the 'LSTM' branch), per-window standard deviation (the 'temporal' branch) and a feature-correlation summary (the 'graph' branch), and scores by reconstruction error against the fitted statistics.",
        practitioners="Engineers working with multichannel sensor windows; reading it needs to know the branch names are shape-aliases for statistics, not trained networks.",
        use_when="Windowed multivariate data where correlation breaks between channels are the anomaly.",
        not_for="Anyone expecting a trained LTG network; see lstm_autoencoder for the trained multivariate path.",
        references=(
            _r(
                "LTG: multivariate time-series anomaly detection combining LSTM, temporal convolution and graph convolution (Springer, 2025).",
                "https://link.springer.com/article/10.1007/s44443-025-00024-3",
            ),
        ),
    ),
    DetectorProfile(
        name="info_geometry",
        discipline="Information geometry",
        lineage="IGEOOD (Gomes et al., ICLR 2022) Fisher-Rao out-of-distribution scoring with Fisher-information adaptive thresholds.",
        mechanism="Fits a reference on normal data, computes Fisher-Rao (geodesic) distance of each row's statistics from the reference on the statistical manifold, and sets or recalibrates the threshold from the Fisher information metric on drift.",
        practitioners="ML researchers and reliability engineers deploying OOD detectors; reading it needs the reference distribution and whether the adaptive threshold is on.",
        use_when="Out-of-distribution rows judged by a metric that respects the geometry of the fitted model.",
        not_for="Temporal structure; it reads rows as samples from a distribution.",
        references=(
            _r(
                "Gomes, E.D.C., Alberge, F., Duhamel, P. & Piantanida, P. (2022). IGEOOD: an information geometry approach to out-of-distribution detection. ICLR 2022.",
                "https://openreview.net/pdf?id=mfwdY3U_9ea",
            ),
            _r(
                "Amari, S. (2016). Information Geometry and Its Applications. Springer.",
                "https://doi.org/10.1007/978-4-431-55978-8",
            ),
        ),
    ),
    # ---------------------------------------------------------------- model
    DetectorProfile(
        name="quantum",
        discipline="Quantum-inspired feature engineering with fitted-reference scoring",
        lineage="A quantum-state feature embedding (amplitude, phase and entanglement statistics) retained from Mercury's early models, re-scored since 2026-10-27 as a calibrated deviation from a reference fitted on normal data.",
        mechanism="Maps each row to quantum-inspired features, fits the FittedReferenceScorer on normal data and scores the standardised deviation of the features from that reference; the old fixed-direction heuristics (measured below chance) are gone.",
        practitioners="Research engineers exploring quantum-inspired representations; reading it needs the knowledge that its feature maps are scale-invariant and therefore blind to amplitude collapse.",
        use_when="Tabular novelty where a non-linear phase/amplitude embedding is an interesting second view.",
        not_for="Low-side anomalies (amplitude collapse), pinned as undetectable in tests/models/test_fitted_reference_models.py.",
        references=(
            _r(
                "Schuld, M. & Petruccione, F. (2018). Supervised Learning with Quantum Computers. Springer.",
                "https://doi.org/10.1007/978-3-319-96424-9",
            ),
            _r(
                "Mercury Agent, tests/models/test_fitted_reference_models.py (fitted-reference pins)."
            ),
        ),
    ),
    DetectorProfile(
        name="astrophysical",
        discipline="Physics-metaphor feature engineering with fitted-reference scoring",
        lineage="A black-hole-physics feature mapping (event-horizon distance, energy and curvature analogues) from Mercury's early models, re-scored since 2026-10-27 as deviation from a fitted normal reference after the original heuristics measured AUROC 0.000 on a planted burst.",
        mechanism="Maps rows to the astrophysical feature set, fits the FittedReferenceScorer on normal data and scores calibrated deviation, which rises for any departure, high- or low-energy, by construction.",
        practitioners="Research engineers; reading it needs the history: the representation is a metaphor, the scoring is ordinary fitted-reference statistics.",
        use_when="Tabular novelty detection where its feature mapping is wanted as an additional voter.",
        not_for="Interpretation in physical terms; the mapping is not a physical model of the data.",
        references=(
            _r(
                "Mercury Agent, tests/models/test_fitted_reference_models.py (the 12-sigma planted-burst pin)."
            ),
            _r(
                "Chandola, V., Banerjee, A. & Kumar, V. (2009). Anomaly detection: a survey. ACM Computing Surveys 41(3), 1-58.",
                "https://doi.org/10.1145/1541880.1541882",
            ),
        ),
    ),
    DetectorProfile(
        name="biometric",
        discipline="Biometrics and face analysis",
        lineage="Mercury's biometric anomaly model: window-level statistics over biometric feature streams, in the tradition of biometric performance evaluation (Jain, Ross & Prabhakar, 2004).",
        mechanism="Aggregates a biometric feature series over windows and scores distributional departure from the reference window; a planted burst separates at the window level, but there is no per-point contract.",
        practitioners="Biometric-system engineers and forensic analysts; reading it needs to know the score is per window and no biometric-domain corpus has been evaluated.",
        use_when="Window-level screening of biometric telemetry (template quality, match-score drift) for gross departures.",
        not_for="Identity decisions or per-sample alarms; the FINDΩYOU bridge handles evidence packets separately.",
        references=(
            _r(
                "Jain, A.K., Ross, A. & Prabhakar, S. (2004). An introduction to biometric recognition. IEEE Transactions on Circuits and Systems for Video Technology 14(1), 4-20.",
                "https://doi.org/10.1109/TCSVT.2003.818349",
            ),
            _r(
                "ISO/IEC 19795-1:2021, Information technology - Biometric performance testing and reporting - Part 1."
            ),
        ),
    ),
    DetectorProfile(
        name="affective",
        discipline="Affective computing",
        lineage="Picard's affective-computing programme (1997) and the circumplex model of affect (Russell, 1980), implemented as a deterministic distress pipeline over declared emotion-probability series.",
        mechanism="For declared affective input (an emotion-probability time series under the documented key) it aggregates emotions per sample, computes entropy and negative-affect distress and scores distress; for anything else it emits a neutral 0.5 and a one-time notice, never fabricating an affective reading.",
        practitioners="Human-factors and mental-health technology researchers; reading it needs the emotion vocabulary and the knowledge that generic input yields the neutral prior.",
        use_when="Declared emotion-distribution streams where sustained negative affect or high-entropy states are the concern.",
        not_for="Point or drift detection on numeric series; declared not detected, and undeclared input is neutral by design.",
        references=(
            _r("Picard, R.W. (1997). Affective Computing. MIT Press."),
            _r(
                "Russell, J.A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology 39(6), 1161-1178.",
                "https://doi.org/10.1037/h0077714",
            ),
        ),
    ),
    DetectorProfile(
        name="neural_cognitive",
        discipline="Clinical neurophysiology (EEG analysis)",
        lineage="Window-level EEG-style feature scoring in the tradition of clinical EEG quantification; the ontology notes CHB-MIT as the intended labeled corpus.",
        mechanism="Computes band-power and variability features over windows of a brain-activity series and scores departure from the reference window; a planted burst separates (0.999) at window level only.",
        practitioners="Clinical neurophysiologists and BCI engineers; reading it needs the montage and sampling rate, and the fact that no EEG-domain evaluation is committed.",
        use_when="Window-level screening of EEG-like series for gross activity changes.",
        not_for="Seizure detection claims; there is no per-point contract and no committed clinical evaluation.",
        references=(
            _r(
                "Schomer, D.L. & Lopes da Silva, F.H. (eds.) (2011). Niedermeyer's Electroencephalography: Basic Principles, Clinical Applications, and Related Fields (6th ed.). Lippincott Williams & Wilkins."
            ),
            _r(
                "Shoeb, A.H. (2009). Application of machine learning to epileptic seizure onset detection and treatment. PhD thesis, MIT (the CHB-MIT Scalp EEG Database).",
                "https://physionet.org/content/chbmit/",
            ),
        ),
    ),
    DetectorProfile(
        name="consciousness",
        discipline="Pattern-coherence feature engineering with fitted-reference scoring",
        lineage="Fourier pattern-coherence and eigen-entropy features from Mercury's early models, re-scored since 2026-10-27 as deviation from a fitted normal reference after the constant-threshold rule separated nothing on a planted burst.",
        mechanism="Encodes rows as coherence and entanglement-style features, fits the FittedReferenceScorer on normal data and scores calibrated deviation; like quantum, the scale-invariant encoding cannot see amplitude collapse.",
        practitioners="Research engineers; reading it needs the fitted-reference history and the declared blind spot.",
        use_when="Tabular novelty detection as an additional coherence-based voter.",
        not_for="Amplitude collapse (declared and pinned) or any claim about consciousness; the name is historical.",
        references=(
            _r(
                "Mercury Agent, tests/models/test_fitted_reference_models.py (coherence encoding, fitted reference)."
            ),
            _r(
                "Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory (2nd ed.). Wiley (entropy of eigen-spectra)."
            ),
        ),
    ),
    DetectorProfile(
        name="neurosymbolic",
        discipline="Neuro-symbolic AI (rules over neural confidence)",
        lineage="A symbolic-rules plus ethical-rules reference surface descended from Logic Tensor Networks (Badreddine et al., 2022); the never-trained random-init network was removed on 2026-06-02 and the canonical co-trained path is ml.symbolic_constraint.",
        mechanism="Applies knowledge-base rules and ethical constraints to a deterministic statistical confidence over the series, records a reasoning chain and returns a rule-grounded verdict with human-readable explanations.",
        practitioners="Explainability and governance engineers; reading it needs to know this is the legacy reference surface, while SymbolicConstraintModule inside fit_fusion is the trained one.",
        use_when="An auditable rules-plus-confidence verdict with an explanation chain is wanted (POST /api/v1/detect/neurosymbolic).",
        not_for="Benchmark comparisons against trained detectors; it is a cognitive-stack component, graded at engine level.",
        references=(
            _r(
                "Badreddine, S., d'Avila Garcez, A., Serafini, L. & Spranger, M. (2022). Logic Tensor Networks. Artificial Intelligence 303, 103649.",
                "https://doi.org/10.1016/j.artint.2021.103649",
            ),
            _r("Mercury Agent, docs/NEUROSYMBOLIC.md."),
        ),
    ),
    DetectorProfile(
        name="chemistry",
        discipline="Chemistry and nuclear chemistry",
        lineage="Periodic-law relationships (Mendeleev, 1869) and nuclide stability data applied to composition, isotope and reaction-kinetics records.",
        mechanism="Parses chemical composition records, scores elements and isotopes against periodic-table relationships and decay-chain expectations, and flags reaction-rate deviations and structural implausibilities; series input yields no elements and a constant score.",
        practitioners="Analytical and radiochemists, laboratory QA scientists; reading it needs the composition record format and the periodic/nuclide references it checks against.",
        use_when="Composition, isotope-ratio or kinetics records where a physically implausible value is the anomaly.",
        not_for="Numeric time series (constant output) or point/drift detection, both declared not detected.",
        references=(
            _r(
                "Mendeleev, D. (1869). Über die Beziehungen der Eigenschaften zu den Atomgewichten der Elemente. Zeitschrift für Chemie 12, 405-406."
            ),
            _r(
                "National Nuclear Data Center (Brookhaven). NuDat 3 nuclear structure and decay data.",
                "https://www.nndc.bnl.gov/nudat3/",
            ),
            _r(
                "IUPAC Periodic Table of the Elements.",
                "https://iupac.org/what-we-do/periodic-table-of-elements/",
            ),
        ),
    ),
    DetectorProfile(
        name="parapsychology",
        discipline="Statistics of random-event-generator experiments (parapsychology-research instrumentation)",
        lineage="REG deviation statistics as used by the PEAR laboratory (Jahn & Dunne) and the Global Consciousness Project (Nelson et al.), with Mercury's preregistered analysis plan.",
        mechanism="Measures deviation from an exactly known chance baseline (e.g. Binomial(200, 0.5) egg trials) with z-scores, chi-square variance tails and effect sizes over trial records; the neural path stays at the neutral prior until a provenance-logged checkpoint is loaded.",
        practitioners="Experimental parapsychologists and sceptical statisticians alike; reading it needs the null distribution of the instrument and the discipline to separate measurement from interpretation.",
        use_when="Hardware-RNG trial streams where the question is deviation from chance (which can equally be a hardware fault).",
        not_for="Causal claims about consciousness, point or drift detection on ordinary series; the detector reports a deviation, never a cause.",
        references=(
            _r(
                "Nelson, R.D., Radin, D.I., Shoup, R. & Bancel, P.A. (2002). Correlations of continuous random data with major world events. Foundations of Physics Letters 15(6), 537-550.",
                "https://doi.org/10.1023/A:1023983708767",
            ),
            _r(
                "Jahn, R.G. & Dunne, B.J. (1987). Margins of Reality: The Role of Consciousness in the Physical World. Harcourt Brace Jovanovich."
            ),
            _r("Mercury Agent, docs/PARAPSYCH_PREREGISTRATION.md."),
        ),
    ),
    DetectorProfile(
        name="copod",
        discipline="Copula-based multivariate statistics",
        lineage="COPOD (Li et al., ICDM 2020): parameter-free outlier detection from empirical copula tail probabilities.",
        mechanism="Computes per-feature left- and right-tail empirical CDFs, combines the tail probabilities through the empirical copula and scores each row by its joint tail improbability in linear time, with interpretable per-feature contributions.",
        practitioners="Data scientists on tabular problems who want no hyperparameters and per-feature explanations; reading it needs only the knowledge that extreme values in any feature tail raise the score.",
        use_when="Tabular outliers with interpretable per-feature attribution and no tuning budget.",
        not_for="Temporal or contextual anomalies; rows are exchangeable to it.",
        references=(
            _r(
                "Li, Z., Zhao, Y., Botta, N., Ionescu, C. & Hu, X. (2020). COPOD: copula-based outlier detection. IEEE ICDM 2020, 1118-1123.",
                "https://doi.org/10.1109/ICDM50108.2020.00135",
            ),
        ),
    ),
    DetectorProfile(
        name="adversarial_autoencoder",
        discipline="Industrial control-system security (deep generative modelling)",
        lineage="Adversarial autoencoders (Makhzani et al., 2015) and MAD-GAN (Li et al., 2019) applied to multivariate ICS sensor streams such as BATADAL and SWaT.",
        mechanism="Trains an autoencoder with a latent discriminator for distribution matching and covariance-aware encoding over sensor correlations, with multi-scale reconstruction and temporal consistency terms; the anomaly score is the reconstruction error.",
        practitioners="OT/ICS security engineers and process-control data scientists; reading it needs the training window of normal operation and the input dimension (e.g. 51 BATADAL channels).",
        use_when="Multivariate industrial sensor data where attacks manipulate correlated readings.",
        not_for="Small or univariate data; and claims of a mean score until the budgeted re-grade recorded in the ontology is done.",
        references=(
            _r(
                "Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I. & Frey, B. (2015). Adversarial autoencoders. arXiv:1511.05644.",
                "https://arxiv.org/abs/1511.05644",
            ),
            _r(
                "Li, D., Chen, D., Jin, B., Shi, L., Goh, J. & Ng, S.-K. (2019). MAD-GAN: multivariate anomaly detection for time series data with generative adversarial networks. ICANN 2019, LNCS 11730, 703-716.",
                "https://doi.org/10.1007/978-3-030-30490-4_56",
            ),
            _r(
                "Taormina, R. et al. (2018). Battle of the Attack Detection Algorithms. Journal of Water Resources Planning and Management 144(8).",
                "https://doi.org/10.1061/(ASCE)WR.1943-5452.0000969",
            ),
        ),
    ),
    DetectorProfile(
        name="lstm_autoencoder",
        discipline="Deep sequence modelling",
        lineage="LSTM encoder-decoder reconstruction for multi-sensor anomaly detection (Malhotra et al., 2016) on Hochreiter & Schmidhuber's LSTM (1997); the registry's only trained multivariate sequence-reconstruction detector.",
        mechanism="Trains an LSTM autoencoder on sliding windows with a bottleneck and early stopping on a held-out tail, deterministic by seed, and scores each window by reconstruction error; measured AUROC 0.85 on injected multivariate bursts, shifts and frequency changes (0.94 on frequency changes).",
        practitioners="ML engineers with torch and multichannel telemetry; reading it needs the window length and the knowledge that broken cross-channel correlations with plausible individual shapes are not seen (0.47).",
        use_when="Multivariate series where shape-level reconstruction of windows captures the anomaly.",
        not_for="Correlation breaks between channels or CPU-only latency-critical deployments.",
        references=(
            _r(
                "Malhotra, P. et al. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. ICML 2016 Anomaly Detection Workshop, arXiv:1607.00148.",
                "https://arxiv.org/abs/1607.00148",
            ),
            _r(
                "Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. Neural Computation 9(8), 1735-1780.",
                "https://doi.org/10.1162/neco.1997.9.8.1735",
            ),
        ),
    ),
    # ------------------------------------------------------------- security
    DetectorProfile(
        name="realtime_threat",
        discipline="Security analytics (unsupervised ensemble)",
        lineage="A tree-free random-projection isolation detector (the isolation principle of Liu, Ting & Zhou, 2008), a KD-tree local-density estimator and a robust Mahalanobis distance (Rousseeuw & Van Driessen, 1999), combined scale-invariantly.",
        mechanism="Each sub-detector's training scores are robustly standardised (median, 1.4826 MAD) before averaging so no sub-detector dominates by units; the threshold and per-sample probability are calibrated against the training distribution's empirical CDF, so a single sample can be judged.",
        practitioners="SOC detection engineers and platform security teams; reading it needs the training window and the pinned false-alarm rate (tests/core/test_declared_modality_false_alarm_rate.py).",
        use_when="Real-time scoring of feature vectors from telemetry where a calibrated per-sample probability is required.",
        not_for="Sequence or text payload threats; see threat_detection and deeplog_sequence.",
        references=(
            _r(
                "Liu, F.T., Ting, K.M. & Zhou, Z.-H. (2008). Isolation forest. IEEE ICDM 2008, 413-422.",
                "https://doi.org/10.1109/ICDM.2008.17",
            ),
            _r(
                "Rousseeuw, P.J. & Van Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator. Technometrics 41(3), 212-223.",
                "https://doi.org/10.1080/00401706.1999.10485670",
            ),
        ),
    ),
    DetectorProfile(
        name="threat_detection",
        discipline="Application security (injection detection)",
        lineage="Pattern-based screening for SQL injection, cross-site scripting and path traversal in request text, the CWE-89/79/22 classes the OWASP Core Rule Set targets, served through the payload adapter.",
        mechanism="Grades each payload with the linear-time pattern families of ThreatDetector.detect_all; the decision is any category matching and the score is the highest category's fraction of patterns matched, with payloads over the size cap graded on their prefix and marked partial.",
        practitioners="Application-security engineers and WAF operators; reading it needs the knowledge that a pattern match is a screen, and that evasion (encoding, obfuscation) is its known weakness.",
        use_when="Screening request bodies, query strings or log lines for classic injection payloads.",
        not_for="Numeric telemetry, or as the sole control against a determined attacker.",
        references=(
            _r(
                "OWASP Foundation. OWASP Top Ten (A03:2021 Injection).",
                "https://owasp.org/Top10/A03_2021-Injection/",
            ),
            _r(
                "MITRE. CWE-89 SQL Injection, CWE-79 Cross-site Scripting, CWE-22 Path Traversal.",
                "https://cwe.mitre.org/",
            ),
            _r("OWASP ModSecurity Core Rule Set.", "https://coreruleset.org/"),
        ),
    ),
    # --------------------------------------------------------- intelligence
    DetectorProfile(
        name="psyop",
        discipline="Information operations analysis (defensive)",
        lineage="Target-audience, narrative and influence-campaign analysis framed on US Army MISO doctrine (FM 3-53) and joint information-operations doctrine (JP 3-13), implemented as a simulation-grounded research tool.",
        mechanism="Scores narrative content and engagement metrics for coordinated-amplification, cognitive-vulnerability and campaign-progression indicators over a message series and reports campaign-detection and audience-analysis structures.",
        practitioners="Counter-disinformation analysts and information-environment researchers under legal authorisation; reading it needs doctrine vocabulary and the knowledge that the module is simulation-based and defensive.",
        use_when="Research on influence-campaign detection over message and engagement streams.",
        not_for="Operational deployment without legal authorisation and oversight, or any numeric telemetry problem.",
        references=(
            _r(
                "U.S. Department of the Army (2013). FM 3-53, Military Information Support Operations."
            ),
            _r("U.S. Joint Chiefs of Staff (2014). JP 3-13, Information Operations."),
        ),
    ),
    DetectorProfile(
        name="intelligence_fusion",
        discipline="All-source intelligence analysis and multisensor data fusion",
        lineage="The JDL data-fusion model (Steinberg, Bowman & White, 1999) and joint intelligence doctrine (JP 2-0) applied to multi-INT reports (OSINT, SIGINT, GEOINT, HUMINT and others).",
        mechanism="Weights and correlates reports from several intelligence disciplines into a unified threat assessment with source-independence checks, temporal progression modelling and neurosymbolic correlation, emitting fused confidence and threat scores per record.",
        practitioners="Intelligence analysts and fusion-centre staff; reading it needs source reliability grading and the independence structure of the reports (derived_from).",
        use_when="Fusing heterogeneous reports where corroboration across independent sources is the evidence.",
        not_for="Single-source numeric data; it has no committed labeled evaluation.",
        references=(
            _r(
                "Steinberg, A.N., Bowman, C.L. & White, F.E. (1999). Revisions to the JDL data fusion model. Proc. SPIE 3719, Sensor Fusion: Architectures, Algorithms, and Applications III.",
                "https://doi.org/10.1117/12.341367",
            ),
            _r(
                "Hall, D.L. & Llinas, J. (1997). An introduction to multisensor data fusion. Proceedings of the IEEE 85(1), 6-23.",
                "https://doi.org/10.1109/5.554205",
            ),
            _r("U.S. Joint Chiefs of Staff (2013). JP 2-0, Joint Intelligence."),
        ),
    ),
    # ---------------------------------------------------------------- space
    DetectorProfile(
        name="schumann_resonance",
        discipline="Atmospheric electricity and ELF geophysics",
        lineage="Spectral monitoring of the Earth-ionosphere cavity resonances predicted by Schumann (1952), with per-harmonic amplitude and frequency tracking.",
        mechanism="FFTs the ELF series, locates the fundamental (~7.83 Hz) and the harmonics near 14, 20, 26 and 33 Hz, and scores amplitude and frequency deviation of each against a baseline with contextual and variance-change sensitivity.",
        practitioners="ELF observatory scientists and atmospheric-electricity researchers; reading it needs the sampling rate and the knowledge that any geophysical-precursor reading is a preregistered hypothesis.",
        use_when="Operator-supplied ELF field series where harmonic amplitude or frequency departures are the question.",
        not_for="Claims of earthquake prediction; stability tests exist, no labeled corpus does.",
        references=(
            _r(
                "Schumann, W.O. (1952). Über die strahlungslosen Eigenschwingungen einer leitenden Kugel. Zeitschrift für Naturforschung A 7(2), 149-154.",
                "https://doi.org/10.1515/zna-1952-0202",
            ),
            _r(
                "Nickolaenko, A.P. & Hayakawa, M. (2002). Resonances in the Earth-Ionosphere Cavity. Kluwer."
            ),
        ),
    ),
    DetectorProfile(
        name="solar_storm",
        discipline="Space-weather forecasting",
        lineage="Snapshot grading of solar wind, IMF, X-ray flux and proton flux on NOAA's G, R and S scales, with a geomagnetic model trained on OMNI2 (Kp MAE 0.575 on 25,846 held-out hours, 2022-2024).",
        mechanism="Each row (solar-wind speed, Bz and optional IMF, density, temperature, pressure, X-ray and proton flux) is graded by SolarStormDetector.predict_solar_storm on the three NOAA scales; the score is the highest level over 5 and the decision is any scale at level 1 or above.",
        practitioners="Space-weather forecasters and satellite/grid operators; reading it needs the NOAA scales and which inputs were unmeasured (the geomagnetic network fills its own training medians).",
        use_when="Grading L1 solar-wind snapshots for geomagnetic, radio-blackout and radiation-storm potential.",
        not_for="Ground-level dB/dt or GIC estimates; see gic and gic_minute_series.",
        references=(
            _r(
                "NOAA Space Weather Prediction Center. NOAA Space Weather Scales.",
                "https://www.swpc.noaa.gov/noaa-scales-explanation",
            ),
            _r(
                "King, J.H. & Papitashvili, N.E. (2005). Solar wind spatial scales in and comparisons of hourly Wind and ACE plasma and magnetic field data. Journal of Geophysical Research 110, A02104.",
                "https://doi.org/10.1029/2004JA010649",
            ),
        ),
    ),
    DetectorProfile(
        name="solar_flare",
        discipline="Solar physics (X-ray flare classification)",
        lineage="The GOES X-ray flare classification (A, B, C, M, X at 1e-8 to 1e-4 W/m^2) with geomagnetic context taken only from an observed planetary Kp.",
        mechanism="Classifies the 0.1-0.8 nm X-ray flux against the physical GOES thresholds and, when an observed Kp is injected or fetched, attaches geomagnetic context; storm quantities are never fabricated offline.",
        practitioners="Space-weather forecasters and HF-radio operators; reading it needs the flare-class letter scale and the knowledge that only observed Kp is reported.",
        use_when="Flare classification from GOES flux series, as the live-wired canonical flare detector.",
        not_for="Geomagnetic storm prediction without observed Kp; and it is covered by xray_flare through the numeric seams.",
        references=(
            _r(
                "NOAA SWPC. Solar flares (radio blackouts) and the GOES X-ray flare classification.",
                "https://www.swpc.noaa.gov/phenomena/solar-flares-radio-blackouts",
            ),
            _r(
                "NOAA NCEI. GOES X-ray Sensor (XRS) data documentation.",
                "https://www.ngdc.noaa.gov/stp/satellite/goes-r.html",
            ),
        ),
    ),
    DetectorProfile(
        name="cme_arrival",
        discipline="Heliospheric physics (CME propagation)",
        lineage="The Empirical Shock Arrival model (Gopalswamy et al., 2001) and the Drag-Based Model (Vršnak et al., 2013) as an ensemble over DONKI cone-model CME parameters.",
        mechanism="CMEArrivalDetector.predict: from each CME's speed, Stonyhurst latitude/longitude, half-angle and time at 21.5 solar radii, predicts Earth-directedness and arrival; the score is the directedness class on miss < unlikely < flank < head_on over 3, and confidence (model agreement scaled by geometry) is a feature, not the score.",
        practitioners="Space-weather forecasters; reading it needs cone-model geometry and the distinction between directedness and severity (a fast head-on CME can have low confidence).",
        use_when="Screening DONKI CME analyses for Earth-directed events and estimating arrival.",
        not_for="Storm intensity at Earth; arrival is not severity.",
        references=(
            _r(
                "Gopalswamy, N., Lara, A., Yashiro, S., Kaiser, M.L. & Howard, R.A. (2001). Predicting the 1-AU arrival times of coronal mass ejections. Journal of Geophysical Research 106(A12), 29207-29217.",
                "https://doi.org/10.1029/2001JA000177",
            ),
            _r(
                "Vršnak, B. et al. (2013). Propagation of interplanetary coronal mass ejections: the drag-based model. Solar Physics 285, 295-315.",
                "https://doi.org/10.1007/s11207-012-0035-4",
            ),
            _r(
                "NASA CCMC Space Weather Database Of Notifications, Knowledge, Information (DONKI).",
                "https://kauai.ccmc.gsfc.nasa.gov/DONKI/",
            ),
        ),
    ),
    DetectorProfile(
        name="sep_storm",
        discipline="Space radiation (solar energetic particles)",
        lineage="NOAA's proton-event definition (three consecutive 5-minute samples at or above 10 pfu at >= 10 MeV) and the S-scale radiation-storm levels from GOES integral proton channels.",
        mechanism="SEPStormDetector.assess: over a window of epoch, >= 10 MeV and >= 100 MeV fluxes, detects event onset by the three-sample rule and grades the NOAA S level from the peak; the score is S over 5, so a lone spike scores S1 but is not an onset.",
        practitioners="Space-weather forecasters, airline and satellite radiation-risk officers; reading it needs the pfu unit and the S-scale.",
        use_when="GOES proton-flux windows where radiation-storm onset and level are the question.",
        not_for="Geomagnetic or radio effects; those are G and R.",
        references=(
            _r(
                "NOAA SWPC. Solar radiation storms and the NOAA S-scale.",
                "https://www.swpc.noaa.gov/phenomena/solar-radiation-storm",
            ),
            _r(
                "Reames, D.V. (1999). Particle acceleration at the Sun and in the heliosphere. Space Science Reviews 90, 413-491.",
                "https://doi.org/10.1023/A:1005105831781",
            ),
        ),
    ),
    DetectorProfile(
        name="ionospheric_scintillation",
        discipline="Ionospheric physics and GNSS engineering",
        lineage="The S4 amplitude and sigma-phi phase scintillation indices (Van Dierendonck, Klobuchar & Hua, 1993) computed from raw receiver samples over a 60 s window.",
        mechanism="IonosphericScintillationDetector.measure: from intensity and/or carrier-phase samples at a known rate computes S4 and sigma-phi, classes degradation weak / moderate (>= 0.3) / strong (> 0.6); the score is the class over 2 and the decision is at least moderate.",
        practitioners="GNSS engineers, aviation navigation specialists and ionospheric scientists; reading it needs the sampling rate and the index thresholds.",
        use_when="Raw GNSS receiver samples where positioning degradation from scintillation is the concern.",
        not_for="Geomagnetic indices; scintillation is a propagation effect.",
        references=(
            _r(
                "Van Dierendonck, A.J., Klobuchar, J. & Hua, Q. (1993). Ionospheric scintillation monitoring using commercial single frequency C/A code receivers. Proc. ION GPS-93, 1333-1342."
            ),
            _r(
                "Kintner, P.M., Ledvina, B.M. & de Paula, E.R. (2007). GPS and ionospheric scintillations. Space Weather 5(9).",
                "https://doi.org/10.1029/2006SW000260",
            ),
        ),
    ),
    DetectorProfile(
        name="gic",
        discipline="Geomagnetically induced currents (power-grid geophysics)",
        lineage="dB/dt tiers from Pulkkinen et al. (2013) and the Oak Ridge Meta-R-321 study, with a plane-wave geoelectric-field proxy, over a real-timestamped magnetometer window.",
        mechanism="GICDetector.assess: from epoch, Bx and By (nT) with real gaps kept as gaps, computes the horizontal time derivative and a plane-wave geoelectric estimate for the configured conductivity, grades the tier (moderate >= 90, high >= 300, severe >= 480 nT/min); the score is the tier over 3 and the decision is at least moderate.",
        practitioners="Transmission planners and GMD analysts, geomagnetic observatory scientists; reading it needs the observatory latitude and conductivity model.",
        use_when="Magnetometer windows with real sample times where grid GIC risk is the question.",
        not_for="Fixed-cadence series without timestamps; see gic_minute_series.",
        references=(
            _r(
                "Pulkkinen, A. et al. (2013). Generation of 100-year geomagnetically induced current scenarios. Space Weather 11(4), 182-191.",
                "https://doi.org/10.1002/swe.20021",
            ),
            _r(
                "Kappenman, J. (2010). Geomagnetic Storms and Their Impacts on the U.S. Power Grid. Oak Ridge National Laboratory, Meta-R-321."
            ),
            _r(
                "Bolduc, L. (2002). GIC observations and studies in the Hydro-Québec power system. Journal of Atmospheric and Solar-Terrestrial Physics 64(16), 1793-1802.",
                "https://doi.org/10.1016/S1364-6826(02)00128-1",
            ),
        ),
    ),
    DetectorProfile(
        name="solar_gic_cascade",
        discipline="Space-weather cascade analysis (Sun to grid)",
        lineage="A staged escalation from DONKI flare and CME analyses through observed Kp to magnetometer dB/dt, composing the flare, CME, geomagnetic and GIC detectors.",
        mechanism="SolarGICCascadeDetector.evaluate takes, at one evaluation time, flares, CME analyses, a Kp series and magnetometer minute series, awards a stage by strict escalation quiet < watch < warning < storm_in_progress, and scores the stage over 3 with any stage above quiet as the decision.",
        practitioners="Grid operators' space-weather desks and forecasters; reading it needs the stage ladder and which evidence set each stage was awarded on.",
        use_when="Composing live DONKI, Kp and magnetometer evidence into one grid-impact posture.",
        not_for="Any single-signal question; use the component detectors.",
        references=(
            _r(
                "NOAA SWPC. NOAA Space Weather Scales (G scale).",
                "https://www.swpc.noaa.gov/noaa-scales-explanation",
            ),
            _r(
                "Pulkkinen, A. et al. (2013). Generation of 100-year geomagnetically induced current scenarios. Space Weather 11(4), 182-191.",
                "https://doi.org/10.1002/swe.20021",
            ),
        ),
    ),
    DetectorProfile(
        name="solar_wind_geomag",
        discipline="Solar-wind / magnetosphere coupling",
        lineage="Geomagnetic storm level from solar-wind speed and IMF Bz, in the coupling tradition of Burton, McPherron & Russell (1975), served on NOAA's G scale by the hazard adapter.",
        mechanism="For rows of (solar_wind_speed_km_s, bz_imf_nt) returns the SolarStormDetector's own geomagnetic storm level, including its trained operating point that raises a sub-G1 Kp estimate to G1 when the storm probability clears it; score is G over 5, decision G >= 1.",
        practitioners="Space-weather forecasters; reading it needs the knowledge that southward Bz with high speed is the driver and that G is derived from the detector, not from Kp alone.",
        use_when="Two-column solar-wind rows where a quick G-scale grade is wanted.",
        not_for="Snapshots with X-ray or proton context; use solar_storm.",
        references=(
            _r(
                "Burton, R.K., McPherron, R.L. & Russell, C.T. (1975). An empirical relationship between interplanetary conditions and Dst. Journal of Geophysical Research 80(31), 4204-4214.",
                "https://doi.org/10.1029/JA080i031p04204",
            ),
            _r(
                "Newell, P.T., Sotirelis, T., Liou, K., Meng, C.-I. & Rich, F.J. (2007). A nearly universal solar wind-magnetosphere coupling function inferred from 10 magnetospheric state variables. Journal of Geophysical Research 112, A01206.",
                "https://doi.org/10.1029/2006JA012015",
            ),
        ),
    ),
    DetectorProfile(
        name="xray_flare",
        discipline="Solar physics (radio-blackout grading)",
        lineage="GOES long-band X-ray flux graded on NOAA's R scale, served by the hazard adapter over flux rows.",
        mechanism="For each flux_long_wm2 row computes the NOAA R level from the flare class (R1 at M1); the score is R over 5 and the decision is R >= 1.",
        practitioners="HF-communication and aviation operators, forecasters; reading it needs the M/X class to R-level mapping.",
        use_when="Flux rows where the radio-blackout level is the question.",
        not_for="Geomagnetic context; see solar_flare for observed-Kp context.",
        references=(
            _r(
                "NOAA SWPC. Radio blackouts and the NOAA R-scale.",
                "https://www.swpc.noaa.gov/phenomena/solar-flares-radio-blackouts",
            ),
        ),
    ),
    DetectorProfile(
        name="gic_minute_series",
        discipline="Geomagnetically induced currents (fixed-cadence magnetometry)",
        lineage="The same Pulkkinen (2013) / Meta-R-321 / Bolduc (2002) dB/dt tiers as gic, applied to a fixed-cadence (Bx, By) minute series without timestamps, in the INTERMAGNET 1-minute data tradition.",
        mechanism="Treats each row as one consecutive minute (or the configured cadence), assesses the whole window with GICDetector.assess and repeats the window's tier score (low 0, moderate 1/3, high 2/3, severe 1) on every row; the decision is at least moderate (90 nT/min).",
        practitioners="GMD analysts working from observatory minute files; reading it needs the cadence and the fact that the assessment is window-level.",
        use_when="INTERMAGNET-style minute files where a quick window tier is wanted.",
        not_for="Series with gaps or irregular timestamps; use gic.",
        references=(
            _r(
                "INTERMAGNET Technical Reference Manual (1-minute definitive data).",
                "https://intermagnet.org/publications/im_manual.pdf",
            ),
            _r(
                "Pulkkinen, A. et al. (2013). Generation of 100-year geomagnetically induced current scenarios. Space Weather 11(4), 182-191.",
                "https://doi.org/10.1002/swe.20021",
            ),
        ),
    ),
    # -------------------------------------------------------------- medical
    DetectorProfile(
        name="medical_abms",
        discipline="Clinical medicine across the ABMS specialty boards",
        lineage="Specialty-specific anomaly patterns organised by the American Board of Medical Specialties' 24 member boards and their subspecialties, with cross-disciplinary consultation recommendations; a simulation-grounded integration module.",
        mechanism="Routes a clinical record to the matching specialty patterns, scores specialty-specific departures and emits consultation recommendations with a reasoning chain.",
        practitioners="Clinical informaticists and physicians across specialties; reading it needs the specialty taxonomy and the knowledge that the module is decision support without a committed clinical evaluation.",
        use_when="Exploring multi-specialty screening structure over clinical records.",
        not_for="Diagnosis, or any claim beyond decision support; nothing is claimed in the ontology.",
        references=(
            _r(
                "American Board of Medical Specialties. ABMS Guide to Medical Specialties.",
                "https://www.abms.org/member-boards/specialty-subspecialty-certificates/",
            ),
        ),
    ),
    DetectorProfile(
        name="sepsis",
        discipline="Critical-care medicine (Sepsis-3 screening)",
        lineage="The Sepsis-3 consensus (Singer et al., JAMA 2016): the qSOFA bedside screen (Seymour et al., 2016) and the SOFA organ-dysfunction score (Vincent et al., 1996), served through the record adapter.",
        mechanism="From vital signs (qSOFA) and laboratory values (SOFA) computes both instruments; the decision is qSOFA >= 2 or SOFA >= 2 from a zero baseline, and the score is x / (x + 1) with x = max(qSOFA, SOFA) / 2, exactly 1/2 where either instrument reaches 2.",
        practitioners="Intensivists, emergency physicians and sepsis-programme nurses; reading it needs the instruments' components and the knowledge that a flag is a screen for possible sepsis, not a diagnosis.",
        use_when="Hourly ICU or emergency records where a reproducible Sepsis-3 screen is wanted (PhysioNet 2019 rows map directly).",
        not_for="Temporal progression (the trained model ships untrained and abstains) or diagnosis.",
        references=(
            _r(
                "Singer, M. et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA 315(8), 801-810.",
                "https://doi.org/10.1001/jama.2016.0287",
            ),
            _r(
                "Seymour, C.W. et al. (2016). Assessment of clinical criteria for sepsis. JAMA 315(8), 762-774.",
                "https://doi.org/10.1001/jama.2016.0288",
            ),
            _r(
                "Vincent, J.-L. et al. (1996). The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure. Intensive Care Medicine 22(7), 707-710.",
                "https://doi.org/10.1007/BF01709751",
            ),
        ),
    ),
    DetectorProfile(
        name="case_surge",
        discipline="Infectious-disease surveillance",
        lineage="Week-on-week case-count comparison with a growth-implied effective reproduction number via the serial interval (Wallinga & Lipsitch, 2007), served over a window of daily counts.",
        mechanism="CaseSurgeDetector.detect_case_surge compares the mean of the latest 7 days with the 7 before over at least 14 gap-free days (an unreported day is left blank, and a window whose gap-free run is shorter is refused), flags a surge above 1.5x (an experimental ratio) and scores q / (q + 1.5) so the threshold sits at 1/2; growth rate and serial interval give R_e.",
        practitioners="Epidemiologists and surveillance officers; reading it needs the serial interval for the pathogen and the reporting-cadence caveats.",
        use_when="Daily case-count windows where a sustained acceleration is the question.",
        not_for="Sparse or weekly data (needs 14 usable days) or non-count series.",
        references=(
            _r(
                "Wallinga, J. & Lipsitch, M. (2007). How generation intervals shape the relationship between growth rates and reproductive numbers. Proceedings of the Royal Society B 274(1609), 599-604.",
                "https://doi.org/10.1098/rspb.2006.3754",
            ),
            _r(
                "Cori, A., Ferguson, N.M., Fraser, C. & Cauchemez, S. (2013). A new framework and software to estimate time-varying reproduction numbers during epidemics. American Journal of Epidemiology 178(9), 1505-1512.",
                "https://doi.org/10.1093/aje/kwt133",
            ),
        ),
    ),
    # ----------------------------------------------------------- geological
    DetectorProfile(
        name="avalanche",
        discipline="Snow science and avalanche forecasting",
        lineage="Snowpack stability from a layered profile with the SK38 skier-stability index (Föhn, 1987; Jamieson & Johnston, 1998) graded on the EAWS five-level danger scale.",
        mechanism="From layers (thickness, density, temperature, grain type), the weak-layer index, measured strength, ski penetration and 24 h weather, computes stability and a danger level; the decision is avalanche_likely (EAWS >= 3, considerable) and the score is (level - 1) / 4.",
        practitioners="Avalanche forecasters, ski-patrol and mountain-guide observers; reading it needs snow-profile observation practice and the EAWS scale.",
        use_when="Snow-pit profiles plus recent weather where the question is skier-triggered release potential.",
        not_for="Remote-sensing-only inputs; it needs a measured profile.",
        references=(
            _r(
                "Föhn, P.M.B. (1987). The stability index and various triggering mechanisms. IAHS Publication 162, 195-214."
            ),
            _r(
                "Jamieson, B. & Johnston, C.D. (1998). Refinements to the stability index for skier-triggered dry-slab avalanches. Annals of Glaciology 26, 296-302.",
                "https://doi.org/10.3189/1998AoG26-1-296-302",
            ),
            _r(
                "European Avalanche Warning Services. The European avalanche danger scale.",
                "https://www.avalanches.org/standards/avalanche-danger-scale/",
            ),
        ),
    ),
    DetectorProfile(
        name="rockfall",
        discipline="Engineering geology (rock-slope monitoring)",
        lineage="Fukuzono's inverse-velocity failure-time forecast (1985) in the materials-failure tradition of Voight (1988), with freeze-thaw, rainfall and microseismic triggers over a cliff section's monitoring channels.",
        mechanism="RockfallDetector.predict_rockfall: from rock temperature with wetness, rain, displacement with time and microseismic event times, evaluates trigger states and the inverse-velocity trend; the hazard ladder is low < moderate < high < critical, scored as level over 3 with any level above low as the decision.",
        practitioners="Rock-slope and tunnel geotechnical engineers, highway geologists; reading it needs the monitoring channel semantics and the inverse-velocity method's assumptions (accelerating creep).",
        use_when="Instrumented cliffs where accelerating displacement or freeze-thaw cycling signals imminent release.",
        not_for="Uninstrumented slopes or event catalogs.",
        references=(
            _r(
                "Fukuzono, T. (1985). A new method for predicting the failure time of a slope. Proc. 4th International Conference and Field Workshop on Landslides, Tokyo, 145-150."
            ),
            _r(
                "Voight, B. (1988). A method for prediction of volcanic eruptions. Nature 332, 125-130 (the materials failure forecast law).",
                "https://doi.org/10.1038/332125a0",
            ),
        ),
    ),
    DetectorProfile(
        name="subsidence",
        discipline="Geodesy and InSAR analysis",
        lineage="Persistent-scatterer InSAR time-series analysis (Ferretti, Prati & Rocca, 2001) applied to ground subsidence and sinkhole precursors (Galloway & Burbey, 2011).",
        mechanism="From a points x epochs line-of-sight displacement matrix (a missing epoch is left blank), epochs in years and optional coordinates and coherence, estimates velocity, acceleration and spatial localisation; the decision and score are the detector's own anomaly_detected and confidence, with sinkhole_precursor_detected and severity as features.",
        practitioners="Geodesists and InSAR analysts, groundwater and mining regulators; reading it needs line-of-sight geometry and coherence quality.",
        use_when="InSAR scenes over aquifers, mines or karst where accelerating or localised displacement is the concern.",
        not_for="Single-epoch data or levelling without a time series.",
        references=(
            _r(
                "Ferretti, A., Prati, C. & Rocca, F. (2001). Permanent scatterers in SAR interferometry. IEEE Transactions on Geoscience and Remote Sensing 39(1), 8-20.",
                "https://doi.org/10.1109/36.898661",
            ),
            _r(
                "Galloway, D.L. & Burbey, T.J. (2011). Regional land subsidence accompanying groundwater extraction. Hydrogeology Journal 19(8), 1459-1486.",
                "https://doi.org/10.1007/s10040-011-0775-5",
            ),
        ),
    ),
    DetectorProfile(
        name="eq_tsunami_cascade",
        discipline="Tsunami warning operations",
        lineage="PTWC-style screening of an earthquake sequence (magnitude, depth, offshore) escalating to a tsunami watch, confirmed against a DART or coastal gauge series with NOAA's tsunami detection algorithm (Mofjeld).",
        mechanism="EqTsunamiCascadeDetector processes events in arrival order, escalating quiet < evaluating < tsunami_watch < tsunami_threat; at watch, confirms against the water-level series when it is usable (otherwise sets it aside with a reason and marks the verdict partial); score is the stage over 3.",
        practitioners="Tsunami warning-centre duty scientists and coastal emergency managers; reading it needs the screening criteria and what 'partial' means for the water-level confirmation.",
        use_when="Replaying or live-following an earthquake sequence with gauge confirmation.",
        not_for="Inundation forecasting; the cascade screens and confirms, it does not model run-up.",
        references=(
            _r(
                "Mofjeld, H.O. (1997). Tsunami detection algorithm. NOAA Pacific Marine Environmental Laboratory.",
                "https://nctr.pmel.noaa.gov/tda_documentation.html",
            ),
            _r(
                "Titov, V.V. et al. (2005). Real-time tsunami forecasting: challenges and solutions. Natural Hazards 35(1), 41-58.",
                "https://doi.org/10.1007/s11069-004-2403-3",
            ),
        ),
    ),
    DetectorProfile(
        name="fire_debris_flow_cascade",
        discipline="Post-wildfire hydrology and geomorphology",
        lineage="The USGS post-fire debris-flow likelihood model M1 (Staley et al., 2017), Cannon's rainfall intensity-duration thresholds (2008) and Gartner's volume model (2014).",
        mechanism="FireDebrisFlowCascadeDetector.assess: from terrain, burn-severity (dNBR) and soil (KF) terms with rainfall per step and the primary duration, computes the M1 likelihood and I-D threshold exceedance, plus a Gartner volume from burned area and relief; the decision is the USGS p = 0.5 rule and the score is the likelihood.",
        practitioners="USGS and state geologists, burned-area emergency response (BAER) teams, county flood-control engineers; reading it needs the M1 inputs and the storm's duration-intensity.",
        use_when="Burned basins with a forecast or observed storm where debris-flow likelihood is the question.",
        not_for="Unburned basins or long after vegetation recovery.",
        references=(
            _r(
                "Staley, D.M. et al. (2017). Prediction of spatially explicit rainfall intensity-duration thresholds for post-fire debris-flow generation in the western United States. Geomorphology 278, 149-162.",
                "https://doi.org/10.1016/j.geomorph.2016.10.019",
            ),
            _r(
                "Cannon, S.H., Gartner, J.E., Wilson, R.C., Bowers, J.C. & Laber, J.L. (2008). Storm rainfall conditions for floods and debris flows from recently burned areas in southwestern Colorado and southern California. Geomorphology 96(3-4), 250-269.",
                "https://doi.org/10.1016/j.geomorph.2007.03.019",
            ),
            _r(
                "Gartner, J.E., Cannon, S.H. & Santi, P.M. (2014). Empirical models for predicting volumes of sediment deposited by debris flows and sediment-laden floods in the transverse ranges of southern California. Engineering Geology 176, 45-56.",
                "https://doi.org/10.1016/j.enggeo.2014.04.008",
            ),
        ),
    ),
    DetectorProfile(
        name="volcanic",
        discipline="Volcanology (eruption forecasting)",
        lineage="Multi-stream volcanic unrest assessment (seismicity, thermal, gas, InSAR, ELF, precursor rate) graded with the USGS alert-level ladder, in the forecasting tradition reviewed by Sparks (2003).",
        mechanism="VolcanicEruptionDetector.predict_eruption takes a snapshot of monitoring streams and/or the standardised volcano-seismic-v1 fused vector, computes an eruption-imminent decision and confidence from the trained checkpoint and physics rules, and reports the USGS alert level as a feature.",
        practitioners="Volcano-observatory scientists and aviation-ash advisory meteorologists; reading it needs the monitoring-stream semantics and the alert-level ladder.",
        use_when="Observatory snapshots where several unrest indicators must be weighed together.",
        not_for="Alert-level catalog data alone (the volcanic loader); it needs monitoring observables.",
        references=(
            _r(
                "Sparks, R.S.J. (2003). Forecasting volcanic eruptions. Earth and Planetary Science Letters 210(1-2), 1-15.",
                "https://doi.org/10.1016/S0012-821X(03)00124-9",
            ),
            _r(
                "Chouet, B.A. (1996). Long-period volcano seismicity: its source and use in eruption forecasting. Nature 380, 309-316.",
                "https://doi.org/10.1038/380309a0",
            ),
            _r(
                "Gardner, C.A. & Guffanti, M.C. (2006). U.S. Geological Survey's alert notification system for volcanic activity. USGS Fact Sheet 2006-3139.",
                "https://doi.org/10.3133/fs20063139",
            ),
        ),
    ),
    DetectorProfile(
        name="landslide",
        discipline="Engineering geology (slope-failure triggering)",
        lineage="Rainfall intensity-duration thresholds (Caine, 1980; Guzzetti et al., 2008), infiltration-driven triggering (Iverson, 2000) and seismic slope displacement (Newmark, 1965) over one site's monitoring row.",
        mechanism="LandslideDetector.predict_landslide: from rainfall (intensity, duration, antecedent 7-day), seismic (PGA, magnitude, distance), slope angle, soil saturation, displacement rate and snowmelt, with unmeasured channels left out rather than replaced, evaluates trigger states and a slope-failure probability; the decision requires a trigger and the probability clearing its bar.",
        practitioners="Geotechnical engineers, landslide hazard mappers and road-maintenance geologists; reading it needs the trigger semantics and the site's slope and soil context.",
        use_when="Instrumented or forecast-fed slope sites where a rain, seismic or snowmelt trigger is in play.",
        not_for="Catalog records without site measurements (see the landslide loader for that).",
        references=(
            _r(
                "Caine, N. (1980). The rainfall intensity-duration control of shallow landslides and debris flows. Geografiska Annaler A 62(1-2), 23-27.",
                "https://doi.org/10.1080/04353676.1980.11879996",
            ),
            _r(
                "Iverson, R.M. (2000). Landslide triggering by rain infiltration. Water Resources Research 36(7), 1897-1910.",
                "https://doi.org/10.1029/2000WR900090",
            ),
            _r(
                "Newmark, N.M. (1965). Effects of earthquakes on dams and embankments. Géotechnique 15(2), 139-160.",
                "https://doi.org/10.1680/geot.1965.15.2.139",
            ),
        ),
    ),
    DetectorProfile(
        name="wildfire",
        discipline="Fire detection from thermal imagery",
        lineage="Contextual active-fire detection on brightness-temperature imagery (Giglio et al., 2003; Schroeder et al., 2014) with optional weather and fire-history context, over a thermal scene record.",
        mechanism="From a brightness-temperature image (or a wildfire-firms-v1 3 x H x W raster) with pixel size and optional weather, detects active fire or forecasts next-day ignition; a scene without a thermal image is refused because the weather-only decision is constant. Score and decision are the detector's confidence and fire_detected.",
        practitioners="Remote-sensing fire analysts and incident meteorologists; reading it needs the raster's units (K) and the assessment field (active fire vs next-day forecast).",
        use_when="Thermal scenes where an active fire or imminent ignition is the question.",
        not_for="Weather-only inputs; refused by design.",
        references=(
            _r(
                "Giglio, L., Descloitres, J., Justice, C.O. & Kaufman, Y.J. (2003). An enhanced contextual fire detection algorithm for MODIS. Remote Sensing of Environment 87(2-3), 273-282.",
                "https://doi.org/10.1016/S0034-4257(03)00184-6",
            ),
            _r(
                "Schroeder, W., Oliva, P., Giglio, L. & Csiszar, I.A. (2014). The New VIIRS 375 m active fire detection data product. Remote Sensing of Environment 143, 85-96.",
                "https://doi.org/10.1016/j.rse.2013.12.008",
            ),
        ),
    ),
    DetectorProfile(
        name="tornado",
        discipline="Severe-storms meteorology (radar and environment)",
        lineage="Doppler mesocyclone detection (Brown, Lemon & Burgess, 1978; Stumpf et al., 1998), convective-environment parameters (Thompson et al., 2003) and FFT resonance analysis from Mercury's 3R mechanism.",
        mechanism="Analyses Doppler velocity sweeps for rotation signatures, instability and pressure-gradient series for tornadic environments, and applies multi-scale recursive feature extraction with FFT resonance; emits tornado likelihood with the contributing paths.",
        practitioners="Warning meteorologists and radar analysts; reading it needs radar velocity-couplet interpretation and the knowledge that this is research tooling, not an official warning product.",
        use_when="Doppler sweeps (sweeps x 64 gates) or environment series where rotation or tornadic potential is the question.",
        not_for="Replacing NWS warnings; defer to official products.",
        references=(
            _r(
                "Brown, R.A., Lemon, L.R. & Burgess, D.W. (1978). Tornado detection by pulsed Doppler radar. Monthly Weather Review 106(1), 29-38.",
                "https://doi.org/10.1175/1520-0493(1978)106<0029:TDBPDR>2.0.CO;2",
            ),
            _r(
                "Stumpf, G.J. et al. (1998). The National Severe Storms Laboratory Mesocyclone Detection Algorithm for the WSR-88D. Weather and Forecasting 13(2), 304-326.",
                "https://doi.org/10.1175/1520-0434(1998)013<0304:TNSSLM>2.0.CO;2",
            ),
        ),
    ),
    DetectorProfile(
        name="hurricane",
        discipline="Tropical meteorology (intensity and intensification)",
        lineage="Saffir-Simpson classification, the Atkinson & Holliday (1977) pressure-wind relationship, SST-driven potential intensity (Emanuel, 1986) and resonance amplification from Mercury's 3R mechanism.",
        mechanism="Tracks SST, wind patterns and barometric pressure, classifies on the Saffir-Simpson scale, amplifies storm signals in the frequency domain and reports intensification likelihood with cross-domain hooks to surge and flood.",
        practitioners="Hurricane specialists and coastal emergency managers; reading it needs SST and shear context and the knowledge that it is research tooling, not an NHC product.",
        use_when="SST, wind and pressure series where intensification or classification is the question.",
        not_for="Official forecasts; defer to NHC/JTWC.",
        references=(
            _r(
                "Atkinson, G.D. & Holliday, C.R. (1977). Tropical cyclone minimum sea level pressure/maximum sustained wind relationship for the western North Pacific. Monthly Weather Review 105(4), 421-427.",
                "https://doi.org/10.1175/1520-0493(1977)105<0421:TCMSLP>2.0.CO;2",
            ),
            _r(
                "Emanuel, K.A. (1986). An air-sea interaction theory for tropical cyclones. Part I: steady-state maintenance. Journal of the Atmospheric Sciences 43(6), 585-605.",
                "https://doi.org/10.1175/1520-0469(1986)043<0585:AASITF>2.0.CO;2",
            ),
        ),
    ),
    DetectorProfile(
        name="flood",
        discipline="Hydrology (flood prediction)",
        lineage="Precipitation accumulation, river-gauge monitoring and soil-saturation runoff (Horton infiltration; the NRCS curve-number method) with Mercury's refactoring-engine optimisation loop.",
        mechanism="Combines precipitation accumulation, gauge readings and a physics-based runoff coefficient from soil saturation, refines the prediction iteratively and reports flood likelihood with the contributing factors.",
        practitioners="River forecasters, floodplain managers and dam-safety engineers; reading it needs the basin's flood stage and soil state, and the knowledge that it is research tooling, not an NWS product.",
        use_when="Precipitation and gauge series where flood likelihood from multiple parameters is the question.",
        not_for="Official flood warnings; defer to NWS and USGS.",
        references=(
            _r(
                "Horton, R.E. (1933). The role of infiltration in the hydrologic cycle. Eos, Transactions American Geophysical Union 14(1), 446-460.",
                "https://doi.org/10.1029/TR014i001p00446",
            ),
            _r(
                "USDA Natural Resources Conservation Service (1986). Urban Hydrology for Small Watersheds, Technical Release 55 (curve-number runoff)."
            ),
        ),
    ),
    DetectorProfile(
        name="earthquake",
        discipline="Seismology (event detection)",
        lineage="STA/LTA triggering (Allen, 1978; Withers et al., 1998) and P/S-wave spectrogram analysis with scipy.signal, with USGS live-catalog wiring and a trained checkpoint carrying a ratified operating point.",
        mechanism="Computes the short-term/long-term average ratio and spectrogram features of a waveform window (one 5.5 s STA/LTA window at 100 Hz needs at least 550 samples), decides with the checkpoint's operating point on the learned path or the constructor threshold on the physics path, and reports identical confidence either way.",
        practitioners="Seismic-network analysts and early-warning engineers; reading it needs sampling rate, window lengths and the trigger ratio.",
        use_when="Waveform series where event onset (P/S arrivals) is the question.",
        not_for="Catalog data (see the earthquake loader) or magnitude estimation.",
        references=(
            _r(
                "Allen, R.V. (1978). Automatic earthquake recognition and timing from single traces. Bulletin of the Seismological Society of America 68(5), 1521-1532.",
                "https://doi.org/10.1785/BSSA0680051521",
            ),
            _r(
                "Withers, M. et al. (1998). A comparison of select trigger algorithms for automated global seismic phase and event detection. Bulletin of the Seismological Society of America 88(1), 95-106.",
                "https://doi.org/10.1785/BSSA0880010095",
            ),
        ),
    ),
    DetectorProfile(
        name="tsunami",
        discipline="Physical oceanography (deep-ocean tsunami detection)",
        lineage="Waveform-spectrum analysis of detided sea-level series in the DART tradition, with a tsunami_dart training path on NDBC bottom-pressure records labelled by NCEI HazEL arrivals, gated by the merit gate (the 2026-07 candidate was refused for false-alarm regression).",
        mechanism="Takes a detided 96-sample 15-minute window (24 h of water-column residual, metres), computes its FFT spectrum and resonance features, and scores tsunami-band energy and change points; the physics path stays in charge until a checkpoint passes the merit gate.",
        practitioners="Tsunami-warning scientists and ocean-observing engineers; reading it needs the detiding step and the 15-minute cadence contract.",
        use_when="Detided DART or coastal-gauge windows where an arrival signature is the question.",
        not_for="Raw tide-dominated series without detiding.",
        references=(
            _r(
                "González, F.I. et al. (2005). The NTHMP tsunameter network. Natural Hazards 35(1), 25-39.",
                "https://doi.org/10.1007/s11069-004-2402-4",
            ),
            _r(
                "Rabinovich, A.B. (1997). Spectral analysis of tsunami waves: separation of source and topography effects. Journal of Geophysical Research 102(C6), 12663-12676.",
                "https://doi.org/10.1029/97JC00479",
            ),
        ),
    ),
    DetectorProfile(
        name="meteor",
        discipline="Planetary defense (impact-risk assessment)",
        lineage="Bayesian fusion of optical and radar observations with NASA/JPL fireball, NeoWs close-approach and Sentry impact-risk sources, in the hazard framework of Chesley et al. (2002).",
        mechanism="Fuses optical and radar returns with Bayesian inference for detection and trajectory estimation and joins live JPL/NASA sources (dependency-injected clients) to grade threat; fully offline when no clients are injected and use_nasa_data is false.",
        practitioners="Planetary-defense analysts and meteor-network operators; reading it needs the Palermo/Torino scales and the provenance of each fused source.",
        use_when="Radar return series or combined observation sets where impact threat is the question.",
        not_for="Orbit determination; it grades threat, it does not compute orbits.",
        references=(
            _r(
                "Chesley, S.R., Chodas, P.W., Milani, A., Valsecchi, G.B. & Yeomans, D.K. (2002). Quantifying the risk posed by potential Earth impacts. Icarus 159(2), 423-432.",
                "https://doi.org/10.1006/icar.2002.6910",
            ),
            _r(
                "Brown, P., Spalding, R.E., ReVelle, D.O., Tagliaferri, E. & Worden, S.P. (2002). The flux of small near-Earth objects colliding with the Earth. Nature 420, 294-296.",
                "https://doi.org/10.1038/nature01238",
            ),
            _r(
                "NASA JPL CNEOS Sentry: Earth Impact Monitoring.",
                "https://cneos.jpl.nasa.gov/sentry/",
            ),
        ),
    ),
    DetectorProfile(
        name="tornado_conditions",
        discipline="Severe-storms meteorology (sounding parameters)",
        lineage="Convective-parameter climatology for tornadic supercells (Rasmussen & Blanchard, 1998; Thompson et al., 2003), served by the hazard adapter over CAPE, CIN, SRH, bulk shear and LCL rows.",
        mechanism="For rows of (cape_j_kg, cin_j_kg, srh_m2_s2, bulk_shear_kt, lcl_m) returns TornadoDetector.predict_tornado's atmospheric-path confidence and tornado_likely decision.",
        practitioners="Forecasters reading model soundings; reading it needs the parameter space (high CAPE and SRH, low LCL, strong deep shear).",
        use_when="Sounding-derived parameter rows for environmental tornado potential.",
        not_for="Radar velocity data; use tornado.",
        references=(
            _r(
                "Thompson, R.L., Edwards, R., Hart, J.A., Elmore, K.L. & Markowski, P. (2003). Close proximity soundings within supercell environments obtained from the Rapid Update Cycle. Weather and Forecasting 18(6), 1243-1261.",
                "https://doi.org/10.1175/1520-0434(2003)018<1243:CPSWSE>2.0.CO;2",
            ),
            _r(
                "Rasmussen, E.N. & Blanchard, D.O. (1998). A baseline climatology of sounding-derived supercell and tornado forecast parameters. Weather and Forecasting 13(4), 1148-1164.",
                "https://doi.org/10.1175/1520-0434(1998)013<1148:ABCOSD>2.0.CO;2",
            ),
        ),
    ),
    DetectorProfile(
        name="hurricane_pressure",
        discipline="Tropical meteorology (wind-pressure relationships)",
        lineage="The Atkinson & Holliday (1977) pressure-wind relationship and its modern refinements (Knaff & Zehr, 2007), served by the hazard adapter over pressure and SST rows.",
        mechanism="For rows of (central_pressure_mb, environmental_pressure_mb, sst_celsius) returns HurricaneDetector.predict_hurricane's confidence and cyclone_detected, with the pressure-wind estimate as a feature.",
        practitioners="Forecasters and reconnaissance data analysts; reading it needs the basin dependence of wind-pressure relationships.",
        use_when="Pressure and SST rows where intensity from pressure deficit is the question.",
        not_for="Track data (see the hurricane loader) or wind-field structure.",
        references=(
            _r(
                "Atkinson, G.D. & Holliday, C.R. (1977). Tropical cyclone minimum sea level pressure/maximum sustained wind relationship for the western North Pacific. Monthly Weather Review 105(4), 421-427.",
                "https://doi.org/10.1175/1520-0493(1977)105<0421:TCMSLP>2.0.CO;2",
            ),
            _r(
                "Knaff, J.A. & Zehr, R.M. (2007). Reexamination of tropical cyclone wind-pressure relationships. Weather and Forecasting 22(1), 71-88.",
                "https://doi.org/10.1175/WAF965.1",
            ),
        ),
    ),
    DetectorProfile(
        name="flood_precipitation",
        discipline="Hydrometeorology (flash-flood guidance)",
        lineage="Precipitation-depth flood screening in the Flash Flood Guidance tradition (Sweeney, 1992; Georgakakos, 2006), served by the hazard adapter over measured and forecast depth rows.",
        mechanism="For rows of 1-, 6- and 24-hour measured precipitation and 24-hour forecast (inches) returns FloodDetector.predict_flood's precipitation-path confidence and flood_likely decision.",
        practitioners="Forecasters and emergency managers reading QPE/QPF; reading it needs the basin's response time and that depth alone ignores antecedent soil state.",
        use_when="Precipitation depth rows for a quick flood-likelihood screen.",
        not_for="Gauge-based river flooding (see the flood loader) or basins with snowmelt dominance.",
        references=(
            _r(
                "Sweeney, T.L. (1992). Modernized areal flash flood guidance. NOAA Technical Memorandum NWS HYDRO 44."
            ),
            _r(
                "Georgakakos, K.P. (2006). Analytical results for operational flash flood guidance. Journal of Hydrology 317(1-2), 81-103.",
                "https://doi.org/10.1016/j.jhydrol.2005.05.009",
            ),
        ),
    ),
    # ------------------------------------------------------- meteorological
    DetectorProfile(
        name="heatwave",
        discipline="Climatology (heatwave metrics)",
        lineage="The calendar-day 90th-percentile definition (Perkins & Alexander, 2013) and the Excess Heat Factor (Nairn & Fawcett, 2015), computed untrained from a multi-year baseline.",
        mechanism="fit_baseline builds CTX90pct thresholds, T95 and the EHF 85th percentile from a daily baseline; detect_heatwaves marks runs of three or more days above the calendar-day threshold and grades severity by EHF tiers.",
        practitioners="Climatologists and heat-health planners; reading it needs the baseline period and the EHF severity tiers.",
        use_when="Daily temperature series with a baseline where heatwave days and severity are the question.",
        not_for="Sub-daily or humidity-driven heat stress (no dew point).",
        references=(
            _r(
                "Perkins, S.E. & Alexander, L.V. (2013). On the measurement of heat waves. Journal of Climate 26(13), 4500-4517.",
                "https://doi.org/10.1175/JCLI-D-12-00383.1",
            ),
            _r(
                "Nairn, J.R. & Fawcett, R.J.B. (2015). The excess heat factor: a metric for heatwave intensity and its use in classifying heatwave severity. International Journal of Environmental Research and Public Health 12(1), 227-253.",
                "https://doi.org/10.3390/ijerph120100227",
            ),
        ),
    ),
    DetectorProfile(
        name="atmospheric_river",
        discipline="Synoptic meteorology (moisture transport)",
        lineage="Integrated water vapour transport (Zhu & Newell, 1998) and the Ralph et al. (2019) atmospheric-river scale, computed from supplied profiles or a precomputed IVT series.",
        mechanism="Integrates q*u and q*v over supplied pressure levels with the trapezoidal rule to IVT (kg m^-1 s^-1), validates units and level order loudly, then classifies AR category by IVT magnitude and duration; profiles are never fabricated.",
        practitioners="West-coast and European forecasters, water managers; reading it needs IVT thresholds (250, 500, 750, 1000, 1250) and duration rules.",
        use_when="Vertical profiles or IVT series where AR strength and category are the question.",
        not_for="Surface-only data without IVT.",
        references=(
            _r(
                "Ralph, F.M. et al. (2019). A scale to characterize the strength and impacts of atmospheric rivers. Bulletin of the American Meteorological Society 100(2), 269-289.",
                "https://doi.org/10.1175/BAMS-D-18-0023.1",
            ),
            _r(
                "Zhu, Y. & Newell, R.E. (1998). A proposed algorithm for moisture fluxes from atmospheric rivers. Monthly Weather Review 126(3), 725-735.",
                "https://doi.org/10.1175/1520-0493(1998)126<0725:APAFMF>2.0.CO;2",
            ),
        ),
    ),
    DetectorProfile(
        name="lightning",
        discipline="Severe-storms meteorology (total lightning)",
        lineage="The 2-sigma lightning jump algorithm (Schultz, Petersen & Carey, 2009; Schultz et al., 2011) as a severe-weather precursor.",
        mechanism="Averages total flash rate over 2-minute bins, computes DFRDT and the standard deviation over the preceding 12 minutes, and declares a jump when DFRDT >= 2 sigma and the flash rate is at least 10 flashes/min.",
        practitioners="Warning meteorologists with total-lightning data (LMA, GLM); reading it needs the bin and activation thresholds.",
        use_when="Flash-rate series where a rapid increase foreshadows severe weather.",
        not_for="Cloud-to-ground-only networks with low detection efficiency for intracloud flashes.",
        references=(
            _r(
                "Schultz, C.J., Petersen, W.A. & Carey, L.D. (2009). Preliminary development and evaluation of lightning jump algorithms for the real-time detection of severe weather. Journal of Applied Meteorology and Climatology 48(12), 2543-2563.",
                "https://doi.org/10.1175/2009JAMC2237.1",
            ),
            _r(
                "Schultz, C.J., Petersen, W.A. & Carey, L.D. (2011). Lightning and severe weather: a comparison between total and cloud-to-ground lightning trends. Weather and Forecasting 26(5), 744-755.",
                "https://doi.org/10.1175/WAF-D-10-05026.1",
            ),
        ),
    ),
    DetectorProfile(
        name="hail",
        discipline="Severe-storms meteorology (hail environments)",
        lineage="The SPC Significant Hail Parameter (SHIP) with its component clamps, CAPE/shear tiers (Thompson et al., 2003) and NWS severe-thunderstorm criteria, computed untrained from sounding-derived values.",
        mechanism="assess() takes MUCAPE, mixing ratio, 700-500 hPa lapse rate, 500 hPa temperature, 0-6 km shear and freezing level, computes SHIP with the SPC corrections and returns significant-hail favourability and probability tiers.",
        practitioners="Severe-weather forecasters; reading it needs sounding analysis and the SHIP interpretation (above 1 favours significant hail).",
        use_when="Sounding-derived environment values where large-hail potential is the question.",
        not_for="Report data (see the hail loader) or radar hail algorithms (MESH).",
        references=(
            _r(
                "NOAA Storm Prediction Center. Significant Hail Parameter (SHIP) mesoanalysis help.",
                "https://www.spc.noaa.gov/exper/mesoanalysis/help/help_sigh.html",
            ),
            _r(
                "Thompson, R.L., Edwards, R., Hart, J.A., Elmore, K.L. & Markowski, P. (2003). Close proximity soundings within supercell environments obtained from the Rapid Update Cycle. Weather and Forecasting 18(6), 1243-1261.",
                "https://doi.org/10.1175/1520-0434(2003)018<1243:CPSWSE>2.0.CO;2",
            ),
        ),
    ),
    DetectorProfile(
        name="winter_storm",
        discipline="Winter-weather forecasting",
        lineage="Partial-thickness precipitation-type inference (Keeter & Cline, 1991), the Stull (2011) wet-bulb approximation and the Freezing Rain Accumulation Model (Sanders & Barjenbruch, 2016).",
        mechanism="Infers precipitation type from 1000-850 and 850-700 hPa thicknesses with the operational thresholds (1300, 1540/1560 m) or from surface wet-bulb within its validity range, and computes ice accretion with FRAM.",
        practitioners="Winter-weather forecasters and road/utility operators; reading it needs thermal-profile interpretation and FRAM's precipitation-rate dependence.",
        use_when="Thermal profiles or surface T/RH with precipitation where type and ice accretion are the question.",
        not_for="Snowfall accumulation modelling (no snow-ratio model).",
        references=(
            _r(
                "Keeter, K.K. & Cline, J.W. (1991). The objective use of observed and forecast thickness values to predict precipitation type in North Carolina. Weather and Forecasting 6(4), 456-469.",
                "https://doi.org/10.1175/1520-0434(1991)006<0456:TOUOOA>2.0.CO;2",
            ),
            _r(
                "Stull, R. (2011). Wet-bulb temperature from relative humidity and air temperature. Journal of Applied Meteorology and Climatology 50(11), 2267-2269.",
                "https://doi.org/10.1175/JAMC-D-11-0143.1",
            ),
            _r(
                "Sanders, K.J. & Barjenbruch, B.L. (2016). Analysis of ice-to-liquid ratios during freezing rain and the development of an ice accumulation model. Weather and Forecasting 31(4), 1041-1060.",
                "https://doi.org/10.1175/WAF-D-15-0118.1",
            ),
        ),
    ),
    DetectorProfile(
        name="derecho",
        discipline="Severe-storms meteorology (convective windstorms)",
        lineage="The derecho criteria of Johns & Hirt (1987) with the swath-dimension revision of Corfidi et al. (2016), evaluated over a supplied severe-wind report series.",
        mechanism="From reports (time, lat, lon, gust, damage rating) checks swath length >= 650 km, width >= 100 km, no more than 3 h between successive reports and the intensity criteria, and returns the criteria met with the swath geometry.",
        practitioners="Severe-weather climatologists and forecasters; reading it needs report-density awareness and the criteria list.",
        use_when="Post-event or live wind-report series where derecho classification is the question.",
        not_for="Single-station gust data.",
        references=(
            _r(
                "Johns, R.H. & Hirt, W.D. (1987). Derechos: widespread convectively induced windstorms. Weather and Forecasting 2(1), 32-49.",
                "https://doi.org/10.1175/1520-0434(1987)002<0032:DWCIW>2.0.CO;2",
            ),
            _r(
                "Corfidi, S.F., Coniglio, M.C., Cohen, A.E. & Mead, C.M. (2016). A proposed revision to the definition of 'derecho'. Bulletin of the American Meteorological Society 97(6), 935-949.",
                "https://doi.org/10.1175/BAMS-D-14-00254.1",
            ),
        ),
    ),
    DetectorProfile(
        name="dust_storm",
        discipline="Aeolian geomorphology and dust meteorology",
        lineage="WMO sand-and-dust-storm visibility definitions (WMO-No. 306) and friction-velocity emission potential after Fécan et al. (1999) and Shao & Lu (2000).",
        mechanism="Classifies visibility with a concurrent dust-raising wind (default 10 m/s at 10 m) into blowing dust, dust storm (< 1000 m) and severe dust storm (< 200 m), and computes emission potential from friction velocity against a moisture-corrected threshold.",
        practitioners="Aviation and desert-region forecasters, air-quality agencies; reading it needs the wind requirement that keeps fog from being mislabelled.",
        use_when="Visibility and wind observations or surface-layer parameters where dust events are the question.",
        not_for="Fine-particle health grading; see pm25_air_quality.",
        references=(
            _r(
                "World Meteorological Organization. Manual on Codes, WMO-No. 306 (present-weather sand and dust storm definitions).",
                "https://library.wmo.int/idurl/4/35713",
            ),
            _r(
                "Fécan, F., Marticorena, B. & Bergametti, G. (1999). Parametrization of the increase of the aeolian erosion threshold wind friction velocity due to soil moisture for arid and semi-arid areas. Annales Geophysicae 17(1), 149-157.",
                "https://doi.org/10.1007/s00585-999-0149-7",
            ),
            _r(
                "Shao, Y. & Lu, H. (2000). A simple expression for wind erosion threshold friction velocity. Journal of Geophysical Research 105(D17), 22437-22443.",
                "https://doi.org/10.1029/2000JD900304",
            ),
        ),
    ),
    DetectorProfile(
        name="surge_flood_cascade",
        discipline="Coastal hazards (compound flooding)",
        lineage="A hurricane-to-storm-surge-to-compound-flood cascade over NOAA CO-OPS water-level residuals and NWPS river gauges, in the compound-flooding framing of Wahl et al. (2015).",
        mechanism="SurgeFloodCascade: each record carries the hurricane detector's real result, CO-OPS observed and predicted water levels (the residual is the surge) and a river-gauge payload; the cascade awards a stage quiet 0 to compound flood 3, scored over 3, optionally persisting evidence across records in a live feed.",
        practitioners="Coastal emergency managers and NWS coastal-flood forecasters; reading it needs the surge-residual concept and the stage ladder.",
        use_when="Landfalling-storm evidence arriving piecemeal where surge and river flooding may compound.",
        not_for="Open-ocean waves or inland riverine floods alone.",
        references=(
            _r(
                "Wahl, T., Jain, S., Bender, J., Meyers, S.D. & Luther, M.E. (2015). Increasing risk of compound flooding from storm surge and rainfall for major US cities. Nature Climate Change 5, 1093-1097.",
                "https://doi.org/10.1038/nclimate2736",
            ),
            _r(
                "NOAA CO-OPS Tides & Currents water-level products.",
                "https://tidesandcurrents.noaa.gov/",
            ),
        ),
    ),
    DetectorProfile(
        name="pm25_air_quality",
        discipline="Air-quality regulation (EPA AQI)",
        lineage="The EPA Air Quality Index for PM2.5 under 40 CFR Part 58 Appendix G with the 2024 breakpoints, served over 24-hour mean rows.",
        mechanism="AirQualityDetector: for each pm25_24h_ug_m3 row computes the AQI category (good to hazardous); the score is the category over 5 and the decision is AQI >= 101 (unhealthy for sensitive groups).",
        practitioners="Air-quality agencies, public-health officers and wildfire-smoke forecasters; reading it needs the breakpoint table and that AQI is a 24-hour mean, not an instantaneous reading.",
        use_when="Monitor-day PM2.5 means where the regulatory category is the question.",
        not_for="Hourly NowCast values or other pollutants (ozone, NO2).",
        references=(
            _r(
                "40 CFR Part 58, Appendix G: Uniform Air Quality Index (AQI) and Daily Reporting.",
                "https://www.ecfr.gov/current/title-40/chapter-I/subchapter-C/part-58/appendix-Appendix%20G%20to%20Part%2058",
            ),
            _r(
                "U.S. EPA (2024). Technical Assistance Document for the Reporting of Daily Air Quality - the Air Quality Index (AQI).",
                "https://document.airnow.gov/technical-assistance-document-for-the-reporting-of-daily-air-quailty.pdf",
            ),
        ),
    ),
    DetectorProfile(
        name="drought",
        discipline="Hydroclimatology (drought indices)",
        lineage="The Standardized Precipitation Index (McKee et al., 1993) with the Thom (1958) gamma fit and the SPEI water-balance index (Vicente-Serrano et al., 2010) with Thornthwaite PET, mapped to US Drought Monitor categories.",
        mechanism="Aggregates monthly precipitation over the configured windows, fits a gamma distribution per calendar month with the zero-precipitation mixture, transforms to standard normal deviates, optionally computes SPEI from temperature and latitude, and maps the latest indices to USDM categories; works untrained and never fills missing input.",
        practitioners="State climatologists and drought-monitor authors, water managers; reading it needs the time-scale semantics and the category mapping.",
        use_when="Monthly precipitation (and temperature) series where drought onset, intensity and duration are the question.",
        not_for="Daily or sub-monthly data, or soil-moisture drought without precipitation.",
        references=(
            _r(
                "McKee, T.B., Doesken, N.J. & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. 8th Conference on Applied Climatology, 179-184."
            ),
            _r(
                "Vicente-Serrano, S.M., Beguería, S. & López-Moreno, J.I. (2010). A multiscalar drought index sensitive to global warming: the Standardized Precipitation Evapotranspiration Index. Journal of Climate 23(7), 1696-1718.",
                "https://doi.org/10.1175/2009JCLI2909.1",
            ),
        ),
    ),
    # ------------------------------------------------- economic / energy / marine / uav
    DetectorProfile(
        name="financial_crisis",
        discipline="Macroprudential economics (systemic-crisis assessment)",
        lineage="Multi-indicator crisis assessment in the early-warning tradition of Kaminsky & Reinhart (1999) and Reinhart & Rogoff (2009), over market, banking, network, currency and sector sections of one record.",
        mechanism="FinancialCrisisDetector.predict_financial_crisis grades each section (market, banking, systemic-network interconnection, currency, trading once fraud weights load, sector labels), counts crisis indicators and awards the severity ladder stable < stress < crisis < severe_crisis < systemic_collapse; the decision is two or more indicators and the score is the ladder over 4.",
        practitioners="Financial-stability analysts at central banks and supervisors, chief risk officers; reading it needs the indicator definitions and the knowledge that no labeled evaluation is committed.",
        use_when="Whole-record assessments where several market and banking sections are available at once.",
        not_for="Named market and banking rows alone; see financial_stress.",
        references=(
            _r(
                "Kaminsky, G.L. & Reinhart, C.M. (1999). The twin crises: the causes of banking and balance-of-payments problems. American Economic Review 89(3), 473-500.",
                "https://doi.org/10.1257/aer.89.3.473",
            ),
            _r(
                "Reinhart, C.M. & Rogoff, K.S. (2009). This Time Is Different: Eight Centuries of Financial Folly. Princeton University Press."
            ),
        ),
    ),
    DetectorProfile(
        name="financial_stress",
        discipline="Financial-stress measurement",
        lineage="Market and banking stress metrics in the tradition of the Kansas City Financial Stress Index (Hakkio & Keeton, 2009) and the Kaminsky-Lizondo-Reinhart signals approach, served by the hazard adapter over named metric rows.",
        mechanism="For rows of price_change_pct, vix, volume_vs_average, cds_spread_bps, default_rate_pct and liquidity_ratio returns FinancialCrisisDetector.predict_financial_crisis's confidence and crisis_imminent; the untrained fraud network stays unconsulted.",
        practitioners="Market-risk and treasury analysts; reading it needs the metric units and the knowledge that this is a threshold screen over six named series.",
        use_when="Six-column market and banking rows where a quick stress grade is wanted.",
        not_for="Currency or network-contagion sections; see financial_crisis.",
        references=(
            _r(
                "Hakkio, C.S. & Keeton, W.R. (2009). Financial stress: what is it, how can it be measured, and why does it matter? Federal Reserve Bank of Kansas City Economic Review 94(2), 5-50."
            ),
            _r(
                "Kaminsky, G., Lizondo, S. & Reinhart, C.M. (1998). Leading indicators of currency crises. IMF Staff Papers 45(1), 1-48.",
                "https://doi.org/10.2307/3867328",
            ),
        ),
    ),
    DetectorProfile(
        name="emp",
        discipline="Electromagnetic-pulse and geomagnetic-threat engineering",
        lineage="The HEMP E1/E2/E3 environment of IEC 61000-2-9 and the EMP Commission's grid assessments, with time-difference-of-arrival localisation over a sensor array.",
        mechanism="From an E1 waveform or summary values, E2 values, magnetometer dB/dt (E3 and grid impact), signature data and sensor-array timings, grades the threat ladder benign < anomalous < suspicious < threat < critical; the decision is the THREAT level (a detected EMP or confirmed intentional EMI) and emp_detected is a feature.",
        practitioners="Critical-infrastructure protection engineers and grid EMP/GMD planners; reading it needs the E1/E2/E3 distinction and the difference between EMP, IEMI and a geomagnetic storm.",
        use_when="Field-sensor and magnetometer records where an electromagnetic attack or storm must be told apart.",
        not_for="Routine space-weather grading; see gic and solar_storm.",
        references=(
            _r(
                "IEC 61000-2-9:1996. Electromagnetic compatibility (EMC) - Part 2: Environment - Section 9: Description of HEMP environment - Radiated disturbance."
            ),
            _r(
                "Foster, J.S. et al. (2008). Report of the Commission to Assess the Threat to the United States from Electromagnetic Pulse (EMP) Attack: Critical National Infrastructures."
            ),
        ),
    ),
    DetectorProfile(
        name="marine_biodiversity",
        discipline="Marine ecology (coral thermal stress and ocean chemistry)",
        lineage="NOAA Coral Reef Watch thermal-stress metrics (SST anomaly, Degree Heating Weeks; Liu et al., 2014) and ocean-acidification indicators (Hoegh-Guldberg et al., 2007) over indicator rows.",
        mechanism="MarineBiodiversityDetector.predict_biodiversity_threat: for rows of sst_anomaly_c and degree_heating_weeks (measured together or absent together), ph and temperature_anomaly_c, assesses the measured indicators and awards the health ladder healthy < stressed < critical; a row with nothing measured is refused, never graded healthy.",
        practitioners="Reef managers and marine ecologists; reading it needs the DHW bleaching thresholds (4 and 8 degree C-weeks).",
        use_when="Reef-site indicator rows where bleaching or acidification stress is the question.",
        not_for="Occurrence-record richness (see the marine loader).",
        references=(
            _r(
                "Liu, G. et al. (2014). Reef-scale thermal stress monitoring of coral ecosystems: new 5-km global products from NOAA Coral Reef Watch. Remote Sensing 6(11), 11579-11606.",
                "https://doi.org/10.3390/rs61111579",
            ),
            _r(
                "Hoegh-Guldberg, O. et al. (2007). Coral reefs under rapid climate change and ocean acidification. Science 318(5857), 1737-1742.",
                "https://doi.org/10.1126/science.1152509",
            ),
        ),
    ),
    DetectorProfile(
        name="drone",
        discipline="UAV flight safety and telemetry analysis",
        lineage="Mission-phase invariant rules (RADD) over the UAV state vector, the Mercury in-house ensemble on a rolling telemetry window, and DronLomaly-style structured-log analysis; evaluated on the ALFA fixed-wing fault dataset.",
        mechanism="Evaluates phase-specific invariants (battery, GPS satellites, altitude rate, distance to home), fits MercuryAnomalyDetector on the rolling telemetry window for ML scores, and extracts keyword features from sensor-log entries; the three channels are combined into a fault decision.",
        practitioners="UAV operators, flight-test and autopilot engineers; reading it needs the mission-phase model and the ALFA evaluation record (artifacts/alfa_uav.eval.json).",
        use_when="Autopilot telemetry streams where actuator or sensor faults must be caught in flight.",
        not_for="Airspace-level traffic anomalies; see the airspace loader and geo_movement.",
        references=(
            _r(
                "Keipour, A., Mousaei, M. & Scherer, S. (2021). ALFA: a dataset for UAV fault and anomaly detection. International Journal of Robotics Research 40(2-3), 515-520.",
                "https://doi.org/10.1177/0278364920966642",
            ),
            _r("Mercury Agent, artifacts/alfa_uav.eval.json (aviation_uav validation)."),
        ),
    ),
    # --------------------------------------------------------------- visual
    DetectorProfile(
        name="patchcore",
        discipline="Industrial visual inspection (deep feature memory banks)",
        lineage="PatchCore (Roth et al., CVPR 2022): locally aware patch features from a pretrained backbone with greedy coreset subsampling, benchmarked on MVTec AD.",
        mechanism="Extracts mid-level patch features from normal images, subsamples a coreset memory bank, and scores test patches by nearest-neighbour distance to the bank, giving an image score and an anomaly map.",
        practitioners="Machine-vision and quality engineers; reading it needs the backbone layer choice and the coreset ratio, and the image_batch contract.",
        use_when="Few-shot industrial defect detection with pixel-level localisation and no defect examples.",
        not_for="Video or non-visual data; validated on the visual_industrial fixtures only.",
        references=(
            _r(
                "Roth, K. et al. (2022). Towards total recall in industrial anomaly detection. IEEE/CVF CVPR 2022, 14318-14328.",
                "https://doi.org/10.1109/CVPR52688.2022.01392",
            ),
            _r(
                "Bergmann, P., Fauser, M., Sattlegger, D. & Steger, C. (2019). MVTec AD - a comprehensive real-world dataset for unsupervised anomaly detection. IEEE/CVF CVPR 2019, 9592-9600.",
                "https://doi.org/10.1109/CVPR.2019.00982",
            ),
        ),
    ),
    DetectorProfile(
        name="padim",
        discipline="Industrial visual inspection (patch distribution modelling)",
        lineage="PaDiM (Defard et al., 2021): a multivariate Gaussian per patch position over concatenated backbone features, scored by Mahalanobis distance.",
        mechanism="Fits a mean and covariance for every patch position from normal images' pretrained features (randomly reduced), and scores test patches by Mahalanobis distance, producing an anomaly map and image score.",
        practitioners="Machine-vision engineers; reading it needs aligned imagery (position-specific statistics assume registration).",
        use_when="Registered industrial images where per-position normality is a valid assumption.",
        not_for="Unaligned or rotating parts.",
        references=(
            _r(
                "Defard, T., Setkov, A., Loesch, A. & Audigier, R. (2021). PaDiM: a patch distribution modeling framework for anomaly detection and localization. ICPR 2020 Workshops, LNCS 12664, 475-489.",
                "https://doi.org/10.1007/978-3-030-68799-1_35",
            ),
        ),
    ),
    DetectorProfile(
        name="stfpm",
        discipline="Industrial visual inspection (knowledge distillation)",
        lineage="Student-Teacher Feature Pyramid Matching (Wang et al., BMVC 2021).",
        mechanism="Trains a student network to match a frozen pretrained teacher's multi-scale features on normal images; at test time the feature discrepancy across the pyramid is the anomaly map.",
        practitioners="Machine-vision engineers; reading it needs the pyramid levels used and training epochs.",
        use_when="Defect localisation where a trainable, compact student is preferred to a memory bank.",
        not_for="Domains where the teacher's pretraining features are irrelevant (non-natural imagery).",
        references=(
            _r(
                "Wang, G., Han, S., Ding, E. & Huang, D. (2021). Student-teacher feature pyramid matching for anomaly detection. BMVC 2021.",
                "https://arxiv.org/abs/2103.04257",
            ),
        ),
    ),
    DetectorProfile(
        name="reverse_distillation",
        discipline="Industrial visual inspection (reverse knowledge distillation)",
        lineage="Reverse Distillation (Deng & Li, CVPR 2022) with a one-class embedding (OCE) bottleneck.",
        mechanism="A frozen teacher encodes the image, an OCE bottleneck compresses it, and a student decoder reconstructs the teacher's multi-scale features; discrepancy between teacher and student features marks anomalies.",
        practitioners="Machine-vision engineers; reading it needs the bottleneck's role (it discards anomalous detail by design).",
        use_when="Defects that a forward student would copy from the teacher; the reversed flow suppresses them.",
        not_for="Real-time constraints with large decoders.",
        references=(
            _r(
                "Deng, H. & Li, X. (2022). Anomaly detection via reverse distillation from one-class embedding. IEEE/CVF CVPR 2022, 9737-9746.",
                "https://doi.org/10.1109/CVPR52688.2022.00951",
            ),
        ),
    ),
    DetectorProfile(
        name="cflow",
        discipline="Industrial visual inspection (normalizing flows)",
        lineage="CFLOW-AD (Gudovskiy, Ishizaka & Kozuka, WACV 2022): conditional normalizing flows over positional-encoded backbone features.",
        mechanism="Fits conditional normalizing-flow decoders on multi-scale features with positional conditioning so the likelihood of each patch is estimated; low likelihood is the anomaly map.",
        practitioners="Machine-vision engineers; reading it needs the knowledge that scores are likelihoods, so calibration is position-aware.",
        use_when="Precise localisation with a probabilistic score per patch.",
        not_for="Tiny training sets where flow fitting is unstable.",
        references=(
            _r(
                "Gudovskiy, D., Ishizaka, S. & Kozuka, K. (2022). CFLOW-AD: real-time unsupervised anomaly detection with localization via conditional normalizing flows. IEEE/CVF WACV 2022, 1819-1828.",
                "https://doi.org/10.1109/WACV51458.2022.00188",
            ),
        ),
    ),
    # ------------------------------------------------------------------ vlm
    DetectorProfile(
        name="anyanomaly",
        discipline="Video anomaly detection with vision-language models",
        lineage="AnyAnomaly (WACV 2026): zero-shot, customisable video anomaly detection with context-aware visual question answering on a large vision-language model backend.",
        mechanism="Builds context (scene, position, temporal) for each frame and asks the LVLM backend whether the user-described anomaly type is present, returning calibrated frame scores; requires an external LVLM backend, without which nothing is claimed.",
        practitioners="Surveillance and safety analysts who can describe the anomaly in words; reading it needs the backend's limits and the prompt wording.",
        use_when="Describing a novel anomaly type at run time without training data.",
        not_for="Deployments without an LVLM backend or with strict latency budgets.",
        references=(
            _r(
                "AnyAnomaly: zero-shot customizable video anomaly detection with LVLM. IEEE/CVF WACV 2026.",
                "https://arxiv.org/abs/2503.04504",
            ),
        ),
    ),
    DetectorProfile(
        name="lavad",
        discipline="Video anomaly detection with language models",
        lineage="LAVAD (Zanella et al., CVPR 2024): training-free video anomaly detection by captioning frames and scoring them with a large language model.",
        mechanism="Captions video frames, cleans captions by cross-modal similarity, prompts an LLM to score anomalousness over a temporal window and aggregates into frame-level scores without any training.",
        practitioners="Surveillance analysts and researchers; reading it needs the captioner and LLM provenance and the temporal window.",
        use_when="Training-free screening of surveillance video for unusual events.",
        not_for="Fine-grained localisation or offline environments without model backends.",
        references=(
            _r(
                "Zanella, L., Menapace, W., Mancini, M., Wang, Y. & Ricci, E. (2024). Harnessing large language models for training-free video anomaly detection. IEEE/CVF CVPR 2024, 18527-18536.",
                "https://doi.org/10.1109/CVPR52733.2024.01753",
            ),
        ),
    ),
    # ----------------------------------------------------------- foundation
    DetectorProfile(
        name="timegpt",
        discipline="Time-series foundation models (hosted)",
        lineage="Nixtla's TimeGPT (Garza & Mergenthaler-Canseco, 2023), a hosted forecasting foundation model, used as a forecast-residual anomaly adapter.",
        mechanism="Sends the series to the TimeGPT API, receives forecasts and anomaly flags, and maps residuals to [0, 1]; external service, no seams declared, nothing claimed without it.",
        practitioners="Forecasting teams with an API subscription; reading it needs the service's data-handling terms and the residual semantics.",
        use_when="Comparing a hosted foundation forecaster's residuals against Mercury's local detectors.",
        not_for="Offline or data-sovereign deployments; the data leaves the host.",
        references=(
            _r(
                "Garza, A. & Mergenthaler-Canseco, M. (2023). TimeGPT-1. arXiv:2310.03589.",
                "https://arxiv.org/abs/2310.03589",
            ),
        ),
    ),
    DetectorProfile(
        name="chronos",
        discipline="Time-series foundation models (local)",
        lineage="Amazon Chronos (Ansari et al., 2024): pretrained probabilistic forecasters built on T5 over tokenised series, run locally.",
        mechanism="Tokenises the series, samples probabilistic forecasts from the pretrained model and scores observations by their position in the forecast distribution.",
        practitioners="ML engineers with GPU/CPU budget for a local model; reading it needs the sweep result (0.371 on NAB, 0.11 on AWS telemetry): the label outranks the evidence today.",
        use_when="Benchmarking foundation forecasting against task-specific detectors.",
        not_for="Change points (declared not detected) or production alerting on its current evidence.",
        references=(
            _r(
                "Ansari, A.F. et al. (2024). Chronos: learning the language of time series. Transactions on Machine Learning Research.",
                "https://arxiv.org/abs/2403.07815",
            ),
        ),
    ),
    DetectorProfile(
        name="matrix_profile",
        discipline="Time-series data mining (matrix profile)",
        lineage="The Matrix Profile (Yeh et al., ICDM 2016) for discord discovery, STUMPY-accelerated when installed (Law, 2019).",
        mechanism="Computes the z-normalised nearest-neighbour distance of every subsequence to its best match; subsequences with large profile values (discords) are anomalies, and motifs have small values.",
        practitioners="Data miners and analysts who think in subsequence shapes; reading it needs the subsequence length and the knowledge that a repeating anomaly earns a low discord score.",
        use_when="Finding unique, never-repeated subsequences in long series.",
        not_for="Level shifts (declared not detected) or anomalies that recur; sweep 0.07 on realTraffic.",
        references=(
            _r(
                "Yeh, C.-C.M. et al. (2016). Matrix profile I: all pairs similarity joins for time series. IEEE ICDM 2016, 1317-1322.",
                "https://doi.org/10.1109/ICDM.2016.0179",
            ),
            _r(
                "Law, S.M. (2019). STUMPY: a powerful and scalable Python library for time series data mining. Journal of Open Source Software 4(39), 1504.",
                "https://doi.org/10.21105/joss.01504",
            ),
        ),
    ),
    DetectorProfile(
        name="foundation_ensemble",
        discipline="Ensemble methods over foundation adapters",
        lineage="A rank-combining ensemble (Dietterich, 2000) of the foundation adapters (matrix_profile, chronos, timegpt when available).",
        mechanism="Runs the available foundation adapters, normalises their scores and combines them; it inherits the matrix profile's measured weaknesses (sweep 0.356) because that adapter is always present.",
        practitioners="Researchers comparing foundation approaches; reading it needs to know which adapters were actually available at run time.",
        use_when="One call that exercises every installed foundation adapter.",
        not_for="Level shifts or production alerting on current evidence.",
        references=(
            _r(
                "Dietterich, T.G. (2000). Ensemble methods in machine learning. Multiple Classifier Systems, LNCS 1857, 1-15.",
                "https://doi.org/10.1007/3-540-45014-9_1",
            ),
        ),
    ),
    # --------------------------------------------------------- distillation
    DetectorProfile(
        name="dual_student",
        discipline="Knowledge distillation for visual anomaly detection",
        lineage="Knowledge distillation (Hinton, Vinyals & Dean, 2015) in the uninformed-students tradition (Bergmann et al., CVPR 2020), with two students trained to agree on normal images and disagree elsewhere.",
        mechanism="Two student networks distil a teacher on normal images; at test time their mutual disagreement and teacher discrepancy form the anomaly map, so regions one student cannot reproduce stand out.",
        practitioners="Machine-vision researchers; reading it needs the teacher backbone and student capacities.",
        use_when="Visual anomaly localisation where student disagreement is a better signal than a single reconstruction.",
        not_for="Non-image data; no committed evaluation yet.",
        references=(
            _r(
                "Hinton, G., Vinyals, O. & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv:1503.02531.",
                "https://arxiv.org/abs/1503.02531",
            ),
            _r(
                "Bergmann, P., Fauser, M., Sattlegger, D. & Steger, C. (2020). Uninformed students: student-teacher anomaly detection with discriminative latent embeddings. IEEE/CVF CVPR 2020, 4183-4192.",
                "https://doi.org/10.1109/CVPR42600.2020.00424",
            ),
        ),
    ),
    # -------------------------------------------------------------- physics
    DetectorProfile(
        name="spectral_vibration",
        discipline="Vibration-based condition monitoring",
        lineage="Frequency-domain fault signature analysis (Randall, 2011) with wavelet denoising (Donoho & Johnstone, 1994), graph-Laplacian modelling of frequency interactions and CNN pattern recognition, framed on MLIP/phonon analogies.",
        mechanism="FFTs the vibration signal, denoises, builds a graph over frequency bins to model their interactions, classifies the spectral pattern and emits an anomaly score with a signature type (e.g. bearing, imbalance).",
        practitioners="Rotating-machinery reliability engineers and vibration analysts; reading it needs the sample rate and the fault-frequency vocabulary (CWRU-style bearing fixtures exist under tests/fixtures/cwru).",
        use_when="Machine vibration series where fault signatures or variance changes matter.",
        not_for="Non-oscillatory data; and no labeled evaluation pack is committed yet.",
        references=(
            _r(
                "Randall, R.B. (2011). Vibration-based Condition Monitoring: Industrial, Aerospace and Automotive Applications. Wiley.",
                "https://doi.org/10.1002/9780470977668",
            ),
            _r(
                "Donoho, D.L. & Johnstone, I.M. (1994). Ideal spatial adaptation by wavelet shrinkage. Biometrika 81(3), 425-455.",
                "https://doi.org/10.1093/biomet/81.3.425",
            ),
            _r(
                "Case Western Reserve University Bearing Data Center.",
                "https://engineering.case.edu/bearingdatacenter",
            ),
        ),
    ),
    DetectorProfile(
        name="acceleration_dynamics",
        discipline="Nonlinear dynamics and kinematics",
        lineage="Kinematic derivatives (velocity, acceleration, jerk), energy and momentum analogues, phase-space reconstruction (Takens, 1981) and Lyapunov-exponent estimation from a time series (Wolf et al., 1985).",
        mechanism="Differentiates the series to velocity, acceleration and jerk, tracks energy-conservation analogues, embeds the trajectory in phase space and estimates the largest Lyapunov exponent; scores sudden rate changes, energy violations and chaotic transitions with a motion-state label.",
        practitioners="Control engineers and physicists, and analysts who read rates of change (network, financial, system metrics) through the kinematic metaphor; reading it needs the time step and the metaphor's limits.",
        use_when="Series where the rate and acceleration of change, or a transition to chaotic behaviour, is the anomaly.",
        not_for="Static distributional outliers; no labeled evaluation is committed.",
        references=(
            _r(
                "Wolf, A., Swift, J.B., Swinney, H.L. & Vastano, J.A. (1985). Determining Lyapunov exponents from a time series. Physica D 16(3), 285-317.",
                "https://doi.org/10.1016/0167-2789(85)90011-9",
            ),
            _r(
                "Takens, F. (1981). Detecting strange attractors in turbulence. Dynamical Systems and Turbulence, LNM 898, 366-381.",
                "https://doi.org/10.1007/BFb0091924",
            ),
        ),
    ),
    DetectorProfile(
        name="physics_integrated",
        discipline="Physics-inspired detector integration (Mercury 3R and GOSNN)",
        lineage="Mercury's unified physics detector combining spectral_vibration, acceleration_dynamics and uiux_anomaly through the 3R mechanism (Recursion, Resonance, Refactoring), GOSNN governance scalars and the Omni-Ava Equation fusion.",
        mechanism="Runs the enabled physics detectors, enhances their features recursively, fuses their scores with the OAE and applies GOSNN governance, returning a set-level anomaly score and the fusion breakdown; the set-level false-alarm rate is pinned in tests.",
        practitioners="Engine operators and Mercury developers; reading it needs the enabled-detector list and docs/MATH_SPEC.md for the fusion equation.",
        use_when="One physics-flavoured verdict over a batch when several physics detectors apply.",
        not_for="Per-sample scores (set-level verdict) or domain grading, which is pending.",
        references=(
            _r("Mercury Agent, docs/MATH_SPEC.md (Omni-Ava Equation, 3R mechanism)."),
            _r(
                "Mercury Agent, tests/core/test_declared_modality_false_alarm_rate.py (set-level pin)."
            ),
        ),
    ),
    DetectorProfile(
        name="spectral_domain_frequency",
        discipline="Spectral analysis with post-selection inference",
        lineage="Windowed DFT band analysis (Harris, 1978) with binary-segmentation CUSUM change points and exact post-selection inference (Lee et al., 2016), cepstral harmonic fingerprinting and spectral flux, as a neuro-symbolic Oracle feeding the fusion layer.",
        mechanism="Slides Hann windows with 50 percent overlap, z-scores per-band power against reference statistics, runs recursive CUSUM change-point detection with truncated-normal selective p-values, measures spectral flux and cepstral structure, and emits per-band scores plus a FrequencyInfluenceVector; phase coherence abstains.",
        practitioners="Signal-processing engineers and statisticians who care about Type I error after selection; reading it needs the band definitions and the abstention semantics.",
        use_when="Series where spectral change in specific bands, with controlled false positives, is the question.",
        not_for="Non-stationary broadband noise without band structure; no labeled evaluation is committed.",
        references=(
            _r(
                "Lee, J.D., Sun, D.L., Sun, Y. & Taylor, J.E. (2016). Exact post-selection inference, with application to the lasso. Annals of Statistics 44(3), 907-927.",
                "https://doi.org/10.1214/15-AOS1371",
            ),
            _r(
                "Harris, F.J. (1978). On the use of windows for harmonic analysis with the discrete Fourier transform. Proceedings of the IEEE 66(1), 51-83.",
                "https://doi.org/10.1109/PROC.1978.10837",
            ),
        ),
    ),
    # ----------------------------------------------------------------- uiux
    DetectorProfile(
        name="uiux_anomaly",
        discipline="Human-computer interaction analytics",
        lineage="Usability-engineering indicators (Nielsen, 1994) and interaction-timing models (Card, Moran & Newell, 1983) applied to click, scroll, cursor, keyboard and navigation streams, with bot-versus-human behaviour classification.",
        mechanism="Extracts interaction features (click frequency and location, scroll speed, cursor velocity, dwell, navigation loops, response times, layout shifts), scores frustration indicators such as rage clicks and abandonment, and classifies the session behaviour (human, bot, confused) with an anomaly score.",
        practitioners="UX researchers, product analysts and fraud/bot-mitigation engineers; reading it needs the session model and the rage-click threshold.",
        use_when="Session interaction logs where usability failure, frustration or automation is the question.",
        not_for="Numeric sensor data; no labeled evaluation is committed.",
        references=(
            _r("Nielsen, J. (1994). Usability Engineering. Morgan Kaufmann."),
            _r(
                "Card, S.K., Moran, T.P. & Newell, A. (1983). The Psychology of Human-Computer Interaction. Lawrence Erlbaum Associates."
            ),
        ),
    ),
)

DETECTOR_PROFILES: dict[str, DetectorProfile] = {p.name: p for p in _DETECTORS}


# --------------------------------------------------------------------------- #
# Lookup, serialisation and validation
# --------------------------------------------------------------------------- #


def domain_profile(domain: str) -> DomainProfile | None:
    """The profile of a loader or hazard domain, or ``None`` when none is declared."""
    return DOMAIN_PROFILES.get(domain)


def detector_profile(name: str) -> DetectorProfile | None:
    """The profile of a manifest detector, or ``None`` when none is declared."""
    return DETECTOR_PROFILES.get(name)


def _listify(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_listify(v) for v in value]
    if isinstance(value, dict):
        return {k: _listify(v) for k, v in value.items()}
    return value


def profile_dict(profile: DomainProfile | DetectorProfile) -> dict[str, Any]:
    """A JSON-ready dictionary of a profile (tuples become lists, nested ones included)."""
    return dict(_listify(asdict(profile)))


_PLACEHOLDERS = ("todo", "tbd", "lorem", "placeholder", "xxx")


def validate_profiles(
    detector_names: set[str] | frozenset[str] | None = None,
    domain_names: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Every defect in the registry, as human-readable sentences; empty when it is sound.

    ``detector_names`` and ``domain_names`` are the sets the registry must cover
    exactly (the manifest and the loader/hazard domains); when omitted only the
    internal rules are checked. The rules are the authoring rules in the module
    docstring: exact coverage, no shared mechanism/lineage/method/label rule,
    at least one reference each, at least one ``https`` institution per domain,
    no placeholder text, and no profile that merely repeats its name.
    """
    problems: list[str] = []
    if detector_names is not None:
        missing = sorted(set(detector_names) - set(DETECTOR_PROFILES))
        extra = sorted(set(DETECTOR_PROFILES) - set(detector_names))
        if missing:
            problems.append(f"detectors without a knowledge profile: {missing}")
        if extra:
            problems.append(f"detector profiles for names not in the manifest: {extra}")
    if domain_names is not None:
        missing = sorted(set(domain_names) - set(DOMAIN_PROFILES))
        extra = sorted(set(DOMAIN_PROFILES) - set(domain_names))
        if missing:
            problems.append(f"domains without a knowledge profile: {missing}")
        if extra:
            problems.append(f"domain profiles for unknown domains: {extra}")

    def _dupes(label: str, pairs: list[tuple[str, str]]) -> None:
        seen: dict[str, str] = {}
        for owner, text in pairs:
            key = " ".join(text.lower().split())
            if key in seen:
                problems.append(f"{label} shared by {seen[key]} and {owner}")
            seen[key] = owner

    def _text_ok(owner: str, field_name: str, text: str, minimum: int) -> None:
        if len(text) < minimum:
            problems.append(
                f"{owner}.{field_name} is too short to distinguish it ({len(text)} chars)"
            )
        low = text.lower()
        if any(p in low for p in _PLACEHOLDERS):
            problems.append(f"{owner}.{field_name} carries placeholder text")

    for name, d in DETECTOR_PROFILES.items():
        if d.name != name:
            problems.append(f"detector profile keyed {name} names itself {d.name}")
        for field_name, minimum in (
            ("discipline", 8),
            ("lineage", 40),
            ("mechanism", 60),
            ("practitioners", 40),
            ("use_when", 20),
            ("not_for", 20),
        ):
            _text_ok(f"detector {name}", field_name, getattr(d, field_name), minimum)
        if not d.references:
            problems.append(f"detector {name} cites no reference")
        for ref in d.references:
            if len(ref.citation) < 12:
                problems.append(f"detector {name} has an empty citation")
            if ref.url and not ref.url.startswith("https://"):
                problems.append(f"detector {name} cites a non-https URL {ref.url}")
    _dupes("detector mechanism", [(n, d.mechanism) for n, d in DETECTOR_PROFILES.items()])
    _dupes("detector lineage", [(n, d.lineage) for n, d in DETECTOR_PROFILES.items()])
    _dupes("detector practitioners", [(n, d.practitioners) for n, d in DETECTOR_PROFILES.items()])

    for domain, p in DOMAIN_PROFILES.items():
        if p.domain != domain:
            problems.append(f"domain profile keyed {domain} names itself {p.domain}")
        for field_name, minimum in (
            ("title", 6),
            ("discipline", 8),
            ("observables", 40),
            ("method", 60),
            ("label_rule", 30),
        ):
            _text_ok(f"domain {domain}", field_name, getattr(p, field_name), minimum)
        if not p.professions:
            problems.append(f"domain {domain} names no profession")
        if not p.expertise or len(p.expertise) < 2:
            problems.append(
                f"domain {domain} states fewer than two items of practitioner expertise"
            )
        if not p.limits:
            problems.append(f"domain {domain} states no limits")
        if not p.references:
            problems.append(f"domain {domain} cites no reference")
        if not p.institutions:
            problems.append(f"domain {domain} names no institution")
        if not any(i.url.startswith("https://") for i in p.institutions):
            problems.append(f"domain {domain} names no institution with an https URL")
        for inst in p.institutions:
            if not inst.role:
                problems.append(f"domain {domain}: institution {inst.name} has no stated role")
        if not p.detectors:
            problems.append(f"domain {domain} names no detector that serves it")
        if detector_names is not None:
            unknown = sorted(set(p.detectors) - set(detector_names))
            if unknown:
                problems.append(f"domain {domain} names detectors not in the manifest: {unknown}")
        for ref in p.references:
            if ref.url and not ref.url.startswith("https://"):
                problems.append(f"domain {domain} cites a non-https URL {ref.url}")
    _dupes("domain method", [(n, p.method) for n, p in DOMAIN_PROFILES.items()])
    _dupes("domain label rule", [(n, p.label_rule) for n, p in DOMAIN_PROFILES.items()])
    _dupes("domain observables", [(n, p.observables) for n, p in DOMAIN_PROFILES.items()])
    return problems
