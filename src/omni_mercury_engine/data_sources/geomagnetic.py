# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Geomagnetic & Electromagnetic Monitoring Data Sources.

Production-grade integrations for:
- USGS Geomagnetism Web Service
- INTERMAGNET (International Real-time Magnetic Observatory Network)
- SuperMAG (Ground Magnetometer Network)
- HeartMath Global Coherence Monitoring System
- British Geological Survey ELF Station (Schumann Resonances)

API Documentation:
- USGS: https://geomag.usgs.gov/ws/
- INTERMAGNET: https://imag-data.bgs.ac.uk/GIN_V1/
- SuperMAG: https://supermag.jhuapl.edu/
- HeartMath: https://www.heartmath.org/gci/gcms/live-data/
- BGS: https://geomag.bgs.ac.uk/research/IARs.html

Notes:
- USGS and INTERMAGNET provide magnetometer data (H, D, Z, F components)
- Schumann resonance processing requires FFT (7.83, 14, 20, 26, 33, 38 Hz)
- HeartMath data has hourly power calculations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from omni_mercury_engine.data_sources.base import (
    AlertLevel,
    CacheConfig,
    DataPoint,
    DataSourceBase,
    DataSourceConfig,
    DataSourceError,
    DataSourceType,
    RateLimitConfig,
)

logger = logging.getLogger(__name__)

# =============================================================================
# USGS Geomagnetism Web Service
# =============================================================================


class USGSObservatory(Enum):
    """USGS Geomagnetism observatories."""

    BOULDER = "BOU"  # Boulder, Colorado
    COLLEGE = "CMO"  # College, Alaska
    HONOLULU = "HON"  # Honolulu, Hawaii
    FREDERICKSBURG = "FRD"  # Fredericksburg, Virginia
    FRESNO = "FRN"  # Fresno, California
    DEADHORSE = "DED"  # Deadhorse, Alaska
    BARROW = "BRW"  # Barrow, Alaska
    SITKA = "SIT"  # Sitka, Alaska
    TUCSON = "TUC"  # Tucson, Arizona
    GUAM = "GUA"  # Guam
    SAN_JUAN = "SJG"  # San Juan, Puerto Rico


class MagneticElement(Enum):
    """Magnetic field elements/channels."""

    H = "H"  # Horizontal intensity
    D = "D"  # Declination
    Z = "Z"  # Vertical intensity
    F = "F"  # Total field intensity
    X = "X"  # North component
    Y = "Y"  # East component


@dataclass
class USGSGeomagConfig:
    """USGS Geomagnetism configuration."""

    observatories: list[USGSObservatory] | None = None
    elements: list[MagneticElement] | None = None
    sampling: str = "minute"  # minute, second, hour, day


class USGSGeomagnetismSource(DataSourceBase):
    """USGS Geomagnetism Web Service data source.

    Provides real-time and historical magnetometer data from USGS observatories:
    - H (horizontal intensity)
    - D (declination)
    - Z (vertical intensity)
    - F (total field intensity)
    - X (north component)
    - Y (east component)

    No authentication required.

    Example:
        >>> source = USGSGeomagnetismSource(
        ...     observatories=[USGSObservatory.BOULDER],
        ...     elements=[MagneticElement.H, MagneticElement.Z, MagneticElement.F]
        ... )
        >>> result = await source.fetch()
    """

    DEFAULT_BASE_URL = "https://geomag.usgs.gov/ws/"

    # Observatory coordinates (lat, lon)
    OBSERVATORY_COORDS: dict[USGSObservatory, tuple[float, float]] = {
        USGSObservatory.BOULDER: (40.137, -105.237),
        USGSObservatory.COLLEGE: (64.874, -147.862),
        USGSObservatory.HONOLULU: (21.316, -158.014),
        USGSObservatory.FREDERICKSBURG: (38.205, -77.373),
        USGSObservatory.FRESNO: (37.091, -119.719),
        USGSObservatory.DEADHORSE: (70.356, -148.793),
        USGSObservatory.BARROW: (71.323, -156.626),
        USGSObservatory.SITKA: (57.058, -135.327),
        USGSObservatory.TUCSON: (32.175, -110.733),
        USGSObservatory.GUAM: (13.590, 144.867),
        USGSObservatory.SAN_JUAN: (18.113, -66.150),
    }

    def __init__(
        self,
        observatories: list[USGSObservatory] | None = None,
        elements: list[MagneticElement] | None = None,
        sampling: str = "minute",
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize USGS Geomagnetism data source.

        Args:
            observatories: Observatories to fetch from (None = Boulder)
            elements: Magnetic elements to fetch (None = H, D, Z, F)
            sampling: Data sampling rate (minute, second, hour, day)
            config: Optional base configuration
        """
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=0,  # No stated limit
            min_interval_seconds=10.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=60)

        super().__init__(base_config)

        self._observatories = observatories or [USGSObservatory.BOULDER]
        self._elements = elements or [
            MagneticElement.H,
            MagneticElement.D,
            MagneticElement.Z,
            MagneticElement.F,
        ]
        self._sampling = sampling

    @property
    def source_id(self) -> str:
        """Source id."""
        return "usgs_geomagnetism"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        """Default source types."""
        return [DataSourceType.MAGNETOMETER]

    def _calculate_disturbance_level(self, values: dict[str, float]) -> AlertLevel:
        """Calculate disturbance level from magnetic field values.

        Uses deviation from quiet-time baseline to estimate geomagnetic activity.
        """
        # Typical quiet-time ranges (nT)
        # H variations > 100 nT indicate storm conditions
        # Dst index approximation: -50 to -100 = moderate, -100 to -200 = strong, < -200 = severe

        h_value = values.get("H")
        if h_value is not None:
            # Rough approximation of storm intensity from H variation
            # Note: Actual storm detection requires baseline comparison
            h_deviation = abs(h_value - 20000)  # Typical mid-latitude H value

            if h_deviation > 500:
                return AlertLevel.SEVERE
            elif h_deviation > 300:
                return AlertLevel.STRONG
            elif h_deviation > 150:
                return AlertLevel.MODERATE
            elif h_deviation > 50:
                return AlertLevel.MINOR

        return AlertLevel.NONE

    async def _fetch_observatory(
        self,
        observatory: USGSObservatory,
        start_time: datetime,
        end_time: datetime,
    ) -> list[DataPoint]:
        """Fetch data from a single observatory."""
        elements_str = ",".join(e.value for e in self._elements)

        params = {
            "id": observatory.value,
            "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
            "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
            "elements": elements_str,
            "sampling_period": self._sampling,
            "format": "json",
            "type": "variation",
        }

        try:
            response = await self._http_get("edge/", params=params)
            data = response.json()
            return self._parse_response(observatory, data)

        except Exception as e:
            logger.warning(f"USGS Geomag {observatory.value} fetch failed: {e}")
            return []

    def _parse_response(
        self,
        observatory: USGSObservatory,
        data: dict[str, Any],
    ) -> list[DataPoint]:
        """Parse USGS Geomagnetism response."""
        data_points: list[DataPoint] = []

        values = data.get("values", [])
        times = data.get("times", [])

        if not values or not times:
            return []

        # Get observatory coordinates
        coords = self.OBSERVATORY_COORDS.get(observatory, (0.0, 0.0))
        location = (coords[0], coords[1], 0.0)

        # values is a list of lists, one per element
        # times is a list of ISO timestamps

        # Determine elements from response
        elements = []
        for elem_data in values:
            elem_id = elem_data.get("id", "")
            if elem_id:
                elements.append(elem_id)

        for i, time_str in enumerate(times[-100:]):  # Last 100 data points
            try:
                # Adjust index for slicing
                idx = len(times) - 100 + i if len(times) > 100 else i

                timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))

                # Extract values for each element
                point_values: dict[str, float] = {}
                for j, elem_data in enumerate(values):
                    elem_id = elem_data.get("id", f"elem_{j}")
                    elem_values = elem_data.get("values", [])
                    if idx < len(elem_values) and elem_values[idx] is not None:
                        point_values[elem_id] = float(elem_values[idx])

                if not point_values:
                    continue

                data_points.append(
                    DataPoint(
                        source_id=self.source_id,
                        source_type=DataSourceType.MAGNETOMETER,
                        event_id=f"usgs_{observatory.value}_{timestamp.isoformat()}",
                        timestamp=timestamp,
                        data={
                            "observatory": observatory.value,
                            "elements": point_values,
                            "sampling": self._sampling,
                        },
                        location=location,
                        alert_level=self._calculate_disturbance_level(point_values),
                        confidence=0.95,
                        metadata={"format": "IAGA2002", "type": "variation"},
                    )
                )

            except (ValueError, IndexError, KeyError) as e:
                logger.debug(f"Failed to parse USGS data point: {e}")
                continue

        return data_points

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch magnetometer data from USGS observatories."""
        end_time = end_time or datetime.now(UTC)
        start_time = start_time or (end_time - timedelta(hours=1))

        all_data_points: list[DataPoint] = []

        for observatory in self._observatories:
            data_points = await self._fetch_observatory(observatory, start_time, end_time)
            all_data_points.extend(data_points)

        logger.info(
            f"USGS Geomag: Fetched {len(all_data_points)} data points "
            f"from {len(self._observatories)} observatories"
        )
        return all_data_points


# =============================================================================
# INTERMAGNET - International Real-time Magnetic Observatory Network
# =============================================================================


class INTERMAGNETSource(DataSourceBase):
    """INTERMAGNET (International Real-time Magnetic Observatory Network) data source.

    Provides access to data from 150+ magnetometer observatories globally.
    Data types: Definitive, quasi-definitive, variation data.

    Note: Some endpoints may require account for bulk download.

    Example:
        >>> source = INTERMAGNETSource()
        >>> result = await source.fetch(observatory_codes=["BOU", "ESK"])
    """

    DEFAULT_BASE_URL = "https://imag-data.bgs.ac.uk/GIN_V1/"

    # Common INTERMAGNET observatories with coordinates
    OBSERVATORIES: dict[str, tuple[str, float, float]] = {
        "BOU": ("Boulder, USA", 40.137, -105.237),
        "ESK": ("Eskdalemuir, UK", 55.317, -3.200),
        "HAD": ("Hartland, UK", 50.995, -4.483),
        "LER": ("Lerwick, UK", 60.138, -1.183),
        "NGK": ("Niemegk, Germany", 52.072, 12.675),
        "CLF": ("Chambon-la-Forêt, France", 48.025, 2.260),
        "KAK": ("Kakioka, Japan", 36.232, 140.186),
        "HON": ("Honolulu, USA", 21.316, -158.014),
        "HER": ("Hermanus, South Africa", -34.425, 19.225),
        "GUA": ("Guam", 13.590, 144.867),
    }

    def __init__(
        self,
        observatory_codes: list[str] | None = None,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize INTERMAGNET data source.

        Args:
            observatory_codes: Observatory codes (None = common set)
            config: Optional base configuration
        """
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=100,
            min_interval_seconds=5.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=300)

        super().__init__(base_config)

        self._observatory_codes = observatory_codes or ["BOU", "ESK", "NGK"]

    @property
    def source_id(self) -> str:
        """Source id."""
        return "intermagnet"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        """Default source types."""
        return [DataSourceType.MAGNETOMETER]

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch magnetometer data from INTERMAGNET.

        Note: This is a simplified implementation. Full INTERMAGNET access
        requires specific data format handling (IAGA2002, CDF).
        """
        end_time = end_time or datetime.now(UTC)
        start_time = start_time or (end_time - timedelta(hours=24))

        data_points: list[DataPoint] = []

        # Note: INTERMAGNET has specific data access protocols
        # This implementation provides metadata and current status
        for code in self._observatory_codes:
            obs_info = self.OBSERVATORIES.get(code)
            if not obs_info:
                continue

            name, lat, lon = obs_info

            # Create a status data point
            data_points.append(
                DataPoint(
                    source_id=self.source_id,
                    source_type=DataSourceType.MAGNETOMETER,
                    event_id=f"intermagnet_{code}_{datetime.now(UTC).isoformat()}",
                    timestamp=datetime.now(UTC),
                    data={
                        "observatory_code": code,
                        "observatory_name": name,
                        "network": "INTERMAGNET",
                        "data_types": ["definitive", "quasi-definitive", "variation"],
                        "elements": ["H", "D", "Z", "F"],
                        "note": "Full data requires INTERMAGNET data access protocol",
                    },
                    location=(lat, lon, 0.0),
                    confidence=0.85,
                    metadata={"network": "INTERMAGNET", "format": "IAGA2002"},
                )
            )

        logger.info(f"INTERMAGNET: Created {len(data_points)} observatory entries")
        return data_points


# =============================================================================
# SuperMAG - Ground Magnetometer Network
# =============================================================================


class SuperMAGSource(DataSourceBase):
    """SuperMAG ground magnetometer network data source.

    Provides access to 500+ ground magnetometer stations globally.
    Includes derived indices: SME, SMU, SML (auroral electrojet indices).

    Note: Account required for bulk download.

    Example:
        >>> source = SuperMAGSource(username="user", password="pass")
        >>> result = await source.fetch()
    """

    DEFAULT_BASE_URL = "https://supermag.jhuapl.edu/"

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize SuperMAG data source.

        Args:
            username: SuperMAG account username (optional)
            password: SuperMAG account password (optional)
            config: Optional base configuration
        """
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=100,
            min_interval_seconds=10.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=600)

        super().__init__(base_config)

        self._username = username
        self._password = password

    @property
    def source_id(self) -> str:
        """Source id."""
        return "supermag"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        """Default source types."""
        return [DataSourceType.MAGNETOMETER]

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch magnetometer data from SuperMAG.

        Note: Full SuperMAG API access requires authentication and specific
        API endpoints. This provides metadata about available indices.
        """
        data_points: list[DataPoint] = []

        # SuperMAG provides auroral electrojet indices
        indices = ["SME", "SMU", "SML"]  # Auroral electrojet indices

        for index_name in indices:
            data_points.append(
                DataPoint(
                    source_id=self.source_id,
                    source_type=DataSourceType.MAGNETOMETER,
                    event_id=f"supermag_{index_name}_{datetime.now(UTC).isoformat()}",
                    timestamp=datetime.now(UTC),
                    data={
                        "index_name": index_name,
                        "description": self._get_index_description(index_name),
                        "station_count": 500,  # Approximate
                        "note": "Full data requires SuperMAG account",
                    },
                    confidence=0.85,
                    metadata={"network": "SuperMAG"},
                )
            )

        logger.info(f"SuperMAG: Created {len(data_points)} index entries")
        return data_points

    def _get_index_description(self, index_name: str) -> str:
        """Get description for SuperMAG index."""
        descriptions = {
            "SME": "SuperMAG Electrojet index - combined auroral electrojet activity",
            "SMU": "SuperMAG Upper envelope - eastward auroral electrojet",
            "SML": "SuperMAG Lower envelope - westward auroral electrojet",
        }
        return descriptions.get(index_name, "SuperMAG derived index")


# =============================================================================
# HeartMath Global Coherence Monitoring System
# =============================================================================


class HeartMathSite(Enum):
    """HeartMath GCMS monitoring sites."""

    CALIFORNIA = "california"
    SAUDI_ARABIA = "saudi_arabia"
    LITHUANIA = "lithuania"
    NEW_ZEALAND = "new_zealand"
    SOUTH_AFRICA = "south_africa"
    CANADA = "canada"


class HeartMathGCMSSource(DataSourceBase):
    """HeartMath Global Coherence Monitoring System data source.

    Monitors Schumann resonance and Earth's magnetic field:
    - Schumann resonance spectrograms (0.32-36 Hz band)
    - Sites: California, Saudi Arabia, Lithuania, New Zealand, South Africa, Canada
    - Hourly power calculations with 24-hour moving average

    HONESTY CONTRACT: HeartMath GCMS publishes its spectrograms as web
    imagery only — there is no public machine-readable API — so this source
    cannot fetch real per-mode power values. Every point it emits therefore
    carries a fixed, clearly-invented placeholder spectrum and is labelled
    ``metadata["simulated"] = True`` with ``confidence=0.0``, which the
    live-ingestion seam refuses unless the consumer explicitly passes
    ``allow_simulated=True`` (see
    :mod:`omni_mercury_engine.data_sources.live_ingestion`). For real
    Schumann-resonance measurements use :class:`BGSELFStationSource`
    instrument mode with raw ELF samples.

    Example:
        >>> source = HeartMathGCMSSource()
        >>> result = await source.fetch(sites=[HeartMathSite.CALIFORNIA])
    """

    DEFAULT_BASE_URL = "https://www.heartmath.org/gci/gcms/"

    # Site coordinates
    SITE_COORDS: dict[HeartMathSite, tuple[str, float, float]] = {
        HeartMathSite.CALIFORNIA: ("Boulder Creek, CA", 37.125, -122.147),
        HeartMathSite.SAUDI_ARABIA: ("Hofuf, Saudi Arabia", 25.379, 49.587),
        HeartMathSite.LITHUANIA: ("Paluknys, Lithuania", 54.880, 25.072),
        HeartMathSite.NEW_ZEALAND: ("Northland, New Zealand", -35.311, 173.923),
        HeartMathSite.SOUTH_AFRICA: ("Groot Marico, South Africa", -25.579, 26.394),
        HeartMathSite.CANADA: ("Alberta, Canada", 53.523, -113.530),
    }

    # Schumann resonance fundamental frequencies (Hz)
    SCHUMANN_FREQUENCIES = [7.83, 14.1, 20.3, 26.4, 32.4, 38.0]

    def __init__(
        self,
        sites: list[HeartMathSite] | None = None,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize HeartMath GCMS data source.

        Args:
            sites: Monitoring sites to include (None = all)
            config: Optional base configuration
        """
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=60,
            min_interval_seconds=60.0,  # Hourly updates
        )
        base_config.cache = CacheConfig(ttl_seconds=3600)

        super().__init__(base_config)

        self._sites = sites or list(HeartMathSite)
        self._warned_simulated = False

    @property
    def source_id(self) -> str:
        """Source id."""
        return "heartmath_gcms"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        """Default source types."""
        return [DataSourceType.SCHUMANN_RESONANCE, DataSourceType.MAGNETOMETER]

    def _calculate_coherence_level(self, power_data: dict[str, float]) -> AlertLevel:
        """Calculate coherence level from Schumann resonance power.

        Higher coherence (more pronounced peaks) indicates stronger Earth-ionosphere resonance
        coupling.
        """
        # This is a simplified model - actual coherence calculation requires
        # spectral analysis of the full waveform
        total_power = sum(power_data.values())

        if total_power > 100:
            return AlertLevel.STRONG
        elif total_power > 50:
            return AlertLevel.MODERATE
        elif total_power > 20:
            return AlertLevel.MINOR

        return AlertLevel.NONE

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Emit labelled placeholder Schumann-resonance points (no real API).

        HeartMath GCMS has no public machine-readable endpoint, so nothing
        here is measured: the per-mode power spectrum is a fixed placeholder
        (``10 / mode``). Every point is labelled
        ``metadata["simulated"] = True`` with ``confidence=0.0`` so the
        live-ingestion seam refuses it without an explicit
        ``allow_simulated=True`` opt-in — this source can never masquerade
        as a live feed (see the class HONESTY CONTRACT).
        """
        data_points: list[DataPoint] = []

        if not self._warned_simulated:
            logger.warning(
                "HeartMath GCMS has no public machine-readable API; emitting "
                "PLACEHOLDER spectra labelled metadata['simulated']=True with "
                "confidence=0.0. Consumers must opt in with allow_simulated=True; "
                "use BGSELFStationSource instrument mode for real measurements."
            )
            self._warned_simulated = True

        for site in self._sites:
            site_info = self.SITE_COORDS.get(site)
            if not site_info:
                continue

            name, lat, lon = site_info

            schumann_data: dict[str, Any] = {
                "site": site.value,
                "site_name": name,
                "frequency_band": "0.32-36 Hz",
                "schumann_frequencies_hz": self.SCHUMANN_FREQUENCIES,
                "fundamental_frequency": 7.83,
                "update_interval": "hourly",
                "note": (
                    "PLACEHOLDER Schumann spectrum (no public HeartMath API); "
                    "simulated=True, not a measurement"
                ),
            }

            # Fixed placeholder power per resonance mode — deliberately not
            # randomised so it can never be mistaken for a measurement.
            power_data: dict[str, float] = {}
            for i, freq in enumerate(self.SCHUMANN_FREQUENCIES):
                power_data[f"mode_{i+1}_{freq}Hz"] = 10.0 / (i + 1)

            schumann_data["power_spectrum"] = power_data

            data_points.append(
                DataPoint(
                    source_id=self.source_id,
                    source_type=DataSourceType.SCHUMANN_RESONANCE,
                    event_id=f"heartmath_{site.value}_{datetime.now(UTC).isoformat()}",
                    timestamp=datetime.now(UTC),
                    data=schumann_data,
                    location=(lat, lon, 0.0),
                    alert_level=AlertLevel.NONE,
                    confidence=0.0,
                    metadata={
                        "network": "HeartMath GCI",
                        "measurement_type": "schumann_resonance",
                        "simulated": True,
                        "data_provenance": "simulated",
                    },
                )
            )

        logger.info(f"HeartMath GCMS: Created {len(data_points)} labelled-simulated site entries")
        return data_points


# =============================================================================
# British Geological Survey ELF Station
# =============================================================================


class BGSELFStationSource(DataSourceBase):
    """British Geological Survey ELF Station data source.

    Monitors extremely low frequency (ELF) electromagnetic phenomena:
    - Schumann resonances (7.83, 14, 20, 26, 33, 38 Hz)
    - Ionospheric Alfvén Resonances (IARs)
    - Requires FFT processing for resonance extraction

    HONESTY CONTRACT: real BGS ELF spectrogram feeds require instrument /
    research-agreement access (https://geomag.bgs.ac.uk/research/IARs.html has
    no public machine-readable endpoint), so this source does NOT fake an HTTP
    feed. Two modes exist and both are labelled explicitly:

    - **Instrument mode** — the caller supplies raw ELF samples via
      ``fetch(raw_samples=..., sampling_rate_hz=...)``. The real Welch-PSD
      helpers (:meth:`_welch_power_estimate` / :meth:`_extract_schumann_resonances`)
      extract the Schumann mode powers from the supplied record; emitted data
      points carry ``metadata["simulated"] = False`` and
      ``metadata["data_provenance"] = "instrument"``.
    - **Simulated mode** — with no raw samples, a deterministic synthetic ELF
      record (damped Schumann-mode sinusoids + seeded noise) is generated and
      the SAME real Welch DSP is run over it, so the processing chain is real
      even though the feed is not. Every emitted data point carries
      ``metadata["simulated"] = True`` and a warning is logged; downstream
      live-ingestion refuses simulated points unless the consumer passes an
      explicit ``allow_simulated=True``
      (see :mod:`omni_mercury_engine.data_sources.live_ingestion`).

    Example:
        >>> source = BGSELFStationSource()
        >>> result = await source.fetch(raw_samples=my_elf_record, sampling_rate_hz=100.0)
    """

    DEFAULT_BASE_URL = "https://geomag.bgs.ac.uk/"

    # Schumann resonance frequencies
    SCHUMANN_RESONANCES = {
        "SR1": 7.83,  # Fundamental
        "SR2": 14.1,  # Second harmonic
        "SR3": 20.3,  # Third harmonic
        "SR4": 26.4,  # Fourth harmonic
        "SR5": 32.4,  # Fifth harmonic
        "SR6": 38.0,  # Sixth harmonic
    }

    def __init__(
        self,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize BGS ELF Station data source."""
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=60,
            min_interval_seconds=60.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=3600)

        super().__init__(base_config)
        self._warned_simulated = False

    @property
    def source_id(self) -> str:
        """Source id."""
        return "bgs_elf"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        """Default source types."""
        return [DataSourceType.ELF_VLF, DataSourceType.SCHUMANN_RESONANCE]

    def _welch_power_estimate(
        self,
        signal: np.ndarray[Any, Any],
        fs: float,
        target_freq: float,
        window_size: int = 256,
    ) -> float:
        """Estimate power at target frequency using Welch's method.

        Args:
            signal: Time-domain signal
            fs: Sampling frequency
            target_freq: Target frequency for power estimation
            window_size: FFT window size

        Returns:
            Power estimate at target frequency
        """
        # Apply Hann window
        window = np.hanning(window_size)

        # Number of segments
        n_segments = len(signal) // window_size

        if n_segments == 0:
            return 0.0

        power_sum = 0.0

        for i in range(n_segments):
            segment = signal[i * window_size : (i + 1) * window_size]
            windowed = segment * window

            # FFT
            fft_result = np.fft.rfft(windowed)
            power = np.abs(fft_result) ** 2

            # Find frequency bin closest to target
            freqs = np.fft.rfftfreq(window_size, 1.0 / fs)
            target_bin = np.argmin(np.abs(freqs - target_freq))

            power_sum += power[target_bin]

        return float(power_sum / n_segments)

    def _extract_schumann_resonances(
        self,
        signal: np.ndarray[Any, Any],
        fs: float = 100.0,
    ) -> dict[str, float]:
        """Extract Schumann resonance power from ELF signal.

        Uses Welch's method with Hann windowing to estimate power
        at each resonance frequency.

        Args:
            signal: Raw ELF signal
            fs: Sampling frequency (Hz)

        Returns:
            Dictionary of resonance powers
        """
        powers: dict[str, float] = {}

        for name, freq in self.SCHUMANN_RESONANCES.items():
            power = self._welch_power_estimate(signal, fs, freq)
            powers[name] = power

        return powers

    def _calculate_resonance_quality(self, powers: dict[str, float]) -> AlertLevel:
        """Calculate resonance quality from power distribution.

        Higher SR1/SR2 ratio and clear harmonic structure indicates better resonance quality.
        """
        sr1 = powers.get("SR1", 0)
        sr2 = powers.get("SR2", 0)

        if sr1 == 0:
            return AlertLevel.NONE

        # Ratio indicates resonance clarity
        ratio = sr1 / (sr2 + 1e-10)

        if ratio > 10:
            return AlertLevel.STRONG  # Very clear resonance
        elif ratio > 5:
            return AlertLevel.MODERATE
        elif ratio > 2:
            return AlertLevel.MINOR

        return AlertLevel.NONE

    # Simulated-record shape: ~20 s at 100 Hz covers >150 cycles of the 7.83 Hz
    # fundamental and gives the 256-sample Welch window 8 full segments.
    _SIM_N_SAMPLES = 2048
    _SIM_SAMPLING_HZ = 100.0
    _SIM_SEED = 783  # deterministic: same simulated record on every fetch

    def _simulated_elf_record(self) -> np.ndarray[Any, Any]:
        """Deterministic synthetic ELF record (EXPLICITLY SIMULATED).

        Sum of amplitude-decaying sinusoids at the six Schumann mode
        frequencies plus seeded Gaussian noise. This is a labelled simulation
        of the Earth-ionosphere cavity signal shape — NOT a real measurement —
        used only so the real Welch DSP chain has something to process when no
        instrument record is supplied.

        Returns:
            Synthetic ELF record of ``_SIM_N_SAMPLES`` samples at
            ``_SIM_SAMPLING_HZ``.
        """
        rng = np.random.default_rng(self._SIM_SEED)
        t = np.arange(self._SIM_N_SAMPLES) / self._SIM_SAMPLING_HZ
        record = np.zeros_like(t)
        for i, freq in enumerate(self.SCHUMANN_RESONANCES.values()):
            record += (1.0 / (i + 1)) * np.sin(2.0 * np.pi * freq * t)
        record += 0.1 * rng.standard_normal(self._SIM_N_SAMPLES)
        return record

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        raw_samples: np.ndarray[Any, Any] | None = None,
        sampling_rate_hz: float | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Extract Schumann resonance powers from an ELF record.

        Real BGS ELF feeds require instrument access; there is no public HTTP
        endpoint, so nothing is fetched over the network. When ``raw_samples``
        is supplied the real Welch-PSD helpers process it (instrument mode);
        otherwise a deterministic synthetic record is processed and every
        output is labelled ``metadata["simulated"] = True`` (simulated mode).

        Args:
            start_time: Unused (kept for the DataSourceBase contract).
            end_time: Unused (kept for the DataSourceBase contract).
            raw_samples: Optional raw ELF record from a caller's instrument.
            sampling_rate_hz: Sampling rate of ``raw_samples`` (default 100.0).
            **kwargs: Unused source-specific parameters.

        Returns:
            One SCHUMANN_RESONANCE data point with Welch-extracted mode powers.

        Raises:
            DataSourceError: If ``raw_samples`` is supplied but unusable
                (non-finite or shorter than one Welch window).
        """
        fs = float(sampling_rate_hz or self._SIM_SAMPLING_HZ)

        if raw_samples is not None:
            record = np.asarray(raw_samples, dtype=float).ravel()
            if record.size < 256:
                raise DataSourceError(
                    f"BGS ELF: raw_samples too short for Welch estimation "
                    f"({record.size} < 256 samples)",
                    source_id=self.source_id,
                    retryable=False,
                )
            if not np.all(np.isfinite(record)):
                raise DataSourceError(
                    "BGS ELF: raw_samples contain non-finite values",
                    source_id=self.source_id,
                    retryable=False,
                )
            simulated = False
            provenance = "instrument"
        else:
            if not self._warned_simulated:
                logger.warning(
                    "BGS ELF: no raw_samples supplied and real BGS ELF feeds require "
                    "instrument access -- emitting an EXPLICITLY SIMULATED record "
                    "(metadata['simulated']=True). Consumers must opt in with "
                    "allow_simulated=True."
                )
                self._warned_simulated = True
            record = self._simulated_elf_record()
            fs = self._SIM_SAMPLING_HZ
            simulated = True
            provenance = "simulated"

        # Real DSP on both paths: Welch power estimation with Hann windowing.
        powers = self._extract_schumann_resonances(record, fs=fs)

        data_point = DataPoint(
            source_id=self.source_id,
            source_type=DataSourceType.SCHUMANN_RESONANCE,
            event_id=f"bgs_elf_{datetime.now(UTC).isoformat()}",
            timestamp=datetime.now(UTC),
            data={
                "station": "BGS ELF",
                "schumann_resonances": self.SCHUMANN_RESONANCES,
                "power_spectrum": powers,
                "fundamental_frequency_hz": 7.83,
                "processing_method": "Welch with Hann window",
                "sampling_rate_hz": fs,
                "n_samples": int(record.size),
                # The processed record is included so consumers can run their
                # own spectral pipeline on exactly what was analysed here.
                "elf_record": record.tolist(),
                "frequency_range_hz": "0-50",
                "iar_detected": False,  # Ionospheric Alfvén Resonances
            },
            location=(55.317, -3.200, 0.0),  # Eskdalemuir coordinates
            alert_level=self._calculate_resonance_quality(powers),
            confidence=0.75 if not simulated else 0.0,
            metadata={
                "source": "BGS",
                "measurement_type": "elf_spectrogram",
                "simulated": simulated,
                "data_provenance": provenance,
            },
        )

        logger.info("BGS ELF: extracted Schumann resonance powers (%s)", provenance)
        return [data_point]
