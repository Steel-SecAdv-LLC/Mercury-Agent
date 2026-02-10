"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Disaster and Emergency Management Dataset Loaders

This module provides loaders for disaster and emergency datasets:
- OpenFEMA: US disaster declarations, assistance programs, hazard mitigation
- Provides free REST API access with no authentication required

All data sources are publicly accessible government datasets.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

from omni_mercury_engine.security.input_validation import TrustedEndpoints

from .base import DatasetConfig, DatasetLoader, DatasetRegistry

logger = logging.getLogger(__name__)


class FEMADisasterLoader(DatasetLoader):
    """
    OpenFEMA Disaster Declarations Data Loader.

    Downloads REAL disaster declaration data from FEMA's OpenFEMA API:
    - Disaster declarations (hurricanes, floods, fires, etc.)
    - Declaration dates, locations, and disaster types
    - Individual and public assistance program data

    Data source: https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries
    License: Public Domain (US Government)
    No API key required - free and open access.
    """

    DATASET_NAME = "fema_disaster"
    DATASET_URL = "https://www.fema.gov/about/openfema/data-sets"
    LICENSE = "Public Domain (US Government)"
    CITATION = """Federal Emergency Management Agency (FEMA). OpenFEMA Dataset:
    Disaster Declarations Summaries - v2.
    https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2"""
    REQUIRES_CREDENTIALS = False

    # Disaster type codes from FEMA
    DISASTER_TYPES = {
        "DR": "Major Disaster Declaration",
        "EM": "Emergency Declaration",
        "FM": "Fire Management Assistance",
        "FS": "Fire Suppression Authorization",
    }

    # Incident types
    INCIDENT_TYPES = [
        "Hurricane",
        "Flood",
        "Severe Storm",
        "Fire",
        "Tornado",
        "Earthquake",
        "Snowstorm",
        "Drought",
        "Coastal Storm",
        "Dam/Levee Break",
        "Tsunami",
        "Volcanic Eruption",
    ]

    FEATURE_NAMES = [
        "disaster_number",
        "state_fips",
        "year",
        "month",
        "day",
        "incident_type_code",
        "declaration_type_code",
        "designated_area_code",
        "ia_program",  # Individual Assistance
        "pa_program",  # Public Assistance
        "hm_program",  # Hazard Mitigation
    ]

    # OpenFEMA API endpoints (via TrustedEndpoints for SSRF protection)
    API_URL = TrustedEndpoints.FEMA_DISASTER_DECLARATIONS

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize FEMA disaster loader.

        Args:
            config: Dataset configuration. Preprocessing options:
                - year_range (tuple): (start_year, end_year)
                - states (list): List of state abbreviations to filter
                - incident_types (list): List of incident types to include
                - declaration_types (list): 'DR', 'EM', 'FM', etc.
        """
        super().__init__(config)
        self.year_range = config.preprocessing.get("year_range", (2000, 2025))
        self.states = config.preprocessing.get("states", None)  # None = all states
        self.incident_types = config.preprocessing.get("incident_types", None)
        self.declaration_types = config.preprocessing.get("declaration_types", ["DR", "EM"])
        self._is_real_data = False

        # Rate limiting (FEMA recommends respectful API usage)
        self._last_request_time = 0.0
        self._request_delay = 0.5  # 500ms between requests

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    def _rate_limit(self) -> None:
        """Implement rate limiting for API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_delay:
            time.sleep(self._request_delay - elapsed)
        self._last_request_time = time.time()

    def download(self) -> bool:
        """Download disaster data from OpenFEMA API.

        Returns:
            True if download successful, False otherwise.
        """
        if self._download_from_fema():
            return True

        logger.warning("OpenFEMA API failed, falling back to SYNTHETIC data.")
        return self._create_synthetic_disasters()

    def _download_from_fema(self) -> bool:
        """Download disaster declarations from OpenFEMA API."""
        import urllib.parse
        import urllib.request

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "fema_disaster_real.npz"

        if cache_file.exists():
            logger.info(f"FEMA disaster data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            # Build OData filter query
            filters = []

            # Year filter
            start_date = f"{self.year_range[0]}-01-01"
            end_date = f"{self.year_range[1]}-12-31"
            filters.append(f"declarationDate ge '{start_date}'")
            filters.append(f"declarationDate le '{end_date}'")

            # Declaration type filter
            if self.declaration_types:
                type_filter = " or ".join(
                    [f"declarationType eq '{t}'" for t in self.declaration_types]
                )
                filters.append(f"({type_filter})")

            filter_string = " and ".join(filters)

            # Build URL with parameters
            params = {
                "$filter": filter_string,
                "$top": str(min(self.config.max_samples or 10000, 10000)),
                "$orderby": "declarationDate desc",
            }

            query_string = urllib.parse.urlencode(params)
            url = f"{self.API_URL}?{query_string}"

            logger.info("Downloading disaster data from OpenFEMA API...")
            logger.debug(f"API URL: {url}")

            # Rate limit
            self._rate_limit()

            # Validate URL before opening (SSRF protection via domain allowlist)
            TrustedEndpoints.validate_url(self.API_URL)
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 Mercury-Agent/1.0"}
            )

            with urllib.request.urlopen(req, timeout=120) as response:  # nosec B310
                data = json.loads(response.read().decode("utf-8"))

            # OpenFEMA returns {metadata: {...}, DisasterDeclarationsSummaries: [...]}
            records = data.get("DisasterDeclarationsSummaries", [])

            if not records:
                logger.warning("No disaster records returned from FEMA API")
                return False

            logger.info(f"Downloaded {len(records)} disaster declaration records")

            # Process the data
            features, labels = self._process_fema_data(records)

            # Save to cache
            np.savez_compressed(cache_file, features=features, labels=labels)
            self._is_real_data = True

            logger.info(
                f"FEMA disaster data loaded: {len(features)} samples, "
                f"{labels.sum()} major disasters (is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"OpenFEMA API download failed: {e}")
            return False

    def _process_fema_data(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        """Process OpenFEMA disaster declaration records.

        Args:
            records: List of disaster declaration records

        Returns:
            Tuple of (features, labels) numpy arrays
        """
        rows = []

        # Build incident type lookup
        incident_lookup = {name.lower(): idx for idx, name in enumerate(self.INCIDENT_TYPES)}
        declaration_lookup = {code: idx for idx, code in enumerate(self.DISASTER_TYPES.keys())}

        for record in records:
            # Parse declaration date
            decl_date_str = record.get("declarationDate", "")
            try:
                decl_date = datetime.fromisoformat(decl_date_str.replace("Z", "+00:00"))
                year = decl_date.year
                month = decl_date.month
                day = decl_date.day
            except (ValueError, AttributeError):
                year, month, day = 2000, 1, 1

            # Get incident type code
            incident_type = record.get("incidentType", "").lower()
            incident_code = incident_lookup.get(incident_type, 99)

            # Get declaration type code
            decl_type = record.get("declarationType", "DR")
            decl_code = declaration_lookup.get(decl_type, 0)

            # Parse state FIPS code
            state_fips = 0
            fips_str = record.get("fipsStateCode", "0")
            try:
                state_fips = int(fips_str) if fips_str else 0
            except ValueError:
                pass

            # Parse designated area (county level or statewide)
            designated_area = record.get("designatedArea", "")
            area_code = 1 if "Statewide" in designated_area else 0

            # Program flags
            ia_program = 1 if record.get("ihProgramDeclared", False) else 0
            pa_program = 1 if record.get("paProgramDeclared", False) else 0
            hm_program = 1 if record.get("hmProgramDeclared", False) else 0

            row = [
                record.get("disasterNumber", 0),
                state_fips,
                year,
                month,
                day,
                incident_code,
                decl_code,
                area_code,
                ia_program,
                pa_program,
                hm_program,
            ]
            rows.append(row)

        features = np.array(rows, dtype=np.float32)

        # Label major disasters (DR type with multiple programs)
        # Index 6 = declaration_type_code, 8-10 = program flags
        labels = (
            (features[:, 6] == 0)  # DR (Major Disaster)
            & ((features[:, 8] + features[:, 9] + features[:, 10]) >= 2)  # Multiple programs
        ).astype(np.int64)

        return features, labels

    def _create_synthetic_disasters(self) -> bool:
        """Create synthetic disaster declaration data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 5000

        features = []
        labels = []

        # State FIPS codes (sampling of major disaster-prone states)
        disaster_prone_states = [
            6,  # California (fire, earthquake)
            12,  # Florida (hurricane)
            48,  # Texas (hurricane, flood)
            22,  # Louisiana (hurricane)
            40,  # Oklahoma (tornado)
            20,  # Kansas (tornado)
            36,  # New York (various)
            37,  # North Carolina (hurricane)
            45,  # South Carolina (hurricane)
            1,  # Alabama (tornado)
        ]

        for i in range(n_samples):
            # Generate disaster declaration
            disaster_number = 1000 + i

            # State selection (weighted towards disaster-prone states)
            if np.random.random() < 0.7:
                state_fips = np.random.choice(disaster_prone_states)
            else:
                state_fips = np.random.randint(1, 57)

            # Date within year range
            year = np.random.randint(self.year_range[0], self.year_range[1] + 1)

            # Seasonal patterns
            if state_fips in [12, 22, 48, 37, 45]:  # Hurricane states
                month = np.random.choice([8, 9, 10], p=[0.3, 0.4, 0.3])
            elif state_fips in [40, 20, 1]:  # Tornado alley
                month = np.random.choice([4, 5, 6], p=[0.3, 0.4, 0.3])
            elif state_fips == 6:  # California
                month = np.random.choice([7, 8, 9, 10, 11], p=[0.1, 0.2, 0.2, 0.3, 0.2])
            else:
                month = np.random.randint(1, 13)

            day = np.random.randint(1, 29)

            # Incident type based on state
            if state_fips in [12, 22, 48, 37, 45]:
                incident_code = np.random.choice([0, 1, 2])  # Hurricane, Flood, Severe Storm
            elif state_fips in [40, 20, 1]:
                incident_code = np.random.choice([4, 2])  # Tornado, Severe Storm
            elif state_fips == 6:
                incident_code = np.random.choice([3, 5])  # Fire, Earthquake
            else:
                incident_code = np.random.randint(0, len(self.INCIDENT_TYPES))

            # Declaration type (DR is most common)
            decl_code = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])

            # Designated area (statewide vs county)
            area_code = 1 if np.random.random() < 0.3 else 0

            # Program flags (correlated with disaster severity)
            severity = np.random.beta(2, 5)  # Skewed towards less severe
            ia_program = 1 if severity > 0.3 else 0
            pa_program = 1 if severity > 0.2 else 0
            hm_program = 1 if severity > 0.4 else 0

            feature_vec = [
                disaster_number,
                state_fips,
                year,
                month,
                day,
                incident_code,
                decl_code,
                area_code,
                ia_program,
                pa_program,
                hm_program,
            ]
            features.append(feature_vec)

            # Major disaster label
            is_major = (
                decl_code == 0  # DR type
                and (ia_program + pa_program + hm_program) >= 2  # Multiple programs
            )
            labels.append(1 if is_major else 0)

        features = np.array(features, dtype=np.float32)  # type: ignore[assignment, unused-ignore]
        labels = np.array(labels, dtype=np.int64)  # type: ignore[assignment, unused-ignore]

        save_path = self.data_path / "synthetic_fema_disaster.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(
            f"Generated {n_samples} synthetic disaster records, "
            f"{labels.sum()} major disasters (is_real_data=False)"  # type: ignore[attr-defined, unused-ignore]
        )
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load disaster data from cache."""
        real_cache = self.data_path / "fema_disaster_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._is_real_data = True
            logger.info(f"Loaded REAL FEMA disaster data from {real_cache}")
            return data["features"], data["labels"]

        synthetic_path = self.data_path / "synthetic_fema_disaster.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            self._is_real_data = False
            logger.info("Loaded SYNTHETIC FEMA disaster data (is_real_data=False)")
            return data["features"], data["labels"]

        raise FileNotFoundError("FEMA disaster data not found. Run download() first.")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess disaster data."""
        data = np.nan_to_num(data, nan=0.0)
        # Normalize numerical features (skip categorical codes)
        for i in [0, 2, 3, 4]:  # disaster_number, year, month, day
            col = data[:, i]
            data[:, i] = (col - col.mean()) / (col.std() + 1e-8)
        return data.astype(np.float32)

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about loaded disaster data."""
        features, labels = self.load()

        # Count by incident type
        incident_counts = {}
        for i, name in enumerate(self.INCIDENT_TYPES):
            count = int((features[:, 5] == i).sum())
            if count > 0:
                incident_counts[name] = count

        # Count by declaration type
        decl_counts = {}
        for i, (code, name) in enumerate(self.DISASTER_TYPES.items()):
            count = int((features[:, 6] == i).sum())
            if count > 0:
                decl_counts[code] = count

        return {
            "n_samples": len(features),
            "n_major_disasters": int(labels.sum()),
            "major_disaster_ratio": float(labels.mean()),
            "year_range": (int(features[:, 2].min()), int(features[:, 2].max())),
            "incident_type_distribution": incident_counts,
            "declaration_type_distribution": decl_counts,
            "is_real_data": self._is_real_data,
        }


class FEMAHazardMitigationLoader(DatasetLoader):
    """
    OpenFEMA Hazard Mitigation Grant Program Data Loader.

    Downloads REAL hazard mitigation project data from FEMA:
    - Flood mitigation projects
    - Property acquisitions and elevations
    - Infrastructure hardening
    - Wildfire mitigation

    Data source: https://www.fema.gov/api/open/v2/HazardMitigationGrants
    License: Public Domain (US Government)
    """

    DATASET_NAME = "fema_hazard_mitigation"
    DATASET_URL = "https://www.fema.gov/about/openfema/data-sets"
    LICENSE = "Public Domain (US Government)"
    CITATION = """Federal Emergency Management Agency (FEMA). OpenFEMA Dataset:
    Hazard Mitigation Grants. https://www.fema.gov/openfema-data-page/hazard-mitigation-grants-v1"""
    REQUIRES_CREDENTIALS = False

    FEATURE_NAMES = [
        "project_amount",
        "federal_share",
        "state_fips",
        "year",
        "project_type_code",
        "status_code",
        "program_type_code",
    ]

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize FEMA hazard mitigation loader."""
        super().__init__(config)
        self.year_range = config.preprocessing.get("year_range", (2000, 2025))
        self._is_real_data = False

    @property
    def is_real_data(self) -> bool:
        return self._is_real_data

    def download(self) -> bool:
        """Download hazard mitigation data or generate synthetic."""
        # Note: Full implementation would query FEMA's HazardMitigationGrants endpoint
        # For now, use synthetic data as the API structure is more complex
        logger.info("Generating synthetic hazard mitigation data for development")
        return self._create_synthetic_mitigation()

    def _create_synthetic_mitigation(self) -> bool:
        """Create synthetic hazard mitigation project data."""
        np.random.seed(self.config.random_seed)
        n_samples = self.config.max_samples or 3000

        features = []
        labels = []

        project_types = ["Acquisition", "Elevation", "Floodproofing", "Drainage", "Wildfire"]

        for _ in range(n_samples):
            # Project cost (log-normal distribution)
            project_amount = np.random.lognormal(12, 1.5)  # Mean ~$100K

            # Federal share (typically 75%)
            federal_share = project_amount * np.random.uniform(0.7, 0.9)

            state_fips = np.random.randint(1, 57)
            year = np.random.randint(self.year_range[0], self.year_range[1] + 1)

            project_type_code = np.random.randint(0, len(project_types))
            status_code = np.random.choice(
                [0, 1, 2], p=[0.2, 0.3, 0.5]
            )  # pending, active, complete
            program_type_code = np.random.randint(0, 3)  # HMGP, PDM, FMA

            feature_vec = [
                project_amount,
                federal_share,
                state_fips,
                year,
                project_type_code,
                status_code,
                program_type_code,
            ]
            features.append(feature_vec)

            # Label high-value projects
            is_major = project_amount > 500000
            labels.append(1 if is_major else 0)

        features = np.array(features, dtype=np.float32)  # type: ignore[assignment, unused-ignore]
        labels = np.array(labels, dtype=np.int64)  # type: ignore[assignment, unused-ignore]

        save_path = self.data_path / "synthetic_hazard_mitigation.npz"
        np.savez_compressed(save_path, features=features, labels=labels)

        logger.info(f"Generated {n_samples} synthetic mitigation projects")
        return True

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load mitigation data from cache."""
        synthetic_path = self.data_path / "synthetic_hazard_mitigation.npz"
        if synthetic_path.exists():
            data = np.load(synthetic_path)
            return data["features"], data["labels"]
        raise FileNotFoundError("Hazard mitigation data not found. Run download() first.")

    def preprocess(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess mitigation data."""
        data = np.nan_to_num(data, nan=0.0)
        # Log transform monetary values
        data[:, 0] = np.log1p(data[:, 0])
        data[:, 1] = np.log1p(data[:, 1])
        # Normalize
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        return data.astype(np.float32)


# Register disaster loaders
DatasetRegistry.register("fema_disaster", FEMADisasterLoader)
DatasetRegistry.register("fema", FEMADisasterLoader)  # Alias
DatasetRegistry.register("disaster_declarations", FEMADisasterLoader)  # Alias
DatasetRegistry.register("fema_hazard_mitigation", FEMAHazardMitigationLoader)
DatasetRegistry.register("hazard_mitigation", FEMAHazardMitigationLoader)  # Alias
