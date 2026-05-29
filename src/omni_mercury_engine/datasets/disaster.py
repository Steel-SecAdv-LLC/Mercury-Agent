"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

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
from .exceptions import ALLOW_SYNTHETIC, DataSourceUnavailableError, check_synthetic_allowed

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

    Note:
        This dataset does not include ground-truth anomaly labels.
        Use for unsupervised deployment only.
    """

    DATASET_NAME = "fema_disaster"
    LABEL_SOURCE = "statistical"  # no ground-truth anomaly labels; heuristic polarity selection
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
        """
        Initialize FEMA disaster loader.

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
        # Tracks the v1.7.0 FEMA label-polarity correction; populated
        # by `_select_anomaly_polarity` on every load.  Exposed via
        # the `labels_inverted` property for callers that want to
        # surface the polarity flip in their telemetry.
        self._labels_inverted = False

        # Rate limiting (FEMA recommends respectful API usage)
        self._last_request_time = 0.0
        self._request_delay = 0.5  # 500ms between requests

    @property
    def is_real_data(self) -> bool:
        """Return True if real data was loaded."""
        return self._is_real_data

    @property
    def labels_inverted(self) -> bool:
        """Return True if `_select_anomaly_polarity` flipped the label mask.

        See the docstring on `_select_anomaly_polarity` and
        `tests/datasets/test_disaster.py::TestFEMAInvertedScoresCorrection`
        for the contract.  Useful for benchmark reporters that want
        to footnote the polarity flip alongside AUC.
        """
        return self._labels_inverted

    def _rate_limit(self) -> None:
        """Implement rate limiting for API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_delay:
            time.sleep(self._request_delay - elapsed)
        self._last_request_time = time.time()

    def download(self) -> bool:
        """Download."""
        if self._download_from_fema():
            return True
        if ALLOW_SYNTHETIC:
            check_synthetic_allowed("FEMADisaster", "OpenFEMA API failed")
            return self._create_synthetic_disasters()
        raise DataSourceUnavailableError(
            loader_name="FEMADisaster",
            source_url=str(self.API_URL),
            reason="OpenFEMA API unavailable",
        )

    # OpenFEMA enforces a hard server-side cap of 1000 records per request
    # for v2 endpoints. Requesting larger $top returns HTTP 400. To respect
    # this while still allowing larger requested sample counts we paginate
    # via $skip.
    _OPENFEMA_PAGE_SIZE = 1000

    def _download_from_fema(self) -> bool:
        """Download disaster declarations from OpenFEMA API with pagination."""
        import urllib.parse

        from .base import http_get_with_retry

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "fema_disaster_real.npz"

        if cache_file.exists():
            logger.info(f"FEMA disaster data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            # OpenFEMA OData v4 requires ISO-8601 with millisecond precision
            # and a lowercase 'z' literal. The previous bare 'YYYY-MM-DD' form
            # was silently rejected by the v2 endpoint and returned an empty
            # DisasterDeclarationsSummaries array, which manifested as the
            # documented "FEMA Disaster — known broken" loader.
            start_iso = f"{self.year_range[0]}-01-01T00:00:00.000z"
            end_iso = f"{self.year_range[1]}-12-31T23:59:59.999z"
            filters = [
                f"declarationDate ge '{start_iso}'",
                f"declarationDate le '{end_iso}'",
            ]
            if self.declaration_types:
                type_filter = " or ".join(
                    f"declarationType eq '{t}'" for t in self.declaration_types
                )
                filters.append(f"({type_filter})")
            filter_string = " and ".join(filters)

            target = self.config.max_samples or 10000
            page_size = self._OPENFEMA_PAGE_SIZE

            # SSRF: validate the canonical endpoint once; per-page URLs only
            # vary by query string under the same host.
            TrustedEndpoints.validate_url(self.API_URL)

            all_records: list[dict[str, Any]] = []
            skip = 0
            while len(all_records) < target:
                page_top = min(page_size, target - len(all_records))
                params = {
                    "$filter": filter_string,
                    "$top": str(page_top),
                    "$skip": str(skip),
                    "$orderby": "declarationDate desc",
                }
                url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"

                self._rate_limit()
                logger.info(
                    "OpenFEMA page: $skip=%d $top=%d (collected %d/%d)",
                    skip,
                    page_top,
                    len(all_records),
                    target,
                )
                content = http_get_with_retry(url, timeout=120)
                page = json.loads(content.decode("utf-8")).get("DisasterDeclarationsSummaries", [])

                if not page:
                    # Empty page = end of result set; stop paginating.
                    break

                all_records.extend(page)
                if len(page) < page_top:
                    # Server returned fewer than asked → no more pages.
                    break
                skip += len(page)

            if not all_records:
                logger.warning("No disaster records returned from FEMA API")
                return False

            logger.info(f"Downloaded {len(all_records)} disaster declaration records")
            features, labels = self._process_fema_data(all_records)

            np.savez_compressed(
                cache_file,
                features=features,
                labels=labels,
                labels_inverted=np.array(self._labels_inverted, dtype=bool),
            )
            self._is_real_data = True

            major_count = int(self._major_disaster_mask_from_features(features).sum())
            anomaly_count = int(labels.sum())
            logger.info(
                f"FEMA disaster data loaded: {len(features)} samples, "
                f"{major_count} DR multi-program major-disaster records, "
                f"{anomaly_count} anomaly labels "
                f"(labels_inverted={self._labels_inverted}, is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"OpenFEMA API download failed: {e}")
            return False

    def _process_fema_data(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        """
        Process OpenFEMA disaster declaration records.

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

        # Label major disasters (DR type with multiple programs).
        candidate_major = self._major_disaster_mask_from_features(features)

        # ---- v1.7.0: FEMA Disaster label-polarity correction ----
        #
        # CHANGELOG previously flagged this loader as "known broken
        # — produces inverted scores".  Root cause: in the historical
        # FEMA Disaster Declarations record (1990s–present, filtered
        # to declarationType in {"DR","EM"}) a majority of records
        # *are* Major Disaster declarations that activate at least
        # two of the IA / PA / HM programs — hurricanes and major
        # floods routinely trigger all three.  Marking that class as
        # the positive (anomaly) class hands an unsupervised
        # anomaly detector inverted labels and the ensemble's
        # rank-based AUC collapses below 0.5.
        #
        # Fix: the anomaly class is the empirical minority.  If the
        # candidate "major-disaster" mask covers more than half the
        # records on the loaded slice we invert it so that the
        # *rarer* event (Emergency declarations, single-program
        # activations, fire-management-only) carries label==1, which
        # is the unsupervised-anomaly convention used throughout
        # the rest of Mercury's loaders.  The decision is recorded
        # on the instance (`self._labels_inverted`) so callers and
        # tests can introspect it, and emitted at INFO so production
        # operators see the polarity flip in their logs.
        labels = self._select_anomaly_polarity(candidate_major)

        return features, labels

    @staticmethod
    def _major_disaster_mask_from_features(features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return the hand-defined DR + multi-program major-disaster mask."""
        if features.size == 0:
            return np.array([], dtype=bool)
        # Index 6 = declaration_type_code, 8-10 = program flags.
        return (features[:, 6] == 0) & (  # DR (Major Disaster)
            (features[:, 8] + features[:, 9] + features[:, 10]) >= 2
        )  # Multiple programs

    def _select_anomaly_polarity(
        self, candidate_anomaly_mask: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Return labels with the minority-class-as-anomaly convention.

        Args:
            candidate_anomaly_mask: Boolean array marking the
                hand-defined "of-interest" class (DR + multi-program).

        Returns:
            int64 labels where 1 marks the minority class.  Sets
            ``self._labels_inverted`` to True when the input mask had
            to be flipped to satisfy the convention.

        The minority-as-anomaly invariant is what the rest of the
        Mercury detection stack (`detectors/statistical.py`'s
        ``is_inverted`` sanity check at line 1479) expects.  Loaders
        that hand the detector a majority-as-positive label set
        cause ``adaptive_weights`` to zero out otherwise-correct
        components — see `tests/datasets/test_disaster.py
        ::TestFEMAInvertedScoresCorrection`.
        """
        self._labels_inverted = False
        n = int(candidate_anomaly_mask.size)
        if n == 0:
            return candidate_anomaly_mask.astype(np.int64)

        positive_rate = float(candidate_anomaly_mask.mean())
        if positive_rate > 0.5:
            self._labels_inverted = True
            logger.info(
                "FEMA Disaster labels inverted: hand-defined "
                "'major-disaster' class covered %.1f%% of %d records "
                "(majority); flipping so the rarer event is the "
                "anomaly class (label==1) per the unsupervised "
                "convention.  See `_select_anomaly_polarity` docstring.",
                positive_rate * 100,
                n,
            )
            return (~candidate_anomaly_mask).astype(np.int64)
        return candidate_anomaly_mask.astype(np.int64)

    def _create_synthetic_disasters(self) -> bool:
        """Create synthetic disaster declaration data."""
        rng = np.random.default_rng(self.config.random_seed)
        n_samples = self.config.max_samples or 5000

        feature_rows = []
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
            if rng.random() < 0.7:
                state_fips = rng.choice(disaster_prone_states)
            else:
                state_fips = rng.integers(1, 57)

            # Date within year range
            year = rng.integers(self.year_range[0], self.year_range[1] + 1)

            # Seasonal patterns
            if state_fips in [12, 22, 48, 37, 45]:  # Hurricane states
                month = rng.choice([8, 9, 10], p=[0.3, 0.4, 0.3])
            elif state_fips in [40, 20, 1]:  # Tornado alley
                month = rng.choice([4, 5, 6], p=[0.3, 0.4, 0.3])
            elif state_fips == 6:  # California
                month = rng.choice([7, 8, 9, 10, 11], p=[0.1, 0.2, 0.2, 0.3, 0.2])
            else:
                month = rng.integers(1, 13)

            day = rng.integers(1, 29)

            # Incident type based on state
            if state_fips in [12, 22, 48, 37, 45]:
                incident_code = rng.choice([0, 1, 2])  # Hurricane, Flood, Severe Storm
            elif state_fips in [40, 20, 1]:
                incident_code = rng.choice([4, 2])  # Tornado, Severe Storm
            elif state_fips == 6:
                incident_code = rng.choice([3, 5])  # Fire, Earthquake
            else:
                incident_code = rng.integers(0, len(self.INCIDENT_TYPES))

            # Declaration type (DR is most common)
            decl_code = rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1])

            # Designated area (statewide vs county)
            area_code = 1 if rng.random() < 0.3 else 0

            # Program flags (correlated with disaster severity)
            severity = rng.beta(2, 5)  # Skewed towards less severe
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
            feature_rows.append(feature_vec)

            # Major disaster mask (DR type + multiple programs).  Final
            # label polarity is decided after the loop via
            # `_select_anomaly_polarity`, matching the real-data path.
            is_major = (
                decl_code == 0  # DR type
                and (ia_program + pa_program + hm_program) >= 2  # Multiple programs
            )
            labels.append(1 if is_major else 0)

        features = np.array(feature_rows, dtype=np.float32)
        labels_arr = np.array(labels, dtype=bool)
        labels = self._select_anomaly_polarity(labels_arr)  # type: ignore[assignment]

        save_path = self.data_path / "synthetic_fema_disaster.npz"
        np.savez_compressed(
            save_path,
            features=features,
            labels=labels,
            labels_inverted=np.array(self._labels_inverted, dtype=bool),
        )

        major_count = int(self._major_disaster_mask_from_features(features).sum())
        anomaly_count = int(labels.sum())  # type: ignore[attr-defined, unused-ignore]
        logger.info(
            f"Generated {n_samples} synthetic disaster records, "
            f"{major_count} DR multi-program major-disaster records, "
            f"{anomaly_count} anomaly labels "
            f"(labels_inverted={self._labels_inverted}, is_real_data=False)"
        )
        return True

    def _normalise_cached_labels(
        self,
        features: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
        cache_path: Any,
    ) -> np.ndarray[Any, Any]:
        """Restore v1.7.0 label polarity metadata for cached FEMA data."""
        candidate_major = self._major_disaster_mask_from_features(features)
        selected_labels = self._select_anomaly_polarity(candidate_major)
        cached_labels = labels.astype(np.int64)

        if cached_labels.shape != selected_labels.shape or not np.array_equal(
            cached_labels, selected_labels
        ):
            logger.warning(
                "FEMA disaster cache at %s had stale or non-canonical labels; "
                "using v1.7.0 minority-as-anomaly labels for this load.",
                cache_path,
            )
            return selected_labels

        # `_select_anomaly_polarity` above also restored `_labels_inverted`
        # for caches written before the metadata sidecar existed.
        return cached_labels

    def _load_raw(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Load disaster data from cache."""
        real_cache = self.data_path / "fema_disaster_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._is_real_data = True
            features = data["features"]
            labels = data["labels"]
            if "labels_inverted" in data.files:
                self._labels_inverted = bool(data["labels_inverted"])
            else:
                labels = self._normalise_cached_labels(features, labels, real_cache)
            logger.info(f"Loaded REAL FEMA disaster data from {real_cache}")
            return features, labels

        synthetic_path = self.data_path / "synthetic_fema_disaster.npz"
        if synthetic_path.exists() and ALLOW_SYNTHETIC:
            data = np.load(synthetic_path)
            self._is_real_data = False
            features = data["features"]
            labels = data["labels"]
            if "labels_inverted" in data.files:
                self._labels_inverted = bool(data["labels_inverted"])
            else:
                labels = self._normalise_cached_labels(features, labels, synthetic_path)
            logger.warning("Loaded SYNTHETIC FEMA disaster data (MERCURY_ALLOW_SYNTHETIC=1)")
            return features, labels

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

        major_mask = self._major_disaster_mask_from_features(features)
        anomaly_count = int(labels.sum())
        major_count = int(major_mask.sum())

        return {
            "n_samples": len(features),
            "n_major_disasters": major_count,
            "major_disaster_ratio": float(major_mask.mean()),
            "n_anomaly_labels": anomaly_count,
            "anomaly_label_ratio": float(labels.mean()),
            "labels_inverted": self._labels_inverted,
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

    Note:
        This dataset does not include ground-truth anomaly labels.
        Use for unsupervised deployment only.
    """

    DATASET_NAME = "fema_hazard_mitigation"
    LABEL_SOURCE = "statistical"  # no ground-truth anomaly labels; heuristic polarity selection
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
        """Is real data."""
        return self._is_real_data

    def download(self) -> bool:
        """Download hazard mitigation data from OpenFEMA API."""
        if self._download_from_fema():
            return True
        if ALLOW_SYNTHETIC:
            check_synthetic_allowed("FEMAHazardMitigation", "OpenFEMA Hazard Mitigation API failed")
            return self._create_synthetic_mitigation()
        raise DataSourceUnavailableError(
            loader_name="FEMAHazardMitigation",
            source_url="https://www.fema.gov/api/open/v2/HazardMitigationGrants",
            reason="OpenFEMA Hazard Mitigation API unavailable",
        )

    _OPENFEMA_PAGE_SIZE = 1000

    def _download_from_fema(self) -> bool:
        """Download hazard mitigation grants from OpenFEMA API with pagination."""
        import urllib.parse

        from omni_mercury_engine.security.input_validation import TrustedEndpoints

        from .base import http_get_with_retry

        dataset_dir = self.data_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "fema_hazard_mitigation_real.npz"

        if cache_file.exists():
            logger.info(f"FEMA hazard mitigation data already cached at {cache_file}")
            self._is_real_data = True
            return True

        try:
            api_url = TrustedEndpoints.FEMA_HAZARD_MITIGATION
            TrustedEndpoints.validate_url(api_url)

            target = self.config.max_samples or 5000
            page_size = self._OPENFEMA_PAGE_SIZE

            all_records: list[dict[str, Any]] = []
            skip = 0
            while len(all_records) < target:
                page_top = min(page_size, target - len(all_records))
                # OpenFEMA HazardMitigationGrants v2 date field is
                # "lastRefresh" (ISO-8601). Avoid $orderby on optional fields
                # that may be absent in some records — sort by id for safety.
                params = {
                    "$top": str(page_top),
                    "$skip": str(skip),
                    "$orderby": "id asc",
                    "$format": "json",
                }
                url = f"{api_url}?{urllib.parse.urlencode(params)}"
                logger.info(
                    "OpenFEMA HMG page: $skip=%d $top=%d (collected %d/%d)",
                    skip,
                    page_top,
                    len(all_records),
                    target,
                )
                content = http_get_with_retry(url, timeout=120)
                page_data = json.loads(content.decode("utf-8"))
                # OpenFEMA v2 wraps records under the dataset name; fall back
                # to a top-level list if the key is absent (API version drift).
                page = page_data.get("HazardMitigationGrants", page_data if isinstance(page_data, list) else [])
                if not page:
                    break
                all_records.extend(page)
                if len(page) < page_top:
                    break
                skip += len(page)

            if not all_records:
                logger.warning("No hazard mitigation records returned from FEMA API")
                return False

            logger.info(f"Downloaded {len(all_records)} hazard mitigation records")
            features, labels = self._process_mitigation_data(all_records)

            np.savez_compressed(cache_file, features=features, labels=labels)
            self._is_real_data = True

            logger.info(
                f"FEMA hazard mitigation data loaded: {len(features)} samples, "
                f"{labels.sum()} high-value projects (is_real_data=True)"
            )
            return True

        except Exception as e:
            logger.warning(f"OpenFEMA Hazard Mitigation API download failed: {e}")
            return False

    def _process_mitigation_data(
        self, records: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process OpenFEMA hazard mitigation grant records."""
        rows = []

        for record in records:
            project_amount = float(record.get("projectAmount", 0) or 0)
            federal_share = float(record.get("federalShareObligated", 0) or 0)

            state_str = record.get("state", "0")
            try:
                state_fips = int(state_str) if state_str and state_str.isdigit() else 0
            except (ValueError, TypeError):
                state_fips = 0

            date_str = record.get("dateApproved", "")
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                year = dt.year
            except (ValueError, AttributeError):
                year = 2000

            project_type_code = hash(record.get("projectType", "")) % 10
            status_code = hash(record.get("status", "")) % 3
            program_type_code = hash(record.get("programArea", "")) % 3

            row = [
                project_amount,
                federal_share,
                state_fips,
                year,
                project_type_code,
                status_code,
                program_type_code,
            ]
            rows.append(row)

        features = np.array(rows, dtype=np.float32)
        labels = (features[:, 0] > 500000).astype(np.int64)
        return features, labels

    def _create_synthetic_mitigation(self) -> bool:
        """Create synthetic hazard mitigation project data."""
        rng = np.random.default_rng(self.config.random_seed)
        n_samples = self.config.max_samples or 3000

        features = []
        labels = []

        project_types = ["Acquisition", "Elevation", "Floodproofing", "Drainage", "Wildfire"]

        for _ in range(n_samples):
            # Project cost (log-normal distribution)
            project_amount = rng.lognormal(12, 1.5)  # Mean ~$100K

            # Federal share (typically 75%)
            federal_share = project_amount * rng.uniform(0.7, 0.9)

            state_fips = rng.integers(1, 57)
            year = rng.integers(self.year_range[0], self.year_range[1] + 1)

            project_type_code = rng.integers(0, len(project_types))
            status_code = rng.choice([0, 1, 2], p=[0.2, 0.3, 0.5])  # pending, active, complete
            program_type_code = rng.integers(0, 3)  # HMGP, PDM, FMA

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
        real_cache = self.data_path / "fema_hazard_mitigation_real.npz"
        if real_cache.exists():
            data = np.load(real_cache)
            self._is_real_data = True
            logger.info(f"Loaded REAL FEMA hazard mitigation data from {real_cache}")
            return data["features"], data["labels"]

        synthetic_path = self.data_path / "synthetic_hazard_mitigation.npz"
        if synthetic_path.exists() and ALLOW_SYNTHETIC:
            data = np.load(synthetic_path)
            self._is_real_data = False
            logger.warning("Loaded SYNTHETIC hazard mitigation data (MERCURY_ALLOW_SYNTHETIC=1)")
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
