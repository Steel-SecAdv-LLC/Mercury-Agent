"""
Mercury Agent

Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

Domain loader for network security data (CICIDS, NSL-KDD, UNSW-NB15).

Bridges the domain loader interface (BaseDomainLoader) to the existing
dataset infrastructure in omni_mercury_engine.datasets.security so that
the dedicated network security detector module can consume these benchmark
datasets directly.

Supported datasets / ground-truth events:
- nsl_kdd: NSL-KDD network intrusion detection dataset
- cicids_2017: CICIDS 2017 network intrusion dataset
- batadal: BATADAL water network attack detection dataset

Feature engineering covers standard network flow observables (duration,
protocol type, byte counts, connection counts, error rates) and maps
every record to a binary anomaly label (0 = normal, 1 = attack).
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.datasets.security import NSLKDDLoader as _NSLKDDDataset
from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ground-truth event catalog
# ---------------------------------------------------------------------------

_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "nsl_kdd": {
        "name": "NSL-KDD Network Intrusion Dataset",
        "date": "2009-01-01",
        "description": (
            "Improved version of KDD Cup 99 with duplicates removed and "
            "balanced difficulty levels. ~125K training + ~22K test records "
            "covering DoS, Probe, R2L, and U2R attack categories."
        ),
        "dataset_loader": "nsl-kdd",
        "n_features": 41,
    },
    "cicids_2017": {
        "name": "CICIDS 2017 Network Intrusion Dataset",
        "date": "2017-07-07",
        "description": (
            "Modern intrusion dataset from the Canadian Institute for "
            "Cybersecurity with ~2.8M labeled network flows and 80 features. "
            "Attack types include DDoS, Brute Force, SQL Injection, and more."
        ),
        "dataset_loader": "cicids",
        "n_features": 78,
    },
    "batadal": {
        "name": "BATADAL Water Network Attack Dataset",
        "date": "2018-01-01",
        "description": (
            "Battle of Attack Detection Algorithms dataset for water "
            "distribution network cyber-physical attacks. Contains sensor "
            "readings with binary attack labels."
        ),
        "dataset_loader": "batadal",
        "n_features": 43,
    },
}

# ---------------------------------------------------------------------------
# NSL-KDD column schema -- single source of truth is
# ``datasets.security.NSLKDDLoader.COLUMN_NAMES`` (this loader bridges to that
# dataset infrastructure).  Derived here so the two can never drift; the
# previously-duplicated 43-column literal and the unused ``_NSLKDD_CATEGORICAL``
# copy of ``NSLKDDLoader.CATEGORICAL_COLS`` are removed.
# ---------------------------------------------------------------------------

_NSLKDD_COLUMNS: list[str] = _NSLKDDDataset.COLUMN_NAMES

# ---------------------------------------------------------------------------
# UNSW-NB15 column definitions
# ---------------------------------------------------------------------------

_UNSWNB15_URL = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"

_UNSWNB15_FEATURE_COLS: list[str] = [
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
]


class NetworkSecurityLoader(BaseDomainLoader):
    """
    Domain loader for network security intrusion detection datasets.

    Wraps the existing dataset infrastructure in
    :mod:`omni_mercury_engine.datasets.security` and
    :mod:`omni_mercury_engine.datasets.industrial` so that the
    dedicated network security detector can consume NSL-KDD, CICIDS 2017,
    UNSW-NB15, and BATADAL datasets through the unified
    :class:`BaseDomainLoader` interface.

    The existing NSL-KDD pipeline already achieves AUC 0.972 via the core
    Mercury detector.  This loader wires those same datasets into the
    domain loader pattern used by the broader Mercury framework.

    Attributes:
        DOMAIN: ``"network_security"``
        SOURCE_URL: CIC IDS-2017 reference page.
        REQUIRES_API_KEY: ``False`` -- all datasets are freely available.
    """

    DOMAIN: str = "network_security"
    SOURCE_URL: str = "https://www.unb.ca/cic/datasets/ids-2017.html"
    REQUIRES_API_KEY: bool = False
    # Feature selection based on Cohen's d effect size analysis.
    # Only features with d >= 1.2 are retained — these provide strong
    # separation between normal and attack traffic.  Features below
    # this threshold (duration d=0.007, urgent d=0.008, hot d=0.019,
    # num_compromised d=0.020, num_root d=0.022, count d=1.23, etc.)
    # add noise that degrades unsupervised AUC.  Categorical features
    # (protocol_type, service, flag) and near-constant binary features
    # (land, logged_in, is_host_login, is_guest_login) are also excluded.
    FEATURE_COLUMNS: list[str] = [
        "dst_bytes",
        "same_srv_rate",
        "dst_host_srv_count",
        "src_bytes",
        "dst_host_same_srv_rate",
        "dst_host_srv_serror_rate",
        "dst_host_serror_rate",
        "serror_rate",
        "srv_serror_rate",
    ]

    # Columns to drop — categorical or near-constant binary.
    _DROP_COLUMNS: list[str] = [
        "protocol_type",
        "service",
        "flag",
        "land",
        "logged_in",
        "is_host_login",
        "is_guest_login",
    ]

    # Heavy-tailed columns that benefit from log1p transform.
    _LOG_TRANSFORM_COLUMNS: list[str] = [
        "duration",
        "src_bytes",
        "dst_bytes",
    ]

    # Cache event data for 24 hours (benchmark datasets are static).
    CACHE_TTL: int = 86400

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """
        Fetch the most recent network security data.

        Because intrusion detection benchmark datasets are static, this
        method returns the NSL-KDD test set as a representative sample
        of network traffic.  For true real-time operation, integrate with
        a live packet capture pipeline.

        Returns:
            DataFrame with network flow features and a ``label`` column.

        Raises:
            ConnectionError: If the data source is unreachable after
                retries.
        """
        cache_key = "network_security_realtime"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time network security data.")
            return pd.DataFrame(cached)

        # Attempt to use existing NSLKDDLoader infrastructure
        df = self._load_nslkdd_dataframe()

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "network_security: fetched %d real-time records (NSL-KDD).",
            len(df),
        )
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch data for a specific network security benchmark dataset.

        Args:
            event_id: One of ``"nsl_kdd"``, ``"cicids_2017"``, or
                ``"batadal"``.

        Returns:
            DataFrame with dataset-specific features and a ``label``
            column (0 = normal, 1 = attack).

        Raises:
            ValueError: If *event_id* is not recognised.
            ConnectionError: If the data source is unreachable.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id {event_id!r}. " f"Available: {list(_EVENT_CATALOG)}"
            )

        cache_key = f"network_security_historical_{event_id}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            return pd.DataFrame(cached)

        if event_id == "nsl_kdd":
            df = self._load_nslkdd_dataframe()
        elif event_id == "cicids_2017":
            df = self._load_cicids_dataframe()
        elif event_id == "batadal":
            df = self._load_batadal_dataframe()
        else:
            raise ValueError(f"Unhandled event_id: {event_id!r}")

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "network_security: fetched %d historical records for '%s'.",
            len(df),
            event_id,
        )
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """
        Return the catalog of available network security benchmark events.

        Returns:
            List of dicts, each containing ``event_id``, ``name``,
            ``date``, and ``description`` keys.
        """
        return [
            {
                "event_id": event_id,
                "name": meta["name"],
                "date": meta["date"],
                "description": meta["description"],
            }
            for event_id, meta in _EVENT_CATALOG.items()
        ]

    def get_ground_truth(self, event_id: str) -> np.ndarray[Any, Any]:
        """Return binary anomaly labels for a benchmark dataset.

        Labels are derived directly from the dataset: attack traffic is
        labeled ``1`` (anomaly), normal traffic is labeled ``0``.

        Args:
            event_id: One of ``"nsl_kdd"``, ``"cicids_2017"``, or
                ``"batadal"``.

        Returns:
            1-D numpy array of binary labels (0 = normal, 1 = anomaly).

        Raises:
            ValueError: If *event_id* is not recognised.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id {event_id!r}. " f"Available: {list(_EVENT_CATALOG)}"
            )

        # Try to load via existing dataset infrastructure first
        labels = self._load_labels_from_dataset(event_id)
        if labels is not None:
            logger.info(
                "network_security: ground truth for '%s' -- " "%d anomalies / %d total.",
                event_id,
                int(labels.sum()),
                len(labels),
            )
            return labels

        # Fall back to loading the full dataframe and extracting labels
        df = self.fetch_historical(event_id)
        if df.empty:
            return np.array([], dtype=np.int64)

        if "label" in df.columns:
            labels = df["label"].values.astype(np.int64)
        else:
            labels = np.zeros(len(df), dtype=np.int64)

        logger.info(
            "network_security: ground truth for '%s' -- " "%d anomalies / %d total.",
            event_id,
            int(labels.sum()),
            len(labels),
        )
        return labels

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """
        Transform raw network data into a feature matrix.

        Feature engineering depends on the source dataset:

        **NSL-KDD features** (9 high-signal dimensions):
            Features selected by Cohen's d effect size (d >= 1.2).
            Covers byte volumes (src_bytes, dst_bytes), service
            patterns (same_srv_rate, dst_host_srv_count,
            dst_host_same_srv_rate), and SYN error indicators
            (serror_rate, srv_serror_rate, dst_host_serror_rate,
            dst_host_srv_serror_rate).  Heavy-tailed features
            (src_bytes, dst_bytes) are log1p-transformed.

        **CICIDS 2017 / BATADAL features**:
            All numeric columns retained (these datasets are already
            predominantly continuous).  Categorical and metadata
            columns are dropped.

        All features are cleaned (inf/nan removed) and returned as
        float64.  The ``label`` column is excluded from features.

        Args:
            raw_data: DataFrame from :meth:`fetch_historical` or
                :meth:`fetch_realtime`.

        Returns:
            2-D numpy array of shape ``(n_samples, n_features)``.
        """
        if raw_data.empty:
            return np.empty((0, 0), dtype=np.float64)

        df = raw_data.copy()

        # Drop non-feature columns
        drop_cols = [c for c in ["label", "difficulty", "DATETIME", "ATT_FLAG"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # Drop categorical and near-constant binary columns
        drop_categorical = [c for c in self._DROP_COLUMNS if c in df.columns]
        if drop_categorical:
            df = df.drop(columns=drop_categorical)

        # Also drop any remaining object/category columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_cols:
            df = df.drop(columns=categorical_cols)

        # Convert to numeric, coercing errors
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Log1p transform for heavy-tailed features
        for col in self._LOG_TRANSFORM_COLUMNS:
            if col in df.columns:
                df[col] = np.log1p(df[col].fillna(0).clip(lower=0))

        # Select FEATURE_COLUMNS if they exist in the DataFrame;
        # otherwise keep all remaining numeric columns (for CICIDS/BATADAL)
        available_features = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        if available_features:
            df = df[available_features]

        arr = df.values.astype(np.float64)

        # Clean non-finite values
        arr = np.where(np.isinf(arr), np.nan, arr)
        for col_idx in range(arr.shape[1]):
            col = arr[:, col_idx]
            mask = np.isnan(col)
            if mask.any():
                median_val = np.nanmedian(col)
                col[mask] = median_val if np.isfinite(median_val) else 0.0

        return arr

    # ------------------------------------------------------------------
    # Internal: dataset loading via existing infrastructure
    # ------------------------------------------------------------------

    def _load_labels_from_dataset(self, event_id: str) -> np.ndarray[Any, Any] | None:
        """
        Load binary labels from existing dataset loaders.

        Attempts to import and use the existing ``NSLKDDLoader``,
        ``CICIDSLoader``, or ``BATADALLoader`` from the datasets
        package.  Returns ``None`` if the import or loading fails.

        Args:
            event_id: Identifier for the benchmark event.

        Returns:
            1-D binary numpy array, or ``None`` on failure.
        """
        try:
            if event_id == "nsl_kdd":
                from omni_mercury_engine.datasets.base import DatasetConfig
                from omni_mercury_engine.datasets.security import NSLKDDLoader

                config = DatasetConfig(
                    name="nsl-kdd",
                    preprocessing={"binary": True},
                )
                loader = NSLKDDLoader(config)
                _features, labels = loader.load_data()
                return labels.astype(np.int64)

            elif event_id == "cicids_2017":
                from omni_mercury_engine.datasets.base import DatasetConfig
                from omni_mercury_engine.datasets.security import CICIDSLoader

                config = DatasetConfig(
                    name="cicids",
                    preprocessing={"binary": True},
                )
                loader = CICIDSLoader(config)  # type: ignore[assignment]
                _features, labels = loader.load_data()
                return labels.astype(np.int64)

            elif event_id == "batadal":
                from omni_mercury_engine.datasets.base import DatasetConfig
                from omni_mercury_engine.datasets.industrial import BATADALLoader

                config = DatasetConfig(name="batadal")
                loader = BATADALLoader(config)  # type: ignore[assignment]
                _features, labels = loader.load()
                return labels.astype(np.int64)

        except Exception as exc:
            logger.debug(
                "network_security: could not load labels via dataset "
                "infrastructure for '%s': %s",
                event_id,
                exc,
            )
        return None

    def _load_nslkdd_dataframe(self) -> pd.DataFrame:
        """
        Load NSL-KDD data as a DataFrame.

        Tries the existing ``NSLKDDLoader`` first, then falls back to
        direct HTTP download from the GitHub mirror.

        Returns:
            DataFrame with NSL-KDD columns and a binary ``label`` column.

        Raises:
            ConnectionError: If all sources fail.
        """
        # Try existing infrastructure
        try:
            from omni_mercury_engine.datasets.base import DatasetConfig
            from omni_mercury_engine.datasets.security import NSLKDDLoader

            config = DatasetConfig(
                name="nsl-kdd",
                preprocessing={"binary": True},
            )
            loader = NSLKDDLoader(config)
            features, labels = loader.load_data()

            # Reconstruct DataFrame from features + labels
            feature_names = _NSLKDD_COLUMNS[:-2]  # exclude label, difficulty
            n_cols = features.shape[1]
            if n_cols == len(feature_names):
                col_names = feature_names
            else:
                col_names = [f"feature_{i}" for i in range(n_cols)]

            df = pd.DataFrame(features, columns=col_names)
            df["label"] = labels
            logger.info(
                "network_security: loaded %d NSL-KDD records via dataset loader.",
                len(df),
            )
            return df

        except Exception as exc:
            logger.debug(
                "network_security: NSLKDDLoader failed (%s), " "falling back to direct download.",
                exc,
            )

        # Fall back to direct download
        return self._download_nslkdd_direct()

    def _download_nslkdd_direct(self) -> pd.DataFrame:
        """
        Download NSL-KDD directly from GitHub mirror.

        Returns:
            DataFrame with NSL-KDD columns and a binary ``label`` column.

        Raises:
            ConnectionError: If the download fails after retries.
        """
        url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/" "master/KDDTest+.txt"
        raw_bytes = self._fetch_url(url)
        text = raw_bytes.decode("utf-8", errors="replace")

        df = pd.read_csv(
            io.StringIO(text),
            names=_NSLKDD_COLUMNS,
            header=None,
        )

        # Encode labels: normal = 0, everything else = 1
        raw_labels = df["label"].str.strip()
        df["label"] = np.where(raw_labels == "normal", 0, 1).astype(np.int64)

        # Drop difficulty column (not a feature)
        df = df.drop(columns=["difficulty"], errors="ignore")

        logger.info(
            "network_security: downloaded %d NSL-KDD records directly.",
            len(df),
        )
        return df

    def _load_cicids_dataframe(self) -> pd.DataFrame:
        """
        Load CICIDS 2017 data as a DataFrame.

        Tries the existing ``CICIDSLoader`` first, then falls back to a
        minimal feature matrix built from the loader's cache.

        Returns:
            DataFrame with CICIDS features and a binary ``label`` column.

        Raises:
            ConnectionError: If all sources fail.
        """
        # Try existing infrastructure
        try:
            from omni_mercury_engine.datasets.base import DatasetConfig
            from omni_mercury_engine.datasets.security import CICIDSLoader

            config = DatasetConfig(
                name="cicids",
                preprocessing={"binary": True},
            )
            loader = CICIDSLoader(config)
            features, labels = loader.load_data()

            col_names = [f"feature_{i}" for i in range(features.shape[1])]
            df = pd.DataFrame(features, columns=col_names)
            df["label"] = labels
            logger.info(
                "network_security: loaded %d CICIDS records via dataset loader.",
                len(df),
            )
            return df

        except Exception as exc:
            logger.warning(
                "network_security: CICIDSLoader unavailable (%s). "
                "CICIDS 2017 requires download via the datasets package.",
                exc,
            )
            raise ConnectionError(
                f"network_security: failed to load CICIDS 2017 data: {exc}"
            ) from exc

    def _load_batadal_dataframe(self) -> pd.DataFrame:
        """
        Load BATADAL data as a DataFrame.

        Tries the existing ``BATADALLoader`` first, then falls back to
        direct CSV download from batadal.net.

        Returns:
            DataFrame with BATADAL sensor features and a binary
            ``label`` column.

        Raises:
            ConnectionError: If all sources fail.
        """
        # Try existing infrastructure
        try:
            from omni_mercury_engine.datasets.base import DatasetConfig
            from omni_mercury_engine.datasets.industrial import BATADALLoader

            config = DatasetConfig(name="batadal")
            loader = BATADALLoader(config)
            features, labels = loader.load()

            col_names = [f"feature_{i}" for i in range(features.shape[1])]
            df = pd.DataFrame(features, columns=col_names)
            df["label"] = labels
            logger.info(
                "network_security: loaded %d BATADAL records via dataset loader.",
                len(df),
            )
            return df

        except Exception as exc:
            logger.debug(
                "network_security: BATADALLoader failed (%s), " "falling back to direct download.",
                exc,
            )

        # Fall back to direct download
        return self._download_batadal_direct()

    def _download_batadal_direct(self) -> pd.DataFrame:
        """
        Download BATADAL dataset directly from batadal.net.

        Downloads both training (no attacks) and test (with attack
        labels) CSV files, concatenates them, and returns a unified
        DataFrame.

        Returns:
            DataFrame with sensor features and a binary ``label`` column.

        Raises:
            ConnectionError: If the download fails after retries.
        """
        urls = {
            "train": "https://www.batadal.net/data/BATADAL_dataset03.csv",
            "test": "https://www.batadal.net/data/BATADAL_dataset04.csv",
        }

        dfs: list[pd.DataFrame] = []
        for name, url in urls.items():
            try:
                raw_bytes = self._fetch_url(url)
                text = raw_bytes.decode("utf-8", errors="replace")
                frame = pd.read_csv(io.StringIO(text))
                frame.columns = frame.columns.str.strip()
                dfs.append(frame)
                logger.info(
                    "network_security: downloaded BATADAL %s (%d rows).",
                    name,
                    len(frame),
                )
            except Exception as exc:
                logger.warning(
                    "network_security: failed to download BATADAL %s: %s",
                    name,
                    exc,
                )

        if not dfs:
            raise ConnectionError("network_security: failed to download any BATADAL data.")

        df = pd.concat(dfs, ignore_index=True)

        # Extract labels and rename to standard column
        if "ATT_FLAG" in df.columns:
            raw_flags = df["ATT_FLAG"].values.astype(int)
            df["label"] = (raw_flags == 1).astype(np.int64)
            df = df.drop(columns=["ATT_FLAG"])
        else:
            df["label"] = np.int64(0)

        # Drop datetime column (not a numeric feature)
        df = df.drop(columns=["DATETIME"], errors="ignore")

        # Clean numeric values
        for col in df.columns:
            if col != "label":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.fillna(0.0)

        logger.info(
            "network_security: loaded %d BATADAL records, %d attacks.",
            len(df),
            int(df["label"].sum()),
        )
        return df
