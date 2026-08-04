# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Domain loader for sepsis / critical care data from PhysioNet.

Connects to the PhysioNet / Computing in Cardiology Challenge 2019 dataset for early prediction of
sepsis from clinical data.  The challenge dataset is openly available (no credentials required) and
ships per-patient files in PSV (pipe-separated values) format with a ``SepsisLabel`` column
indicating sepsis onset.

Training sets A and B are exposed as separate ground truth events.  Feature engineering produces
vital-sign dynamics and laboratory time-series features suitable for the Mercury KinematicScore
anomaly detector.

Note: MIMIC-III requires credentialed access.  This loader exclusively uses the OPEN PhysioNet
Challenge 2019 dataset to avoid credential requirements. If PhysioNet is unreachable,
``DataSourceUnavailableError`` is raised.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.loaders.base import BaseDomainLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PhysioNet Challenge 2019 endpoints
# ---------------------------------------------------------------------------
# PhysioNet serves the training sets as DIRECTORIES of per-patient ``p*.psv``
# files, one file per ICU stay -- there is no ``training_setA.zip`` on the
# server (the previous zip URLs returned HTTP 404 on every fetch, so this
# loader's download path was a dead wire). The transport lists the directory
# index once, then fetches individual PSV files up to ``max_patients``.
_BASE_DATA_URL = "https://physionet.org/files/challenge-2019/1.0.0/"
_TRAINING_A_URL = f"{_BASE_DATA_URL}training/training_setA/"
_TRAINING_B_URL = f"{_BASE_DATA_URL}training/training_setB/"

# ---------------------------------------------------------------------------
# Canonical column names from the challenge PSV files
# ---------------------------------------------------------------------------
_VITAL_SIGN_COLS: list[str] = [
    "HR",  # Heart rate (beats/min)
    "O2Sat",  # Pulse oximetry SpO2 (%)
    "Temp",  # Temperature (deg C)
    "SBP",  # Systolic blood pressure (mmHg)
    "MAP",  # Mean arterial pressure (mmHg)
    "DBP",  # Diastolic blood pressure (mmHg)
    "Resp",  # Respiration rate (breaths/min)
]

_LAB_VALUE_COLS: list[str] = [
    "WBC",  # White blood cell count (10^3/uL)
    "Lactate",  # Lactate (mmol/L)
    "Creatinine",  # Creatinine (mg/dL)
    "Platelets",  # Platelets (10^3/uL)
    "Bilirubin_total",  # Total bilirubin (mg/dL)
]

_SOFA_COLS: list[str] = [
    "FiO2",  # Fraction of inspired oxygen (%)
    "pH",  # Arterial pH
    "PaCO2",  # Partial pressure of CO2 (mmHg)
    "SaO2",  # Arterial oxygen saturation (%)
    "BUN",  # Blood urea nitrogen (mg/dL)
]

_DEMOGRAPHIC_COLS: list[str] = [
    "Age",
    "Gender",
    "HospAdmTime",
    "ICULOS",  # ICU length of stay (hours)
]

_LABEL_COL = "SepsisLabel"

# All feature columns used in engineer_features (vital signs + labs + SOFA)
_FEATURE_COLS: list[str] = _VITAL_SIGN_COLS + _LAB_VALUE_COLS + _SOFA_COLS

# ---------------------------------------------------------------------------
# Ground truth event catalog
# ---------------------------------------------------------------------------
_EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "physionet_challenge_2019_A": {
        "name": "PhysioNet Challenge 2019 - Training Set A",
        "date": "2019-02-08",
        "description": (
            "Training set A from the PhysioNet/CinC Challenge 2019: Early "
            "Prediction of Sepsis from Clinical Data.  Contains ~20,000 ICU "
            "patient records with hourly vital signs, lab values, and a binary "
            "SepsisLabel column indicating sepsis onset."
        ),
        "url": _TRAINING_A_URL,
        "set_label": "A",
    },
    "physionet_challenge_2019_B": {
        "name": "PhysioNet Challenge 2019 - Training Set B",
        "date": "2019-02-08",
        "description": (
            "Training set B from the PhysioNet/CinC Challenge 2019: Early "
            "Prediction of Sepsis from Clinical Data.  Contains ~20,000 ICU "
            "patient records from a second hospital system with the same "
            "schema as set A."
        ),
        "url": _TRAINING_B_URL,
        "set_label": "B",
    },
}

# Maximum number of patient files to load per training set.  The full
# datasets are very large; cap at a reasonable number for benchmarking.
_MAX_PATIENTS_DEFAULT = 500


class SepsisLoader(BaseDomainLoader):
    """Loader for sepsis / critical care data from PhysioNet Challenge 2019.

    Uses the openly available PhysioNet/Computing in Cardiology Challenge 2019
    dataset which provides per-patient ICU records in PSV (pipe-separated
    values) format.  Each file contains hourly observations of vital signs,
    lab values, demographics, and a ``SepsisLabel`` column (binary: 0 = no
    sepsis, 1 = sepsis onset).

    Two ground truth events are exposed:

    * ``physionet_challenge_2019_A`` -- Training set A
    * ``physionet_challenge_2019_B`` -- Training set B

    Feature engineering focuses on vital-sign dynamics and derivatives,
    which are ideally suited for the Mercury KinematicScore detector:

    * Raw vital signs and lab values (forward-filled, then zero-filled)
    * First-order derivatives (rate of change per hour)
    * Rolling standard deviation (variability over a 6-hour window)
    * Interaction terms between heart rate and mean arterial pressure

    No credentials are required.  If PhysioNet is unreachable,
    :class:`DataSourceUnavailableError` is raised.
    """

    DOMAIN: str = "sepsis"
    SOURCE_URL: str = "https://physionet.org/content/challenge-2019/"
    # Labels come directly from the ``SepsisLabel`` column in the PhysioNet
    # Challenge 2019 PSV files — a clinician-derived ground truth annotation
    # independent of any scored vital-sign feature.
    LABEL_SOURCE: str = "ground_truth"
    REQUIRES_API_KEY: bool = False
    FEATURE_COLUMNS: list[str] = [
        # Raw vital signs (7)
        "HR",
        "O2Sat",
        "Temp",
        "SBP",
        "MAP",
        "DBP",
        "Resp",
        # Raw lab values (5)
        "WBC",
        "Lactate",
        "Creatinine",
        "Platelets",
        "Bilirubin_total",
        # Raw SOFA-related (5)
        "FiO2",
        "pH",
        "PaCO2",
        "SaO2",
        "BUN",
        # First derivatives (17)
        "d_HR",
        "d_O2Sat",
        "d_Temp",
        "d_SBP",
        "d_MAP",
        "d_DBP",
        "d_Resp",
        "d_WBC",
        "d_Lactate",
        "d_Creatinine",
        "d_Platelets",
        "d_Bilirubin_total",
        "d_FiO2",
        "d_pH",
        "d_PaCO2",
        "d_SaO2",
        "d_BUN",
        # Rolling standard deviation (17)
        "std_HR",
        "std_O2Sat",
        "std_Temp",
        "std_SBP",
        "std_MAP",
        "std_DBP",
        "std_Resp",
        "std_WBC",
        "std_Lactate",
        "std_Creatinine",
        "std_Platelets",
        "std_Bilirubin_total",
        "std_FiO2",
        "std_pH",
        "std_PaCO2",
        "std_SaO2",
        "std_BUN",
        # Interaction terms (2)
        "hr_map_product",
        "resp_o2sat_product",
    ]

    #: Cache TTL -- 24 hours (challenge data is static)
    CACHE_TTL: int = 86400

    def __init__(
        self,
        *args: Any,
        max_patients: int = _MAX_PATIENTS_DEFAULT,
        **kwargs: Any,
    ) -> None:
        """Initialize the sepsis loader.

        Args:
            *args: Positional arguments forwarded to
                :class:`BaseDomainLoader`.
            max_patients: Maximum number of patient files to parse per
                training set.  Use ``0`` or a negative value to load all
                available files (may be slow).
            **kwargs: Keyword arguments forwarded to
                :class:`BaseDomainLoader`.
        """
        super().__init__(*args, **kwargs)
        self.max_patients: int = max_patients if max_patients > 0 else 0

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def fetch_realtime(self) -> pd.DataFrame:
        """Fetch the most recent available sepsis data.

        The PhysioNet Challenge 2019 dataset is static (not a live feed),
        so this method returns a sample from Training Set A as a proxy for
        "live" data.  In a production setting this would connect to an
        EHR system or hospital data warehouse.

        Returns:
            DataFrame with vital signs, lab values, demographics, and the
            ``SepsisLabel`` column.

        Raises:
            DataSourceUnavailableError: If PhysioNet is unreachable.
        """
        cache_key = "sepsis_realtime"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached real-time sepsis data.")
            return pd.DataFrame(cached)

        # Use a small slice of training set A as a proxy for real-time data.
        df = self._download_and_parse_training_set(
            url=_TRAINING_A_URL,
            set_label="A",
            max_patients=min(self.max_patients or 50, 50),
        )

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "Fetched %d real-time sepsis records (%d patients) from PhysioNet.",
            len(df),
            df["patient_id"].nunique() if "patient_id" in df.columns else 0,
        )
        return df

    def fetch_historical(self, event_id: str) -> pd.DataFrame:
        """Fetch data for a specific PhysioNet Challenge 2019 training set.

        Args:
            event_id: One of ``"physionet_challenge_2019_A"`` or
                ``"physionet_challenge_2019_B"``.

        Returns:
            DataFrame with vital signs, lab values, demographics,
            ``SepsisLabel``, and ``patient_id`` columns.

        Raises:
            ValueError: If *event_id* is not recognized.
            DataSourceUnavailableError: If PhysioNet is unreachable.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        cache_key = f"sepsis_historical_{event_id}"
        cached = self._read_cache(cache_key)

        if cached is not None:
            logger.debug("Returning cached historical data for '%s'.", event_id)
            return pd.DataFrame(cached)

        event = _EVENT_CATALOG[event_id]
        df = self._download_and_parse_training_set(
            url=event["url"],
            set_label=event["set_label"],
            max_patients=self.max_patients,
        )

        if df.empty:
            logger.warning("PhysioNet returned no data for event '%s'.", event_id)
            return df

        self._write_cache(cache_key, df.to_dict(orient="list"))
        logger.info(
            "Fetched %d historical records (%d patients) for event '%s'.",
            len(df),
            df["patient_id"].nunique() if "patient_id" in df.columns else 0,
            event_id,
        )
        return df

    def list_events(self) -> list[dict[str, Any]]:
        """Return the catalog of ground truth sepsis events.

        Returns:
            List of dicts each containing *event_id*, *name*, *date*,
            and *description* keys.
        """
        events: list[dict[str, Any]] = []
        for event_id, meta in _EVENT_CATALOG.items():
            events.append(
                {
                    "event_id": event_id,
                    "name": meta["name"],
                    "date": meta["date"],
                    "description": meta["description"],
                }
            )
        return events

    def get_ground_truth(self, event_id: str) -> np.ndarray[Any, Any]:
        """Return binary sepsis onset labels for a training set.

        Labels come directly from the ``SepsisLabel`` column in the
        PhysioNet Challenge 2019 PSV files.  Each hourly observation
        is labeled ``1`` at and after the point of sepsis onset, and
        ``0`` otherwise.

        Args:
            event_id: Key into the ground truth catalog.

        Returns:
            1-D binary numpy array of shape ``(n_observations,)`` where
            ``1`` indicates sepsis onset and ``0`` indicates no sepsis.

        Raises:
            ValueError: If *event_id* is not recognized or no data is
                available.
        """
        if event_id not in _EVENT_CATALOG:
            raise ValueError(
                f"Unknown event_id '{event_id}'. " f"Available: {list(_EVENT_CATALOG.keys())}"
            )

        df = self.fetch_historical(event_id)
        if df.empty:
            return np.array([], dtype=np.int64)

        if _LABEL_COL not in df.columns:
            logger.warning(
                "SepsisLabel column not found in data for event '%s'. "
                "Returning all-zero labels.",
                event_id,
            )
            return np.zeros(len(df), dtype=np.int64)

        labels = df[_LABEL_COL].fillna(0).values.astype(np.int64)
        n_positive = int(labels.sum())
        logger.info(
            "Ground truth for '%s': %d sepsis-positive / %d total observations "
            "(%.1f%% positive rate).",
            event_id,
            n_positive,
            len(labels),
            100.0 * n_positive / max(len(labels), 1),
        )
        return np.asarray(labels)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, raw_data: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Transform raw sepsis data into a feature matrix for Mercury.

        Engineered features per hourly observation:

        1. **Raw vital signs** (7): HR, O2Sat, Temp, SBP, MAP, DBP, Resp
        2. **Raw lab values** (5): WBC, Lactate, Creatinine, Platelets,
           Bilirubin_total
        3. **SOFA-related** (5): FiO2, pH, PaCO2, SaO2, BUN
        4. **First derivatives** (17): Hourly rate of change for each
           vital sign, lab value, and SOFA-related measure
        5. **Rolling std** (17): 6-hour rolling standard deviation for
           each vital sign, lab value, and SOFA-related measure
        6. **Interaction terms** (2): HR*MAP product, Resp*O2Sat product

        Total: 17 raw + 17 derivatives + 17 rolling_std + 2 interactions = 53

        Missing values are forward-filled per patient, then remaining NaN
        values are filled with column medians.  Infinite values are replaced
        with NaN before median imputation.

        KinematicScore relevance: HIGH.  The first-derivative features
        capture velocity of physiological deterioration, making them
        ideal inputs for the kinematic anomaly detector.

        Args:
            raw_data: DataFrame from :meth:`fetch_realtime` or
                :meth:`fetch_historical`.

        Returns:
            2-D numpy array of shape ``(n_samples, 53)``.
        """
        if raw_data.empty:
            return np.empty((0, 53), dtype=np.float64)

        df = raw_data.copy()

        # Determine which feature columns are actually present
        missing_cols = [c for c in _FEATURE_COLS if c not in df.columns]
        if missing_cols:
            logger.debug(
                "Sepsis feature engineering: %d/%d expected columns missing: %s",
                len(missing_cols),
                len(_FEATURE_COLS),
                missing_cols,
            )
            # Add missing columns as NaN so downstream logic works uniformly
            for col in missing_cols:
                df[col] = np.nan

        # Forward-fill within each patient to propagate last-known values
        if "patient_id" in df.columns:
            df[_FEATURE_COLS] = df.groupby("patient_id")[_FEATURE_COLS].ffill()

        # --- Raw feature values ---
        raw_values = df[_FEATURE_COLS].values.astype(np.float64)

        # --- First derivatives (rate of change per hour) ---
        derivatives = self._compute_derivatives(df, _FEATURE_COLS)

        # --- Rolling standard deviation (6-hour window) ---
        rolling_std = self._compute_rolling_std(df, _FEATURE_COLS, window=6)

        # --- Interaction terms ---
        hr_map_product = df["HR"].values.astype(np.float64) * df["MAP"].values.astype(np.float64)
        resp_o2sat_product = df["Resp"].values.astype(np.float64) * df["O2Sat"].values.astype(
            np.float64
        )
        interactions = np.column_stack([hr_map_product, resp_o2sat_product])

        # --- Combine all features ---
        features = np.column_stack(
            [
                raw_values,
                derivatives,
                rolling_std,
                interactions,
            ]
        )

        # --- Clean non-finite values ---
        features = np.where(np.isinf(features), np.nan, features)
        for col_idx in range(features.shape[1]):
            col_data = features[:, col_idx]
            mask = np.isnan(col_data)
            if mask.any():
                median_val = np.nanmedian(col_data)
                col_data[mask] = median_val if np.isfinite(median_val) else 0.0

        return features

    # ------------------------------------------------------------------
    # Private helpers -- data download and parsing
    # ------------------------------------------------------------------

    def _download_and_parse_training_set(
        self,
        url: str,
        set_label: str,
        max_patients: int = 0,
    ) -> pd.DataFrame:
        """Download and parse a PhysioNet Challenge 2019 training set.

        PhysioNet serves each training set as a directory of per-patient
        PSV (pipe-separated values) files named ``p{NNNNNN}.psv`` — not as
        a ZIP archive (the historical zip URLs 404). The directory index
        is fetched once, the ``p*.psv`` entries are enumerated in sorted
        order, and individual files are fetched up to ``max_patients``.

        Args:
            url: URL of the training set directory (trailing slash).
            set_label: Human-readable label for the set (``"A"`` or ``"B"``).
            max_patients: Maximum number of patient files to fetch.
                ``0`` means all files in the index — for the full sets that
                is ~20k HTTP fetches; callers should cap.

        Returns:
            Concatenated DataFrame with all patient records.  Includes a
            ``patient_id`` column extracted from the filename.

        Raises:
            DataSourceUnavailableError: If PhysioNet is unreachable, the
                index lists no PSV files, or any listed file fails to
                fetch. A listed-but-unfetchable file is upstream
                inconsistency and fails loud rather than silently
                shrinking the cohort.
        """
        try:
            index_html = self._fetch_url(url).decode("utf-8", errors="replace")
        except ConnectionError as exc:
            raise DataSourceUnavailableError(
                loader_name="SepsisLoader",
                source_url=url,
                reason=(f"Failed to list training set {set_label} from " f"PhysioNet: {exc}"),
            ) from exc

        psv_names = sorted(set(re.findall(r'href="(p\d+\.psv)"', index_html)))
        if not psv_names:
            raise DataSourceUnavailableError(
                loader_name="SepsisLoader",
                source_url=url,
                reason=(
                    f"Training set {set_label} index at PhysioNet lists no "
                    "p*.psv files; the upstream layout has changed."
                ),
            )
        if max_patients > 0:
            psv_names = psv_names[:max_patients]

        frames: list[pd.DataFrame] = []
        for psv_name in psv_names:
            try:
                raw = self._fetch_url(f"{url}{psv_name}")
            except ConnectionError as exc:
                raise DataSourceUnavailableError(
                    loader_name="SepsisLoader",
                    source_url=url,
                    reason=(
                        f"Training set {set_label}: listed file "
                        f"'{psv_name}' failed to fetch: {exc}"
                    ),
                ) from exc
            patient_df = pd.read_csv(
                io.BytesIO(raw),
                sep="|",
                na_values=["NaN", "nan", ""],
            )
            patient_df["patient_id"] = self._extract_patient_id(psv_name)
            frames.append(patient_df)

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "Parsed %d patient files (%d total observations).",
            len(frames),
            len(combined),
        )
        return combined

    @staticmethod
    def _extract_patient_id(filename: str) -> str:
        """Extract patient identifier from a PSV filename.

        PhysioNet Challenge 2019 files are named ``p{NNNNNN}.psv`` where
        ``NNNNNN`` is a zero-padded numeric patient ID.

        Args:
            filename: PSV filename (may include directory prefix).

        Returns:
            Patient ID string (e.g. ``"p000001"``).
        """
        basename = Path(filename).stem  # e.g. "p000001"
        return basename

    # ------------------------------------------------------------------
    # Private helpers -- feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_derivatives(
        df: pd.DataFrame,
        columns: list[str],
    ) -> np.ndarray[Any, Any]:
        """Compute first-order derivatives (hourly rate of change).

        Within each patient, the derivative is ``value[t] - value[t-1]``.
        The first observation per patient gets a derivative of zero.

        Args:
            df: DataFrame with a ``patient_id`` column and the specified
                feature columns.
            columns: Column names to differentiate.

        Returns:
            2-D numpy array of shape ``(n_rows, len(columns))``.
        """
        result = np.zeros((len(df), len(columns)), dtype=np.float64)

        if "patient_id" not in df.columns or df.empty:
            return result

        for col_idx, col_name in enumerate(columns):
            if col_name not in df.columns:
                continue
            values = df[col_name].values.astype(np.float64)
            patient_ids = df["patient_id"].values

            # Compute diff; set to 0 where patient changes
            diff = np.zeros(len(values), dtype=np.float64)
            if len(values) > 1:
                diff[1:] = np.diff(values)
                # Zero out transitions between patients
                patient_change = patient_ids[1:] != patient_ids[:-1]
                diff[1:][patient_change] = 0.0
            result[:, col_idx] = diff

        return result

    @staticmethod
    def _compute_rolling_std(
        df: pd.DataFrame,
        columns: list[str],
        window: int = 6,
    ) -> np.ndarray[Any, Any]:
        """Compute rolling standard deviation within each patient.

        Args:
            df: DataFrame with a ``patient_id`` column and the specified
                feature columns.
            columns: Column names to compute rolling std for.
            window: Rolling window size in hours.

        Returns:
            2-D numpy array of shape ``(n_rows, len(columns))``.
        """
        result = np.zeros((len(df), len(columns)), dtype=np.float64)

        if "patient_id" not in df.columns or df.empty:
            return result

        for col_idx, col_name in enumerate(columns):
            if col_name not in df.columns:
                continue

            # Compute rolling std per patient group
            rolling_vals = (
                df.groupby("patient_id")[col_name]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )

            # Align back to original index
            result[:, col_idx] = rolling_vals.reindex(df.index).fillna(0.0).values

        return result
