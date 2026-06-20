# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for ADRepository dataset loaders."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.datasets import (
    ADREPOSITORY_DATASETS,
    ADRepositoryLoader,
    DatasetConfig,
    DataSourceUnavailableError,
    list_available_datasets,
    load_dataset,
)


def _raise_offline(*_args: object, **_kwargs: object) -> bool:
    """Stand-in for an unreachable real source (simulated offline)."""
    raise RuntimeError("simulated offline / real source unreachable")


class TestADRepositoryMetadata:
    """Test ADRepository dataset metadata."""

    def test_datasets_defined(self) -> None:
        """Verify all expected datasets are defined."""
        expected = {
            "fraud",
            "backdoor",
            "campaign",
            "thyroid",
            "donors",
            "census",
            "celeba",
            "smd",
            "swat",
            "dsads",
            "epilepsy",
        }
        assert expected.issubset(set(ADREPOSITORY_DATASETS.keys()))

    def test_dataset_info_complete(self) -> None:
        """Verify all datasets have required metadata fields."""
        required_fields = {"samples", "features", "anomaly_ratio", "domain", "description", "url"}

        for name, info in ADREPOSITORY_DATASETS.items():
            for field in required_fields:
                assert field in info, f"Dataset '{name}' missing field '{field}'"

    def test_list_available_datasets(self) -> None:
        """Test convenience function to list datasets."""
        datasets = list_available_datasets()
        assert isinstance(datasets, dict)
        assert len(datasets) >= 11
        assert "fraud" in datasets
        assert "thyroid" in datasets


class TestADRepositoryLoader:
    """Test ADRepository loader functionality."""

    def test_init_valid_dataset(self) -> None:
        """Test loader initialization with valid dataset."""
        config = DatasetConfig(name="thyroid", data_dir="./data/test")
        loader = ADRepositoryLoader(config, dataset_name="thyroid")

        assert loader.dataset_name == "thyroid"
        assert loader.dataset_info["samples"] == 7200
        assert loader.dataset_info["features"] == 21

    def test_init_invalid_dataset(self) -> None:
        """Test loader raises error for invalid dataset."""
        config = DatasetConfig(name="invalid", data_dir="./data/test")

        with pytest.raises(ValueError, match="Unknown dataset"):
            ADRepositoryLoader(config, dataset_name="nonexistent")

    def test_get_metadata(self) -> None:
        """Test metadata retrieval."""
        config = DatasetConfig(name="fraud", data_dir="./data/test")
        loader = ADRepositoryLoader(config, dataset_name="fraud")
        metadata = loader.get_metadata()

        assert metadata["name"] == "fraud"
        assert metadata["source"] == "ADRepository"
        assert metadata["samples"] == 284807
        assert metadata["features"] == 29
        assert metadata["domain"] == "finance"
        assert "Pang" in metadata["citation"]

    def test_synthetic_fallback(self) -> None:
        """Test synthetic fallback when download fails."""
        config = DatasetConfig(
            name="thyroid",
            data_dir="./data/test_synthetic",
            max_samples=1000,
        )
        loader = ADRepositoryLoader(config, dataset_name="thyroid")

        # Force synthetic fallback
        loader._create_synthetic_fallback()

        X, y = loader.load_data()

        assert X.shape[0] == 1000
        assert X.shape[1] == 21  # thyroid has 21 features
        assert len(y) == 1000
        assert y.sum() > 0  # Should have some anomalies
        assert not loader.is_real_data

    def test_load_with_max_samples(self) -> None:
        """Test loading with sample limit."""
        config = DatasetConfig(
            name="donors",
            data_dir="./data/test_limited",
            max_samples=500,
        )
        loader = ADRepositoryLoader(config, dataset_name="donors")
        loader._create_synthetic_fallback()

        X, y = loader.load_data()

        assert X.shape[0] <= 500
        assert len(y) <= 500

    def test_get_statistics(self) -> None:
        """Test statistics calculation."""
        config = DatasetConfig(
            name="campaign",
            data_dir="./data/test_stats",
            max_samples=1000,
        )
        loader = ADRepositoryLoader(config, dataset_name="campaign")
        loader._create_synthetic_fallback()
        loader.load_data()

        stats = loader.get_statistics()

        assert "n_samples" in stats
        assert "n_features" in stats
        assert "n_anomalies" in stats
        assert "anomaly_ratio" in stats
        assert "is_real_data" in stats
        assert stats["n_samples"] == 1000
        assert stats["n_features"] == 62


class TestLoadDatasetConvenience:
    """Test the load_dataset convenience function."""

    def test_load_thyroid_synthetic(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        """The convenience loader returns usable data via the synthetic path.

        Forces the offline + policy-on branch so the test is deterministic and
        never depends on the network: real fetch is stubbed unreachable, an
        isolated ``tmp_path`` data dir guarantees no cached real file is hit,
        and ``MERCURY_ALLOW_SYNTHETIC=1`` permits the gated synthetic fallback.
        """
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
        monkeypatch.setattr(ADRepositoryLoader, "_download_from_repository", _raise_offline)

        X, y, meta = load_dataset(
            "thyroid",
            data_dir=str(tmp_path / "adrepo"),
            max_samples=500,
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(meta, dict)
        assert X.shape[0] == len(y)
        assert meta["name"] == "thyroid"
        assert meta["synthetic"] is True

    def test_load_invalid_dataset(self) -> None:
        """Test error on invalid dataset name."""
        with pytest.raises(ValueError):
            load_dataset("not_a_real_dataset")


class TestIntegrationWithEngine:
    """Integration tests with Mercury Agent engine."""

    def test_adrepository_with_detector(self) -> None:
        """Test using ADRepository data with anomaly detector."""
        from omni_mercury_engine import OmniMercuryEngine

        # Load synthetic data (faster for tests)
        config = DatasetConfig(
            name="backdoor",
            data_dir="./data/test_integration",
            max_samples=200,
        )
        loader = ADRepositoryLoader(config, dataset_name="backdoor")
        loader._create_synthetic_fallback()
        X, y = loader.load_data()

        # Run detection
        engine = OmniMercuryEngine()
        result = engine.detect(X)

        assert "detectors" in result
        assert "is_anomaly" in result

    def test_multiple_datasets_benchmark(self) -> None:
        """Test benchmarking across multiple ADRepository datasets."""
        from omni_mercury_engine.ml.mercury_ml import roc_auc_score

        datasets_to_test = ["thyroid", "backdoor", "campaign"]
        results = {}

        for name in datasets_to_test:
            config = DatasetConfig(
                name=name,
                data_dir=f"./data/test_benchmark_{name}",
                max_samples=500,
            )
            loader = ADRepositoryLoader(config, dataset_name=name)
            loader._create_synthetic_fallback()
            X, y = loader.load_data()

            # Simple detector test using MADDetector's actual API
            from omni_mercury_engine.detectors.enhanced_statistical import MADDetector

            clf = MADDetector()
            clf.fit(X)
            result = clf.detect(X)
            scores = result.scores

            if len(np.unique(y)) > 1:
                auc = roc_auc_score(y, scores)
                results[name] = auc

        # All datasets should produce valid AUC scores
        assert len(results) == 3
        for name, auc in results.items():
            assert 0 <= auc <= 1, f"Invalid AUC for {name}: {auc}"


class TestSyntheticPolicyGate:
    """ADRepository must fail loud — never silently fabricate — by default.

    Closes the silent-synthetic foot-gun: the loader historically called
    ``_create_synthetic_fallback`` directly in three ``except``/fallback sites
    with no policy check, so a moved mirror or unreachable host returned
    fabricated data that looked real (the exact trap that contaminated an
    earlier benchmark headline). Every synthetic path now routes through the
    single ``_create_synthetic_fallback`` chokepoint, which gates on
    ``MERCURY_ALLOW_SYNTHETIC`` via ``check_synthetic_allowed``.
    """

    def test_chokepoint_raises_when_policy_denied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling the synthetic chokepoint with policy off raises loudly."""
        monkeypatch.delenv("MERCURY_ALLOW_SYNTHETIC", raising=False)
        loader = ADRepositoryLoader(
            DatasetConfig(name="thyroid", data_dir="./data/test_failloud"),
            dataset_name="thyroid",
        )
        with pytest.raises(DataSourceUnavailableError, match="ADRepository-thyroid"):
            loader._create_synthetic_fallback()

    def test_production_load_fails_loud_on_unreachable_source(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A failed real fetch with policy off raises, never returns synthetic."""
        monkeypatch.delenv("MERCURY_ALLOW_SYNTHETIC", raising=False)
        loader = ADRepositoryLoader(
            DatasetConfig(name="thyroid", data_dir=str(tmp_path / "a")),
            dataset_name="thyroid",
        )
        monkeypatch.setattr(loader, "_download_from_repository", _raise_offline)
        with pytest.raises(DataSourceUnavailableError):
            loader.load_data()

    def test_opt_in_synthetic_is_marked_synthetic(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """With policy on, the gated synthetic path is honoured and labelled."""
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
        loader = ADRepositoryLoader(
            DatasetConfig(name="thyroid", data_dir=str(tmp_path / "b"), max_samples=200),
            dataset_name="thyroid",
        )
        monkeypatch.setattr(loader, "_download_from_repository", _raise_offline)

        X, _y = loader.load_data()
        meta = loader.get_metadata()

        assert X.shape[0] == 200
        assert meta["synthetic"] is True
        assert meta["is_real_data"] is False

    def test_time_series_set_has_no_mirror_fails_loud(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """smd/swat/dsads/epilepsy aren't in the DevNet mirror -> loud error.

        These sets are served by their dedicated loaders (SMDLoader, ...); the
        ADRepository loader must not fabricate them.
        """
        monkeypatch.delenv("MERCURY_ALLOW_SYNTHETIC", raising=False)
        loader = ADRepositoryLoader(
            DatasetConfig(name="smd", data_dir=str(tmp_path / "c")),
            dataset_name="smd",
        )
        with pytest.raises(DataSourceUnavailableError):
            loader.load_data()


@pytest.mark.network
class TestADRepositoryRealMirror:
    """Repoint regression: real data flows from the mala-lab raw mirror.

    Locks the GuansongPang->mala-lab repoint, the ``raw.githubusercontent``
    pin (no 301), and the ``.tar.xz`` extraction path. Runs only in the
    network lane (``-m network``); the default offline suite deselects it.
    Synthetic is denied so a fallback would *raise* — proving the data is real.
    """

    def test_real_thyroid_csv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
        loader = ADRepositoryLoader(
            DatasetConfig(name="thyroid", data_dir="./data/net_thyroid"),
            dataset_name="thyroid",
        )
        X, y = loader.load_data()
        assert X.shape[1] == 21
        assert loader.is_real_data is True
        assert 0 < int(np.asarray(y).sum()) < len(y)

    def test_real_backdoor_tar_xz(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
        loader = ADRepositoryLoader(
            DatasetConfig(name="backdoor", data_dir="./data/net_backdoor"),
            dataset_name="backdoor",
        )
        X, _y = loader.load_data()
        assert X.shape[1] == 196
        assert loader.is_real_data is True
