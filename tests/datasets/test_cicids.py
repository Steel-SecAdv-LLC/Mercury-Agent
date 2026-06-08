# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for CICIDS 2017 dataset loader - REAL network intrusion data."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import numpy as np

from omni_mercury_engine.datasets import CICIDSLoader, DatasetConfig
from omni_mercury_engine.security.input_validation import TrustedEndpoints


class TestCICIDSMetadata:
    """Test CICIDS dataset metadata and configuration."""

    def test_attack_labels_defined(self) -> None:
        """Verify all expected attack labels are defined."""
        expected_attacks = {
            "BENIGN",
            "DDoS",
            "PortScan",
            "Bot",
            "Infiltration",
            "FTP-Patator",
            "SSH-Patator",
            "DoS slowloris",
            "DoS Slowhttptest",
            "DoS Hulk",
            "DoS GoldenEye",
            "Heartbleed",
        }
        defined_attacks = set(CICIDSLoader.ATTACK_LABELS.keys())
        # Check that expected attacks are subset (there may be variations for web attacks)
        assert expected_attacks.issubset(defined_attacks)

    def test_data_sources_defined(self) -> None:
        """Verify all data sources are properly defined."""
        assert "huggingface" in CICIDSLoader.DATA_SOURCES
        assert "distrinet" in CICIDSLoader.DATA_SOURCES
        assert "cic_official" in CICIDSLoader.DATA_SOURCES

        # Hugging Face should have dataset_id
        assert "dataset_id" in CICIDSLoader.DATA_SOURCES["huggingface"]
        assert CICIDSLoader.DATA_SOURCES["huggingface"]["dataset_id"] == "bvk/CICIDS-2017"

        # URL sources should have url
        assert "url" in CICIDSLoader.DATA_SOURCES["distrinet"]
        assert "url" in CICIDSLoader.DATA_SOURCES["cic_official"]

    def test_cic_official_source_uses_trusted_hostname_not_raw_ip(self) -> None:
        """The CIC fallback must not bypass SSRF allowlisting via a raw IP URL."""
        url = CICIDSLoader.DATA_SOURCES["cic_official"]["url"]
        parsed = urlparse(url)

        assert parsed.scheme == "https"
        assert parsed.hostname == "cicresearch.ca"
        assert parsed.hostname in TrustedEndpoints.TRUSTED_DOMAINS
        assert "205.174.165.80" not in url

    def test_cicids_files_defined(self) -> None:
        """Verify CICIDS CSV file names are defined."""
        assert "ddos" in CICIDSLoader.CICIDS_FILES
        assert "portscan" in CICIDSLoader.CICIDS_FILES
        assert "infiltration" in CICIDSLoader.CICIDS_FILES
        assert "webattacks" in CICIDSLoader.CICIDS_FILES
        assert "all" in CICIDSLoader.CICIDS_FILES


class TestCICIDSLoader:
    """Test CICIDS loader functionality."""

    def test_init_default_config(self) -> None:
        """Test loader initialization with default config."""
        config = DatasetConfig(name="cicids", data_dir="./data/test_cicids")
        loader = CICIDSLoader(config)

        assert loader.binary_labels is True  # Default
        assert loader.subset == "all"  # Default
        assert loader._is_real_data is False  # Not loaded yet
        assert loader._features is None
        assert loader._raw_labels is None

    def test_init_binary_false(self) -> None:
        """Test loader with multi-class classification."""
        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids",
            preprocessing={"binary": False},
        )
        loader = CICIDSLoader(config)
        assert loader.binary_labels is False

    def test_init_specific_subset(self) -> None:
        """Test loader with specific subset."""
        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids",
            preprocessing={"subset": "ddos"},
        )
        loader = CICIDSLoader(config)
        assert loader.subset == "ddos"

    def test_is_real_data_property(self) -> None:
        """Test is_real_data property."""
        config = DatasetConfig(name="cicids", data_dir="./data/test_cicids")
        loader = CICIDSLoader(config)

        # Before loading, should be False
        assert loader.is_real_data is False

    def test_synthetic_fallback(self) -> None:
        """Test synthetic fallback when download fails."""
        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids_synthetic",
            max_samples=1000,
        )
        loader = CICIDSLoader(config)

        # Force synthetic fallback
        result = loader._create_synthetic_fallback()

        assert result is True
        assert loader._features is not None
        assert loader._raw_labels is not None
        assert loader._features.shape[0] == 1000
        assert loader._features.shape[1] == 78  # Typical CICIDS feature count
        assert len(loader._raw_labels) == 1000
        assert loader.is_real_data is False

    def test_synthetic_attack_distribution(self) -> None:
        """Test synthetic data has reasonable attack distribution."""
        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids_dist",
            max_samples=10000,
            random_seed=42,
        )
        loader = CICIDSLoader(config)
        loader._create_synthetic_fallback()

        X, y = loader.load_data()

        # CICIDS is heavily imbalanced (mostly benign)
        attack_ratio = y.mean()
        assert 0.1 <= attack_ratio <= 0.3  # ~20% attacks in synthetic

    def test_get_metadata(self) -> None:
        """Test metadata retrieval."""
        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids_meta",
            max_samples=500,
        )
        loader = CICIDSLoader(config)
        loader._create_synthetic_fallback()

        metadata = loader.get_metadata()

        assert metadata["name"] == "CICIDS 2017"
        assert metadata["source"] == "Canadian Institute for Cybersecurity"
        assert metadata["n_samples"] == 500
        assert metadata["n_features"] == 78
        assert metadata["label_type"] == "binary"
        assert metadata["is_real_data"] is False
        assert "citation" in metadata

    def test_get_statistics(self) -> None:
        """Test statistics calculation."""
        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids_stats",
            max_samples=1000,
        )
        loader = CICIDSLoader(config)
        loader._create_synthetic_fallback()
        loader.load_data()

        stats = loader.get_statistics()

        assert "n_samples" in stats
        assert "n_features" in stats
        assert "n_attacks" in stats
        assert "attack_ratio" in stats
        assert "class_distribution" in stats
        assert "is_real_data" in stats
        assert stats["n_samples"] == 1000
        assert stats["n_features"] == 78
        assert stats["is_real_data"] is False

    def test_preprocess(self) -> None:
        """Test preprocessing transforms data correctly."""
        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids_preprocess",
            max_samples=100,
        )
        loader = CICIDSLoader(config)
        loader._create_synthetic_fallback()

        X, _ = loader.load_data()

        # Add some problematic values
        X_test = X.copy()
        X_test[0, 0] = np.inf
        X_test[1, 1] = -np.inf
        X_test[2, 2] = np.nan

        X_processed = loader.preprocess(X_test)

        # Should have no inf/nan values
        assert not np.any(np.isinf(X_processed))
        assert not np.any(np.isnan(X_processed))

        # Should be float32
        assert X_processed.dtype == np.float32


class TestCICIDSLabelEncoding:
    """Test CICIDS label encoding."""

    def test_binary_encoding(self) -> None:
        """Test binary label encoding."""
        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids_binary",
            preprocessing={"binary": True},
        )
        loader = CICIDSLoader(config)

        # Test encoding
        assert loader.binary_labels is True

    def test_multiclass_encoding(self) -> None:
        """Test multi-class label encoding."""
        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids_multiclass",
            preprocessing={"binary": False},
        )
        loader = CICIDSLoader(config)

        # Test known labels
        assert loader._encode_label("BENIGN") == 0
        assert loader._encode_label("DDoS") == 1
        assert loader._encode_label("PortScan") == 2
        assert loader._encode_label("Bot") == 3
        assert loader._encode_label("Infiltration") == 4

    def test_unknown_label_handling(self) -> None:
        """Test handling of unknown attack types."""
        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids_unknown",
            preprocessing={"binary": False},
        )
        loader = CICIDSLoader(config)

        # Unknown labels should get code 15
        result = loader._encode_label("UnknownAttackType")
        assert result == 15


class TestCICIDSDataCleaning:
    """Test CICIDS data cleaning functionality."""

    def test_clean_infinity_values(self) -> None:
        """Test infinity values are handled."""
        import pandas as pd

        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids_clean",
        )
        loader = CICIDSLoader(config)

        # Create test dataframe with infinity
        df = pd.DataFrame(
            {
                "col1": [1.0, np.inf, 3.0],
                "col2": [-np.inf, 2.0, 4.0],
                "col3": [1.0, 2.0, 3.0],
            }
        )

        cleaned = loader._clean_cicids_data(df)

        # Rows with inf should be dropped
        assert len(cleaned) == 1
        assert not np.any(np.isinf(cleaned.values))

    def test_clean_negative_duration(self) -> None:
        """Test negative duration values are clipped."""
        import pandas as pd

        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids_duration",
        )
        loader = CICIDSLoader(config)

        # Create test dataframe with negative duration
        df = pd.DataFrame(
            {
                "flow_duration": [-100, 50, 200],
                "other_col": [1.0, 2.0, 3.0],
            }
        )

        cleaned = loader._clean_cicids_data(df)

        # Negative duration should be clipped to 0
        assert cleaned["flow_duration"].min() >= 0


class TestCICIDSIntegration:
    """Integration tests with Mercury Agent engine."""

    def test_cicids_with_detector(self) -> None:
        """Test using CICIDS data with anomaly detector."""
        from omni_mercury_engine import OmniMercuryEngine

        # Load synthetic data (faster for tests)
        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids_integration",
            max_samples=200,
        )
        loader = CICIDSLoader(config)
        loader._create_synthetic_fallback()
        X, y = loader.load_data()

        # Run detection
        engine = OmniMercuryEngine()
        result = engine.detect(X)

        assert "detectors" in result
        assert "is_anomaly" in result

    def test_cicids_benchmark(self) -> None:
        """Test benchmarking CICIDS with Isolation Forest."""
        from omni_mercury_engine.detectors.enhanced_statistical import MADDetector
        from omni_mercury_engine.ml.mercury_ml import roc_auc_score

        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids_benchmark",
            max_samples=1000,
            random_seed=42,
        )
        loader = CICIDSLoader(config)
        loader._create_synthetic_fallback()
        X, y = loader.load_data()

        # Preprocess
        X_processed = loader.preprocess(X)

        # Train and evaluate using MADDetector's actual API
        clf = MADDetector()
        clf.fit(X_processed)
        result = clf.detect(X_processed)
        scores = result.scores

        if len(np.unique(y)) > 1:
            auc = roc_auc_score(y, scores)
            # Should achieve reasonable performance even on synthetic
            assert 0.5 <= auc <= 1.0


class TestCICIDSDataSourcePriority:
    """Test data source priority and fallback behavior."""

    def test_data_sources_order(self) -> None:
        """Test data sources are in correct priority order."""
        sources = list(CICIDSLoader.DATA_SOURCES.keys())

        # Hugging Face should be first (most reliable)
        assert sources[0] == "huggingface"

        # Distrinet second (improved version)
        assert sources[1] == "distrinet"

        # Official CIC last (often unreliable)
        assert sources[2] == "cic_official"

    def test_synthetic_fallback_warning(self, caplog: Any) -> None:
        """Test synthetic fallback logs warning."""
        import logging

        config = DatasetConfig(
            name="cicids",
            data_dir="./data/test_cicids_warning",
            max_samples=100,
        )
        loader = CICIDSLoader(config)

        with caplog.at_level(logging.WARNING):
            loader._create_synthetic_fallback()

        # Should log warning about synthetic data
        assert any("SYNTHETIC" in record.message for record in caplog.records)
