# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for ADRepository dataset loaders."""

from __future__ import annotations

import io
import tarfile
import zipfile
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
from omni_mercury_engine.datasets.adrepository import (
    _extract_tar_members_guarded,
    _fresh_extract_dir,
    _safe_extract_tar,
    _safe_extract_zip,
    _select_single_csv,
)

# Sentinel: distinguishes "extract() called with no filter= kwarg" (the bare,
# default-resolving call) from "filter=None passed explicitly" in spy assertions.
_MISSING = object()


def _raise_offline(*_args: object, **_kwargs: object) -> bool:
    """Stand-in for an unreachable real source (simulated offline)."""
    raise RuntimeError("simulated offline / real source unreachable")


def _write_tar(path: Any, members: dict[str, bytes]) -> None:
    """Write a tar at ``path`` from an ``arcname -> payload`` mapping."""
    with tarfile.open(path, "w") as tf:
        for arcname, payload in members.items():
            info = tarfile.TarInfo(name=arcname)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))


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

    def test_delegated_timeseries_returns_real_data(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """smd is delegated to SMDLoader; its real (X, y) flows back, marked real.

        The dedicated loader is stubbed at its boundary so the *wiring* is tested
        without the network — a test double, not a benchmark result (the real
        end-to-end load lives in the network lane).
        """
        from omni_mercury_engine.datasets import timeseries

        rng = np.random.RandomState(0)
        fake_x = rng.randn(120, 38).astype(np.float32)
        fake_y = (np.arange(120) % 12 == 0).astype(np.int64)
        monkeypatch.setattr(timeseries.SMDLoader, "download", lambda self: True)
        monkeypatch.setattr(timeseries.SMDLoader, "_load_raw", lambda self: (fake_x, fake_y))

        loader = ADRepositoryLoader(
            DatasetConfig(name="smd", data_dir=str(tmp_path / "d")), dataset_name="smd"
        )
        X, y = loader.load_data()

        assert X.shape == (120, 38)
        assert loader.is_real_data is True
        assert loader.get_metadata()["synthetic"] is False
        assert 0 < int(y.sum()) < len(y)

    def test_delegated_timeseries_fails_loud_when_delegate_cannot(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """If the dedicated loader can't get real data, fail loud — never fabricate."""
        from omni_mercury_engine.datasets import timeseries

        def _boom(_self: Any) -> Any:
            raise FileNotFoundError("SMD data not found")

        monkeypatch.delenv("MERCURY_ALLOW_SYNTHETIC", raising=False)
        monkeypatch.setattr(timeseries.SMDLoader, "download", lambda self: False)
        monkeypatch.setattr(timeseries.SMDLoader, "_load_raw", _boom)
        loader = ADRepositoryLoader(
            DatasetConfig(name="smd", data_dir=str(tmp_path / "e")), dataset_name="smd"
        )
        with pytest.raises(DataSourceUnavailableError, match="SMDLoader"):
            loader.load_data()

    def test_credentialed_timeseries_fails_loud(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """swat -> SWaTLoader, which needs iTrust credentials: loud, no fabrication."""
        monkeypatch.delenv("MERCURY_ALLOW_SYNTHETIC", raising=False)
        loader = ADRepositoryLoader(
            DatasetConfig(name="swat", data_dir=str(tmp_path / "f")), dataset_name="swat"
        )
        with pytest.raises(DataSourceUnavailableError, match="credentials"):
            loader.load_data()

    @pytest.mark.parametrize("name", ["dsads", "epilepsy"])
    def test_no_loader_timeseries_fails_loud_with_closing_step(
        self, name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """dsads/epilepsy have a documented upstream but no dedicated loader yet.

        The loud error must name the real source and the exact closing step.
        """
        monkeypatch.delenv("MERCURY_ALLOW_SYNTHETIC", raising=False)
        loader = ADRepositoryLoader(
            DatasetConfig(name=name, data_dir=str(tmp_path / name)), dataset_name=name
        )
        with pytest.raises(DataSourceUnavailableError) as exc:
            loader.load_data()
        msg = str(exc.value)
        assert "Closing step" in msg and "github.com" in msg

    def test_no_loader_timeseries_opt_in_synthetic_is_marked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Policy on → a no-loader time-series set yields *marked* synthetic
        (same single chokepoint as every other set), never silent real-looking data."""
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
        loader = ADRepositoryLoader(
            DatasetConfig(name="epilepsy", data_dir=str(tmp_path / "g"), max_samples=100),
            dataset_name="epilepsy",
        )
        X, _y = loader.load_data()
        assert X.shape[0] == 100
        assert loader.get_metadata()["synthetic"] is True
        assert loader.is_real_data is False


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

    # (name, expected feature count, anomalies dense enough to land in first 2k rows)
    # Closes the open gap: campaign/donors/celeba/census/fraud were only
    # reachability-checked (HTTP 200), never load-verified end-to-end. census's
    # 500-feature claim is asserted here against the real CSV.
    _DEVNET_REAL = [
        ("campaign", 62, True),
        ("donors", 10, True),
        ("celeba", 39, True),
        ("census", 500, True),
        ("fraud", 29, False),  # 0.17% anomaly ratio: too sparse to require in 2k rows
    ]

    @pytest.mark.parametrize("name, n_features, has_anomalies", _DEVNET_REAL)
    def test_real_devnet_set_load_verified(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        name: str,
        n_features: int,
        has_anomalies: bool,
    ) -> None:
        """Each DevNet tabular set loads real data with its documented feature count.

        Synthetic is denied so any fallback would *raise*; ``max_samples`` bounds
        the parse (the loader reads only the first rows of large CSVs like the
        299k x 500 census set) so the network lane stays fast.
        """
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
        loader = ADRepositoryLoader(
            DatasetConfig(name=name, data_dir=str(tmp_path / name), max_samples=2000),
            dataset_name=name,
        )
        X, y = loader.load_data()
        assert X.shape[1] == n_features, f"{name}: {X.shape[1]} features, expected {n_features}"
        assert loader.is_real_data is True
        assert set(np.unique(np.asarray(y))).issubset({0, 1}), f"{name}: non-binary labels"
        if has_anomalies:
            assert int(np.asarray(y).sum()) > 0, f"{name}: no real anomalies in first 2000 rows"

    def test_real_smd_via_delegation(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        """smd is delegated to SMDLoader and loads real server telemetry end-to-end."""
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
        loader = ADRepositoryLoader(
            DatasetConfig(
                name="smd",
                data_dir=str(tmp_path / "smd"),
                preprocessing={"machines": ["machine-1-1"]},
            ),
            dataset_name="smd",
        )
        X, y = loader.load_data()
        assert X.shape[1] == 38
        assert loader.is_real_data is True
        assert int(np.asarray(y).sum()) > 0  # machine-1-1 test split carries real anomalies


class TestArchiveExtractionSafety:
    """External-mirror archive extraction must be slip-safe, version-portable,
    and stale-free.

    These guard the ``.tar.xz`` / ``.zip`` path in ``_load_from_file``: a
    hostile or malformed archive must never write outside the extraction dir
    (tar-slip / zip-slip), the tar path must not depend on a ``filter=`` kwarg
    that pre-3.11.4 interpreters lack, and a re-used extraction dir must not
    leak stale members into the ``rglob('*.csv')`` member search.
    """

    # ---- zip-slip --------------------------------------------------------
    def test_zip_traversal_member_is_refused(self, tmp_path: Any) -> None:
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("ok.csv", "a,b\n1,2\n")
            zf.writestr("../escape.csv", "pwned")
        dest = tmp_path / "out"
        dest.mkdir()
        with (
            zipfile.ZipFile(archive, "r") as zf,
            pytest.raises(RuntimeError, match="escapes the extraction directory"),
        ):
            _safe_extract_zip(zf, dest)
        assert not (tmp_path / "escape.csv").exists()

    def test_zip_clean_archive_extracts(self, tmp_path: Any) -> None:
        archive = tmp_path / "clean.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("data.csv", "a,b\n1,2\n")
        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(archive, "r") as zf:
            _safe_extract_zip(zf, dest)
        assert (dest / "data.csv").read_text() == "a,b\n1,2\n"

    def test_zip_refuses_backslash_traversal(self, tmp_path: Any) -> None:
        """A ``..\\escape.csv`` member uses a backslash separator that
        ``zipfile.extract`` honours on Windows but POSIX ``Path`` treats as a
        literal char. The guard normalises ``\\``->``/`` for the escape check so
        the member is refused on every platform, not silently written outside
        ``dest`` on Windows (cross-platform zip-slip)."""
        archive = tmp_path / "evil_bs.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("ok.csv", "a,b\n1,2\n")
            zf.writestr("..\\escape.csv", "pwned")
        dest = tmp_path / "out"
        dest.mkdir()
        with (
            zipfile.ZipFile(archive, "r") as zf,
            pytest.raises(RuntimeError, match="escapes the extraction directory"),
        ):
            _safe_extract_zip(zf, dest)
        assert not (tmp_path / "escape.csv").exists()

    # ---- tar-slip (legacy guard, exercised on every interpreter) ---------
    def test_legacy_tar_guard_refuses_traversal(self, tmp_path: Any) -> None:
        archive = tmp_path / "evil.tar"
        _write_tar(archive, {"ok.csv": b"x", "../escape.csv": b"pwned"})
        dest = tmp_path / "out"
        dest.mkdir()
        with (
            tarfile.open(archive, "r") as tf,
            pytest.raises(RuntimeError, match="escapes the extraction directory"),
        ):
            _extract_tar_members_guarded(tf, dest)
        assert not (tmp_path / "escape.csv").exists()

    def test_legacy_tar_guard_refuses_symlink(self, tmp_path: Any) -> None:
        archive = tmp_path / "link.tar"
        with tarfile.open(archive, "w") as tf:
            info = tarfile.TarInfo(name="link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)
        dest = tmp_path / "out"
        dest.mkdir()
        with (
            tarfile.open(archive, "r") as tf,
            pytest.raises(RuntimeError, match="links are not allowed"),
        ):
            _extract_tar_members_guarded(tf, dest)

    def test_legacy_tar_guard_extracts_clean(self, tmp_path: Any) -> None:
        archive = tmp_path / "clean.tar"
        _write_tar(archive, {"data.csv": b"a,b\n1,2\n"})
        dest = tmp_path / "out"
        dest.mkdir()
        with tarfile.open(archive, "r") as tf:
            _extract_tar_members_guarded(tf, dest)
        assert (dest / "data.csv").read_bytes() == b"a,b\n1,2\n"

    # ---- feature-detect fallback: the fix for pre-3.11.4 `filter=` -------
    def test_safe_extract_tar_falls_back_when_filter_absent(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate a pre-3.11.4 interpreter (no ``tarfile.data_filter``).

        The public entry must route through the manual guard instead of raising
        ``TypeError`` on ``filter="data"`` — still extracting clean archives and
        still refusing traversal — so dataset loading does not silently collapse
        into the synthetic fallback on old patch releases.

        The feature-absence is forced at the routing seam
        (``_tar_supports_data_filter``) rather than by deleting
        ``tarfile.data_filter``: on CPython 3.14 the ``data`` filter is the
        PEP 706 default that ``TarFile.extract`` looks up by that global name, so
        deleting it breaks extraction itself (``NameError``) and models no real
        interpreter.
        """
        monkeypatch.setattr(
            "omni_mercury_engine.datasets.adrepository._tar_supports_data_filter",
            lambda: False,
        )

        clean = tmp_path / "clean.tar"
        _write_tar(clean, {"d.csv": b"hello"})
        dest = tmp_path / "out"
        dest.mkdir()
        with tarfile.open(clean, "r") as tf:
            _safe_extract_tar(tf, dest)
        assert (dest / "d.csv").read_bytes() == b"hello"

        evil = tmp_path / "evil.tar"
        _write_tar(evil, {"../escape.csv": b"pwned"})
        with (
            tarfile.open(evil, "r") as tf,
            pytest.raises(RuntimeError, match="escapes the extraction directory"),
        ):
            _safe_extract_tar(tf, dest)
        assert not (tmp_path / "escape.csv").exists()

    def test_safe_extract_tar_clean_roundtrip_modern(self, tmp_path: Any) -> None:
        """On an interpreter that has the filter, the public entry still works."""
        archive = tmp_path / "clean.tar"
        _write_tar(archive, {"d.csv": b"a,b\n1,2\n"})
        dest = tmp_path / "out"
        dest.mkdir()
        with tarfile.open(archive, "r") as tf:
            _safe_extract_tar(tf, dest)
        assert (dest / "d.csv").read_bytes() == b"a,b\n1,2\n"

    def test_guard_passes_explicit_filter_when_runtime_supports_it(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard must pass ``filter="data"`` explicitly per member whenever
        the runtime exposes it, never a bare ``extract()``.

        A bare ``extract()`` resolves the default filter by looking up the
        module-global ``tarfile.data_filter`` *by name* on CPython 3.14 — the
        exact path that NameErrored when the old fallback test deleted that
        global. Pinning the explicit kwarg keeps the guard deterministic and
        free of that default-resolution dependency on every modern interpreter.
        """
        if not hasattr(tarfile, "data_filter"):
            pytest.skip("interpreter predates the tarfile data filter")

        real_extract = tarfile.TarFile.extract
        seen: list[Any] = []

        def _spy_extract(self: Any, member: Any, path: Any = "", **kwargs: Any) -> Any:
            seen.append(kwargs.get("filter", _MISSING))
            return real_extract(self, member, path, **kwargs)

        monkeypatch.setattr(tarfile.TarFile, "extract", _spy_extract)
        archive = tmp_path / "clean.tar"
        _write_tar(archive, {"d.csv": b"hello"})
        dest = tmp_path / "out"
        dest.mkdir()
        with tarfile.open(archive, "r") as tf:
            _extract_tar_members_guarded(tf, dest)
        assert seen == ["data"], f"guard relied on the default filter: {seen}"
        assert (dest / "d.csv").read_bytes() == b"hello"

    # ---- single-CSV selection (multi-member archives must fail loud) ------
    def test_select_single_csv_returns_sole_member(self, tmp_path: Any) -> None:
        only = tmp_path / "annthyroid_normalised.csv"
        only.write_text("a,b\n1,2\n")
        assert _select_single_csv([only], "thyroid.tar.xz") == only

    def test_select_single_csv_fails_loud_on_multiple(self, tmp_path: Any) -> None:
        """A train/test bundle must not be silently narrowed to one split."""
        train = tmp_path / "train.csv"
        test = tmp_path / "test.csv"
        train.write_text("x")
        test.write_text("y")
        with pytest.raises(RuntimeError, match="multiple CSV members"):
            _select_single_csv([train, test], "UNSW_NB15_traintest_backdoor.tar.xz")

    def test_select_single_csv_fails_loud_on_none(self) -> None:
        with pytest.raises(RuntimeError, match="No CSV member"):
            _select_single_csv([], "empty.tar.xz")

    # ---- stale-dir clearing ----------------------------------------------
    def test_fresh_extract_dir_clears_stale_members(self, tmp_path: Any) -> None:
        d = tmp_path / "archive.tar.xz_extracted"
        d.mkdir()
        (d / "stale.csv").write_text("from a previous, different archive")
        out = _fresh_extract_dir(d)
        assert out == d
        assert d.is_dir()
        assert list(d.iterdir()) == []  # the stale CSV is gone

    def test_fresh_extract_dir_creates_when_absent(self, tmp_path: Any) -> None:
        d = tmp_path / "nested" / "x_extracted"
        assert not d.exists()
        out = _fresh_extract_dir(d)
        assert out == d
        assert d.is_dir()
