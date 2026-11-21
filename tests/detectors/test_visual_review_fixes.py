# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the reviewed visual-detector findings.

* Config aliases -- ``PatchCoreConfig`` / ``CFlowConfig`` resolved their alias
  fields by comparing against the default value, so "explicitly set to the
  default" was indistinguishable from "unset": ``dataclasses.replace`` back to
  a default was ignored, and a config mutated back to a default reverted on an
  ``asdict`` -> reconstruct (save/load) round trip.
* Checkpoint compatibility -- a visual checkpoint carried no scoring/layout
  version, so a PaDiM state fitted on the old 14x14 grid died with an opaque
  tensor-size ``RuntimeError`` and a PatchCore threshold calibrated on the old
  un-weighted score was compared with re-weighted scores without a word.
* PaDiM fit memory -- ``fit`` accumulated the UNPROJECTED ``[N, H*W, C]``
  features of every image before reducing them.
* Loaded settings -- found while hardening ``load``: it replaced
  ``detector.config`` but left the detector's own typed reference
  (``patchcore_config`` ...), which scoring reads, on the constructor's
  config, so a checkpoint loaded into a differently-configured detector was
  scored with the wrong settings against its saved threshold.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from omni_mercury_engine.detectors import visual
from omni_mercury_engine.detectors.visual.cflow import CFlowConfig
from omni_mercury_engine.detectors.visual.patchcore import PatchCoreConfig
from omni_mercury_engine.security.safe_torch import safe_torch_load

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.visual

SIZE = 32
TINY: dict[str, Any] = {
    "backbone": "resnet18",
    "image_size": (SIZE, SIZE),
    "threshold": 1.0e9,
    "batch_size": 2,
}

DETECTORS: dict[str, tuple[Any, dict[str, Any]]] = {
    "patchcore": (visual.PatchCoreDetector, {"seed": 0}),
    "padim": (visual.PaDiMDetector, {}),
    "stfpm": (visual.STFPMDetector, {"num_epochs": 1}),
    "reverse_distillation": (visual.ReverseDistillationDetector, {"num_epochs": 1}),
    "cflow": (visual.CFlowDetector, {"num_epochs": 1}),
}


def _tiny_detector(name: str, **overrides: Any) -> Any:
    cls, extra = DETECTORS[name]
    return cls({**TINY, **extra, **overrides})


def _images(n: int, seed: int = 0) -> Any:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, SIZE, SIZE, generator=generator)


# ---------------------------------------------------------------------------
# Config aliases: "explicitly the default" is not "unset"
# ---------------------------------------------------------------------------


class TestPatchCoreConfigAliases:
    """``coreset_ratio`` / ``k_nearest`` mirror their canonical fields."""

    def test_replace_back_to_the_default_takes_effect(self) -> None:
        """``replace`` to the canonical DEFAULT used to be ignored (kept 0.25 / 5)."""
        config = replace(
            PatchCoreConfig(coreset_ratio=0.25, k_nearest=5),
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        )
        assert config.coreset_sampling_ratio == 0.1
        assert config.num_neighbors == 9
        assert config.coreset_ratio == 0.1
        assert config.k_nearest == 9

    def test_a_value_mutated_back_to_the_default_survives_the_round_trip(self) -> None:
        """``asdict`` -> reconstruct (the save/load path) used to revert 9 to 5."""
        config = PatchCoreConfig(num_neighbors=5)
        config.num_neighbors = 9
        rebuilt = PatchCoreConfig(**asdict(config))
        assert rebuilt.num_neighbors == 9
        assert rebuilt.k_nearest == 9

    def test_the_canonical_field_wins_even_at_its_default_value(self) -> None:
        config = PatchCoreConfig(
            coreset_sampling_ratio=0.1, coreset_ratio=0.25, num_neighbors=9, k_nearest=5
        )
        assert config.coreset_sampling_ratio == 0.1
        assert config.num_neighbors == 9
        assert (config.coreset_ratio, config.k_nearest) == (0.1, 9)

    def test_an_alias_given_alone_takes_effect_and_reads_back(self) -> None:
        config = PatchCoreConfig(coreset_ratio=0.25, k_nearest=5)
        assert (config.coreset_sampling_ratio, config.num_neighbors) == (0.25, 5)
        assert (config.coreset_ratio, config.k_nearest) == (0.25, 5)

    def test_unset_aliases_read_back_the_released_defaults(self) -> None:
        config = PatchCoreConfig()
        assert (config.coreset_sampling_ratio, config.coreset_ratio) == (0.1, 0.1)
        assert (config.num_neighbors, config.k_nearest) == (9, 9)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"coreset_ratio": 0.25, "k_nearest": 5},
            {"coreset_sampling_ratio": 0.3, "num_neighbors": 4},
            {"coreset_sampling_ratio": 0.1, "coreset_ratio": 0.25},
            {"num_neighbors": 9, "k_nearest": 3, "seed": 7},
        ],
    )
    def test_asdict_and_replace_round_trip(self, kwargs: dict[str, Any]) -> None:
        config = PatchCoreConfig(**kwargs)
        assert PatchCoreConfig(**asdict(config)) == config
        assert replace(config) == config
        assert copy.deepcopy(config) == config

    def test_either_name_can_be_mutated_after_construction(self) -> None:
        config = PatchCoreConfig()
        config.k_nearest = 4
        assert config.num_neighbors == 4
        config.num_neighbors = 6
        assert config.k_nearest == 6
        config.coreset_ratio = 0.5
        assert config.coreset_sampling_ratio == 0.5

    def test_positional_arguments_count_as_given(self) -> None:
        """The first positional field is ``backbone``; nothing here is an alias."""
        config = PatchCoreConfig("resnet18", (SIZE, SIZE))
        assert config.backbone == "resnet18"
        assert config.image_size == (SIZE, SIZE)
        assert (config.coreset_ratio, config.k_nearest) == (0.1, 9)

    def test_the_detector_checkpoint_keeps_a_value_set_back_to_the_default(
        self, tmp_path: Path
    ) -> None:
        """End to end: fit, set ``num_neighbors`` back to 9, save, load."""
        detector = _tiny_detector("patchcore", num_neighbors=5)
        detector.fit(_images(4))
        detector.patchcore_config.num_neighbors = 9
        path = tmp_path / "patchcore.pt"
        detector.save(str(path))
        fresh = _tiny_detector("patchcore").load(str(path))
        assert fresh.config.num_neighbors == 9


class TestCFlowConfigAliases:
    """``n_flows`` mirrors ``num_flows``; ``hidden_ratio`` scales ``hidden_dim``."""

    def test_replacing_hidden_dim_back_to_the_default_takes_effect(self) -> None:
        """``replace(..., hidden_dim=256)`` used to stay at 128."""
        config = replace(CFlowConfig(hidden_ratio=0.5), hidden_dim=256)
        assert config.hidden_dim == 256
        assert config.hidden_ratio == 1.0

    def test_replacing_num_flows_back_to_the_default_takes_effect(self) -> None:
        config = replace(CFlowConfig(n_flows=4), num_flows=8)
        assert (config.num_flows, config.n_flows) == (8, 8)

    def test_an_explicit_hidden_dim_wins_over_the_ratio(self) -> None:
        config = CFlowConfig(hidden_dim=256, hidden_ratio=0.5)
        assert config.hidden_dim == 256
        assert config.hidden_ratio == 1.0

    def test_the_ratio_given_alone_scales_the_default_width(self) -> None:
        config = CFlowConfig(n_flows=4, hidden_ratio=0.5)
        assert (config.num_flows, config.n_flows) == (4, 4)
        assert config.hidden_dim == 128
        assert config.hidden_ratio == 0.5

    def test_unset_aliases_read_back_the_released_defaults(self) -> None:
        config = CFlowConfig()
        assert (config.num_flows, config.n_flows) == (8, 8)
        assert (config.hidden_dim, config.hidden_ratio) == (256, 1.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"n_flows": 4, "hidden_ratio": 0.5},
            {"num_flows": 6, "hidden_dim": 100},
            {"hidden_ratio": 0.3},
            {"hidden_dim": 256, "hidden_ratio": 0.5},
        ],
    )
    def test_asdict_and_replace_round_trip(self, kwargs: dict[str, Any]) -> None:
        config = CFlowConfig(**kwargs)
        assert CFlowConfig(**asdict(config)) == config
        assert replace(config) == config
        assert copy.deepcopy(config) == config

    def test_either_name_can_be_mutated_after_construction(self) -> None:
        config = CFlowConfig()
        config.n_flows = 3
        assert config.num_flows == 3
        config.hidden_ratio = 0.25
        assert config.hidden_dim == 64
        config.hidden_dim = 128
        assert config.hidden_ratio == 0.5


# ---------------------------------------------------------------------------
# Checkpoint compatibility: a scoring/layout version, checked on load
# ---------------------------------------------------------------------------


def _rewrite_checkpoint(path: Path, mutate: Any) -> None:
    """Load a checkpoint as plain data, let ``mutate`` edit it, write it back."""
    checkpoint = safe_torch_load(str(path), map_location="cpu")
    mutate(checkpoint)
    torch.save(checkpoint, str(path))


def _fitted_checkpoint(name: str, tmp_path: Path, **overrides: Any) -> tuple[Any, Path]:
    torch.manual_seed(0)
    detector = _tiny_detector(name, **overrides)
    detector.fit(_images(4))
    path = tmp_path / f"{name}.pt"
    detector.save(str(path))
    return detector, path


class TestCheckpointCompatibility:
    """An incompatible checkpoint is refused with an actionable error."""

    @pytest.mark.parametrize("name", sorted(DETECTORS))
    def test_a_checkpoint_carries_its_scoring_version(self, name: str, tmp_path: Path) -> None:
        detector, path = _fitted_checkpoint(name, tmp_path)
        checkpoint = safe_torch_load(str(path), map_location="cpu")
        assert checkpoint["scoring_version"] == detector.scoring_version
        fresh = _tiny_detector(name).load(str(path))
        test = _images(2, seed=5)
        torch.testing.assert_close(
            torch.from_numpy(fresh.detect(test)["scores"]),
            torch.from_numpy(detector.detect(test)["scores"]),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_padim_refuses_the_old_small_grid_checkpoint(self, tmp_path: Path) -> None:
        """Old format: no version stamp, statistics on the smallest grid.

        This used to load and then die inside ``detect`` with ``The size of
        tensor a (64) must match the size of tensor b (4)``.
        """
        detector, path = _fitted_checkpoint("padim", tmp_path)
        h, w = detector._spatial_shape
        small = (h // 4) * (w // 4)

        def to_old_format(checkpoint: dict[str, Any]) -> None:
            del checkpoint["scoring_version"]
            state = checkpoint["fitted_state"]
            state["mean"] = state["mean"][:small].clone()
            state["cov_inv"] = state["cov_inv"][:small].clone()
            state["spatial_shape"] = (h // 4, w // 4)

        _rewrite_checkpoint(path, to_old_format)
        fresh = _tiny_detector("padim")
        with pytest.raises(visual.IncompatibleCheckpointError) as raised:
            fresh.load(str(path))
        message = str(raised.value)
        assert "PaDiM" in message
        assert "grid" in message
        assert "refit" in message.lower()
        assert not fresh._is_fitted
        assert fresh.mean is None

    def test_patchcore_refuses_a_checkpoint_scored_before_the_reweighting(
        self, tmp_path: Path
    ) -> None:
        """An unstamped PatchCore threshold is on the old, un-weighted score scale."""
        _, path = _fitted_checkpoint("patchcore", tmp_path)
        _rewrite_checkpoint(path, lambda checkpoint: checkpoint.pop("scoring_version"))
        fresh = _tiny_detector("patchcore")
        with pytest.raises(visual.IncompatibleCheckpointError) as raised:
            fresh.load(str(path))
        message = str(raised.value)
        assert "PatchCore" in message
        assert "re-weighted" in message
        assert "squared" in message
        assert "refit" in message.lower()
        assert not fresh._is_fitted
        assert fresh.memory_bank is None

    @pytest.mark.parametrize("name", ["cflow", "reverse_distillation", "stfpm"])
    def test_an_unstamped_checkpoint_loads_where_scoring_did_not_change(
        self, name: str, tmp_path: Path
    ) -> None:
        detector, path = _fitted_checkpoint(name, tmp_path)
        _rewrite_checkpoint(path, lambda checkpoint: checkpoint.pop("scoring_version"))
        fresh = _tiny_detector(name).load(str(path))
        test = _images(2, seed=5)
        torch.testing.assert_close(
            torch.from_numpy(fresh.detect(test)["scores"]),
            torch.from_numpy(detector.detect(test)["scores"]),
            rtol=1e-5,
            atol=1e-6,
        )

    @pytest.mark.parametrize("name", ["padim", "stfpm"])
    def test_a_checkpoint_from_a_newer_scoring_version_is_refused(
        self, name: str, tmp_path: Path
    ) -> None:
        detector, path = _fitted_checkpoint(name, tmp_path)

        def bump(checkpoint: dict[str, Any]) -> None:
            checkpoint["scoring_version"] = detector.scoring_version + 1

        _rewrite_checkpoint(path, bump)
        with pytest.raises(visual.IncompatibleCheckpointError, match="newer"):
            _tiny_detector(name).load(str(path))

    def test_padim_refuses_statistics_on_a_grid_it_does_not_extract(self, tmp_path: Path) -> None:
        """A stamped checkpoint whose grid still differs is refused, not mis-scored."""
        detector, path = _fitted_checkpoint("padim", tmp_path)
        h, w = detector._spatial_shape
        small = (h // 2) * (w // 2)

        def shrink_grid(checkpoint: dict[str, Any]) -> None:
            state = checkpoint["fitted_state"]
            state["mean"] = state["mean"][:small].clone()
            state["cov_inv"] = state["cov_inv"][:small].clone()
            state["spatial_shape"] = (h // 2, w // 2)

        _rewrite_checkpoint(path, shrink_grid)
        fresh = _tiny_detector("padim")
        with pytest.raises(visual.IncompatibleCheckpointError, match="grid"):
            fresh.load(str(path))
        assert not fresh._is_fitted
        assert fresh.mean is None

    @pytest.mark.parametrize(
        ("name", "built_with"),
        [
            ("patchcore", {"layers": ["layer1", "layer2"]}),
            ("padim", {"layers": ["layer2", "layer3"]}),
            ("stfpm", {"layers": ["layer3"]}),
        ],
    )
    def test_a_checkpoint_for_other_layers_is_refused(
        self, name: str, built_with: dict[str, Any], tmp_path: Path
    ) -> None:
        """The feature extractor is built at construction; ``load`` cannot swap it.

        Loading such a checkpoint used to end in a tensor-size error deep in
        ``detect`` or ``load_state_dict`` -- or, where the widths happened to
        agree, in scores from features the fitted state was never built on.
        """
        _, path = _fitted_checkpoint(name, tmp_path)
        other = _tiny_detector(name, **built_with)
        with pytest.raises(visual.IncompatibleCheckpointError, match="layers") as raised:
            other.load(str(path))
        assert "construct" in str(raised.value).lower()
        assert not other._is_fitted
        assert other.config.layers == built_with["layers"]  # untouched

    def test_a_checkpoint_for_another_backbone_is_refused(self, tmp_path: Path) -> None:
        _, path = _fitted_checkpoint("patchcore", tmp_path)

        def other_backbone(checkpoint: dict[str, Any]) -> None:
            checkpoint["config"]["backbone"] = "wide_resnet50_2"

        _rewrite_checkpoint(path, other_backbone)
        fresh = _tiny_detector("patchcore")
        with pytest.raises(visual.IncompatibleCheckpointError, match="wide_resnet50_2"):
            fresh.load(str(path))
        assert fresh.config.backbone == "resnet18"

    def test_a_loaded_detector_scores_with_the_checkpoints_own_settings(
        self, tmp_path: Path
    ) -> None:
        """Scoring settings come from the checkpoint, not from the loading instance.

        Regression: ``load`` replaced ``detector.config`` but left
        ``patchcore_config`` -- the reference ``detect`` reads -- on the
        constructor's config.  A bank fitted and calibrated with
        ``num_neighbors=1`` was re-weighted over 9 neighbours after loading
        into a default detector (measured: 2.61 saved vs 2.54 reloaded).
        """
        torch.manual_seed(0)
        detector = _tiny_detector("patchcore", num_neighbors=1)
        detector.fit(_images(4))
        path = tmp_path / "patchcore.pt"
        detector.save(str(path))

        fresh = _tiny_detector("patchcore").load(str(path))
        assert fresh.patchcore_config.num_neighbors == 1
        test = _images(2, seed=5)
        torch.testing.assert_close(
            torch.from_numpy(fresh.detect(test)["scores"]),
            torch.from_numpy(detector.detect(test)["scores"]),
            rtol=1e-6,
            atol=1e-6,
        )

    @pytest.mark.parametrize(
        ("name", "attribute"),
        [
            ("cflow", "cflow_config"),
            ("padim", "padim_config"),
            ("patchcore", "patchcore_config"),
            ("reverse_distillation", "rd_config"),
            ("stfpm", "stfpm_config"),
        ],
    )
    def test_every_config_reference_follows_the_loaded_config(
        self, name: str, attribute: str, tmp_path: Path
    ) -> None:
        _, path = _fitted_checkpoint(name, tmp_path, false_alarm_rate=0.2)
        fresh = _tiny_detector(name).load(str(path))
        assert getattr(fresh, attribute) is fresh.config
        assert fresh.visual_config is fresh.config
        assert getattr(fresh, attribute).false_alarm_rate == 0.2

    def test_a_refused_load_leaves_the_detector_as_it_was(self, tmp_path: Path) -> None:
        _, path = _fitted_checkpoint("patchcore", tmp_path)
        _rewrite_checkpoint(path, lambda checkpoint: checkpoint.pop("scoring_version"))
        torch.manual_seed(1)
        fitted = _tiny_detector("patchcore", num_neighbors=3)
        fitted.fit(_images(4, seed=2))
        test = _images(2, seed=5)
        before = fitted.detect(test)["scores"]
        with pytest.raises(visual.IncompatibleCheckpointError):
            fitted.load(str(path))
        assert fitted.config.num_neighbors == 3
        torch.testing.assert_close(
            torch.from_numpy(fitted.detect(test)["scores"]), torch.from_numpy(before)
        )


# ---------------------------------------------------------------------------
# PaDiM fit: project per batch, accumulate only the reduced features
# ---------------------------------------------------------------------------


def _reference_padim_statistics(detector: Any, train: Any) -> tuple[Any, Any, Any]:
    """The one-shot algorithm: stack every unprojected feature, then reduce.

    Features are extracted in the detector's own batches (convolution output
    depends on the batch at the 1e-6 level), so the only difference from
    ``fit`` is where the projection is applied.
    """
    data = detector.preprocess(train)
    batch_size = detector.padim_config.batch_size
    collected = []
    for start in range(0, len(data), batch_size):
        with torch.no_grad():
            features = detector.backbone(data[start : start + batch_size])
        collected.append(detector._aggregate_features(features)[0])
    patches = torch.cat(collected, dim=0)
    generator = torch.Generator(device="cpu").manual_seed(detector.padim_config.seed)
    projection = None
    if patches.shape[2] > detector.padim_config.d_reduced:
        projection = torch.randn(
            patches.shape[2], detector.padim_config.d_reduced, generator=generator
        )
        projection, _ = torch.linalg.qr(projection)
        patches = torch.matmul(patches, projection)
    mean = patches.mean(dim=0)
    centered = patches - mean.unsqueeze(0)
    cov = torch.einsum("npd,npe->pde", centered, centered) / (patches.shape[0] - 1)
    identity = torch.eye(patches.shape[2]).unsqueeze(0)
    return projection, mean, torch.linalg.inv(cov + detector.padim_config.epsilon * identity)


class TestPaDiMFitMemory:
    """``fit`` reduces each batch before it is accumulated."""

    def test_fit_never_holds_more_than_a_batch_of_unprojected_features(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every tensor wider than ``d_reduced`` is at most one batch long.

        Regression: ``torch.cat`` built an ``[N, H*W, C]`` tensor of the
        unprojected features of ALL images, and one ``matmul`` reduced it.
        """
        detector = _tiny_detector("padim", d_reduced=16, batch_size=2)
        train = _images(6)
        wide: list[tuple[int, ...]] = []
        real_cat, real_matmul = torch.cat, torch.matmul

        def _note(tensor: Any) -> None:
            if tensor.dim() == 3 and tensor.shape[2] > 16:
                wide.append(tuple(tensor.shape))

        def _cat(tensors: Any, *args: Any, **kwargs: Any) -> Any:
            out = real_cat(tensors, *args, **kwargs)
            _note(out)
            return out

        def _matmul(left: Any, right: Any) -> Any:
            _note(left)
            return real_matmul(left, right)

        # ``padim`` calls these as ``torch.cat`` / ``torch.matmul``.
        monkeypatch.setattr(torch, "cat", _cat)
        monkeypatch.setattr(torch, "matmul", _matmul)
        detector.fit(train)

        assert wide, "the spies saw no unprojected features at all"
        assert max(shape[0] for shape in wide) <= 2, wide

    def test_the_accumulated_features_have_the_projected_width(self) -> None:
        detector = _tiny_detector("padim", d_reduced=16, batch_size=2)
        accumulated = detector._accumulate_reduced_features(detector.preprocess(_images(5)))
        h, w = detector._spatial_shape
        assert accumulated.shape == (5, h * w, 16)
        assert detector.projection is not None
        assert detector.projection.shape[1] == 16

    def test_no_projection_when_the_features_are_already_narrow(self) -> None:
        detector = _tiny_detector("padim", d_reduced=4096, batch_size=2)
        accumulated = detector._accumulate_reduced_features(detector.preprocess(_images(3)))
        assert detector.projection is None
        assert accumulated.shape[2] == detector.feature_dim

    @pytest.mark.parametrize(("d_reduced", "batch_size"), [(16, 2), (100, 3), (4096, 4)])
    def test_per_batch_projection_matches_the_one_shot_algorithm(
        self, d_reduced: int, batch_size: int
    ) -> None:
        """Same seed, same data: identical projection and Gaussians (atol 1e-6)."""
        train = _images(7)
        detector = _tiny_detector("padim", d_reduced=d_reduced, batch_size=batch_size, seed=3)
        detector.fit(train)
        projection, mean, cov_inv = _reference_padim_statistics(detector, train)

        if projection is None:
            assert detector.projection is None
        else:
            torch.testing.assert_close(detector.projection, projection, rtol=0.0, atol=0.0)
        torch.testing.assert_close(detector.mean, mean, rtol=0.0, atol=1e-6)
        torch.testing.assert_close(detector.cov_inv, cov_inv, rtol=1e-5, atol=1e-6)

    def test_a_refit_replaces_the_projection_and_the_grid(self) -> None:
        """A second fit must not keep the first fit's projection or grid."""
        detector = _tiny_detector("padim", d_reduced=16)
        detector.fit(_images(4))
        detector.padim_config.d_reduced = 4096
        detector.fit(_images(4))
        assert detector.projection is None
        assert detector.mean is not None
        assert detector.mean.shape[1] == detector.feature_dim
