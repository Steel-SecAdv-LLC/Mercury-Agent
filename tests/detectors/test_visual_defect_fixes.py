# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the audited visual-detector defects (A1-A11).

Each test here was written to fail on the defective code and passes once the
defect is fixed:

* A1  -- the Reverse Distillation student encoder never trained because
  ``FeatureExtractor.forward`` always ran under ``torch.no_grad``.
* A2  -- every detector read ``original_size`` from the raw tensor before
  ``preprocess`` permuted channel-last input, so ``[B, H, W, C]`` batches got
  anomaly maps resized to ``(W, C)``.
* A3  -- CFlow's 2D positional encoding ignored horizontal position.
* A4  -- ``extract_features`` lazily registered an unseeded
  ``_fusion_projection`` Linear that broke strict ``load_state_dict`` and made
  fusion features irreproducible; PatchCore also skipped the neighborhood
  aggregation that ``detect`` applies.
* A5/A6 -- PatchCore coreset selection and k-NN querying were quadratic; the
  faiss branch crashed on CPU-only faiss builds and returned squared L2 while
  the fallback returned plain L2.
* A7  -- the PatchCore image score was a plain 1-NN max despite the
  re-weighting claim; alias config fields were asserted but never read.
* A9  -- PaDiM's random projection was unseeded and its covariance used a
  Python loop.
* A10 -- ``preprocess`` scaled by the batch max (batch-dependent); dead
  config fields ``normalize_features``/``pool_strategy`` were never read.
  The "uint8 is always divided by 255" rule is tested through the public
  ``fit`` / ``detect`` / ``extract_features`` path: every entry point used to
  cast NumPy input to float before ``preprocess`` could see its dtype.
* A11 -- repeated detector construction re-loaded the backbone from disk.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from scipy.spatial.distance import cdist

from omni_mercury_engine.detectors import visual
from omni_mercury_engine.detectors.visual.backbone import FeatureExtractor

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.visual

SIZE = 32
#: Shared tiny-budget settings: fixed threshold skips split-conformal
#: calibration (not under test here) and keeps every fit within seconds.
TINY: dict[str, Any] = {
    "backbone": "resnet18",
    "image_size": (SIZE, SIZE),
    "threshold": 1.0e9,
    "batch_size": 4,
}

#: detector name -> (class, extra tiny-budget config)
DETECTORS: dict[str, tuple[Any, dict[str, Any]]] = {
    "patchcore": (visual.PatchCoreDetector, {"seed": 0}),
    "padim": (visual.PaDiMDetector, {}),
    "stfpm": (visual.STFPMDetector, {"num_epochs": 1}),
    "reverse_distillation": (visual.ReverseDistillationDetector, {"num_epochs": 1}),
    "cflow": (visual.CFlowDetector, {"num_epochs": 1}),
}


def _tiny_detector(name: str, **overrides: Any) -> Any:
    """Build a detector under the shared tiny-budget config."""
    cls, extra = DETECTORS[name]
    return cls({**TINY, **extra, **overrides})


def _train_batch(n: int = 4) -> Any:
    """A deterministic channel-first training batch."""
    generator = torch.Generator().manual_seed(0)
    return torch.rand(n, 3, 64, 64, generator=generator)


# ---------------------------------------------------------------------------
# A1: the student encoder must actually train
# ---------------------------------------------------------------------------


class TestStudentTraining:
    """A1: trainable FeatureExtractor paths must carry gradients."""

    def test_feature_extractor_trainable_flag_gives_grad(self) -> None:
        """``trainable=True`` outputs carry a graph; the default stays frozen."""
        trainable = FeatureExtractor(
            backbone_name="resnet18",
            layers=["layer1", "layer2"],
            pretrained=False,
            trainable=True,
        )
        features = trainable(torch.rand(1, 3, SIZE, SIZE))
        assert all(feat.requires_grad for feat in features.values())
        assert all(p.requires_grad for p in trainable.parameters())

        frozen = FeatureExtractor(
            backbone_name="resnet18", layers=["layer1", "layer2"], pretrained=False
        )
        features = frozen(torch.rand(1, 3, SIZE, SIZE))
        assert not any(feat.requires_grad for feat in features.values())
        assert not any(p.requires_grad for p in frozen.parameters())

    def test_reverse_distillation_student_encoder_trains(self) -> None:
        """After ``fit`` the student encoder's weights changed and grads flowed.

        Regression: ``FeatureExtractor.forward`` always ran under
        ``torch.no_grad``, so the student encoder passed to Adam received no
        gradient and stayed byte-identical to its initialization.
        """
        torch.manual_seed(0)
        detector = _tiny_detector("reverse_distillation", batch_size=2)
        before = {
            name: param.detach().clone()
            for name, param in detector.student_encoder.named_parameters()
        }
        detector.fit(_train_batch(2))
        after = dict(detector.student_encoder.named_parameters())
        assert any(not torch.equal(before[name], after[name].detach()) for name in before)
        grads = [p.grad for p in detector.student_encoder.parameters()]
        assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)

    def test_stfpm_student_trains(self) -> None:
        """STFPM's student (its own ``StudentNetwork``) trains; guard it stays so."""
        torch.manual_seed(0)
        detector = _tiny_detector("stfpm", batch_size=2)
        before = {
            name: param.detach().clone() for name, param in detector.student.named_parameters()
        }
        detector.fit(_train_batch(2))
        after = dict(detector.student.named_parameters())
        assert any(not torch.equal(before[name], after[name].detach()) for name in before)


# ---------------------------------------------------------------------------
# A2: channel-last input must yield (H, W) anomaly maps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_channel_last_input_yields_hw_anomaly_maps(name: str) -> None:
    """A [2, 64, 64, 3] channel-last batch gets (2, 64, 64) anomaly maps.

    Regression: every detector read ``original_size`` from the raw tensor
    before ``preprocess`` permuted channel-last input, producing maps resized
    to (W, C) = (64, 3).
    """
    torch.manual_seed(0)
    detector = _tiny_detector(name)
    detector.fit(_train_batch(4))

    generator = torch.Generator().manual_seed(7)
    channel_last = torch.rand(2, 64, 64, 3, generator=generator)
    result = detector.detect(channel_last)
    assert result["anomaly_maps"].shape == (2, 64, 64)


# ---------------------------------------------------------------------------
# A3: CFlow positional encoding must vary along both axes
# ---------------------------------------------------------------------------


def test_cflow_positional_encoding_varies_along_both_axes() -> None:
    """The x-channels vary across columns and the y-channels across rows.

    Regression: ``x_pos.T`` had shape [max, 1] exactly like ``y_pos``, so the
    sin/cos x-terms varied along rows and horizontal position was ignored.
    """
    from omni_mercury_engine.detectors.visual.cflow import PositionalEncoding2D

    encoding = PositionalEncoding2D(8, max_size=16)
    pe = encoding(16, 16)[0]  # [8, 16, 16]

    # Channels 4i and 4i+1 encode y: they vary down rows.
    for channel in (0, 1, 4, 5):
        assert float(pe[channel].var(dim=0).max()) > 1e-8, f"y-channel {channel} flat over rows"
    # Channels 4i+2 and 4i+3 encode x: they must vary across columns.
    for channel in (2, 3, 6, 7):
        assert (
            float(pe[channel].var(dim=1).max()) > 1e-8
        ), f"x-channel {channel} ignores horizontal position"


# ---------------------------------------------------------------------------
# A4: deterministic fusion projection, clean save/load
# ---------------------------------------------------------------------------


class TestFusionProjection:
    """A4: the fusion projection must be deterministic and survive save/load."""

    def test_fit_extract_save_load_detect_round_trip(self, tmp_path: Path) -> None:
        """fit -> extract_features -> save -> load into a fresh instance -> detect.

        Regression: ``extract_features`` registered a lazy ``_fusion_projection``
        Linear on the module, which landed in ``state_dict`` and crashed strict
        ``load_state_dict`` on a fresh instance (Unexpected key(s)).
        """
        torch.manual_seed(0)
        detector = _tiny_detector("patchcore")
        train = _train_batch(4)
        detector.fit(train)
        features_before = detector.extract_features(train[:2])

        path = tmp_path / "patchcore.pt"
        detector.save(str(path))
        fresh = _tiny_detector("patchcore").load(str(path))
        result = fresh.detect(train[:2])
        assert np.isfinite(result["scores"]).all()

        features_after = fresh.extract_features(train[:2])
        torch.testing.assert_close(features_after, features_before, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("name", ["patchcore", "padim"])
    def test_extract_features_identical_across_instances(self, name: str) -> None:
        """Two instances give identical fusion features despite different RNG state.

        Regression: the lazy projection was initialized from the ambient global
        RNG at first call, so two detectors disagreed on the same input.
        """
        first = _tiny_detector(name)
        second = _tiny_detector(name)
        generator = torch.Generator().manual_seed(3)
        images = torch.rand(2, 3, 64, 64, generator=generator)

        torch.manual_seed(111)
        features_first = first.extract_features(images)
        torch.manual_seed(2222)
        torch.randn(1000)  # perturb the global RNG between the two calls
        features_second = second.extract_features(images)
        torch.testing.assert_close(features_second, features_first, rtol=0.0, atol=0.0)

    def test_patchcore_extract_features_applies_neighborhood_aggregation(self) -> None:
        """``extract_features`` pools the same locally-aware patches as ``detect``.

        Regression: ``detect`` applied ``_apply_local_neighborhood_aggregation``
        but ``extract_features`` skipped it, so fusion features came from a
        different embedding than the scores.
        """
        detector = _tiny_detector("patchcore")
        calls: list[tuple[int, ...]] = []
        original = detector._apply_local_neighborhood_aggregation

        def _spy(patches: Any, patch_shape: tuple[int, int], **kwargs: Any) -> Any:
            calls.append(tuple(patches.shape))
            return original(patches, patch_shape, **kwargs)

        detector._apply_local_neighborhood_aggregation = _spy
        detector.extract_features(_train_batch(2))
        assert calls, "extract_features never applied the neighborhood aggregation"


# ---------------------------------------------------------------------------
# A5/A6: coreset and k-NN querying
# ---------------------------------------------------------------------------


class TestPatchCoreNeighbours:
    """A5/A6: coreset selection and the two k-NN query branches."""

    def test_coreset_rows_come_from_the_input_deterministically(self) -> None:
        """The coreset is a unique subset of the input and seeded-deterministic."""
        detector = _tiny_detector("patchcore", seed=5)
        generator = torch.Generator().manual_seed(1)
        embeddings = torch.rand(200, 8, generator=generator)
        first = detector._greedy_coreset_sampling(embeddings, 0.1)
        second = _tiny_detector("patchcore", seed=5)._greedy_coreset_sampling(embeddings, 0.1)
        assert first.shape == (20, 8)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        rows = {tuple(np.round(r, 6)) for r in first.cpu().numpy()}
        pool = {tuple(np.round(r, 6)) for r in embeddings.cpu().numpy()}
        assert len(rows) == 20 and rows <= pool

    def test_fallback_knn_matches_naive_cdist(self) -> None:
        """The chunked fallback equals naive plain-L2 distances (atol 1e-6)."""
        detector = _tiny_detector("patchcore")
        rng = np.random.default_rng(0)
        bank = rng.normal(size=(157, 12)).astype(np.float32)
        detector.memory_bank = torch.from_numpy(bank)
        detector._build_nn_index(detector.memory_bank)

        queries = rng.normal(size=(73, 12)).astype(np.float32)
        distances, indices = detector._query_nn(torch.from_numpy(queries), k=3)
        naive = np.sort(cdist(queries.astype(np.float64), bank.astype(np.float64)), axis=1)[:, :3]
        np.testing.assert_allclose(distances, naive, atol=1e-6)
        assert indices.shape == (73, 3)

    def test_cpu_only_faiss_returns_plain_l2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A CPU-only faiss (no GpuIndexFlatL2) works and returns plain L2.

        Regression: ``isinstance(..., (faiss.IndexFlatL2, faiss.GpuIndexFlatL2))``
        raised AttributeError on faiss-cpu and the code fell through to a
        ``.kneighbors`` call that a faiss index does not have; and the faiss
        branch returned SQUARED L2 while the fallback returned plain L2.
        """

        class _StubIndexFlatL2:
            """Mimics faiss.IndexFlatL2: ``search`` returns SQUARED L2."""

            def __init__(self, dimension: int) -> None:
                self.dimension = dimension
                self._bank: np.ndarray[Any, Any] | None = None

            def add(self, vectors: np.ndarray[Any, Any]) -> None:
                self._bank = np.asarray(vectors, dtype=np.float32)

            def search(
                self, queries: np.ndarray[Any, Any], k: int
            ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
                assert self._bank is not None
                squared = ((queries[:, None, :] - self._bank[None]) ** 2).sum(axis=-1)
                order = np.argsort(squared, axis=1)[:, :k]
                return np.take_along_axis(squared, order, axis=1), order

        stub = types.ModuleType("faiss")
        stub.IndexFlatL2 = _StubIndexFlatL2  # type: ignore[attr-defined]
        # Deliberately no GpuIndexFlatL2 / StandardGpuResources: faiss-cpu.
        monkeypatch.setitem(sys.modules, "faiss", stub)

        detector = _tiny_detector("patchcore")
        rng = np.random.default_rng(1)
        bank = rng.normal(size=(50, 6)).astype(np.float32)
        detector.memory_bank = torch.from_numpy(bank)
        detector._build_nn_index(detector.memory_bank)
        assert isinstance(detector._nn_index, _StubIndexFlatL2)

        queries = rng.normal(size=(9, 6)).astype(np.float32)
        distances, _ = detector._query_nn(torch.from_numpy(queries), k=2)
        naive = np.sort(cdist(queries.astype(np.float64), bank.astype(np.float64)), axis=1)[:, :2]
        np.testing.assert_allclose(distances, naive, atol=1e-5)


# ---------------------------------------------------------------------------
# A7: re-weighted image score and alias config fields
# ---------------------------------------------------------------------------


class TestPatchCoreScoringAndAliases:
    """A7: the image score matches the published re-weighting; aliases work."""

    def test_image_score_is_reweighted_below_plain_max(self) -> None:
        """detect() scores are the re-weighted max-patch distance, not the plain max."""
        torch.manual_seed(0)
        detector = _tiny_detector("patchcore", num_neighbors=5)
        train = _train_batch(4)
        detector.fit(train)

        generator = torch.Generator().manual_seed(9)
        test = torch.rand(2, 3, 64, 64, generator=generator)
        scores = detector.detect(test)["scores"]

        data = detector.preprocess(test)
        with torch.no_grad():
            features = detector.backbone(data)
        patches, patch_shape = detector._aggregate_features(features)
        patches = detector._apply_local_neighborhood_aggregation(patches, patch_shape)
        distances, _ = detector._query_nn(patches.reshape(-1, patches.shape[-1]), k=1)
        plain_max = distances.reshape(2, -1).max(axis=1)

        assert np.all(plain_max > 0)
        assert np.all(scores < plain_max), "image score is the plain 1-NN max, not re-weighted"
        assert np.all(scores > 0)

    def test_patchcore_alias_fields_are_wired(self) -> None:
        """Setting coreset_ratio / k_nearest affects the canonical fields."""
        from omni_mercury_engine.detectors.visual.patchcore import PatchCoreConfig

        config = PatchCoreConfig(coreset_ratio=0.05, k_nearest=3)
        assert config.coreset_sampling_ratio == 0.05
        assert config.num_neighbors == 3
        # Values asserted by the released test surface stay readable.
        assert config.coreset_ratio == 0.05
        assert config.k_nearest == 3
        # Canonical fields win and the aliases follow them.
        canonical = PatchCoreConfig(coreset_sampling_ratio=0.2, num_neighbors=4)
        assert canonical.coreset_ratio == 0.2
        assert canonical.k_nearest == 4

    def test_cflow_alias_fields_are_wired(self) -> None:
        """Setting n_flows / hidden_ratio affects the canonical fields."""
        from omni_mercury_engine.detectors.visual.cflow import CFlowConfig

        config = CFlowConfig(n_flows=4, hidden_ratio=0.5)
        assert config.num_flows == 4
        assert config.hidden_dim == 128
        assert config.n_flows == 4
        assert config.hidden_ratio == 0.5
        canonical = CFlowConfig(num_flows=6)
        assert canonical.n_flows == 6


# ---------------------------------------------------------------------------
# A9: PaDiM seeding and covariance
# ---------------------------------------------------------------------------


class TestPaDiM:
    """A9: seeded projection and einsum covariance."""

    def test_projection_is_seeded_from_config(self) -> None:
        """Two fits under the same config seed agree despite global RNG drift."""
        train = _train_batch(4)
        torch.manual_seed(100)
        first = _tiny_detector("padim", seed=1)
        first.fit(train)
        torch.manual_seed(4242)
        torch.randn(500)  # perturb the global RNG between the two fits
        second = _tiny_detector("padim", seed=1)
        second.fit(train)
        torch.testing.assert_close(second.projection, first.projection, rtol=0.0, atol=0.0)
        torch.testing.assert_close(second.mean, first.mean, rtol=0.0, atol=0.0)

        other = _tiny_detector("padim", seed=2)
        other.fit(train)
        assert not torch.equal(other.projection, first.projection)

    def test_aggregation_uses_largest_layer_resolution(self) -> None:
        """Per-layer maps are fused at layer1's (largest) grid, per the paper.

        Measured on the real-image task (tests/detectors/test_visual_real_images.py
        data): AUROC 1.0000 at both resolutions, so the paper-correct largest
        grid is kept for its finer localization.
        """
        detector = _tiny_detector("padim")
        detector.fit(_train_batch(4))
        with torch.no_grad():
            features = detector.backbone(detector.preprocess(_train_batch(1)))
        largest = max(tuple(feat.shape[-2:]) for feat in features.values())
        assert detector._spatial_shape == largest

    def test_covariance_matches_per_position_loop(self) -> None:
        """The einsum covariance equals the per-position loop (atol 1e-6)."""
        train = _train_batch(4)
        detector = _tiny_detector("padim", d_reduced=16, seed=0)
        detector.fit(train)

        data = detector.preprocess(train)
        with torch.no_grad():
            features = detector.backbone(data)
        patches, _ = detector._aggregate_features(features)
        if detector.projection is not None:
            patches = torch.matmul(patches, detector.projection)
        centered = patches - patches.mean(dim=0).unsqueeze(0)
        n_samples, n_positions, dim = patches.shape
        reference = torch.zeros(n_positions, dim, dim)
        for position in range(n_positions):
            pos_features = centered[:, position, :]
            reference[position] = torch.matmul(pos_features.T, pos_features) / (n_samples - 1)
        identity = torch.eye(dim).unsqueeze(0)
        reference_inv = torch.linalg.inv(reference + detector.padim_config.epsilon * identity)
        torch.testing.assert_close(detector.cov_inv, reference_inv, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# A10: explicit input scaling and the normalize/pool config fields
# ---------------------------------------------------------------------------


class TestPreprocessScaling:
    """A10: the [0, 1] scaling decision is explicit and per-sample."""

    @staticmethod
    def _expected(scaled: Any) -> Any:
        """ImageNet-normalize an already [0, 1]-scaled channel-first batch."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (scaled - mean) / std

    def test_float_input_slightly_above_one_is_not_divided(self) -> None:
        """Float input with max in (1, 1.5] is taken as-is, not divided by 255."""
        detector = _tiny_detector("padim", image_size=(SIZE, SIZE))
        generator = torch.Generator().manual_seed(0)
        images = torch.rand(2, 3, SIZE, SIZE, generator=generator) * 1.2
        torch.testing.assert_close(
            detector.preprocess(images), self._expected(images), rtol=1e-5, atol=1e-6
        )

    @staticmethod
    def _uint8_batch_with_a_near_black_frame(n: int, seed: int) -> Any:
        """Ordinary 8-bit frames, the first replaced by one whose values are 0/1.

        The near-black frame is the case a value heuristic gets wrong: its
        maximum is 1, so read as float it looks already normalized.
        """
        rng = np.random.default_rng(seed)
        images = rng.integers(0, 256, size=(n, 3, 64, 64)).astype(np.uint8)
        images[0] = rng.integers(0, 2, size=(3, 64, 64)).astype(np.uint8)
        return images

    @pytest.mark.parametrize("name", sorted(DETECTORS))
    def test_uint8_input_is_divided_by_255_through_detect(self, name: str) -> None:
        """detect() scores NumPy uint8 and torch uint8 input identically.

        Regression: every entry point ran ``torch.from_numpy(data).float()``
        before ``preprocess``, erasing the dtype, so a NumPy near-black frame
        fell to the ``max > 1.5`` rule and was scored as a full-contrast
        image -- while the same bytes as a torch tensor were divided by 255
        (measured with PaDiM: first-frame score 41.3 vs 68.7).
        """
        torch.manual_seed(0)
        detector = _tiny_detector(name)
        detector.fit(_train_batch(4))

        images = self._uint8_batch_with_a_near_black_frame(3, seed=0)
        from_numpy = detector.detect(images)
        from_torch = detector.detect(torch.from_numpy(images))
        as_unit_float = detector.detect(images.astype(np.float32) / 255.0)

        np.testing.assert_allclose(from_numpy["scores"], from_torch["scores"], rtol=0, atol=0)
        np.testing.assert_allclose(from_numpy["scores"], as_unit_float["scores"], rtol=1e-5)
        np.testing.assert_allclose(
            from_numpy["anomaly_maps"], from_torch["anomaly_maps"], rtol=0, atol=0
        )

    @pytest.mark.parametrize("name", sorted(DETECTORS))
    def test_uint8_input_is_divided_by_255_through_extract_features(self, name: str) -> None:
        """extract_features() agrees on NumPy uint8 and torch uint8 input."""
        torch.manual_seed(0)
        detector = _tiny_detector(name)
        detector.fit(_train_batch(4))
        images = self._uint8_batch_with_a_near_black_frame(3, seed=1)
        torch.testing.assert_close(
            detector.extract_features(images),
            detector.extract_features(torch.from_numpy(images)),
            rtol=0.0,
            atol=0.0,
        )

    @pytest.mark.parametrize("name", sorted(DETECTORS))
    def test_uint8_input_is_divided_by_255_through_fit(self, name: str) -> None:
        """fit() learns the same model from NumPy uint8 and torch uint8 images."""
        images = self._uint8_batch_with_a_near_black_frame(5, seed=2)
        torch.manual_seed(0)
        from_numpy = _tiny_detector(name).fit(images)
        torch.manual_seed(0)
        from_torch = _tiny_detector(name).fit(torch.from_numpy(images))
        if name in ("padim", "patchcore"):
            learned = "mean" if name == "padim" else "memory_bank"
            torch.testing.assert_close(
                getattr(from_numpy, learned), getattr(from_torch, learned), rtol=0.0, atol=0.0
            )
        test = torch.from_numpy(self._uint8_batch_with_a_near_black_frame(3, seed=3))
        np.testing.assert_allclose(
            from_numpy.detect(test)["scores"], from_torch.detect(test)["scores"], rtol=1e-5
        )

    def test_preprocess_itself_divides_uint8_of_either_kind(self) -> None:
        """The rule also holds for a direct ``preprocess`` call."""
        detector = _tiny_detector("padim", image_size=(SIZE, SIZE))
        rng = np.random.default_rng(0)
        images = rng.integers(0, 2, size=(2, 3, SIZE, SIZE)).astype(np.uint8)
        expected = self._expected(torch.from_numpy(images).float() / 255.0)
        torch.testing.assert_close(detector.preprocess(images), expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            detector.preprocess(torch.from_numpy(images)), expected, rtol=1e-5, atol=1e-6
        )

    def test_float_scaling_is_decided_per_sample(self) -> None:
        """A [0, 1] sample next to a [0, 255] sample keeps its own scale."""
        detector = _tiny_detector("padim", image_size=(SIZE, SIZE))
        generator = torch.Generator().manual_seed(0)
        unit = torch.rand(1, 3, SIZE, SIZE, generator=generator)
        eight_bit = torch.rand(1, 3, SIZE, SIZE, generator=generator) * 255.0
        out = detector.preprocess(torch.cat([unit, eight_bit]))
        expected = self._expected(torch.cat([unit, eight_bit / 255.0]))
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


class TestFusionConfigFields:
    """A10: ``normalize_features`` and ``pool_strategy`` are read."""

    def test_normalize_features_false_skips_l2_normalization(self) -> None:
        """With normalize_features=False the fusion embedding is not unit-norm."""
        normalized = _tiny_detector("patchcore")
        raw = _tiny_detector("patchcore", normalize_features=False)
        images = _train_batch(2)
        norms_on = torch.linalg.norm(normalized.extract_features(images), dim=1)
        norms_off = torch.linalg.norm(raw.extract_features(images), dim=1)
        torch.testing.assert_close(norms_on, torch.ones_like(norms_on), rtol=1e-4, atol=1e-4)
        assert not torch.allclose(norms_off, torch.ones_like(norms_off), atol=1e-3)

    def test_pool_strategy_max_changes_the_fusion_embedding(self) -> None:
        """pool_strategy='max' yields a different embedding than the default."""
        avg = _tiny_detector("patchcore")
        maxed = _tiny_detector("patchcore", pool_strategy="max")
        images = _train_batch(2)
        assert not torch.allclose(
            avg.extract_features(images), maxed.extract_features(images), atol=1e-6
        )


# ---------------------------------------------------------------------------
# A11: backbone cache
# ---------------------------------------------------------------------------


class TestBackboneCache:
    """A11: repeated construction reuses the loaded backbone without leaks."""

    def test_pretrained_backbone_is_loaded_once_and_isolated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two extractors trigger one torchvision load; copies stay independent."""
        from torchvision import models as tv_models

        from omni_mercury_engine.detectors.visual import backbone as backbone_module

        cache = getattr(backbone_module, "_BACKBONE_CACHE", None)
        assert isinstance(cache, dict), "module-level backbone cache is missing"
        cache.clear()

        calls: list[Any] = []
        real = tv_models.resnet18

        def _counting(*args: Any, **kwargs: Any) -> Any:
            calls.append(kwargs.get("weights"))
            return real(*args, **kwargs)

        monkeypatch.setattr(tv_models, "resnet18", _counting)

        first = FeatureExtractor(backbone_name="resnet18", layers=["layer1"])
        second = FeatureExtractor(backbone_name="resnet18", layers=["layer1"])
        assert len(calls) == 1, "second construction reloaded the backbone"

        # Mutating one copy must not leak into the cache or other copies.
        with torch.no_grad():
            next(iter(first.backbone.parameters())).add_(1.0)
        third = FeatureExtractor(backbone_name="resnet18", layers=["layer1"])
        p_second = next(iter(second.backbone.parameters()))
        p_third = next(iter(third.backbone.parameters()))
        torch.testing.assert_close(p_third, p_second, rtol=0.0, atol=0.0)
        assert not torch.equal(next(iter(first.backbone.parameters())), p_third)
        cache.clear()

    def test_trainable_student_does_not_come_from_the_cache(self) -> None:
        """A trainable (from-scratch) student bypasses the frozen cache."""
        from omni_mercury_engine.detectors.visual import backbone as backbone_module

        cache = getattr(backbone_module, "_BACKBONE_CACHE", None)
        assert isinstance(cache, dict)
        cache.clear()
        student = FeatureExtractor(
            backbone_name="resnet18", layers=["layer1"], pretrained=False, trainable=True
        )
        assert all(p.requires_grad for p in student.parameters())
        assert not cache, "a trainable / non-pretrained backbone must not be cached"


# ---------------------------------------------------------------------------
# A8: honest timm backbone support
# ---------------------------------------------------------------------------


class TestTimmBackbones:
    """A8: timm backbones honour the layer-dict contract or are not offered."""

    def test_efficientnet_b4_returns_named_feature_dict(self) -> None:
        """efficientnet_b4 yields layer2/layer3 maps with the documented dims.

        Regression: the timm features_only path registered hooks on layer2 /
        layer3 attributes EfficientNet lacks, so ``forward`` returned ``{}``
        and every detector crashed downstream with IndexError.
        """
        pytest.importorskip("timm")
        extractor = FeatureExtractor(
            backbone_name="efficientnet_b4",
            layers=["layer2", "layer3"],
            pretrained=False,
        )
        features = extractor(torch.rand(1, 3, 64, 64))
        assert set(features) == {"layer2", "layer3"}
        assert features["layer2"].shape[1] == 56
        assert features["layer3"].shape[1] == 160
        assert extractor.feature_dims == {"layer2": 56, "layer3": 160}

    def test_vit_base_returns_named_feature_dict(self) -> None:
        """vit_base_patch16_224 yields 768-channel maps for each named layer."""
        pytest.importorskip("timm")
        extractor = FeatureExtractor(
            backbone_name="vit_base_patch16_224",
            layers=["layer2", "layer3"],
            pretrained=False,
        )
        features = extractor(torch.rand(1, 3, 224, 224))
        assert set(features) == {"layer2", "layer3"}
        assert features["layer2"].shape == (1, 768, 14, 14)
        assert features["layer3"].shape == (1, 768, 14, 14)

    def test_dinov2_member_retained_but_construction_errors_clearly(self) -> None:
        """``DINO_V2`` is a released member, so it is kept, not removed.

        ``dinov2_vitb14`` is not a timm 1.x model name, so it never loaded.
        Rather than drop the released enum member (which would break any code
        that references it), it is retained and its construction raises a clear,
        actionable error naming the real timm DINOv2 variant.
        """
        from omni_mercury_engine.detectors.visual.backbone import get_backbone
        from omni_mercury_engine.detectors.visual.base_visual import BackboneType

        assert "dinov2_vitb14" in {member.value for member in BackboneType}
        with pytest.raises(ValueError, match="vit_base_patch14_dinov2"):
            get_backbone("dinov2_vitb14", pretrained=False)
