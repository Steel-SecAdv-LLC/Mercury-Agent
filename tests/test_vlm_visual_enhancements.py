"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisory LLC

Tests for VLM/Visual detector enhancements including:
- Advanced context providers (Semantic, Frequency, Appearance)
- LVLM backend cache with pre-warming
- Multi-modal fusion optimizer
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Multi-modal Fusion
from omni_mercury_engine.detectors.fusion.multimodal_fusion import (
    AdaptiveFusion,
    AttentionFusion,
    DecisionConfidenceFusion,
    FeatureConcatFusion,
    FusionResult,
    FusionStrategy,
    ModalityInput,
    MultiModalFusionOptimizer,
    ScoreWeightedFusion,
    create_fusion_optimizer,
)

# Context Providers
from omni_mercury_engine.detectors.vlm.advanced_context_providers import (
    AppearanceContextProvider,
    EnhancedCombinedContextProvider,
    FrequencyContextProvider,
    SemanticContextProvider,
)
from omni_mercury_engine.detectors.vlm.context_providers import ContextInfo

# LVLM Cache
from omni_mercury_engine.detectors.vlm.lvlm_cache import (
    CacheStatistics,
    LVLMBackendCache,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a sample frame for testing."""
    # Create 3-channel 64x64 image
    frame = np.random.rand(3, 64, 64).astype(np.float32)
    return frame


@pytest.fixture
def sample_video() -> np.ndarray:
    """Create sample video frames for testing."""
    # 16 frames, 3 channels, 64x64
    video = np.random.rand(16, 3, 64, 64).astype(np.float32)
    return video


@pytest.fixture
def sample_vlm_input() -> ModalityInput:
    """Create sample VLM modality input."""
    return ModalityInput(
        modality_type="vlm",
        detector_name="blip_vlm",
        features=torch.randn(4, 128),
        scores=np.array([0.3, 0.7, 0.5, 0.9]),
        predictions=np.array([0, 1, 1, 1]),
        confidence=np.array([0.8, 0.9, 0.6, 0.95]),
        explanations=[
            "Normal scene",
            "Unusual activity detected",
            "Possible anomaly",
            "Clear anomaly present",
        ],
    )


@pytest.fixture
def sample_visual_input() -> ModalityInput:
    """Create sample Visual modality input."""
    return ModalityInput(
        modality_type="visual",
        detector_name="patchcore",
        features=torch.randn(4, 256),
        scores=np.array([0.2, 0.8, 0.4, 0.85]),
        predictions=np.array([0, 1, 0, 1]),
        confidence=np.array([0.9, 0.85, 0.7, 0.9]),
        anomaly_maps=torch.randn(4, 64, 64),
    )


# =============================================================================
# Semantic Context Provider Tests
# =============================================================================


class TestSemanticContextProvider:
    """Tests for SemanticContextProvider."""

    def test_initialization(self) -> None:
        """Test provider initialization."""
        provider = SemanticContextProvider(
            complexity_threshold=0.5,
            edge_kernel_size=3,
            random_state=42,
        )
        assert provider.complexity_threshold == 0.5
        assert provider.edge_kernel_size == 3
        assert provider.rng is not None

    def test_extract_context_single_frame(self, sample_frame: np.ndarray) -> None:
        """Test context extraction from single frame."""
        provider = SemanticContextProvider(random_state=42)
        context = provider.extract_context(sample_frame)

        assert isinstance(context, ContextInfo)
        assert context.context_type == "semantic"
        assert context.description is not None
        assert context.features is not None
        assert len(context.features) == 6  # 6 semantic features
        assert "scene_type" in context.metadata

    def test_extract_context_video(self, sample_video: np.ndarray) -> None:
        """Test context extraction from video."""
        provider = SemanticContextProvider()
        context = provider.extract_context(sample_video)

        assert context.context_type == "semantic"
        assert context.metadata["scene_type"] in ["sparse", "moderate", "dense", "cluttered"]

    def test_extract_context_torch_tensor(self, sample_frame: np.ndarray) -> None:
        """Test context extraction from torch tensor."""
        provider = SemanticContextProvider()
        tensor = torch.from_numpy(sample_frame)
        context = provider.extract_context(tensor)

        assert context.context_type == "semantic"

    def test_format_context_prompt(self, sample_frame: np.ndarray) -> None:
        """Test prompt formatting."""
        provider = SemanticContextProvider()
        context = provider.extract_context(sample_frame)
        prompt = provider.format_context_prompt(context)

        assert "[Semantic Context:" in prompt
        assert context.description in prompt

    def test_scene_classification(self) -> None:
        """Test scene type classification."""
        provider = SemanticContextProvider()

        # Sparse scene (mostly uniform)
        sparse_frame = np.ones((3, 64, 64), dtype=np.float32) * 0.5
        context = provider.extract_context(sparse_frame)
        # Should be sparse or moderate
        assert context.metadata["scene_type"] in ["sparse", "moderate"]

    def test_reproducibility_with_seed(self, sample_frame: np.ndarray) -> None:
        """Test reproducibility with random state."""
        provider1 = SemanticContextProvider(random_state=42)
        provider2 = SemanticContextProvider(random_state=42)

        context1 = provider1.extract_context(sample_frame)
        context2 = provider2.extract_context(sample_frame)

        np.testing.assert_array_equal(context1.features, context2.features)


# =============================================================================
# Frequency Context Provider Tests
# =============================================================================


class TestFrequencyContextProvider:
    """Tests for FrequencyContextProvider."""

    def test_initialization(self) -> None:
        """Test provider initialization."""
        provider = FrequencyContextProvider(
            frequency_bins=32,
            periodicity_threshold=0.3,
            flicker_threshold=0.2,
        )
        assert provider.frequency_bins == 32
        assert provider.periodicity_threshold == 0.3

    def test_extract_context_single_frame(self, sample_frame: np.ndarray) -> None:
        """Test frequency analysis on single frame."""
        provider = FrequencyContextProvider()
        context = provider.extract_context(sample_frame)

        assert context.context_type == "frequency"
        assert context.features is not None
        assert len(context.features) == 6  # 6 frequency features
        assert "periodic_score" in context.metadata

    def test_extract_context_video(self, sample_video: np.ndarray) -> None:
        """Test temporal frequency analysis."""
        provider = FrequencyContextProvider()
        context = provider.extract_context(sample_video)

        assert context.context_type == "frequency"
        assert "temporal_periodicity" in context.metadata
        assert "flicker_detected" in context.metadata

    def test_periodic_pattern_detection(self) -> None:
        """Test detection of periodic patterns."""
        provider = FrequencyContextProvider()

        # Create striped pattern (periodic)
        frame = np.zeros((3, 64, 64), dtype=np.float32)
        for i in range(64):
            if i % 8 < 4:
                frame[:, i, :] = 1.0

        context = provider.extract_context(frame)
        # Should detect periodicity
        assert context.metadata["periodic_score"] > 0.1

    def test_flicker_detection(self) -> None:
        """Test temporal flicker detection."""
        provider = FrequencyContextProvider(flicker_threshold=0.1)

        # Create flickering video (alternating bright/dark)
        video = np.zeros((16, 3, 64, 64), dtype=np.float32)
        for t in range(16):
            video[t] = 1.0 if t % 2 == 0 else 0.0

        context = provider.extract_context(video)
        assert context.metadata["flicker_detected"] is True


# =============================================================================
# Appearance Context Provider Tests
# =============================================================================


class TestAppearanceContextProvider:
    """Tests for AppearanceContextProvider."""

    def test_initialization(self) -> None:
        """Test provider initialization."""
        provider = AppearanceContextProvider(
            color_bins=16,
            num_dominant_colors=5,
        )
        assert provider.color_bins == 16
        assert provider.num_dominant_colors == 5

    def test_extract_context(self, sample_frame: np.ndarray) -> None:
        """Test appearance feature extraction."""
        provider = AppearanceContextProvider()
        context = provider.extract_context(sample_frame)

        assert context.context_type == "appearance"
        assert context.features is not None
        assert "brightness_mean" in context.metadata
        assert "dominant_colors" in context.metadata

    def test_dominant_color_detection(self) -> None:
        """Test dominant color extraction."""
        provider = AppearanceContextProvider()

        # Create mostly red image
        frame = np.zeros((3, 64, 64), dtype=np.float32)
        frame[0] = 1.0  # Red channel

        context = provider.extract_context(frame)
        dominant = context.metadata["dominant_colors"]
        assert len(dominant) > 0

    def test_brightness_analysis(self) -> None:
        """Test brightness statistics."""
        provider = AppearanceContextProvider()

        # Dark image
        dark_frame = np.ones((3, 64, 64), dtype=np.float32) * 0.1
        dark_context = provider.extract_context(dark_frame)
        assert dark_context.metadata["brightness_mean"] < 0.3

        # Bright image
        bright_frame = np.ones((3, 64, 64), dtype=np.float32) * 0.9
        bright_context = provider.extract_context(bright_frame)
        assert bright_context.metadata["brightness_mean"] > 0.7

    def test_texture_entropy(self) -> None:
        """Test texture entropy calculation."""
        provider = AppearanceContextProvider()

        # Uniform texture (low entropy)
        uniform = np.ones((3, 64, 64), dtype=np.float32) * 0.5
        uniform_context = provider.extract_context(uniform)

        # Random texture (high entropy)
        random_tex = np.random.rand(3, 64, 64).astype(np.float32)
        random_context = provider.extract_context(random_tex)

        assert (
            random_context.metadata["texture_entropy"]
            >= uniform_context.metadata["texture_entropy"]
        )


# =============================================================================
# Enhanced Combined Context Provider Tests
# =============================================================================


class TestEnhancedCombinedContextProvider:
    """Tests for EnhancedCombinedContextProvider."""

    def test_initialization_all_enabled(self) -> None:
        """Test with all providers enabled."""
        provider = EnhancedCombinedContextProvider(
            enable_position=True,
            enable_temporal=True,
            enable_semantic=True,
            enable_frequency=True,
            enable_appearance=True,
        )
        assert len(provider.providers) == 5

    def test_initialization_partial(self) -> None:
        """Test with some providers disabled."""
        provider = EnhancedCombinedContextProvider(
            enable_position=True,
            enable_temporal=False,
            enable_semantic=True,
            enable_frequency=False,
            enable_appearance=True,
        )
        assert len(provider.providers) == 3

    def test_extract_all_context(self, sample_video: np.ndarray) -> None:
        """Test extracting all context types."""
        provider = EnhancedCombinedContextProvider()
        contexts = provider.extract_all_context(sample_video)

        assert "position" in contexts
        assert "temporal" in contexts
        assert "semantic" in contexts
        assert "frequency" in contexts
        assert "appearance" in contexts

    def test_extract_selective(self, sample_frame: np.ndarray) -> None:
        """Test selective context extraction."""
        provider = EnhancedCombinedContextProvider()
        contexts = provider.extract_all_context(
            sample_frame,
            context_types=["semantic", "appearance"],
        )

        assert "semantic" in contexts
        assert "appearance" in contexts
        assert "position" not in contexts

    def test_format_combined_prompt(self, sample_frame: np.ndarray) -> None:
        """Test combined prompt formatting."""
        provider = EnhancedCombinedContextProvider()
        contexts = provider.extract_all_context(sample_frame)
        prompt = provider.format_combined_prompt(contexts)

        assert "[Semantic Context:" in prompt or "[Appearance Analysis:" in prompt


# =============================================================================
# LVLM Backend Cache Tests
# =============================================================================


class TestLVLMBackendCache:
    """Tests for LVLMBackendCache."""

    @pytest.fixture(autouse=True)
    def reset_cache(self) -> None:
        """Reset cache singleton before each test."""
        LVLMBackendCache._instance = None

    def test_singleton_pattern(self) -> None:
        """Test singleton implementation."""
        cache1 = LVLMBackendCache()
        cache2 = LVLMBackendCache()
        assert cache1 is cache2

    def test_get_instance(self) -> None:
        """Test get_instance class method."""
        cache = LVLMBackendCache.get_instance()
        assert isinstance(cache, LVLMBackendCache)

    def test_cache_key_generation(self) -> None:
        """Test cache key generation."""
        cache = LVLMBackendCache()
        key = cache._cache_key("mock", "test-model")
        assert key == "mock:test-model"

        key_default = cache._cache_key("mock", None)
        assert key_default == "mock:mock"

    @patch("omni_mercury_engine.detectors.vlm.lvlm_cache.get_lvlm_backend")
    def test_get_creates_and_caches(self, mock_get_backend: MagicMock) -> None:
        """Test that get creates and caches backends."""
        mock_backend = MagicMock()
        mock_backend.model = MagicMock()
        mock_get_backend.return_value = mock_backend

        cache = LVLMBackendCache()

        # First call should load
        result1 = cache.get("mock", "test-model", device="cpu")
        assert mock_get_backend.called
        assert mock_backend.initialize.called

        # Second call should hit cache
        mock_get_backend.reset_mock()
        mock_backend.initialize.reset_mock()

        result2 = cache.get("mock", "test-model", device="cpu")
        assert result1 is result2
        assert not mock_get_backend.called

    def test_cache_statistics(self) -> None:
        """Test cache statistics tracking."""
        cache = LVLMBackendCache()
        stats = cache.get_stats()

        assert isinstance(stats, CacheStatistics)
        assert stats.total_requests >= 0
        assert stats.cache_hits >= 0

    def test_is_loaded(self) -> None:
        """Test is_loaded check."""
        cache = LVLMBackendCache()
        assert not cache.is_loaded("nonexistent", "model")

    def test_list_models(self) -> None:
        """Test listing cached models."""
        cache = LVLMBackendCache()
        models = cache.list_models()
        assert isinstance(models, list)

    @patch("omni_mercury_engine.detectors.vlm.lvlm_cache.get_lvlm_backend")
    def test_eviction(self, mock_get_backend: MagicMock) -> None:
        """Test model eviction."""
        mock_backend = MagicMock()
        mock_backend.model = MagicMock()
        mock_get_backend.return_value = mock_backend

        cache = LVLMBackendCache(max_models=1)

        # Load first model
        cache.get("mock", "model1", device="cpu")
        assert cache.is_loaded("mock", "model1")

        # Manually evict
        result = cache.evict("mock", "model1")
        assert result is True
        assert not cache.is_loaded("mock", "model1")

    def test_clear(self) -> None:
        """Test clearing all cached models."""
        cache = LVLMBackendCache()
        cache.clear()
        stats = cache.get_stats()
        assert stats.models_loaded == 0

    def test_callbacks(self) -> None:
        """Test callback registration."""
        cache = LVLMBackendCache()

        load_called = []
        evict_called = []
        error_called = []

        cache.on_load(lambda key, time: load_called.append(key))
        cache.on_evict(lambda key: evict_called.append(key))
        cache.on_error(lambda key, err: error_called.append(key))

        assert len(cache._on_load_callbacks) > 0
        assert len(cache._on_evict_callbacks) > 0
        assert len(cache._on_error_callbacks) > 0


# =============================================================================
# Multi-Modal Fusion Tests
# =============================================================================


class TestFeatureConcatFusion:
    """Tests for FeatureConcatFusion."""

    def test_basic_fusion(
        self,
        sample_vlm_input: ModalityInput,
        sample_visual_input: ModalityInput,
    ) -> None:
        """Test basic feature concatenation."""
        fusion = FeatureConcatFusion()
        result = fusion.fuse([sample_vlm_input, sample_visual_input])

        assert isinstance(result, FusionResult)
        assert result.fused_scores is not None
        assert result.fused_predictions is not None
        assert result.fused_features is not None
        # 128 (VLM) + 256 (Visual) = 384
        assert result.fused_features.shape[-1] == 384

    def test_with_projection(
        self,
        sample_vlm_input: ModalityInput,
        sample_visual_input: ModalityInput,
    ) -> None:
        """Test with output projection."""
        fusion = FeatureConcatFusion(output_dim=128, normalize=True)
        result = fusion.fuse([sample_vlm_input, sample_visual_input])

        assert result.fused_features.shape[-1] == 128
        # Check normalization
        norms = torch.norm(result.fused_features, dim=-1)
        np.testing.assert_array_almost_equal(norms.numpy(), np.ones(4), decimal=5)


class TestScoreWeightedFusion:
    """Tests for ScoreWeightedFusion."""

    def test_equal_weights(
        self,
        sample_vlm_input: ModalityInput,
        sample_visual_input: ModalityInput,
    ) -> None:
        """Test with equal weights."""
        fusion = ScoreWeightedFusion(uncertainty_weighting=False)
        result = fusion.fuse([sample_vlm_input, sample_visual_input])

        # Should be average of scores
        expected = (sample_vlm_input.scores + sample_visual_input.scores) / 2
        np.testing.assert_array_almost_equal(result.fused_scores, expected)

    def test_custom_weights(
        self,
        sample_vlm_input: ModalityInput,
        sample_visual_input: ModalityInput,
    ) -> None:
        """Test with custom weights."""
        fusion = ScoreWeightedFusion(
            weights={"blip_vlm": 0.7, "patchcore": 0.3},
            uncertainty_weighting=False,
        )
        result = fusion.fuse([sample_vlm_input, sample_visual_input])

        expected = 0.7 * sample_vlm_input.scores + 0.3 * sample_visual_input.scores
        np.testing.assert_array_almost_equal(result.fused_scores, expected)

    def test_uncertainty_weighting(
        self,
        sample_vlm_input: ModalityInput,
        sample_visual_input: ModalityInput,
    ) -> None:
        """Test uncertainty-aware weighting."""
        fusion = ScoreWeightedFusion(uncertainty_weighting=True)
        result = fusion.fuse([sample_vlm_input, sample_visual_input])

        # Weights should be influenced by confidence
        assert len(result.modality_weights) == 2


class TestDecisionConfidenceFusion:
    """Tests for DecisionConfidenceFusion."""

    def test_basic_fusion(
        self,
        sample_vlm_input: ModalityInput,
        sample_visual_input: ModalityInput,
    ) -> None:
        """Test basic decision fusion."""
        fusion = DecisionConfidenceFusion()
        result = fusion.fuse([sample_vlm_input, sample_visual_input])

        assert result.fused_predictions is not None
        assert len(result.fused_predictions) == 4

    def test_consensus_required(
        self,
        sample_vlm_input: ModalityInput,
        sample_visual_input: ModalityInput,
    ) -> None:
        """Test with consensus requirement."""
        fusion = DecisionConfidenceFusion(
            require_consensus=True,
            consensus_threshold=0.6,
        )
        result = fusion.fuse([sample_vlm_input, sample_visual_input])

        # Consensus reduces positive predictions
        assert result.fused_predictions is not None


class TestAdaptiveFusion:
    """Tests for AdaptiveFusion."""

    def test_adaptive_weights(
        self,
        sample_vlm_input: ModalityInput,
        sample_visual_input: ModalityInput,
    ) -> None:
        """Test adaptive weight calculation."""
        fusion = AdaptiveFusion()
        result = fusion.fuse([sample_vlm_input, sample_visual_input])

        assert "vlm_weight" in result.metadata
        assert "visual_weight" in result.metadata
        assert "agreement_score" in result.metadata

    def test_explanation_generation(
        self,
        sample_vlm_input: ModalityInput,
        sample_visual_input: ModalityInput,
    ) -> None:
        """Test explanation generation."""
        fusion = AdaptiveFusion()
        result = fusion.fuse([sample_vlm_input, sample_visual_input])

        assert result.explanation is not None
        assert len(result.explanation) > 0


class TestAttentionFusion:
    """Tests for AttentionFusion."""

    def test_attention_mechanism(
        self,
        sample_vlm_input: ModalityInput,
        sample_visual_input: ModalityInput,
    ) -> None:
        """Test attention-based fusion."""
        fusion = AttentionFusion(feature_dim=128, num_heads=4)

        # Need to project features to same dimension
        sample_vlm_input.features = torch.randn(4, 128)
        sample_visual_input.features = torch.randn(4, 128)

        result = fusion.fuse([sample_vlm_input, sample_visual_input])

        assert result.fused_features is not None
        assert result.fused_features.shape[-1] == 128
        assert len(result.modality_weights) == 2


class TestMultiModalFusionOptimizer:
    """Tests for MultiModalFusionOptimizer."""

    def test_initialization(self) -> None:
        """Test optimizer initialization."""
        optimizer = MultiModalFusionOptimizer(
            default_strategy=FusionStrategy.ADAPTIVE,
            threshold=0.5,
        )
        assert optimizer.default_strategy == FusionStrategy.ADAPTIVE
        assert optimizer.threshold == 0.5

    def test_fuse_default_strategy(
        self,
        sample_vlm_input: ModalityInput,
        sample_visual_input: ModalityInput,
    ) -> None:
        """Test fusion with default strategy."""
        optimizer = MultiModalFusionOptimizer()
        result = optimizer.fuse([sample_vlm_input, sample_visual_input])

        assert result.fused_scores is not None
        assert result.metadata["fusion_strategy"] == "adaptive"

    def test_fuse_specific_strategy(
        self,
        sample_vlm_input: ModalityInput,
        sample_visual_input: ModalityInput,
    ) -> None:
        """Test fusion with specific strategy."""
        optimizer = MultiModalFusionOptimizer()
        result = optimizer.fuse(
            [sample_vlm_input, sample_visual_input],
            strategy=FusionStrategy.SCORE_AVERAGE,
        )

        assert result.metadata["fusion_strategy"] == "score_average"

    def test_fuse_all_strategies(
        self,
        sample_vlm_input: ModalityInput,
        sample_visual_input: ModalityInput,
    ) -> None:
        """Test applying all fusion strategies."""
        optimizer = MultiModalFusionOptimizer()
        results = optimizer.fuse_all_strategies([sample_vlm_input, sample_visual_input])

        assert len(results) > 0
        assert FusionStrategy.ADAPTIVE in results

    def test_performance_tracking(self) -> None:
        """Test strategy performance tracking."""
        optimizer = MultiModalFusionOptimizer()

        optimizer.update_performance(FusionStrategy.ADAPTIVE, 0.95)
        optimizer.update_performance(FusionStrategy.ADAPTIVE, 0.92)
        optimizer.update_performance(FusionStrategy.SCORE_WEIGHTED, 0.88)

        best = optimizer.get_best_strategy()
        assert best == FusionStrategy.ADAPTIVE

    def test_get_statistics(self) -> None:
        """Test statistics retrieval."""
        optimizer = MultiModalFusionOptimizer()
        optimizer.update_performance(FusionStrategy.ADAPTIVE, 0.9)

        stats = optimizer.get_statistics()
        assert "adaptive" in stats
        assert stats["adaptive"]["mean_score"] == 0.9


class TestCreateFusionOptimizer:
    """Tests for factory function."""

    def test_create_with_defaults(self) -> None:
        """Test factory with defaults."""
        optimizer = create_fusion_optimizer()
        assert optimizer.default_strategy == FusionStrategy.ADAPTIVE
        assert optimizer.threshold == 0.5

    def test_create_with_custom_strategy(self) -> None:
        """Test factory with custom strategy."""
        optimizer = create_fusion_optimizer(
            strategy="score_weighted",
            threshold=0.6,
        )
        assert optimizer.default_strategy == FusionStrategy.SCORE_WEIGHTED
        assert optimizer.threshold == 0.6


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_context_to_fusion_pipeline(
        self,
        sample_video: np.ndarray,
        sample_vlm_input: ModalityInput,
        sample_visual_input: ModalityInput,
    ) -> None:
        """Test full pipeline from context extraction to fusion."""
        # Extract context
        context_provider = EnhancedCombinedContextProvider()
        contexts = context_provider.extract_all_context(sample_video)

        # Add context to metadata
        sample_vlm_input.metadata["contexts"] = contexts

        # Perform fusion
        fusion = MultiModalFusionOptimizer()
        result = fusion.fuse([sample_vlm_input, sample_visual_input])

        assert result.fused_scores is not None
        assert result.fused_predictions is not None

    def test_multithread_cache_access(self) -> None:
        """Test thread-safe cache access."""
        cache = LVLMBackendCache()
        errors: list[Exception] = []

        def access_cache(thread_id: int) -> None:
            try:
                _ = cache.is_loaded("mock", f"model_{thread_id}")
                _ = cache.get_stats()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_cache, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
