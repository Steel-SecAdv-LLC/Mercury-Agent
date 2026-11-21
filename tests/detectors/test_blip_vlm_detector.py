# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BLIPVLMDetector: the keyword vote offline, the real captioner on real photographs.

The real-model tests use the pinned ``Salesforce/blip-image-captioning-base``
weights and the CC0 photographs in ``tests/fixtures/visual`` (a coffee cup and
scikit-image's "chelsea" cat); they are skipped when the weights are not cached
(``python scripts/prefetch_pinned_models.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
Image = pytest.importorskip("PIL.Image")

from omni_mercury_engine.detectors.vlm.blip_vlm import (
    FEATURE_PROJECTION_SEED,
    BLIPConfig,
    BLIPVLMDetector,
    FeatureProjection,
)
from tests.hf_model_cache import missing_pinned_models

CAPTIONER = "Salesforce/blip-image-captioning-base"
REVISION = "82a37760796d32b1411fe092ab5d4e227313294b"
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "visual"

needs_captioner = pytest.mark.skipif(
    any(m.startswith(CAPTIONER) for m in missing_pinned_models()),
    reason=f"{CAPTIONER} not cached; run scripts/prefetch_pinned_models.py",
)


def _photo(name: str) -> np.ndarray[Any, Any]:
    with Image.open(FIXTURES / name) as image:
        rgb = image.convert("RGB").resize((224, 224))
    return np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0


def _vote(caption: str, **config: Any) -> tuple[bool, float, str]:
    detector = BLIPVLMDetector(BLIPConfig(**config))
    return detector._parse_response(caption)


class TestKeywordVote:
    def test_a_caption_with_no_keyword_scores_neutral_and_does_not_alarm(self) -> None:
        flagged, confidence, _ = _vote("a cup on a table", anomaly_keywords=["fire"])
        assert (flagged, confidence) == (False, 0.5)

    def test_anomaly_majority_flags_and_confidence_is_the_evidence_ratio(self) -> None:
        """Smoke and fire by a calm lake IS an anomaly (B1 audit cure).

        Changed with the B1 fix: this test used to assert the caption was
        NOT flagged, pinning the old ``total / (total + normal + 1)``
        formula whose +1 denominator put two anomaly keywords against one
        normal keyword at exactly 0.5 — below the strict threshold, so
        'smoke' and 'fire' near a 'calm' lake never alarmed.  Under the
        fixed vote, confidence is the plain evidence ratio and strictly
        more anomaly than normal evidence alarms.
        """
        flagged, confidence, _ = _vote(
            "smoke and fire near a calm lake",
            anomaly_keywords=["fire", "smoke"],
            normal_keywords=["calm"],
            anomaly_description="",
            confidence_threshold=0.5,
        )
        assert confidence == pytest.approx(2 / 3)
        assert flagged is True

    def test_one_anomaly_keyword_alone_alarms_and_a_tie_does_not(self) -> None:
        kwargs = {"anomaly_keywords": ["fire", "smoke"], "anomaly_description": ""}
        assert _vote("smoke and fire", **kwargs)[0] is True
        flagged, confidence, explanation = _vote(
            "smoke over a calm lake", normal_keywords=["calm"], **kwargs
        )
        assert flagged is False  # 1-vs-1 tie: no majority
        assert confidence == pytest.approx(0.5)
        assert "Anomaly evidence: 1" in explanation  # distinguishable from no evidence

    def test_keywords_match_whole_words_only(self) -> None:
        flagged, confidence, _ = _vote(
            "an unusual event", anomaly_keywords=["unusual"], normal_keywords=["usual"]
        )
        assert flagged is True  # 'usual' must not match inside 'unusual'
        assert confidence == pytest.approx(1.0)

    def test_the_description_words_count_as_anomaly_evidence(self) -> None:
        flagged, confidence, _ = _vote("a large cat on a rug", anomaly_description="cat")
        assert confidence == pytest.approx(1.0)
        assert flagged is True


class TestFeatureProjection:
    """The 128-D fusion features pass through a fixed random map; it must be reproducible."""

    def test_two_projections_built_under_the_seed_are_identical(self) -> None:
        def build() -> FeatureProjection:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(FEATURE_PROJECTION_SEED)
                return FeatureProjection(768, 128)

        first, second = build(), build()
        for a, b in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
            assert torch.equal(a, b)


@needs_captioner
class TestOnTheRealCaptioner:
    @staticmethod
    def _detector(**config: Any) -> BLIPVLMDetector:
        return BLIPVLMDetector(BLIPConfig(model_name=CAPTIONER, revision=REVISION, **config))

    def test_features_do_not_depend_on_the_instance_or_the_call(self) -> None:
        """Two detectors, two calls, one photograph: one feature vector."""
        photo = torch.from_numpy(_photo("chelsea.png"))
        first, second = self._detector(), self._detector()

        a = first.extract_features(photo)
        b = first.extract_features(photo)
        c = second.extract_features(photo)

        assert a.shape == (1, 128)
        assert torch.allclose(a, b, atol=1e-6)
        assert torch.allclose(a, c, atol=1e-6)
        assert float(a.norm()) == pytest.approx(1.0, abs=1e-5)

    def test_the_global_random_state_is_left_alone(self) -> None:
        torch.manual_seed(1234)
        expected = torch.rand(3)
        torch.manual_seed(1234)
        self._detector().extract_features(torch.from_numpy(_photo("coffee.png")))
        assert torch.equal(torch.rand(3), expected)

    def test_different_photographs_give_different_features(self) -> None:
        detector = self._detector()
        cat = detector.extract_features(torch.from_numpy(_photo("chelsea.png")))
        cup = detector.extract_features(torch.from_numpy(_photo("coffee.png")))
        assert float((cat - cup).abs().max()) > 0.05

    def test_the_cat_photograph_is_captioned_as_a_cat_and_alarms_on_cat(self) -> None:
        detector = self._detector(
            anomaly_keywords=["cat"], anomaly_description="", confidence_threshold=0.3
        )
        result = detector.detect(torch.from_numpy(_photo("chelsea.png")))
        assert "cat" in result["captions"][0].lower()
        assert bool(result["is_anomaly"][0]) is True

    def test_the_coffee_photograph_does_not_alarm_on_cat(self) -> None:
        detector = self._detector(
            anomaly_keywords=["cat"], anomaly_description="", confidence_threshold=0.3
        )
        result = detector.detect(torch.from_numpy(_photo("coffee.png")))
        assert "cat" not in result["captions"][0].lower()
        assert bool(result["is_anomaly"][0]) is False
