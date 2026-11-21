# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the reviewed VLM-detector findings.

* Backend cache -- LRU eviction deleted ``model`` / ``processor`` from a
  backend object that live detectors still held, so the oldest detector went
  blind (``AttributeError`` swallowed into "LVLM query failed").
* AnyAnomaly parsing -- any parsed number overrode an explicit NO, and a
  ``CONFIDENCE`` / "N% confident" value (confidence in the stated verdict) was
  read as the probability of an anomaly.
* Keyword matching -- whole-word matching dropped every inflected form
  ("flooded", "crashes", "emergencies"), losing recall in BLIP and LAVAD.
* StatisticalVLM -- interpolated quantiles lie strictly inside the reference
  range, so ``fit`` flagged its own reference extremes on small sets.
* Exports -- ``vlm.CombinedContextProvider`` had been repointed to another
  class than the released one.
* LAVAD config aliases -- "explicitly the default" was read as "unset".
"""

from __future__ import annotations

import atexit
import copy
import gc
import itertools
import threading
import time
import weakref
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.detectors import vlm
from omni_mercury_engine.detectors.vlm import anyanomaly, blip_vlm, lvlm_cache, text_match
from omni_mercury_engine.detectors.vlm.anyanomaly import AnyAnomalyConfig, AnyAnomalyDetector
from omni_mercury_engine.detectors.vlm.context_providers import (
    CombinedContextProvider as BasicProvider,
)
from omni_mercury_engine.detectors.vlm.context_providers_extended import (
    CombinedContextProvider as ExtendedProvider,
)
from omni_mercury_engine.detectors.vlm.lavad import LAVADConfig, LAVADDetector
from omni_mercury_engine.detectors.vlm.lvlm_backends import BLIPVQABackend, LVLMBackend
from omni_mercury_engine.detectors.vlm.lvlm_cache import LVLMBackendCache
from omni_mercury_engine.detectors.vlm.statistical_vlm import StatisticalVLMDetector

if TYPE_CHECKING:
    from collections.abc import Iterator

# --------------------------------------------------------------------------- #
# Backend cache: eviction drops the cache's reference, nothing else
# --------------------------------------------------------------------------- #


class _StubBackend(LVLMBackend):
    """A loadable backend whose ``generate`` needs its model and processor."""

    def initialize(self) -> None:
        self.model = torch.nn.Linear(4, 4)
        self.processor = {"name": self.model_name}
        self._initialized = True

    def generate(self, images: list[Any], prompt: str) -> str:
        # Touch both attributes exactly as a real backend does: eviction used
        # to delete them from this very object.
        assert self.model.weight.shape == (4, 4)
        assert self.processor["name"] == self.model_name
        if not images:
            return f"text reply from {self.model_name}"
        if len(images) == 1 and "Describe this surveillance camera image" in prompt:
            return f"people walking, captioned by {self.model_name}"
        return "ANOMALY_DETECTED: NO\nANOMALY_PROBABILITY: 0.2\nEXPLANATION: nominal"


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[LVLMBackendCache]:
    """A private three-model cache serving stub backends on a fake clock."""

    def _factory(
        model_type: str, model_name: str | None = None, device: str = "cpu", **kwargs: Any
    ) -> LVLMBackend:
        return _StubBackend(model_name or model_type, device=device, **kwargs)

    # A strictly increasing clock for the cache module only, so "least
    # recently used" never depends on the wall clock's resolution.
    clock = itertools.count(1000)
    fake_time = SimpleNamespace(time=lambda: float(next(clock)), sleep=time.sleep)
    monkeypatch.setattr(lvlm_cache, "time", fake_time)
    monkeypatch.setattr(lvlm_cache, "get_lvlm_backend", _factory)
    monkeypatch.setattr(LVLMBackendCache, "_instance", None)
    instance = LVLMBackendCache(max_models=3)
    atexit.unregister(instance.shutdown)  # torn down here, not at interpreter exit
    yield instance
    # Bounded teardown: a cache deadlock must fail its test, not hang the run.
    teardown = threading.Thread(target=instance.shutdown, daemon=True)
    teardown.start()
    teardown.join(10)


def _anyanomaly(model_name: str) -> AnyAnomalyDetector:
    return AnyAnomalyDetector(
        AnyAnomalyConfig(model_name=model_name, device="cpu", segment_length=4, segment_overlap=0)
    )


FRAMES = np.full((4, 3, 16, 16), 0.2, dtype=np.float32)


class TestCacheEviction:
    """Eviction must never break a backend a detector still holds."""

    def test_an_evicted_backend_still_serves_the_anyanomaly_detector_holding_it(
        self, cache: LVLMBackendCache
    ) -> None:
        """Three detectors, then a fourth distinct model (max_models=3).

        Regression: the fourth load evicted the first key and deleted
        ``model`` / ``processor`` from the backend the first detector still
        held; its next ``detect`` raised ``AttributeError`` inside
        ``generate``, swallowed into "LVLM query failed".
        """
        detectors = [_anyanomaly(f"stub/model-{i}") for i in range(3)]
        before = [d.detect(FRAMES)["scores"] for d in detectors]
        held = [d.backend for d in detectors]

        fourth = _anyanomaly("stub/model-3")
        fourth.detect(FRAMES)
        assert cache.get_stats().evictions == 1

        for detector, backend, scores in zip(detectors, held, before, strict=True):
            assert detector.backend is backend
            assert backend.model is not None
            assert backend.processor is not None
            result = detector.detect(FRAMES)
            np.testing.assert_allclose(result["scores"], scores)
            assert not any(r["backend_failed"] for r in result["segment_results"])

    def test_an_evicted_backend_still_serves_the_lavad_detector_holding_it(
        self, cache: LVLMBackendCache
    ) -> None:
        def lavad(i: int) -> LAVADDetector:
            return LAVADDetector(
                LAVADConfig(
                    caption_backend="stub",
                    caption_model=f"stub/captioner-{i}",
                    use_llm_reasoning=False,
                    device="cpu",
                )
            )

        first = lavad(0)
        captions = first.detect(FRAMES)["captions"]
        for i in range(1, 4):
            lavad(i).detect(FRAMES)
        assert cache.get_stats().evictions == 1
        assert not cache.is_loaded("stub", "stub/captioner-0")
        assert first.detect(FRAMES)["captions"] == captions

    def test_the_lru_victim_is_the_least_recently_used_not_the_least_recently_fetched(
        self, cache: LVLMBackendCache
    ) -> None:
        """A detector that keeps querying must not be the eviction victim.

        Regression: ``last_used`` moved only on ``cache.get``; the detector
        fetched first was the victim however busy it was.
        """
        detectors = [_anyanomaly(f"stub/model-{i}") for i in range(3)]
        for detector in detectors:
            detector.detect(FRAMES)
        detectors[0].detect(FRAMES)  # the first-fetched detector is the busiest

        _anyanomaly("stub/model-3").detect(FRAMES)
        model_type = detectors[0].vlm_config.model_type.value
        assert cache.is_loaded(model_type, "stub/model-0")
        assert not cache.is_loaded(model_type, "stub/model-1")

    def test_touch_refreshes_only_a_backend_the_cache_holds(self, cache: LVLMBackendCache) -> None:
        backend = cache.get("stub", "stub/model-0", device="cpu")
        info = cache.get_model_info("stub", "stub/model-0")
        assert info is not None
        assert lvlm_cache.touch_cached_backend(backend) is True
        touched = cache.get_model_info("stub", "stub/model-0")
        assert touched is not None
        assert touched["last_used"] > info["last_used"]
        assert touched["use_count"] == info["use_count"] + 1
        assert lvlm_cache.touch_cached_backend(object()) is False

    def test_a_model_larger_than_the_memory_cap_keeps_working_once_evicted(
        self, cache: LVLMBackendCache
    ) -> None:
        """A single over-cap model is dropped by the next load; its holder is not harmed."""
        cache.max_memory_bytes = 1  # every stub model exceeds the cap
        first = _anyanomaly("stub/model-0")
        scores = first.detect(FRAMES)["scores"]
        _anyanomaly("stub/model-1").detect(FRAMES)
        model_type = first.vlm_config.model_type.value
        assert not cache.is_loaded(model_type, "stub/model-0")
        np.testing.assert_allclose(first.detect(FRAMES)["scores"], scores)

    def test_manual_evict_and_clear_leave_held_backends_intact(
        self, cache: LVLMBackendCache
    ) -> None:
        one = cache.get("stub", "stub/model-0", device="cpu")
        two = cache.get("stub", "stub/model-1", device="cpu")
        assert cache.evict("stub", "stub/model-0") is True
        assert not cache.is_loaded("stub", "stub/model-0")
        cache.clear()
        assert cache.list_models() == []
        assert cache.get_stats().memory_used_bytes == 0
        for backend in (one, two):
            assert backend.generate([], "ping").startswith("text reply")

    def test_an_evicted_backend_is_freed_once_its_last_holder_lets_go(
        self, cache: LVLMBackendCache
    ) -> None:
        backend = cache.get("stub", "stub/model-0", device="cpu")
        model_ref = weakref.ref(backend.model)
        assert cache.evict("stub", "stub/model-0") is True
        assert model_ref() is not None  # still held here
        del backend
        gc.collect()
        assert model_ref() is None

    def test_a_second_caller_waits_for_a_load_in_progress_without_deadlock(
        self, monkeypatch: pytest.MonkeyPatch, cache: LVLMBackendCache
    ) -> None:
        """Two threads asking for one model while it loads both get that model.

        Regression (found while reworking eviction): ``get`` waited for the
        load in progress while holding the cache lock, and the loading thread
        needs that lock to publish its entry -- both threads hung forever
        (a ``prewarm`` racing the first ``get`` is enough).
        """
        started, release = threading.Event(), threading.Event()

        class _SlowBackend(_StubBackend):
            def initialize(self) -> None:
                started.set()
                release.wait(10)
                super().initialize()

        def _factory(
            model_type: str, model_name: str | None = None, device: str = "cpu", **kwargs: Any
        ) -> LVLMBackend:
            return _SlowBackend(model_name or model_type, device=device, **kwargs)

        monkeypatch.setattr(lvlm_cache, "get_lvlm_backend", _factory)
        fetched: dict[str, Any] = {}

        def _fetch(tag: str) -> None:
            fetched[tag] = cache.get("stub", "stub/slow", device="cpu")

        first = threading.Thread(target=_fetch, args=("first",), daemon=True)
        first.start()
        assert started.wait(10)
        second = threading.Thread(target=_fetch, args=("second",), daemon=True)
        second.start()
        time.sleep(0.2)  # let the second caller reach the load in progress
        release.set()
        first.join(10)
        second.join(10)

        assert not first.is_alive(), "the loading thread never finished"
        assert not second.is_alive(), "the waiting thread never finished"
        assert fetched["first"] is fetched["second"]
        assert cache.get_stats().models_loaded == 1

    def test_a_reload_after_eviction_builds_a_new_backend(self, cache: LVLMBackendCache) -> None:
        """The held, evicted instance is never handed out again as if cached."""
        old = cache.get("stub", "stub/model-0", device="cpu")
        cache.evict("stub", "stub/model-0")
        new = cache.get("stub", "stub/model-0", device="cpu")
        assert new is not old
        assert old.generate([], "ping") == new.generate([], "ping")


# --------------------------------------------------------------------------- #
# AnyAnomaly: which number is P(anomaly), and what an explicit verdict means
# --------------------------------------------------------------------------- #

YES = anyanomaly.YES_DEFAULT_PROBABILITY
NO = anyanomaly.NO_DEFAULT_PROBABILITY
NEUTRAL = anyanomaly.NO_EVIDENCE_PROBABILITY


class TestAnyAnomalyVerdictAndNumber:
    """Only ``ANOMALY_PROBABILITY`` is P(anomaly); confidence is in the verdict."""

    @staticmethod
    def _detail(response: str, **config: Any) -> Any:
        return AnyAnomalyDetector(AnyAnomalyConfig(**config))._parse_response_detail(response)

    @pytest.mark.parametrize(
        ("response", "flagged", "score"),
        [
            # The reviewed regressions: an explicit NO with a confidence.
            (
                "ANOMALY_DETECTED: NO\nEXPLANATION: I am 95% confident the scene is normal.",
                False,
                0.05,
            ),
            ("ANOMALY_DETECTED: NO\nCONFIDENCE: 0.95", False, 0.05),
            ("No. Confidence: 0.9", False, 0.1),
            # Confidence in an explicit YES is P(anomaly) itself.
            ("ANOMALY_DETECTED: YES, CONFIDENCE: 0.9", True, 0.9),
            ("ANOMALY: yes\nCONFIDENCE: 0.9731\nEXPLANATION: p(yes)=0.9731", True, 0.9731),
            ("YES. I am 80% confident there is a fire.", True, 0.8),
            ("ANOMALY_DETECTED: YES\nCONFIDENCE: 0.3", False, 0.3),
            # A low-confidence NO is a likely anomaly, exactly as stated.
            ("ANOMALY_DETECTED: NO\nCONFIDENCE: 0.1", True, 0.9),
            # ANOMALY_PROBABILITY is P(anomaly) for either verdict, or none.
            ("ANOMALY_DETECTED: NO\nANOMALY_PROBABILITY: 0.1", False, 0.1),
            ("ANOMALY_DETECTED: YES\nANOMALY_PROBABILITY: 0.85", True, 0.85),
            ("ANOMALY_PROBABILITY: 0.7", True, 0.7),
            ("ANOMALY_PROBABILITY: 0.2", False, 0.2),
            ("ANOMALY_DETECTED: YES\nANOMALY_PROBABILITY: 85%", True, 0.85),
            ("ANOMALY_DETECTED: YES\nANOMALY_PROBABILITY: 0.9.", True, 0.9),
            ("ANOMALY PROBABILITY: 1", True, 1.0),
            # The explicit field wins over a confidence in the same answer.
            ("ANOMALY_DETECTED: NO\nANOMALY_PROBABILITY: 0.1\nCONFIDENCE: 0.9", False, 0.1),
            # A confidence without a verdict says nothing about the scene.
            ("CONFIDENCE: 0.9", False, NEUTRAL),
            ("I am 95% confident.", False, NEUTRAL),
            # Verdicts without a usable number keep their documented defaults.
            ("YES", True, YES),
            ("NO", False, NO),
            ("", False, NEUTRAL),
        ],
    )
    def test_score_and_flag(self, response: str, flagged: bool, score: float) -> None:
        detail = self._detail(response)
        assert detail.is_anomaly is flagged
        assert detail.probability == pytest.approx(score)
        assert detail.verdict_conflict is False
        assert 0.0 <= detail.probability <= 1.0

    @pytest.mark.parametrize(
        ("response", "flagged", "score"),
        [
            # Out of range and not written as a percentage: discarded, so the
            # explicit verdict stands (this used to rescale 1.5 to 0.015).
            ("ANOMALY_DETECTED: YES\nANOMALY_PROBABILITY: 1.5", True, YES),
            ("ANOMALY_DETECTED: YES\nANOMALY_PROBABILITY: 95", True, YES),
            ("ANOMALY_DETECTED: NO\nANOMALY_PROBABILITY: 7", False, NO),
            ("ANOMALY_DETECTED: NO\nCONFIDENCE: 95", False, NO),
            ("ANOMALY_DETECTED: NO\nCONFIDENCE: 250%", False, NO),
            ("ANOMALY_PROBABILITY: 150%", False, NEUTRAL),
            # A discarded probability falls back to a valid confidence.
            ("ANOMALY_DETECTED: NO\nANOMALY_PROBABILITY: 12\nCONFIDENCE: 0.9", False, 0.1),
        ],
    )
    def test_an_out_of_range_number_is_discarded_never_rescaled(
        self, response: str, flagged: bool, score: float
    ) -> None:
        detail = self._detail(response)
        assert detail.is_anomaly is flagged
        assert detail.probability == pytest.approx(score)
        assert detail.verdict_conflict is False

    @pytest.mark.parametrize(
        ("response", "score"),
        [
            ("ANOMALY_DETECTED: YES\nANOMALY_PROBABILITY: 0.05", YES),
            ("ANOMALY_DETECTED: NO\nANOMALY_PROBABILITY: 0.95", 0.95),
            ("ANOMALY: no\nANOMALY_PROBABILITY: 0.6", 0.6),
        ],
    )
    def test_a_verdict_contradicting_the_probability_alarms_and_is_marked(
        self, response: str, score: float
    ) -> None:
        """Neither reading is silently picked: the alarm one is kept, and flagged."""
        detail = self._detail(response)
        assert detail.is_anomaly is True
        assert detail.verdict_conflict is True
        assert detail.probability == pytest.approx(score)
        assert detail.probability > 0.5

    def test_the_tuple_contract_carries_the_same_decision(self) -> None:
        detector = AnyAnomalyDetector()
        response = "ANOMALY_DETECTED: NO\nCONFIDENCE: 0.95\nEXPLANATION: an empty corridor"
        flagged, score, explanation = detector._parse_response(response)
        assert flagged is False
        assert score == pytest.approx(0.05)
        assert explanation == "an empty corridor"

    def test_detect_marks_a_conflicting_segment(self) -> None:
        detector = AnyAnomalyDetector(AnyAnomalyConfig(segment_length=4, segment_overlap=0))
        detector._backend = _Scripted("ANOMALY_DETECTED: YES\nANOMALY_PROBABILITY: 0.05")
        result = detector.detect(FRAMES)
        assert result["segment_results"][0]["verdict_conflict"] is True
        assert result["segment_results"][0]["is_anomaly"] is True
        assert result["anomaly_frames"] == [0, 1, 2, 3]

        detector._backend = _Scripted("ANOMALY_DETECTED: NO\nCONFIDENCE: 0.95")
        result = detector.detect(FRAMES)
        assert result["segment_results"][0]["verdict_conflict"] is False
        assert np.allclose(result["scores"], 0.05)
        assert result["anomaly_frames"] == []

    def test_the_conflict_is_judged_at_the_configured_threshold(self) -> None:
        response = "ANOMALY_DETECTED: NO\nANOMALY_PROBABILITY: 0.6"
        assert self._detail(response, confidence_threshold=0.5).verdict_conflict is True
        relaxed = self._detail(response, confidence_threshold=0.7)
        assert relaxed.verdict_conflict is False
        assert relaxed.is_anomaly is False
        assert relaxed.probability == pytest.approx(0.6)


class _Scripted:
    """An LVLM stand-in returning one fixed response."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, images: list[Any], prompt: str) -> str:
        return self.response


class TestBlipVqaBackendProtocol:
    """The shipped BLIP-VQA backend states P(anomaly) under its own name."""

    @staticmethod
    def _backend(p_yes: float) -> BLIPVQABackend:
        backend = BLIPVQABackend("Salesforce/blip-vqa-base", device="cpu")
        backend._initialized = True
        backend._yes_probability = lambda pil, question: p_yes  # type: ignore[method-assign]
        return backend

    @pytest.mark.parametrize("p_yes", [0.03, 0.25, 0.5, 0.75, 0.97])
    def test_the_emitted_probability_parses_back_exactly(self, p_yes: float) -> None:
        response = self._backend(p_yes).generate(
            [np.zeros((3, 8, 8), dtype=np.float32)], "Does this image show a cat?"
        )
        assert f"ANOMALY_PROBABILITY: {p_yes:.4f}" in response
        assert "CONFIDENCE" not in response
        detail = AnyAnomalyDetector()._parse_response_detail(response)
        assert detail.probability == pytest.approx(p_yes, abs=1e-4)
        assert detail.is_anomaly is (p_yes > 0.5)
        assert detail.verdict_conflict is False


# --------------------------------------------------------------------------- #
# BLIP conditional captions: the prompt echo is measured in tokens
# --------------------------------------------------------------------------- #

CLS, SEP, BOS = 101, 102, 30522
PROMPT = [CLS, 11, 12, 13, 14, SEP]
CAPTION = [21, 22, 23]


class TestBlipPromptTokens:
    """``_drop_prompt_tokens`` removes exactly the echoed prompt, or nothing."""

    @staticmethod
    def _drop(output: list[int], prompt: list[int] = PROMPT) -> list[int]:
        dropped = blip_vlm.BLIPVLMDetector._drop_prompt_tokens(
            torch.tensor(output), torch.tensor(prompt)
        )
        return [int(v) for v in dropped.tolist()]

    def test_the_blip_layout_loses_the_prompt_and_keeps_every_generated_token(self) -> None:
        """generate() swaps [CLS] for BOS and drops the prompt's [SEP]: one token shorter."""
        output = [BOS, 11, 12, 13, 14, *CAPTION, SEP]
        assert self._drop(output) == [*CAPTION, SEP]
        # A fixed slice at the prompt's length would have eaten token 21.
        assert output[len(PROMPT) :] == [22, 23, SEP]

    def test_a_prompt_echoed_with_its_sep_is_dropped_whole(self) -> None:
        assert self._drop([BOS, 11, 12, 13, 14, SEP, *CAPTION, SEP]) == [*CAPTION, SEP]

    def test_a_verbatim_echo_is_dropped_whole(self) -> None:
        assert self._drop([*PROMPT, *CAPTION]) == CAPTION
        assert self._drop([11, 12, *CAPTION], prompt=[11, 12]) == CAPTION

    def test_an_output_without_an_echo_is_left_alone(self) -> None:
        assert self._drop([BOS, *CAPTION, SEP]) == [BOS, *CAPTION, SEP]

    def test_a_caption_that_merely_starts_like_the_prompt_is_left_alone(self) -> None:
        """Sharing the first word with the prompt is not an echo."""
        output = [BOS, 11, 12, *CAPTION, SEP]
        assert self._drop(output) == output

    def test_an_echo_that_ends_generation_at_once_leaves_no_caption(self) -> None:
        assert self._drop([BOS, 11, 12, 13, 14, SEP]) == []

    def test_an_output_truncated_inside_the_prompt_leaves_no_caption(self) -> None:
        assert self._drop([BOS, 11, 12]) == []

    def test_plain_lists_are_accepted(self) -> None:
        dropped = blip_vlm.BLIPVLMDetector._drop_prompt_tokens(
            [BOS, 11, 12, 13, 14, *CAPTION, SEP], PROMPT
        )
        assert dropped == [*CAPTION, SEP]


# --------------------------------------------------------------------------- #
# Keyword matching: left boundary kept, regular inflections accepted
# --------------------------------------------------------------------------- #

LAVAD_KEYWORDS_SAMPLE = ["fight", "attack", "hit", "kick", "run", "chase", "crash", "fire"]


class TestInflectedKeywords:
    """A keyword matches its regular inflections, never the inside of a word."""

    @pytest.mark.parametrize(
        ("keyword", "text"),
        [
            ("flood", "a flooded street"),
            ("flood", "the river is flooding the town"),
            ("flood", "floods across the valley"),
            ("crash", "a car that crashed into a wall"),
            ("crash", "two cars crashes"),
            ("crash", "a plane crashing"),
            ("damage", "a damaged car"),
            ("damage", "wind damaging the roof"),
            ("damage", "damages to the hull"),
            ("fire", "flames and fires everywhere"),
            ("fire", "a man firing a gun"),
            ("fire", "the fire's glow"),
            ("emergency", "during emergencies"),
            ("accident", "two accidents"),
            ("hit", "a man hitting a wall"),
            ("hit", "he hits the door"),
            ("run", "a child running away"),
            ("run", "she runs"),
            ("chase", "a dog chasing a car"),
            ("chase", "chased by police"),
            ("kick", "kicked the door"),
            ("attack", "a man attacking a woman"),
            ("smoke", "a smoking engine"),
            ("fall", "a person falling"),
            ("FIRE", "Fires Everywhere"),
        ],
    )
    def test_regular_inflections_match(self, keyword: str, text: str) -> None:
        assert text_match.find_word_matches(text, [keyword]) == [keyword.lower()]

    @pytest.mark.parametrize(
        ("keyword", "text"),
        [
            ("usual", "an unusual event"),
            ("normal", "an abnormal reading"),
            ("or", "a tiled floor by the door"),
            ("hit", "a man in a white shirt"),
            ("run", "a brunette woman"),
            ("fire", "a fireman"),  # derivation, not inflection
            ("fire", "a campfire"),
            ("fall", "fallout"),
            ("crash", "a crashworthy design"),
            ("odd", "an oddity"),
            ("gun", "gunmetal grey"),
            ("flood", "a floodlight"),
            ("error", "terrors"),
        ],
    )
    def test_the_inside_of_another_word_never_matches(self, keyword: str, text: str) -> None:
        assert text_match.find_word_matches(text, [keyword]) == []

    @pytest.mark.parametrize(
        ("caption", "evidence"),
        [
            ("a flooded street with a damaged car", 2),
            ("a crashed car in a flooded road during emergencies", 3),
            ("flames and fires everywhere", 1),
            ("a car that crashed into a wall", 1),
        ],
    )
    def test_the_default_blip_keywords_see_inflected_captions(
        self, caption: str, evidence: int
    ) -> None:
        """Each of these captions scored 0 evidence and raised no alarm."""
        config = blip_vlm.BLIPConfig()
        flagged, confidence, _, counts = blip_vlm.caption_keyword_vote(
            caption, config.anomaly_keywords or [], config.normal_keywords or [], ""
        )
        assert counts == {"anomaly_evidence": evidence, "normal_evidence": 0}
        assert flagged is True
        assert confidence == pytest.approx(1.0)

    def test_the_default_blip_keywords_still_ignore_embedded_words(self) -> None:
        config = blip_vlm.BLIPConfig()
        for caption in ("an unusual event", "an abnormal reading"):
            flagged, confidence, _, counts = blip_vlm.caption_keyword_vote(
                caption, config.anomaly_keywords or [], config.normal_keywords or [], ""
            )
            assert counts == {"anomaly_evidence": 1, "normal_evidence": 0}
            assert (flagged, confidence) == (True, 1.0)
        _, _, _, counts = blip_vlm.caption_keyword_vote(
            "a typical door on a tiled floor",
            config.anomaly_keywords or [],
            config.normal_keywords or [],
            "fire or smoke",
        )
        assert counts == {"anomaly_evidence": 0, "normal_evidence": 1}

    def test_a_description_word_is_not_counted_again_as_an_inflected_keyword(self) -> None:
        """'fires' in the description is the keyword 'fire': one piece of evidence."""
        _, _, _, counts = blip_vlm.caption_keyword_vote(
            "fires in a calm valley", ["fire"], ["calm"], "fires or flooding"
        )
        assert counts == {"anomaly_evidence": 1, "normal_evidence": 1}

    @pytest.mark.parametrize(
        ("caption", "keyword"),
        [
            ("two men fighting near the gate", "fight"),
            ("a man hitting another man", "hit"),
            ("a car crashes into a pole", "crash"),
            ("a woman attacked on the street", "attack"),
            ("smoke and fires in a building", "fire"),
            ("a boy kicking a door", "kick"),
        ],
    )
    def test_the_lavad_keywords_see_inflected_captions(self, caption: str, keyword: str) -> None:
        detector = LAVADDetector(LAVADConfig(anomaly_description=""))
        flagged, _, explanation = detector._simple_analysis([caption])
        assert flagged is True
        assert keyword in explanation

    def test_the_lavad_keywords_still_ignore_embedded_words(self) -> None:
        detector = LAVADDetector()
        flagged, confidence, _ = detector._simple_analysis(
            ["a man in a white shirt near a door", "a brunette woman walking"]
        )
        assert flagged is False
        assert confidence == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# StatisticalVLM: the reference set is inside its own interval
# --------------------------------------------------------------------------- #


class TestStatisticalFitInterval:
    """Finite-sample ranks: ``fit`` never flags more than alpha of its reference."""

    @staticmethod
    def _reference(n: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(n)
        # Frames of varied texture so the salience scores are distinct.
        scale = torch.linspace(0.05, 1.0, n).view(n, 1, 1, 1)
        return torch.rand(n, 3, 16, 16, generator=generator) * scale

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_a_small_reference_set_flags_none_of_its_own_frames(self, n: int) -> None:
        """Interpolated quantiles flagged 2 of 2, 2 of 5 and 2 of 10 frames."""
        reference = self._reference(n)
        detector = StatisticalVLMDetector().fit(reference)
        scores = detector.detect(reference)["scores"]
        assert len(np.unique(scores)) == n  # the extremes are real, distinct frames
        assert detector.detect(reference)["is_anomaly"].sum() == 0
        low, high = detector._fitted_score_interval or (np.nan, np.nan)
        assert low == pytest.approx(float(scores.min()))
        assert high == pytest.approx(float(scores.max()))

    @pytest.mark.parametrize("n", [40, 100, 400])
    def test_a_larger_reference_set_flags_at_most_the_nominal_fraction(self, n: int) -> None:
        reference = self._reference(n)
        detector = StatisticalVLMDetector().fit(reference)
        flagged = int(detector.detect(reference)["is_anomaly"].sum())
        assert flagged <= int(0.05 * n)

    def test_frames_outside_the_reference_range_still_flag(self) -> None:
        reference = self._reference(10) * 0.5 + 0.2
        detector = StatisticalVLMDetector().fit(reference)
        flat = torch.full((2, 3, 16, 16), 0.5)
        assert detector.detect(flat)["is_anomaly"].all()


# --------------------------------------------------------------------------- #
# Exports: the released name keeps its released class
# --------------------------------------------------------------------------- #


class TestContextProviderExports:
    """``vlm.CombinedContextProvider`` is the class it was on origin/main."""

    def test_the_released_name_is_the_extended_provider(self) -> None:
        assert vlm.CombinedContextProvider is ExtendedProvider
        # The released constructor signature still works through the name.
        provider = vlm.CombinedContextProvider(
            enable_position=False, enable_temporal=False, enable_frequency=False, random_state=0
        )
        assert "semantic" in provider.providers
        assert "frequency" not in provider.providers

    def test_the_basic_provider_has_its_own_name(self) -> None:
        assert vlm.BasicCombinedContextProvider is BasicProvider
        assert vlm.BasicCombinedContextProvider.__module__.endswith(".context_providers")
        assert vlm.CombinedContextProvider.__module__.endswith(".context_providers_extended")
        assert isinstance(AnyAnomalyDetector().context_provider, vlm.BasicCombinedContextProvider)

    def test_both_names_are_exported(self) -> None:
        assert {
            "BasicCombinedContextProvider",
            "CombinedContextProvider",
            "ExtendedCombinedContextProvider",
        } <= set(vlm.__all__)
        assert vlm.ExtendedCombinedContextProvider is ExtendedProvider


# --------------------------------------------------------------------------- #
# LAVAD config aliases: "explicitly the default" is not "unset"
# --------------------------------------------------------------------------- #

REASONER_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"
CAPTIONER_DEFAULT = "Salesforce/blip-image-captioning-base"
RELEASED_LLM_ALIAS_DEFAULT = "meta-llama/Meta-Llama-3-8B-Instruct"


class TestLAVADConfigAliases:
    """``llm_model`` / ``vlm_model`` / ``temporal_window`` mirror their fields."""

    def test_an_alias_given_its_released_default_value_takes_effect(self) -> None:
        """An explicit value equal to the old alias default used to read as unset."""
        config = LAVADConfig(llm_model=RELEASED_LLM_ALIAS_DEFAULT)
        assert config.reasoning_model == RELEASED_LLM_ALIAS_DEFAULT
        assert config.llm_model == RELEASED_LLM_ALIAS_DEFAULT

    def test_replace_back_to_the_canonical_default_takes_effect(self) -> None:
        config = replace(
            LAVADConfig(llm_model="my/reasoner", vlm_model="my/captioner", temporal_window=3),
            reasoning_model=REASONER_DEFAULT,
            caption_model=CAPTIONER_DEFAULT,
            window_size=5,
        )
        assert (config.reasoning_model, config.llm_model) == (REASONER_DEFAULT, REASONER_DEFAULT)
        assert (config.caption_model, config.vlm_model) == (CAPTIONER_DEFAULT, CAPTIONER_DEFAULT)
        assert (config.window_size, config.temporal_window) == (5, 5)

    def test_a_value_mutated_back_to_the_default_survives_the_round_trip(self) -> None:
        config = LAVADConfig(reasoning_model="a/b", window_size=9)
        config.reasoning_model = REASONER_DEFAULT
        config.window_size = 5
        rebuilt = LAVADConfig(**asdict(config))
        assert rebuilt.reasoning_model == REASONER_DEFAULT
        assert rebuilt.window_size == 5

    def test_the_canonical_field_wins_even_at_its_default_value(self) -> None:
        config = LAVADConfig(
            reasoning_model=REASONER_DEFAULT,
            llm_model="alias/reasoner",
            caption_model=CAPTIONER_DEFAULT,
            vlm_model="alias/captioner",
            window_size=5,
            temporal_window=3,
        )
        assert config.reasoning_model == REASONER_DEFAULT
        assert config.caption_model == CAPTIONER_DEFAULT
        assert config.window_size == 5
        assert (config.llm_model, config.vlm_model) == (REASONER_DEFAULT, CAPTIONER_DEFAULT)
        assert config.temporal_window == 5

    def test_unset_aliases_read_back_the_models_that_load(self) -> None:
        config = LAVADConfig()
        assert config.llm_model == config.reasoning_model == REASONER_DEFAULT
        assert config.vlm_model == config.caption_model == CAPTIONER_DEFAULT
        assert config.temporal_window == config.window_size == 5

    def test_temporal_window_resolves_in_the_config_itself(self) -> None:
        assert LAVADConfig(temporal_window=3).window_size == 3
        assert LAVADDetector(LAVADConfig(temporal_window=3)).lavad_config.window_size == 3
        assert LAVADDetector({"temporal_window": 4}).lavad_config.window_size == 4

    def test_a_window_size_changed_after_construction_reaches_the_detector(self) -> None:
        """The detector must not re-apply a stale alias over the canonical field."""
        config = LAVADConfig(temporal_window=3)
        config.window_size = 7
        assert config.temporal_window == 7
        assert LAVADDetector(config).lavad_config.window_size == 7
        config.temporal_window = 2
        assert LAVADDetector(config).lavad_config.window_size == 2

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"llm_model": "my/reasoner", "vlm_model": "my/captioner", "temporal_window": 3},
            {"reasoning_model": "x/y", "caption_model": "z/w", "window_size": 9},
            {"reasoning_model": REASONER_DEFAULT, "llm_model": "alias/reasoner"},
            {"llm_model": RELEASED_LLM_ALIAS_DEFAULT},
        ],
    )
    def test_asdict_and_replace_round_trip(self, kwargs: dict[str, Any]) -> None:
        config = LAVADConfig(**kwargs)
        assert LAVADConfig(**asdict(config)) == config
        assert replace(config) == config
        assert copy.deepcopy(config) == config
