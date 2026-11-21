# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the audited VLM detector defects (B1-B7).

Every test here runs offline: BLIP detectors are built with stubbed
model/processor objects (the keyword vote, failure marking, encode count and
VQA wiring are pure control flow), AnyAnomaly/LAVAD get scripted backends at
the ``generate`` boundary, and the context providers / statistical detector
are deterministic by construction.
"""

from __future__ import annotations

import dataclasses
import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.detectors.vlm import anyanomaly, blip_vlm, lavad, lvlm_cache
from omni_mercury_engine.detectors.vlm.anyanomaly import AnyAnomalyConfig, AnyAnomalyDetector
from omni_mercury_engine.detectors.vlm.context_providers import (
    CombinedContextProvider as BasicCombinedContextProvider,
    PositionContextProvider,
)
from omni_mercury_engine.detectors.vlm.context_providers_extended import (
    AppearanceContextProvider,
    CombinedContextProvider as ExtCombinedContextProvider,
    SemanticContextProvider,
)
from omni_mercury_engine.detectors.vlm.lavad import LAVADConfig, LAVADDetector
from omni_mercury_engine.detectors.vlm.lvlm_backends import BLIPVQABackend
from omni_mercury_engine.detectors.vlm.lvlm_cache import LVLMBackendCache
from omni_mercury_engine.detectors.vlm.statistical_vlm import StatisticalVLMDetector

# --------------------------------------------------------------------------- #
# BLIP stubs: real detector control flow, no transformers / no weights
# --------------------------------------------------------------------------- #


class _FakeInputs(dict[str, Any]):
    def to(self, device: Any) -> _FakeInputs:
        return self


class _FakeBlipProcessor:
    """Counts encodings; tokenizes and decodes the way BERT-uncased does.

    BLIP's tokenizer lowercases, splits punctuation into tokens of its own,
    and ``decode`` joins tokens with single spaces -- so a prompt's ``":"``
    comes back as ``" : "`` and the decoded text never starts with the
    prompt string.  (The previous fake returned the prompt verbatim, which
    is exactly what the real tokenizer does not do.)
    """

    CLS, SEP, BOS = 101, 102, 30522
    SPECIAL = frozenset({CLS, SEP, BOS})

    def __init__(self, caption: str) -> None:
        self.caption = caption
        self.encode_calls = 0
        self.texts: list[str | None] = []
        self._vocab: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def tokenize(self, text: str) -> list[int]:
        ids = []
        for word in re.findall(r"[a-z0-9]+|[^a-z0-9\s]", text.lower()):
            if word not in self._vocab:
                self._vocab[word] = 1000 + len(self._vocab)
                self._words[self._vocab[word]] = word
            ids.append(self._vocab[word])
        return ids

    def __call__(self, img: Any, text: str | None = None, return_tensors: str = "pt") -> Any:
        self.encode_calls += 1
        self.texts.append(text)
        inputs = _FakeInputs(pixel_values=torch.ones(1, 3, 4, 4))
        if text is not None:
            inputs["input_ids"] = torch.tensor([[self.CLS, *self.tokenize(text), self.SEP]])
        return inputs

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        values = [int(v) for v in (ids.tolist() if hasattr(ids, "tolist") else ids)]
        return " ".join(self._words[v] for v in values if v not in self.SPECIAL)


class _FakeBlipModel:
    """Generates the scripted caption, echoing a conditional prompt like BLIP.

    ``BlipForConditionalGeneration.generate`` overwrites the first prompt
    token with BOS, drops the prompt's trailing SEP and lets the text decoder
    continue it, so the returned ids are ``[BOS, prompt..., caption..., SEP]``
    (``echo="blip"``).  ``echo="full"`` keeps the prompt's SEP as well -- the
    other layout a prompt-length slice would have to guess between.
    """

    def __init__(
        self,
        fail_generate: bool = False,
        processor: _FakeBlipProcessor | None = None,
        echo: str = "blip",
    ) -> None:
        self.fail_generate = fail_generate
        self.processor = processor
        self.echo = echo

    def generate(self, **kwargs: Any) -> Any:
        if self.fail_generate:
            raise RuntimeError("synthetic generation failure")
        assert self.processor is not None
        proc = self.processor
        caption = proc.tokenize(proc.caption)
        prompt_ids = kwargs.get("input_ids")
        if prompt_ids is None:
            return torch.tensor([[proc.BOS, *caption, proc.SEP]])
        prompt = [int(v) for v in prompt_ids[0].tolist()]
        prefix = prompt[1:-1] if self.echo == "blip" else prompt[1:]
        return torch.tensor([[proc.BOS, *prefix, *caption, proc.SEP]])

    def vision_model(self, pixel_values: Any = None) -> Any:
        return SimpleNamespace(pooler_output=torch.ones(1, 768), last_hidden_state=None)


class _FakeProjection(blip_vlm.FeatureProjection):
    """Deterministic projection: slice the first 128 components."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :128]


def _blip_detector(
    monkeypatch: pytest.MonkeyPatch,
    caption: str = "a cup on a table",
    fail_generate: bool = False,
    **config: Any,
) -> tuple[blip_vlm.BLIPVLMDetector, _FakeBlipProcessor]:
    monkeypatch.setattr(blip_vlm, "HAS_TRANSFORMERS", True)
    detector = blip_vlm.BLIPVLMDetector(blip_vlm.BLIPConfig(device="cpu", **config))
    processor = _FakeBlipProcessor(caption)
    detector._processor = processor
    detector._model = _FakeBlipModel(fail_generate, processor)
    detector._feature_projection = _FakeProjection()
    return detector, processor


def _vote(monkeypatch: pytest.MonkeyPatch, caption: str, **config: Any) -> tuple[bool, float, str]:
    detector, _ = _blip_detector(monkeypatch, **config)
    return detector._parse_response(caption)


class TestB1BlipKeywordVote:
    """B1: substring matching and the never-alarming +1 denominator."""

    def test_anomaly_keyword_does_not_hide_behind_its_substring(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'usual' (normal) must not match inside 'unusual' (anomaly)."""
        flagged, confidence, _ = _vote(monkeypatch, "an unusual event")
        assert flagged is True
        assert confidence == pytest.approx(1.0)

    def test_normal_keyword_does_not_match_inside_abnormal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flagged, confidence, _ = _vote(monkeypatch, "an abnormal reading")
        assert flagged is True
        assert confidence == pytest.approx(1.0)

    def test_description_stopword_contributes_no_evidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The description's 'or' must not count as anomaly evidence via 'door'."""
        flagged, confidence, _ = _vote(
            monkeypatch, "a typical door", anomaly_description="fire or smoke"
        )
        assert flagged is False
        assert confidence == pytest.approx(0.0)  # one normal keyword, zero anomaly evidence

    def test_one_anomaly_keyword_with_no_normal_evidence_alarms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The old total/(total+1) formula pinned this at 0.5 and never alarmed."""
        flagged, confidence, _ = _vote(monkeypatch, "smoke rising from the hill")
        assert flagged is True
        assert confidence == pytest.approx(1.0)

    def test_tie_is_distinguishable_from_no_evidence(self) -> None:
        tie = blip_vlm.caption_keyword_vote("fire in a normal room", ["fire"], ["normal"])
        nothing = blip_vlm.caption_keyword_vote("a cup on a table", ["fire"], ["normal"])
        assert tie[0] is False and tie[1] == pytest.approx(0.5)
        assert nothing[0] is False and nothing[1] == pytest.approx(0.5)
        assert tie[3] == {"anomaly_evidence": 1, "normal_evidence": 1}
        assert nothing[3] == {"anomaly_evidence": 0, "normal_evidence": 0}

    def test_majority_evidence_decides_not_the_threshold(self) -> None:
        flagged, confidence, _, _ = blip_vlm.caption_keyword_vote(
            "smoke and fire near a calm lake", ["fire", "smoke"], ["calm"]
        )
        assert confidence == pytest.approx(2 / 3)
        assert flagged is True  # strictly more anomaly than normal evidence


class TestB2BlipDetectPipeline:
    """B2: swallowed inference failures, dead use_vqa, double encoding."""

    def test_caption_failure_is_explicit_never_a_silent_neutral(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        detector, _ = _blip_detector(monkeypatch, fail_generate=True)
        result = detector.detect(np.full((1, 3, 8, 8), 0.4, dtype=np.float32))
        assert result["details"][0]["caption_failed"] is True
        assert result["captions"][0] == ""
        assert result["scores"][0] == pytest.approx(0.5)
        assert bool(result["is_anomaly"][0]) is False
        assert "failed" in result["explanations"][0]

    def test_successful_captions_are_marked_not_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        detector, _ = _blip_detector(monkeypatch, caption="smoke rising")
        result = detector.detect(np.full((1, 3, 8, 8), 0.4, dtype=np.float32))
        assert result["details"][0]["caption_failed"] is False
        assert result["details"][0]["anomaly_evidence"] == 1
        assert bool(result["is_anomaly"][0]) is True

    def test_detect_encodes_each_image_exactly_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """detect used to call extract_features(data), re-encoding every image."""
        detector, processor = _blip_detector(monkeypatch)
        result = detector.detect(np.full((3, 3, 8, 8), 0.4, dtype=np.float32))
        assert processor.encode_calls == 3
        assert result["features"].shape == (3, 128)
        assert torch.allclose(result["features"].norm(dim=-1), torch.ones(3))

    @pytest.mark.parametrize("echo", ["blip", "full"])
    def test_use_vqa_never_lets_the_vote_see_the_prompt(
        self, monkeypatch: pytest.MonkeyPatch, echo: str
    ) -> None:
        """With use_vqa the prompt conditions generation but never biases the vote.

        Regression: the echo was stripped by ``decoded.startswith(prompt)``.
        The real tokenizer decodes ``"...anomalies: fire or smoke"`` as
        ``"... anomalies : fire or smoke"``, so the prefix never matched, the
        description's own words counted as anomaly evidence, and "a man
        walking a dog" scored (True, 1.0) on every frame.
        """
        detector, processor = _blip_detector(
            monkeypatch,
            caption="a man walking a dog",
            use_vqa=True,
            anomaly_description="fire or smoke",
        )
        detector._model.echo = echo
        voted: list[str] = []
        real_vote = blip_vlm.caption_keyword_vote

        def _spy(caption: str, *args: Any, **kwargs: Any) -> Any:
            voted.append(caption)
            return real_vote(caption, *args, **kwargs)

        monkeypatch.setattr(blip_vlm, "caption_keyword_vote", _spy)
        result = detector.detect(np.full((1, 3, 8, 8), 0.4, dtype=np.float32))

        prompt = detector._create_prompt("fire or smoke")
        assert processor.texts[0] == prompt
        # The premise of the regression: decoding the prompt re-spaces it.
        decoded_prompt = processor.decode(processor.tokenize(prompt))
        assert decoded_prompt == "analyze this image for anomalies : fire or smoke"
        assert not decoded_prompt.startswith(prompt.lower())

        assert voted == ["a man walking a dog"]
        for prompt_word in ("analyze", "anomalies", "fire", "smoke", ":"):
            assert prompt_word not in voted[0]
        assert result["captions"][0] == "a man walking a dog"
        assert bool(result["is_anomaly"][0]) is False
        assert result["scores"][0] == pytest.approx(blip_vlm.NO_EVIDENCE_CONFIDENCE)
        assert result["details"][0] == {
            "caption_failed": False,
            "anomaly_evidence": 0,
            "normal_evidence": 0,
        }

    def test_use_vqa_keeps_a_caption_that_really_names_the_anomaly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stripping the echo must not strip evidence the caption itself carries."""
        detector, _ = _blip_detector(
            monkeypatch,
            caption="smoke rising from a burning car",
            use_vqa=True,
            anomaly_description="fire or smoke",
        )
        result = detector.detect(np.full((1, 3, 8, 8), 0.4, dtype=np.float32))
        assert result["captions"][0] == "smoke rising from a burning car"
        assert bool(result["is_anomaly"][0]) is True

    def test_use_vqa_keeps_everything_when_the_model_echoes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No echo in the output ids: no token is dropped on a guess."""
        detector, processor = _blip_detector(
            monkeypatch, caption="a red bus", use_vqa=True, anomaly_description="fire"
        )

        def _no_echo(**kwargs: Any) -> Any:
            return torch.tensor([[processor.BOS, *processor.tokenize("a red bus"), processor.SEP]])

        detector._model.generate = _no_echo
        result = detector.detect(np.full((1, 3, 8, 8), 0.4, dtype=np.float32))
        assert result["captions"][0] == "a red bus"


class _StaticBackend:
    """Scripted LVLM returning one fixed response."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, images: list[Any], prompt: str) -> str:
        return self.response


class TestB3AnyAnomalyParsing:
    """B3: verdict substring matches, confidence defaults and inversion."""

    @staticmethod
    def _parse(response: str) -> tuple[bool, float, str]:
        return AnyAnomalyDetector()._parse_response(response)

    def test_protocol_response_with_probability(self) -> None:
        flagged, confidence, _ = self._parse("ANOMALY_DETECTED: YES, CONFIDENCE: 0.9")
        assert flagged is True
        assert confidence == pytest.approx(0.9)

    def test_blip_vqa_backend_format_is_accepted_explicitly(self) -> None:
        flagged, confidence, _ = self._parse(
            "ANOMALY: yes\nCONFIDENCE: 0.9731\nEXPLANATION: BLIP VQA p(yes)=0.9731"
        )
        assert flagged is True
        assert confidence == pytest.approx(0.9731)

    def test_a_blip_verdict_beyond_fifty_chars_still_parses(self) -> None:
        padding = "The scene shows a quiet street with several parked cars. "
        flagged, confidence, _ = self._parse(padding + "ANOMALY: yes")
        assert flagged is True
        assert confidence == pytest.approx(anyanomaly.YES_DEFAULT_PROBABILITY)

    def test_bare_yes_defaults_above_the_threshold_and_alarms(self) -> None:
        """A parsed YES with no number used to score exactly 0.5: never an alarm."""
        flagged, confidence, _ = self._parse("YES")
        assert flagged is True
        assert confidence == pytest.approx(anyanomaly.YES_DEFAULT_PROBABILITY)
        assert confidence > AnyAnomalyDetector().vlm_config.confidence_threshold

    def test_eyes_is_not_a_yes(self) -> None:
        flagged, confidence, _ = self._parse("EYES ON THE STREET LOOK CALM. NO")
        assert flagged is False
        assert confidence == pytest.approx(anyanomaly.NO_DEFAULT_PROBABILITY)

    def test_empty_and_garbage_abstain_to_neutral_and_never_alarm(self) -> None:
        for response in ("", "   ", "### unparseable ###"):
            flagged, confidence, _ = self._parse(response)
            assert flagged is False
            assert confidence == pytest.approx(anyanomaly.NO_EVIDENCE_PROBABILITY)

    def test_an_echoed_template_is_not_a_verdict(self) -> None:
        flagged, confidence, _ = self._parse("ANOMALY_DETECTED: [YES/NO]\nCONFIDENCE: [0.0-1.0]")
        assert flagged is False
        assert confidence == pytest.approx(anyanomaly.NO_EVIDENCE_PROBABILITY)

    def test_the_prompt_asks_for_the_probability_of_anomaly(self) -> None:
        prompt = AnyAnomalyDetector()._create_prompt("a person falling")
        assert "probability that the scene is anomalous" in prompt
        assert "ANOMALY_PROBABILITY" in prompt
        assert "Rate your confidence" not in prompt

    def test_no_with_low_probability_scores_low_not_inverted(self) -> None:
        """'NO, ANOMALY_PROBABILITY: 0.1' scores 0.1 — the field the prompt asks for.

        Changed with the review of the B3 cure: this test used to send
        ``"NO, CONFIDENCE: 0.1"`` and pin 0.1, which read a model's
        confidence IN ITS VERDICT as the probability of an anomaly (so
        "NO, CONFIDENCE: 0.95" alarmed).  Only ``ANOMALY_PROBABILITY`` is
        P(anomaly); a confidence qualifies the stated verdict.
        """
        detector = AnyAnomalyDetector(AnyAnomalyConfig(segment_length=4, segment_overlap=0))
        detector._backend = _StaticBackend("NO, ANOMALY_PROBABILITY: 0.1")
        result = detector.detect(np.full((4, 3, 16, 16), 0.2, dtype=np.float32))
        assert np.allclose(result["scores"], 0.1)
        assert result["anomaly_frames"] == []

    def test_no_with_high_confidence_is_a_confident_no(self) -> None:
        """'NO, CONFIDENCE: 0.95' is a 5% anomaly, not a 95% one."""
        detector = AnyAnomalyDetector(AnyAnomalyConfig(segment_length=4, segment_overlap=0))
        detector._backend = _StaticBackend("NO, CONFIDENCE: 0.95")
        result = detector.detect(np.full((4, 3, 16, 16), 0.2, dtype=np.float32))
        assert np.allclose(result["scores"], 0.05)
        assert result["anomaly_frames"] == []


class TestB3BlipVQABackendEmission:
    """The backend emits the probability THAT the scene is anomalous."""

    @staticmethod
    def _backend() -> BLIPVQABackend:
        backend = BLIPVQABackend("Salesforce/blip-vqa-base", device="cpu")
        backend._initialized = True
        return backend

    def test_confidence_is_p_anomalous_for_a_no_verdict(self) -> None:
        backend = self._backend()
        backend._yes_probability = lambda pil, question: 0.25  # type: ignore[method-assign]
        response = backend.generate(
            [np.zeros((3, 8, 8), dtype=np.float32)], "Does this image show a cat?"
        )
        assert "ANOMALY: no" in response
        # p(anomalous), not 1 - p, under the one field name that means it.
        assert "ANOMALY_PROBABILITY: 0.2500" in response
        assert "CONFIDENCE" not in response

    def test_yes_probability_encodes_the_image_once(self) -> None:
        """B7: the image+question pair was encoded once per candidate answer."""
        from PIL import Image

        class _Tokenizer:
            def __call__(self, answer: str, return_tensors: str = "pt") -> Any:
                return SimpleNamespace(input_ids=torch.tensor([[1]]))

        class _Encoding:
            def __init__(self) -> None:
                self.input_ids = torch.tensor([[1, 2]])
                self.pixel_values = torch.zeros(1, 3, 4, 4)

            def to(self, device: Any) -> _Encoding:
                return self

        class _Processor:
            def __init__(self) -> None:
                self.image_calls = 0
                self.tokenizer = _Tokenizer()

            def __call__(self, pil: Any, question: str, return_tensors: str = "pt") -> Any:
                self.image_calls += 1
                return _Encoding()

        class _Model:
            def __init__(self) -> None:
                self.losses = iter([0.4, 0.6])

            def __call__(self, **kwargs: Any) -> Any:
                return SimpleNamespace(loss=torch.tensor(next(self.losses)))

        backend = self._backend()
        processor = _Processor()
        backend.processor = processor
        backend.model = _Model()
        p_yes = backend._yes_probability(Image.new("RGB", (8, 8)), "a question?")
        assert 0.0 < p_yes < 1.0
        assert processor.image_calls == 1


class TestB4LAVAD:
    """B4: lost unlocalised YES verdicts, substring keywords, dead config."""

    def test_a_yes_without_a_frame_list_still_alarms(self) -> None:
        detector = LAVADDetector(LAVADConfig(reasoning_mode="window_generate", window_size=4))
        detector._caption_model = _StaticBackend("people walking in a hallway")
        detector._reasoning_model = _StaticBackend(
            "ANOMALY_DETECTED: YES\nANOMALY_FRAMES: unclear\nCONFIDENCE: 0.8\n"
            "EXPLANATION: an unlocalised incident"
        )
        result = detector.detect(np.full((4, 3, 8, 8), 0.3, dtype=np.float32))
        assert np.allclose(result["scores"], 0.8)
        assert result["anomaly_frames"] == [0, 1, 2, 3]
        assert result["explanations"][0]["no_localization"] is True

    def test_simple_analysis_does_not_match_inside_words(self) -> None:
        detector = LAVADDetector()
        flagged, confidence, _ = detector._simple_analysis(
            ["a man in a white shirt near a door", "a brunette woman walking"]
        )
        assert flagged is False  # 'hit' !< 'white', 'run' !< 'brunette', 'or' !< 'door'
        assert confidence == pytest.approx(0.2)

    def test_simple_analysis_still_matches_whole_words(self) -> None:
        detector = LAVADDetector()
        flagged, _, explanation = detector._simple_analysis(["two men fight near the gate"])
        assert flagged is True
        assert "fight" in explanation

    def test_sampling_fps_subsamples_the_caption_stage(self) -> None:
        class _Counting:
            def __init__(self) -> None:
                self.caption_calls = 0
                self.judge_calls = 0

            def generate(self, images: list[Any], prompt: str) -> str:
                self.caption_calls += 1
                return "a quiet street"

            def yes_probability(self, prompt: str) -> float:
                self.judge_calls += 1
                return 0.2

        backend = _Counting()
        detector = LAVADDetector(LAVADConfig(sampling_fps=2.0))
        detector._caption_model = backend
        detector._reasoning_model = backend
        result = detector.detect_video(np.full((8, 3, 8, 8), 0.3, dtype=np.float32), source_fps=8.0)
        assert backend.caption_calls == 2  # stride 4: frames 0 and 4
        assert backend.judge_calls == 2
        assert result["caption_frame_indices"] == [0, 4]
        assert result["scores"].shape == (8,)
        assert np.allclose(result["scores"], 0.2)

    def test_without_source_fps_every_frame_is_captioned(self) -> None:
        class _Counting:
            def __init__(self) -> None:
                self.caption_calls = 0

            def generate(self, images: list[Any], prompt: str) -> str:
                self.caption_calls += 1
                return "a quiet street"

            def yes_probability(self, prompt: str) -> float:
                return 0.2

        backend = _Counting()
        detector = LAVADDetector()
        detector._caption_model = backend
        detector._reasoning_model = backend
        result = detector.detect(np.full((5, 3, 8, 8), 0.3, dtype=np.float32))
        assert backend.caption_calls == 5
        assert result["caption_frame_indices"] == [0, 1, 2, 3, 4]

    def test_llm_model_and_vlm_model_are_live_aliases(self) -> None:
        """The released alias fields were declared but never read (B4): wire them."""
        config = LAVADConfig(llm_model="my/reasoner", vlm_model="my/captioner")
        assert config.reasoning_model == "my/reasoner"
        assert config.caption_model == "my/captioner"

    def test_the_canonical_model_fields_win_over_their_aliases(self) -> None:
        config = LAVADConfig(
            llm_model="alias/reasoner",
            reasoning_model="canonical/reasoner",
            vlm_model="alias/captioner",
            caption_model="canonical/captioner",
        )
        assert config.reasoning_model == "canonical/reasoner"
        assert config.llm_model == "canonical/reasoner"  # the alias follows
        assert config.caption_model == "canonical/captioner"
        assert config.vlm_model == "canonical/captioner"

    def test_the_default_aliases_mirror_the_models_that_actually_load(self) -> None:
        names = {f.name for f in dataclasses.fields(LAVADConfig)}
        assert {"llm_model", "vlm_model"} <= names  # released API fields stay
        config = LAVADConfig()
        assert config.llm_model == config.reasoning_model
        assert config.vlm_model == config.caption_model

    def test_temporal_window_is_a_live_alias_for_window_size(self) -> None:
        assert LAVADDetector(LAVADConfig(temporal_window=3)).lavad_config.window_size == 3
        assert LAVADDetector(LAVADConfig()).lavad_config.window_size == 5


class TestB5PositionContextUint8:
    """B5: uint8 frame differences wrapped modulo 256."""

    def test_uint8_brightness_drop_names_the_right_region(self) -> None:
        provider = PositionContextProvider(grid_size=(1, 2))
        frames = np.zeros((2, 1, 8, 16), dtype=np.uint8)
        frames[0, 0, :, :8] = 200  # left region: 200 -> 10, true change 190
        frames[1, 0, :, :8] = 10  # (wrapped uint8 diff would be 66)
        frames[0, 0, :, 8:] = 10  # right region: 10 -> 110, change 100
        frames[1, 0, :, 8:] = 110
        activity = provider._compute_region_activity(frames)
        assert activity[0, 0] > activity[0, 1]

    def test_matches_the_float_computation(self) -> None:
        provider = PositionContextProvider(grid_size=(2, 2))
        rng = np.random.default_rng(7)
        frames_uint8 = rng.integers(0, 256, size=(3, 3, 12, 12), dtype=np.uint8)
        a = provider._compute_region_activity(frames_uint8)
        b = provider._compute_region_activity(frames_uint8.astype(np.float64))
        np.testing.assert_allclose(a, b, atol=1e-12)


class TestB6StatisticalVLM:
    """B6: negation parse order; uncalibrated constant decisions."""

    def test_a_leading_negation_is_not_an_anomaly(self) -> None:
        detector = StatisticalVLMDetector()
        flagged, confidence, _ = detector._parse_response("No anomaly detected. Confidence: 0.90")
        assert flagged is False
        assert confidence == pytest.approx(0.90)

    def test_affirmatives_and_anchored_answers_still_parse(self) -> None:
        detector = StatisticalVLMDetector()
        assert detector._parse_response("Anomaly detected. Confidence: 0.9")[0] is True
        assert detector._parse_response("Answer: yes. Confidence: 0.83.")[0] is True
        assert detector._parse_response("Answer: no. Confidence: 0.12.")[0] is False

    def test_fit_calibrates_decisions_to_the_reference_distribution(self) -> None:
        torch.manual_seed(0)
        normals = torch.rand(40, 3, 32, 32)  # textured but normal
        smooth_anomalies = torch.full((8, 3, 32, 32), 0.5)

        detector = StatisticalVLMDetector()
        unfitted_norm = detector.detect(normals)["is_anomaly"]
        unfitted_anom = detector.detect(smooth_anomalies)["is_anomaly"]
        # The audited defect, retained as the documented unfitted fallback:
        # textured normal frames all flag, smooth outliers never do.
        assert unfitted_norm.mean() == 1.0
        assert unfitted_anom.sum() == 0

        detector.fit(normals)
        fitted_norm = detector.detect(normals)["is_anomaly"]
        fitted_anom = detector.detect(smooth_anomalies)["is_anomaly"]
        assert fitted_norm.mean() <= 0.1  # ~alpha false alarms on the reference
        assert fitted_anom.all()


class TestB7ExportsCacheAndVectorization:
    """B7: export collision, backend cache wiring, per-pixel loops."""

    def test_the_package_exports_the_provider_the_detectors_use(self) -> None:
        """Both providers are exported, the released name on its released class.

        Changed with the review of the B7 cure: that fix repointed
        ``vlm.CombinedContextProvider`` from the extended provider (what the
        name resolved to on origin/main) to the basic one, silently changing
        a released import to a class with another constructor.  The name is
        restored; the basic provider the detectors use is exported as
        ``BasicCombinedContextProvider``.
        """
        from omni_mercury_engine.detectors import vlm

        assert vlm.CombinedContextProvider is ExtCombinedContextProvider
        assert vlm.ExtendedCombinedContextProvider is ExtCombinedContextProvider
        assert vlm.BasicCombinedContextProvider is BasicCombinedContextProvider
        assert isinstance(AnyAnomalyDetector().context_provider, vlm.BasicCombinedContextProvider)
        assert {"BasicCombinedContextProvider", "ExtendedCombinedContextProvider"} <= set(
            vlm.__all__
        )

    def test_anyanomaly_initializes_through_the_frozen_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentinel = object()
        calls: dict[str, Any] = {}

        def fake(**kwargs: Any) -> Any:
            calls.update(kwargs)
            return sentinel

        monkeypatch.setattr(anyanomaly, "get_frozen_cached_backend", fake)
        detector = AnyAnomalyDetector({"device": "cpu"})
        assert detector.backend is sentinel
        assert calls["model_type"] == "qwen2_vl"
        assert calls["device"] == "cpu"

    def test_lavad_initializes_both_stages_through_the_frozen_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentinel = object()
        model_types: list[str] = []

        def fake(**kwargs: Any) -> Any:
            model_types.append(kwargs["model_type"])
            return sentinel

        monkeypatch.setattr(lavad, "get_frozen_cached_backend", fake)
        detector = LAVADDetector({"device": "cpu"})
        assert detector.caption_model is sentinel
        assert detector.reasoner is sentinel
        assert model_types == ["blip_caption", "hf_text"]

    def test_frozen_cache_never_reuses_a_mismatched_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cached = SimpleNamespace(
            model_name="m",
            device=torch.device("cpu"),
            max_new_tokens=256,
            temperature=0.9,  # differs from the request below
            revision=None,
        )

        class _Dedicated:
            def __init__(self) -> None:
                self.initialized = False

            def initialize(self) -> None:
                self.initialized = True

        dedicated = _Dedicated()
        monkeypatch.setattr(lvlm_cache, "get_cached_backend", lambda *a, **k: cached)
        monkeypatch.setattr(lvlm_cache, "get_lvlm_backend", lambda **k: dedicated)

        out: Any = lvlm_cache.get_frozen_cached_backend(
            "blip_vqa", "m", "cpu", max_new_tokens=256, temperature=0.1
        )
        assert out is dedicated
        assert dedicated.initialized is True

        cached.temperature = 0.1  # now the frozen configuration matches
        out = lvlm_cache.get_frozen_cached_backend(
            "blip_vqa", "m", "cpu", max_new_tokens=256, temperature=0.1
        )
        assert out is cached

    def test_the_cache_keeps_the_anti_mock_contract_typed(self) -> None:
        cache = LVLMBackendCache.get_instance()
        with pytest.raises(NotImplementedError, match="cannot be used in production"):
            cache.get("mock", "mock-model", device="cpu")

    # ---------------- vectorization equivalence (atol 1e-6) ---------------- #

    def test_edge_density_matches_the_per_pixel_reference(self) -> None:
        provider = SemanticContextProvider()
        rng = np.random.default_rng(11)
        gray = rng.random((12, 14))

        pad = provider.edge_kernel_size // 2
        padded = np.pad(gray, pad, mode="reflect")
        ref_gx = np.zeros_like(gray)
        ref_gy = np.zeros_like(gray)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                ref_gx[i, j] = (padded[i + pad, j + pad + 1] - padded[i + pad, j + pad - 1]) / 2
                ref_gy[i, j] = (padded[i + pad + 1, j + pad] - padded[i + pad - 1, j + pad]) / 2
        ref_mag = np.sqrt(ref_gx**2 + ref_gy**2)
        ref_density = ref_mag.mean() / (ref_mag.max() + 1e-8)

        density, grad_x, grad_y = provider._compute_edge_density(gray)
        np.testing.assert_allclose(grad_x, ref_gx, atol=1e-6)
        np.testing.assert_allclose(grad_y, ref_gy, atol=1e-6)
        assert density == pytest.approx(ref_density, abs=1e-6)

    def test_object_count_matches_the_flood_fill_reference(self) -> None:
        provider = SemanticContextProvider()
        gray = np.zeros((24, 24))
        gray[2:6, 2:6] = 1.0
        gray[10:16, 10:16] = 0.8
        gray[20:22, 4:9] = 0.6
        edge_density = 0.2

        # Reference: the original per-pixel flood-fill implementation.
        threshold = 0.1 + edge_density * 0.2
        grad_x = np.diff(gray, axis=1, prepend=gray[:, :1])
        grad_y = np.diff(gray, axis=0, prepend=gray[:1, :])
        mag = np.sqrt(grad_x**2 + grad_y**2)
        binary = (mag > threshold).astype(np.uint8)
        labels = np.zeros_like(binary, dtype=np.int32)
        current = 0
        h, w = binary.shape
        for i in range(h):
            for j in range(w):
                if binary[i, j] == 1 and labels[i, j] == 0:
                    current += 1
                    stack = [(i, j)]
                    while stack:
                        a, b = stack.pop()
                        if a < 0 or a >= h or b < 0 or b >= w:
                            continue
                        if binary[a, b] == 0 or labels[a, b] != 0:
                            continue
                        labels[a, b] = current
                        stack.extend([(a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)])
        _, counts = np.unique(labels, return_counts=True)
        min_size = binary.size * 0.001
        expected = min(sum(1 for c in counts[1:] if c > min_size), 50)

        assert provider._estimate_object_count(gray, edge_density) == expected

    def test_color_histogram_matches_the_per_pixel_reference(self) -> None:
        for bins in (16, 10):
            provider = AppearanceContextProvider(color_bins=bins)
            rng = np.random.default_rng(13)
            frame = rng.integers(0, 256, size=(3, 9, 11), dtype=np.uint8)

            _, h, w = frame.shape
            quantized = (frame // (256 // bins)).astype(np.int64)
            ref = np.zeros((bins, bins, bins))
            for i in range(h):
                for j in range(w):
                    r, g, b = quantized[:, i, j]
                    ref[min(r, bins - 1), min(g, bins - 1), min(b, bins - 1)] += 1
            ref = (ref / (h * w)).flatten()

            np.testing.assert_allclose(provider._compute_color_histogram(frame), ref, atol=1e-6)

    def test_dominant_colors_match_the_per_pixel_reference(self) -> None:
        provider = AppearanceContextProvider()
        frame = np.zeros((3, 8, 8), dtype=np.uint8)
        frame[:, :, :5] = np.array([200, 10, 10]).reshape(3, 1, 1)  # 40 px
        frame[:, :4, 5:] = np.array([10, 200, 10]).reshape(3, 1, 1)  # 12 px
        frame[:, 4:, 5:] = np.array([10, 10, 200]).reshape(3, 1, 1)  # 12 px... adjust
        frame[:, 7:, 5:] = np.array([240, 240, 240]).reshape(3, 1, 1)  # 3 px

        _, h, w = frame.shape
        quantized = (frame // 32) * 32 + 16
        counts: dict[tuple[int, int, int], int] = {}
        for i in range(h):
            for j in range(w):
                color = (
                    int(quantized[0, i, j]),
                    int(quantized[1, i, j]),
                    int(quantized[2, i, j]),
                )
                counts[color] = counts.get(color, 0) + 1
        expected = {color: n / (h * w) for color, n in counts.items()}

        dominant = provider._find_dominant_colors(frame)
        assert {color: share for color, share in dominant} == dict(
            sorted(expected.items(), key=lambda kv: kv[1], reverse=True)[
                : provider.num_dominant_colors
            ]
        )
        shares = [share for _, share in dominant]
        assert shares == sorted(shares, reverse=True)

    def test_glcm_texture_features_match_the_per_pixel_reference(self) -> None:
        provider = AppearanceContextProvider()
        rng = np.random.default_rng(17)
        gray = rng.random((10, 12))

        levels = 8
        quantized = np.clip((gray * (levels - 1)).astype(np.int32), 0, levels - 1)
        ref_glcm = np.zeros((levels, levels))
        for i in range(quantized.shape[0]):
            for j in range(quantized.shape[1] - 1):
                ref_glcm[quantized[i, j], quantized[i, j + 1]] += 1
        ref_glcm = ref_glcm / (ref_glcm.sum() + 1e-8)
        ref_energy = float((ref_glcm**2).sum())
        nonzero = ref_glcm[ref_glcm > 0]
        ref_entropy = float(-np.sum(nonzero * np.log2(nonzero)))

        energy, entropy = provider._compute_texture_features(gray)
        assert energy == pytest.approx(ref_energy, abs=1e-6)
        assert entropy == pytest.approx(ref_entropy, abs=1e-6)
