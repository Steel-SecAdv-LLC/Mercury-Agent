# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Restored domain symbols removed in 7ab59916 and f07b724e.

Defect tests fail on the released code: the NOAA loader sent ``maxSize``,
which HazEL ignores (all 133 events of 2020-2026 came back for
``max_records=3``, checked live), the SST proxy counted the bleaching-tolerant
*Porites* as heat-sensitive, the multi-scale aggregator failed opaquely on a
missing layer, and ``golden_ratio_weights`` was a ``TypeError``.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.datasets.exceptions import OfflineModeError


def test_make_capability_and_operation_for() -> None:
    from omni_mercury_engine.agentic.mercury_a_agent import DomainType
    from omni_mercury_engine.agentic.subagents.base import make_capability
    from omni_mercury_engine.agentic.subagents.operations import OPERATIONS, operation_for

    capability = make_capability("Themis_I", [DomainType.SECURITY], ["Audit", "LAW"])
    assert capability.domains == frozenset({DomainType.SECURITY})
    assert capability.keywords == frozenset({"audit", "law"})
    assert operation_for("Hestia_II") is OPERATIONS["Hestia_II"]
    assert operation_for("Nobody_0") is None


def test_regulatory_intelligence_ledger() -> None:
    from omni_mercury_engine.agentic.subagents.specializations.compliance import (
        RegulatoryIntelligence,
    )

    ledger = RegulatoryIntelligence()
    ledger.monitor_regulation("CCPA", "https://oag.ca.gov/privacy/ccpa")
    ledger.detect_regulatory_change("CCPA", "CPRA amendments in force")
    assert ledger.tracked_regulations["CCPA"]["source_url"] == "https://oag.ca.gov/privacy/ccpa"
    assert ledger.change_history[0]["regulation"] == "CCPA"
    assert ledger.change_history[0]["requires_review"] is True


class _PassThroughBreaker:
    def call(self, fn: Any) -> Any:
        return fn()


class TestNoaaTsunamiRecords:
    @pytest.fixture
    def module(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        from omni_mercury_engine.detectors.geological import disaster_detectors

        monkeypatch.delenv("MERCURY_OFFLINE", raising=False)
        monkeypatch.setattr(
            disaster_detectors, "get_data_loader_breaker", lambda _name: _PassThroughBreaker()
        )
        return disaster_detectors

    def _serve(
        self, module: Any, monkeypatch: pytest.MonkeyPatch, total: int
    ) -> list[dict[str, str]]:
        requests: list[dict[str, str]] = []

        def get_json(_url: str, params: dict[str, str], **_kw: Any) -> dict[str, Any]:
            requests.append(params)
            per_page, page = int(params["itemsPerPage"]), int(params["page"])
            start = (page - 1) * per_page
            items = [{"id": i} for i in range(start, min(start + per_page, total))]
            return {"items": items, "totalPages": -(-total // per_page), "page": page}

        monkeypatch.setattr(module.SafeHTTPClient, "get_json", staticmethod(get_json))
        return requests

    def test_pages_until_max_records(self, module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fails on the released loader, which sent ``maxSize``."""
        requests = self._serve(module, monkeypatch, total=450)
        records = module.load_noaa_tsunami_records(min_year=2020, max_records=250)
        assert [r["id"] for r in records] == list(range(250))
        assert [r["page"] for r in requests] == ["1", "2"]
        assert {r["itemsPerPage"] for r in requests} == {"200"}
        assert requests[0]["minYear"] == "2020"

    def test_stops_at_the_last_page(self, module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        requests = self._serve(module, monkeypatch, total=133)
        assert len(module.load_noaa_tsunami_records(max_records=1000)) == 133
        assert len(requests) == 1

    def test_empty_service_returns_none(self, module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self._serve(module, monkeypatch, total=0)
        assert module.load_noaa_tsunami_records() is None

    def test_offline_is_refused_not_reported_unavailable(
        self, module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_OFFLINE", "1")
        with pytest.raises(OfflineModeError):
            module.load_noaa_tsunami_records()

    def test_max_records_must_be_positive(self, module: Any) -> None:
        with pytest.raises(ValueError, match="max_records"):
            module.load_noaa_tsunami_records(max_records=0)


def test_sst_proxy_counts_the_susceptible_genera() -> None:
    """Fails on the released proxy (Porites counted, Seriatopora not; 0.4 here)."""
    from omni_mercury_engine.loaders.marine_loader import _compute_sst_anomaly_proxy

    cell = pd.DataFrame(
        {
            "scientificName": [
                "Acropora millepora",
                "Porites lobata",
                "Seriatopora hystrix",
                "Platygyra daedalea",
                None,
            ]
        }
    )
    assert _compute_sst_anomaly_proxy(cell) == 0.5
    assert _compute_sst_anomaly_proxy(pd.DataFrame({"scientificName": ["Isopora palifera"]})) == 0
    assert _compute_sst_anomaly_proxy(pd.DataFrame()) == 0.0


def test_result_containers() -> None:
    from omni_mercury_engine.detectors.physics_integration import IntegratedPhysicsResult
    from omni_mercury_engine.detectors.vlm.base_vlm import VQAResult

    vqa = VQAResult(True, 0.8, "smoke", "yes")
    assert vqa.to_dict()["frame_indices"] == []
    torch = pytest.importorskip("torch")
    result = IntegratedPhysicsResult(
        anomaly_score=0.7,
        is_anomaly=True,
        spectral_result=None,
        dynamics_result=None,
        uiux_result=None,
        fusion_result=None,
        recursion_score=0.1,
        resonance_score=0.2,
        ethical_scaling=0.9,
        component_scores={"spectral": 0.7},
        detector_features=torch.zeros(4),
        recommendations=[],
    )
    assert result.metadata == {}


class TestVisualBackbone:
    def test_aggregator_shapes_and_missing_layer(self) -> None:
        torch = pytest.importorskip("torch")
        from omni_mercury_engine.detectors.visual.backbone import MultiScaleFeatureAggregator

        aggregator = MultiScaleFeatureAggregator({"l2": 8, "l3": 16}, output_dim=12)
        features = {"l2": torch.randn(2, 8, 16, 16), "l3": torch.randn(2, 16, 8, 8)}
        assert aggregator(features).shape == (2, 12)
        with pytest.raises(KeyError, match="l3"):  # released: an opaque matmul error
            aggregator({"l2": features["l2"]})

    def test_patch_embedding_grid(self) -> None:
        torch = pytest.importorskip("torch")
        from omni_mercury_engine.detectors.visual.backbone import PatchEmbedding

        patches, grid = PatchEmbedding(4, patch_size=3, stride=1)(torch.randn(2, 4, 10, 8))
        assert grid == (8, 6)
        assert patches.shape == (2, 48, 36)
        projected, _ = PatchEmbedding(4, patch_size=3, output_dim=10)(torch.randn(1, 4, 5, 5))
        assert projected.shape == (1, 9, 10)


def test_text_log_detector_sends_its_context() -> None:
    from omni_mercury_engine.models.foundation.llm_adapter import (
        BaseLLMAdapter,
        LLMConfig,
        TextLogAnomalyDetector,
    )

    class Recording(BaseLLMAdapter):
        prompts: list[str] = []

        def generate(self, prompt: str, system_prompt: str | None = None) -> str:
            self.prompts.append(prompt)
            return json.dumps(
                {
                    "is_anomaly": True,
                    "anomaly_score": 0.9,
                    "confidence": 0.8,
                    "category": "auth",
                    "explanation": "burst of failures",
                }
            )

        def is_available(self) -> bool:
            return True

    adapter = Recording(LLMConfig())
    detector = TextLogAnomalyDetector(
        log_format="syslog", known_patterns=["session opened"], adapter=adapter
    )
    result = detector.detect_log_anomaly("sshd: 50 failed logins", ["sshd: accepted key"])
    assert result["anomaly_score"] == 0.9
    assert "syslog" in adapter.prompts[-1]
    assert "session opened" in adapter.prompts[-1]
    assert "accepted key" in adapter.prompts[-1]
    detector.detect_text_anomaly("lorem", expected_domain="finance")
    assert "finance" in adapter.prompts[-1]


def test_int_source_registry_factory() -> None:
    from omni_mercury_engine.security.int_sources import (
        IntelligenceSourceRegistry,
        create_int_source_registry,
    )

    assert isinstance(create_int_source_registry(), IntelligenceSourceRegistry)


@pytest.mark.parametrize(
    ("module", "factory", "count"),
    [
        ("omni_mercury_engine.medical.abms_disciplines", "create_omni_medical_scalars", 15),
        ("omni_mercury_engine.models.chemistry", "create_omni_chemistry_scalars", 10),
        ("omni_mercury_engine.models.parapsychology", "create_omni_psi_scalars", 10),
        (
            "omni_mercury_engine.security.intelligence_fusion",
            "create_omni_intelligence_scalars",
            13,
        ),
        ("omni_mercury_engine.space.interstellar_objects", "create_omni_interstellar_scalars", 8),
        ("omni_mercury_engine.space.schumann_resonance", "create_omni_resonance_scalars", 10),
    ],
)
def test_scalar_tables_are_fixed_constants(module: str, factory: str, count: int) -> None:
    import importlib

    table = getattr(importlib.import_module(module), factory)()
    assert len(table) == count
    assert all(key.startswith("omni_") for key in table)
    assert all(1.33 * 1.618 - 1e-9 <= v <= 1.50 * 1.618 + 1e-9 for v in table.values())
    assert getattr(importlib.import_module(module), factory)() == table


def test_tide_periods_reexport() -> None:
    from omni_mercury_engine.detectors.geological import sea_level
    from omni_mercury_engine.ml.hazard_training import tsunami_waveform

    assert tsunami_waveform.TIDE_PERIODS_H is sea_level.TIDE_PERIODS_H


def test_golden_ratio_weights_keeps_its_released_position() -> None:
    """Fails without the restored keyword (TypeError), and on positional calls
    that would otherwise switch counterintelligence screening off."""
    from omni_mercury_engine.security.intelligence_fusion import IntelligenceFusionEngine

    with pytest.warns(DeprecationWarning, match="golden_ratio_weights"):
        engine = IntelligenceFusionEngine(False, False, False)
    assert engine.counterintelligence is not None
    with pytest.warns(DeprecationWarning):
        IntelligenceFusionEngine(golden_ratio_weights=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert IntelligenceFusionEngine(counterintelligence=False).counterintelligence is None
    with pytest.raises(TypeError):
        IntelligenceFusionEngine(True, True, None, False)  # type: ignore[call-arg]


def test_restored_arrays_are_numpy() -> None:
    from omni_mercury_engine.core.three_r_mechanism import _LegacyOAEWeightOptimizer

    assert isinstance(_LegacyOAEWeightOptimizer().initial_weights, np.ndarray)
