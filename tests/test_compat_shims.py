# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the backward-compatibility shim modules.

Covers:
- ``detectors/spectral_domain_oracle.py``: the legacy import path resolves and
  every re-exported symbol is identical to its canonical
  ``detectors/spectral_domain_frequency.py`` binding
- ``core/self_healing.py``: legacy import path resolves, re-exports match
  ``resilience/self_healing.py``, the ``CRISPRInspiredSelfHealing`` alias
  points at ``AdaptiveDefenseSystem``, ``__all__`` is exact, the
  ``SelfHealingDeprecationWarning`` fires on import, and
  ``MERCURY_AGENT_SUPPRESS_DEPRECATION_WARNINGS`` suppresses it
- ``anomaly/__init__.py``: the placeholder fusion package imports cleanly and
  deliberately exports nothing
"""

from __future__ import annotations

import importlib
import sys
import warnings
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

_SELF_HEALING_SHIM = "omni_mercury_engine.core.self_healing"
_SELF_HEALING_CANONICAL = "omni_mercury_engine.resilience.self_healing"
_SUPPRESS_ENV = "MERCURY_AGENT_SUPPRESS_DEPRECATION_WARNINGS"

# Symbols the spectral_domain_oracle shim promises to re-export, per its
# explicit import list.
_SPECTRAL_REEXPORTS = [
    "DEFAULT_ALPHA",
    "DOMAIN_ANOMALY_SPECTRAL_HINTS",
    "DOMAIN_FREQUENCY_BANDS",
    "EPSILON",
    "PHI",
    "FrequencyBandResult",
    "FrequencyDomainOracle",
    "FrequencyDomainOracleConfig",
    "FrequencyInfluenceVector",
    "FrequencyWeighting",
    "SpectralDomainFrequency",
    "SpectralDomainFrequencyConfig",
    "SpectralDomainOracle",
    "SpectralDomainOracleConfig",
    "create_frequency_oracle",
    "create_spectral_frequency",
    "create_spectral_oracle",
    "get_domain_frequency_bands",
]


# ---------------------------------------------------------------------------
# spectral_domain_oracle -> spectral_domain_frequency
# ---------------------------------------------------------------------------


class TestSpectralDomainOracleShim:
    def test_legacy_import_path_resolves(self) -> None:
        shim = importlib.import_module("omni_mercury_engine.detectors.spectral_domain_oracle")
        assert shim is not None

    def test_every_reexport_is_the_canonical_object(self) -> None:
        shim = importlib.import_module("omni_mercury_engine.detectors.spectral_domain_oracle")
        canonical = importlib.import_module(
            "omni_mercury_engine.detectors.spectral_domain_frequency"
        )
        for name in _SPECTRAL_REEXPORTS:
            assert getattr(shim, name) is getattr(canonical, name), (
                f"spectral_domain_oracle.{name} is not the canonical "
                f"spectral_domain_frequency.{name}"
            )

    def test_legacy_class_aliases_are_usable(self) -> None:
        shim = importlib.import_module("omni_mercury_engine.detectors.spectral_domain_oracle")
        # The renamed detector must be constructible through the legacy path.
        assert isinstance(shim.SpectralDomainOracle, type)
        assert isinstance(shim.SpectralDomainFrequency, type)


# ---------------------------------------------------------------------------
# core.self_healing -> resilience.self_healing
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_self_healing_shim() -> Iterator[None]:
    """Remove the shim from sys.modules so a test can re-import it, then
    restore whatever was there before."""
    saved = sys.modules.pop(_SELF_HEALING_SHIM, None)
    try:
        yield
    finally:
        sys.modules.pop(_SELF_HEALING_SHIM, None)
        if saved is not None:
            sys.modules[_SELF_HEALING_SHIM] = saved


def _import_shim_capturing_warnings(
    monkeypatch: pytest.MonkeyPatch, *, suppress: str | None
) -> tuple[Any, list[warnings.WarningMessage]]:
    """Freshly import the self_healing shim, recording emitted warnings."""
    if suppress is None:
        monkeypatch.delenv(_SUPPRESS_ENV, raising=False)
    else:
        monkeypatch.setenv(_SUPPRESS_ENV, suppress)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(_SELF_HEALING_SHIM)
    return module, list(caught)


def _deprecation_warnings(
    caught: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    return [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and w.category.__name__ == "SelfHealingDeprecationWarning"
    ]


class TestSelfHealingShim:
    @pytest.mark.usefixtures("fresh_self_healing_shim")
    def test_deprecation_warning_fires_on_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, caught = _import_shim_capturing_warnings(monkeypatch, suppress=None)
        emitted = _deprecation_warnings(caught)
        assert len(emitted) == 1
        message = str(emitted[0].message)
        assert "deprecated" in message
        assert "omni_mercury_engine.resilience.self_healing" in message
        assert _SUPPRESS_ENV in message

    @pytest.mark.parametrize("value", ["1", "true", "YES"])
    @pytest.mark.usefixtures("fresh_self_healing_shim")
    def test_env_var_suppresses_warning(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        _, caught = _import_shim_capturing_warnings(monkeypatch, suppress=value)
        assert _deprecation_warnings(caught) == []

    @pytest.mark.usefixtures("fresh_self_healing_shim")
    def test_unrecognised_suppress_value_still_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, caught = _import_shim_capturing_warnings(monkeypatch, suppress="0")
        assert len(_deprecation_warnings(caught)) == 1

    @pytest.mark.usefixtures("fresh_self_healing_shim")
    def test_reexports_match_canonical_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        shim, _ = _import_shim_capturing_warnings(monkeypatch, suppress="1")
        canonical = importlib.import_module(_SELF_HEALING_CANONICAL)
        assert shim.AdaptiveDefenseSystem is canonical.AdaptiveDefenseSystem
        assert shim.AnomalySignature is canonical.AnomalySignature
        assert shim.SelfHealingEngine is canonical.SelfHealingEngine

    @pytest.mark.usefixtures("fresh_self_healing_shim")
    def test_crispr_alias_points_at_adaptive_defense_system(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shim, _ = _import_shim_capturing_warnings(monkeypatch, suppress="1")
        canonical = importlib.import_module(_SELF_HEALING_CANONICAL)
        assert shim.CRISPRInspiredSelfHealing is canonical.AdaptiveDefenseSystem

    @pytest.mark.usefixtures("fresh_self_healing_shim")
    def test_all_is_exact(self, monkeypatch: pytest.MonkeyPatch) -> None:
        shim, _ = _import_shim_capturing_warnings(monkeypatch, suppress="1")
        assert shim.__all__ == [
            "AdaptiveDefenseSystem",
            "AnomalySignature",
            "CRISPRInspiredSelfHealing",
            "SelfHealingEngine",
        ]
        for name in shim.__all__:
            assert hasattr(shim, name)

    @pytest.mark.usefixtures("fresh_self_healing_shim")
    def test_warning_category_is_importable_and_well_typed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shim, _ = _import_shim_capturing_warnings(monkeypatch, suppress="1")
        category = shim.SelfHealingDeprecationWarning
        assert issubclass(category, DeprecationWarning)


# ---------------------------------------------------------------------------
# anomaly package placeholder
# ---------------------------------------------------------------------------


class TestAnomalyPackageShim:
    def test_import_resolves(self) -> None:
        module = importlib.import_module("omni_mercury_engine.anomaly")
        assert module is not None

    def test_exports_nothing_by_design(self) -> None:
        module = importlib.import_module("omni_mercury_engine.anomaly")
        assert module.__all__ == []

    def test_docstring_documents_the_architectural_intent(self) -> None:
        module = importlib.import_module("omni_mercury_engine.anomaly")
        assert module.__doc__ is not None
        assert "cross-domain" in module.__doc__.lower()
