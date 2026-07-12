# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression guard: repository data-source secrets stay wired end-to-end.

This test exists because two real wiring gaps were found and fixed:

1. **Name drift.** The wildfire loader reads ``NASA_FIRMS_MAP_KEY`` but the
   repository Actions secret is named ``FIRMS_MAP_KEY`` -- so the key never
   reached the loader.
2. **Never injected.** None of the data-source secrets were referenced by any
   workflow, so every keyed domain loader ran without credentials in CI and its
   live-wiring test silently skipped or fail-softed.

The checks are deliberately **hermetic** -- they read source files and the
workflow YAML rather than importing the engine, so the guard runs in every CI
lane regardless of whether the optional ML / native-crypto stacks are installed.
The one behavioural check (that the base loader resolves the fallback env name)
is guarded with ``importorskip`` so it runs where the deps exist and skips
cleanly where they do not.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src" / "omni_mercury_engine"
_LOADERS = _SRC / "loaders"
_NETWORK_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "network-tests.yml"

#: loader source file -> canonical env var it must read for its API key.
CANONICAL_ENV = {
    "wildfire_loader.py": "NASA_FIRMS_MAP_KEY",
    "financial_loader.py": "FRED_API_KEY",
    "energy_loader.py": "EIA_API_KEY",
    "space_weather_loader.py": "NASA_API_KEY",
    "meteor_loader.py": "NASA_API_KEY",
}

#: env var the loader reads -> repository Actions secret that must feed it.
#: (FIRMS is deliberately mapped onto the canonical env name the loader reads.)
REQUIRED_INJECTIONS = {
    "FRED_API_KEY": "FRED_API_KEY",
    "EIA_API_KEY": "EIA_API_KEY",
    "NASA_API_KEY": "NASA_API_KEY",
    "NASA_FIRMS_MAP_KEY": "FIRMS_MAP_KEY",
    "ALPHA_VANTAGE_API_KEY": "ALPHA_VANTAGE_API_KEY",
    "OPENWEATHERMAP_API_KEY": "OPENWEATHERMAP_API_KEY",
}

_SECRET_RE = re.compile(r"\$\{\{\s*secrets\.([A-Z0-9_]+)\s*\}\}")


class TestLoaderSourceContract:
    @pytest.mark.parametrize("filename,expected", list(CANONICAL_ENV.items()))
    def test_canonical_env_var_declared(self, filename: str, expected: str) -> None:
        src = (_LOADERS / filename).read_text()
        # A loader may read its key via the BaseDomainLoader API_KEY_ENV_VAR
        # attribute or via a direct os.environ/os.getenv call (e.g. the energy
        # loader treats EIA as an optional enrichment over keyless NOAA SWPC).
        # Either way the canonical name must appear in an environment-read.
        patterns = [
            rf'API_KEY_ENV_VAR:\s*str\s*=\s*"{expected}"',
            rf'os\.environ\.get\(\s*"{expected}"',
            rf'os\.getenv\(\s*"{expected}"',
        ]
        assert any(re.search(p, src) for p in patterns), (
            f"{filename} must read the canonical env var {expected!r} "
            "(the name the workflow injection and docs agree on)"
        )

    def test_wildfire_declares_firms_fallback(self) -> None:
        src = (_LOADERS / "wildfire_loader.py").read_text()
        assert "API_KEY_ENV_FALLBACKS" in src and "FIRMS_MAP_KEY" in src, (
            "WildfireLoader must accept the FIRMS_MAP_KEY secret name via " "API_KEY_ENV_FALLBACKS"
        )

    def test_base_loader_implements_fallback(self) -> None:
        src = (_LOADERS / "base.py").read_text()
        assert "API_KEY_ENV_FALLBACKS" in src, "base loader must define the fallback attribute"
        # the resolution loop must actually consult the fallbacks
        assert re.search(
            r"for\s+\w+\s+in\s+self\.API_KEY_ENV_FALLBACKS", src
        ), "base loader must iterate API_KEY_ENV_FALLBACKS when the canonical var is unset"


def _load_workflow() -> dict[Any, Any]:
    assert _NETWORK_WORKFLOW.is_file(), f"missing workflow: {_NETWORK_WORKFLOW}"
    # PyYAML parses the bare ``on:`` mapping key as the boolean ``True``, so the
    # workflow dict is keyed by a mix of str and bool -- type it as Any-keyed.
    return cast("dict[Any, Any]", yaml.safe_load(_NETWORK_WORKFLOW.read_text()))


def _collect_step_env(wf: dict[Any, Any]) -> dict[str, str]:
    """Merge the env of every step across every job (flat name->value view)."""
    merged: dict[str, str] = {}
    for job in wf.get("jobs", {}).values():
        for step in job.get("steps", []):
            for k, v in (step.get("env") or {}).items():
                merged[k] = str(v)
    return merged


class TestWorkflowInjection:
    def test_every_secret_injected_onto_loader_env(self) -> None:
        env = _collect_step_env(_load_workflow())
        for env_var, secret in REQUIRED_INJECTIONS.items():
            assert env_var in env, f"network-tests must inject {env_var} for its loader"
            m = _SECRET_RE.search(env[env_var])
            assert m, f"{env_var} must be fed from a repository secret, got {env[env_var]!r}"
            assert (
                m.group(1) == secret
            ), f"{env_var} must be fed from secrets.{secret}, got secrets.{m.group(1)}"

    def test_workflow_not_fork_pr_triggered(self) -> None:
        wf = _load_workflow()
        # PyYAML parses the bare ``on:`` key as boolean True; handle both.
        triggers = wf.get("on", wf.get(True, {}))
        keys = set(triggers) if isinstance(triggers, (dict, list)) else {triggers}
        assert "pull_request" not in keys, "secrets must not be exposed to PR-triggered runs"
        assert "pull_request_target" not in keys


class TestFallbackBehaviour:
    """Behavioural check of the base-loader fallback (guarded on optional deps)."""

    def test_wildfire_resolves_firms_secret_name(self, monkeypatch: Any, tmp_path: Path) -> None:
        pytest.importorskip("torch")  # loaders package eagerly imports torch-gated detectors
        try:
            from omni_mercury_engine.loaders.wildfire_loader import WildfireLoader
        except Exception as exc:  # pragma: no cover - native crypto backend absent locally
            pytest.skip(f"engine import unavailable: {exc}")
        monkeypatch.delenv("NASA_FIRMS_MAP_KEY", raising=False)
        monkeypatch.setenv("FIRMS_MAP_KEY", "firms-secret-value")
        loader = WildfireLoader(cache_dir=tmp_path)
        assert loader._api_key == "firms-secret-value"
