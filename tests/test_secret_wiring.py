# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression guard: repository data-source secrets stay wired end-to-end.

This test exists because two real wiring gaps were found and fixed:

1. **Name drift.** The wildfire loader reads ``NASA_FIRMS_MAP_KEY`` but the
   repository Actions secret is named ``FIRMS_MAP_KEY`` -- so the key never
   reached the loader.
2. **Never injected.** None of the data-source secrets were referenced by any
   workflow, so every keyed domain loader ran without credentials in CI and its
   live-wiring test silently skipped or degraded to a soft failure.

The checks are deliberately **hermetic** -- they read source files and the
workflow YAML rather than importing the engine, so the guard runs in every CI
lane regardless of whether the optional ML / native-crypto stacks are installed.
The one behavioural check (that the base loader resolves the fallback env name)
imports the engine inside a ``try``/``except`` that treats only the two
"environment not provisioned" failures as skippable -- an optional/transitive
``ImportError`` or the native-crypto PQC ``RuntimeError`` gate
(``AMA/PQC is mandatory ...``, mirroring ``tests/test_calibration_brief.py``) --
and re-raises anything else (a ``NameError``/``SyntaxError`` or a non-PQC
``RuntimeError``) so a genuine loader regression fails loudly instead of being
masked as an offline skip. See :func:`_import_skip_reason`.
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

#: The one RuntimeError this guard treats as "environment not provisioned"
#: rather than a regression: the native-crypto PQC import gate raised by
#: ``omni_mercury_engine._pqc_gate`` when no AMA backend is built. Kept in
#: lockstep with the same marker in ``tests/test_calibration_brief.py`` so the
#: skip condition is narrow and identical across the suite.
_PQC_GATE_MARKER = "AMA/PQC is mandatory"


def _import_skip_reason(exc: BaseException) -> str | None:
    """Classify an engine-import failure as skippable or a genuine regression.

    Importing the loaders package can fail for two "environment not provisioned"
    reasons that the hermetic default CI lane legitimately hits: an optional or
    transitive dependency is absent (:class:`ImportError`), or the mandatory
    native post-quantum-crypto backend is not built -- whose import gate raises
    ``RuntimeError("AMA/PQC is mandatory ...")``. Both mean "skip cleanly here".

    Any other failure -- a :class:`NameError`/``SyntaxError`` from a real code
    regression, or a *non-PQC* :class:`RuntimeError` -- is not an environment
    problem and must surface, so this returns ``None`` and the caller re-raises.

    Args:
        exc: The exception raised while importing the engine / loaders package.

    Returns:
        A human-readable skip reason when ``exc`` is an environment-not-
        provisioned import failure, else ``None`` (meaning: re-raise, never mask).
    """
    if isinstance(exc, ImportError):
        return f"optional/transitive dependency unavailable: {exc}"
    if isinstance(exc, RuntimeError) and _PQC_GATE_MARKER in str(exc):
        return f"native-crypto PQC gate not provisioned: {exc}"
    return None


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

    def test_eros_credentials_injected(self) -> None:
        # USGS EROS/EarthExplorer M2M needs a username + token (or password);
        # all three must be injected for the credential-delivery check to run.
        env = _collect_step_env(_load_workflow())
        for env_var in ("EROSERS_USERNAME", "USGS_KEY", "EROSERS_PASSWORD"):
            assert env_var in env, f"network-tests must inject {env_var} for the EROS check"
            assert _SECRET_RE.search(
                env[env_var]
            ), f"{env_var} must be fed from a repository secret"

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
        # Importing the loaders package can fail for "environment not provisioned"
        # reasons only (optional/transitive ImportError, or the native-crypto PQC
        # RuntimeError gate). _import_skip_reason() encodes exactly that policy:
        # skip for those, and re-raise anything else -- a non-PQC RuntimeError or
        # a NameError/SyntaxError -- so a genuine loader regression fails loudly
        # instead of being masked as an offline skip.
        try:
            from omni_mercury_engine.loaders.wildfire_loader import WildfireLoader
        except (ImportError, RuntimeError) as exc:
            reason = _import_skip_reason(exc)
            if reason is None:
                raise
            pytest.skip(f"engine import unavailable: {reason}")
        monkeypatch.delenv("NASA_FIRMS_MAP_KEY", raising=False)
        monkeypatch.setenv("FIRMS_MAP_KEY", "firms-secret-value")
        loader = WildfireLoader(cache_dir=tmp_path)
        assert loader._api_key == "firms-secret-value"


class TestImportSkipClassification:
    """`_import_skip_reason` must skip only 'environment not provisioned' import
    failures and re-raise genuine regressions.

    Copilot review finding: a broad ``except`` around the engine import masks
    real breakage (a non-PQC ``RuntimeError``, a ``NameError`` from a loader
    refactor) as an offline skip, so the network lane goes green while the code
    is broken. These cases pin the narrow contract.
    """

    def test_import_error_is_skippable(self) -> None:
        assert _import_skip_reason(ImportError("No module named 'torch'")) is not None

    def test_module_not_found_is_skippable(self) -> None:
        # ModuleNotFoundError is an ImportError subclass -- also an optional-dep case.
        assert _import_skip_reason(ModuleNotFoundError("No module named 'obspy'")) is not None

    def test_pqc_gate_runtime_error_is_skippable(self) -> None:
        exc = RuntimeError("AMA/PQC is mandatory for Mercury; see docs/INSTALLATION.md")
        assert _import_skip_reason(exc) is not None

    def test_non_pqc_runtime_error_reraises(self) -> None:
        # A RuntimeError that is NOT the PQC gate is a real regression -> re-raise.
        assert _import_skip_reason(RuntimeError("loader misconfigured: bad state")) is None

    def test_name_error_reraises(self) -> None:
        assert _import_skip_reason(NameError("name 'WildfireLoadr' is not defined")) is None

    def test_value_error_reraises(self) -> None:
        assert _import_skip_reason(ValueError("unexpected value")) is None
