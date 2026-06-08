# Copyright (C) 2025 Steel Security Advisors LLC
"""Pytest configuration and fixtures.

Uses DeterministicRNG for reproducible tests.
All test fixtures now use seeded random number generation
to ensure consistent test results across runs.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

# Synthetic fallback contract for the unit-test suite.  The dataset
# loaders are strict-by-default — they raise ``DataSourceUnavailableError``
# rather than silently returning generated data — and that production
# posture is preserved in main code.  The unit-test suite, however,
# deliberately exercises the synthetic-fallback paths offline (no
# network, no API keys), so we opt into synthetic generation here.
#
# This setting must be in the root conftest, not in any individual test
# module, because pytest-xdist (used by the ``ml-tests`` job with
# ``-n 4``) spawns workers that each import this conftest before any
# test module is collected — guaranteeing the flag is set in every
# worker before the dataset loaders read it.  Setting it at the
# module-load time of a single test file (e.g. ``test_datasets.py``)
# only worked when that file happened to be collected first by the
# main process; under xdist the env var would not propagate to
# workers that pick up dataset tests in other files first, and the
# synthetic-fallback tests would non-deterministically fail.
#
# ``setdefault`` (not direct assignment) so the CI workflow YAML can
# still pin the variable to ``"0"`` for an explicitly real-data run.
os.environ.setdefault("MERCURY_ALLOW_SYNTHETIC", "1")

import numpy as np
import pytest

# Centralized availability check -- avoids importing torch at probe time
from omni_mercury_engine._compat import HAS_TORCH
from omni_mercury_engine.utils.rng import DeterministicRNG, set_global_seed

if HAS_TORCH:
    import torch

    if os.environ.get("PYTEST_XDIST_WORKER"):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

# Default seed for reproducibility
DEFAULT_TEST_SEED = 42

# ---------------------------------------------------------------------------
# Session-start ML-extra gate
# ---------------------------------------------------------------------------
#
# Mercury Agent's test suite exercises engine, fusion, and detector code
# whose production modules import torch at module load.  Historically this
# was managed with ~30 class-level ``@pytest.mark.skipif(not _HAS_TORCH,
# ...)`` decorators sprinkled across nine test files: when an operator
# (or a CI lane) ran pytest without the ``[ml]`` extra, those classes
# would silently turn into SKIPPED records and the run would report
# green.  That pattern masked real regressions — a test that had become
# importable because of an upstream API rename, but was now skipped on
# the same lane that should be exercising it, looked identical to a
# clean pass.
#
# This hook is the replacement: a single, declarative env-validation
# gate.  CI lanes that install the ``[ml]`` extra (Core Tests, ML Tests,
# Neuro-Symbolic Tests, the release lane after this PR) set the env
# var ``MERCURY_REQUIRES_ML=1`` to declare the contract.  At session
# start, the gate verifies the contract holds:
#
# * Contract declared + torch present  -> session continues normally.
# * Contract declared + torch missing  -> session aborts at start with
#   one loud, actionable message.  No "1842 tests skipped" green-wash.
# * Contract NOT declared              -> session continues.  Tests
#   whose imports require torch will fail at collection / runtime with
#   the underlying ImportError; that is the correct failure mode for
#   the minimal-install lanes (dataset-reachability, network-tests,
#   pqc-production-check) which intentionally only collect tests their
#   install set covers.
#
# The override env var is read on every session start so the gate
# cannot be sidestepped silently — flipping it off is a deliberate
# operator action visible in the workflow YAML.


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fail-loud env gates: refuse to run if a declared tier's dep is absent.

    Two must-run tier contracts are enforced here, each declared by an
    env var the CI workflow YAML sets:

    * ``MERCURY_REQUIRES_ML=1`` -- :mod:`torch` must import.  Covers the
      engine / fusion / detector / σ_Immutable-gate test surface.
    * ``MERCURY_REQUIRES_LEAN=1`` -- a Lean 4 toolchain must be on PATH.
      Covers the formal-proof theorem tier
      (``tests/test_lean_theorem_verifier.py::TestLiveLeanKernel``), which
      otherwise ``skipif``-s itself away.  Without this gate, a broken
      Lean install in the lane that owns the theorem tier
      (``.github/workflows/verifiers.yml``) would let the live theorem
      tests silently skip while the job still reports green -- "Lean
      missing" masquerading as "Lean tests passed".

    Each gate is unconditional once its flag is set: there is no
    second-chance override that re-enables silent skipping.  The operator
    either provides the dependency the lane declared it needs, or pytest
    refuses to lie about coverage.  Lanes that legitimately do NOT own a
    tier simply leave its flag unset (the tier then skips cleanly by
    design in that thin env).
    """
    if os.environ.get("MERCURY_REQUIRES_ML") == "1" and not HAS_TORCH:
        pytest.exit(
            "\n"
            "===============================================================\n"
            "  Mercury Agent ML-extra gate (MERCURY_REQUIRES_ML=1)\n"
            "===============================================================\n"
            "This pytest session declared the [ml] extra as a hard contract,\n"
            "but `import torch` cannot be resolved in this interpreter.\n"
            "\n"
            "Likely causes:\n"
            "  * The CI step ran `pip install -e '.[dev]'` without `[ml]`.\n"
            "  * A pinned torch wheel failed to resolve and the install step\n"
            "    succeeded with a warning instead of failing.\n"
            "  * The lane was retargeted to a minimal-install image but the\n"
            "    workflow YAML still exports MERCURY_REQUIRES_ML=1.\n"
            "\n"
            "Remediation:\n"
            "  pip install -e '.[ml,dev]'        # full ML test surface\n"
            "  pip install -e '.[all,dev]'       # everything the ml-tests lane installs\n"
            "\n"
            "If this lane intentionally does NOT need [ml], unset\n"
            "MERCURY_REQUIRES_ML in the workflow YAML and select the test\n"
            "subset explicitly (see dataset-reachability.yml for the pattern).\n"
            "===============================================================\n",
            returncode=2,
        )

    if os.environ.get("MERCURY_REQUIRES_LEAN") == "1":
        from omni_mercury_engine.verifiers.lean_theorem import lean_available

        if not lean_available():
            pytest.exit(
                "\n"
                "===============================================================\n"
                "  Mercury Agent Lean theorem-tier gate (MERCURY_REQUIRES_LEAN=1)\n"
                "===============================================================\n"
                "This pytest session declared the Lean 4 theorem tier as a\n"
                "hard contract, but no `lean` executable is on PATH.\n"
                "\n"
                "The live theorem tests "
                "(test_lean_theorem_verifier.py::TestLiveLeanKernel)\n"
                "are skipif(not lean_available()).  Without this gate a\n"
                "failed Lean install would let them SKIP while the job stays\n"
                "green -- 'Lean missing' masquerading as 'Lean tests passed'.\n"
                "\n"
                "Remediation (see .github/workflows/verifiers.yml):\n"
                "  curl -fsSL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -o elan-init.sh\n"
                "  sh elan-init.sh -y --default-toolchain stable\n"
                '  echo "$HOME/.elan/bin" >> "$GITHUB_PATH"\n'
                "\n"
                "If this lane intentionally does NOT own the theorem tier,\n"
                "leave MERCURY_REQUIRES_LEAN unset -- the tier then skips\n"
                "cleanly by design in this thin environment.\n"
                "===============================================================\n",
                returncode=2,
            )


@pytest.fixture(autouse=True)
def set_random_seed() -> Iterator[None]:
    """
    Set a deterministic seed before each test.

    This fixture runs automatically for all tests to ensure
    reproducibility and eliminate test flakiness from RNG.
    """
    set_global_seed(DEFAULT_TEST_SEED)
    np.random.seed(DEFAULT_TEST_SEED)
    if HAS_TORCH:
        torch.manual_seed(DEFAULT_TEST_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(DEFAULT_TEST_SEED)
    yield


@pytest.fixture(autouse=True)
def _restore_engine_logger_propagation() -> Iterator[None]:
    """Keep ``omni_mercury_engine`` logger propagating across every test.

    ``omni_mercury_engine.utils.logging.configure_logging`` (exercised by
    ``tests/test_logging_utils.py``) sets ``propagate = False`` on the
    package's parent logger to prevent double-printing through Python's
    root logger in production.  Pytest's ``caplog`` fixture, however,
    captures records via a handler on the **root** logger and depends on
    the propagation chain to receive them — so once any earlier test
    leaves ``propagate = False`` set, every downstream caplog assertion
    against an ``omni_mercury_engine.*`` logger silently sees an empty
    ``caplog.text``.

    This fixture saves and restores ``propagate`` around every test in
    the suite so that test-isolation regressions cannot break caplog
    capture again.  It does not weaken production behavior — the
    ``configure_logging`` flip remains in place during the body of each
    test that exercises it; we only restore the flag between tests.
    """
    logger = logging.getLogger("omni_mercury_engine")
    previous_propagate = logger.propagate
    logger.propagate = True
    try:
        yield
    finally:
        logger.propagate = previous_propagate


@pytest.fixture
def deterministic_rng() -> DeterministicRNG:
    """
    Provide a DeterministicRNG instance for tests.

    Use this fixture when you need explicit control over
    the RNG in your test.
    """
    return DeterministicRNG(seed=DEFAULT_TEST_SEED)


@pytest.fixture
def sample_data(deterministic_rng: DeterministicRNG) -> np.ndarray:
    """Generate sample data for testing using deterministic RNG."""
    return deterministic_rng.randn(100, 10)


@pytest.fixture
def sample_tensor(set_random_seed: None) -> Any:
    """Generate sample tensor for testing (requires torch)"""
    if not HAS_TORCH:
        pytest.skip("torch not installed - skipping ML test")
    return torch.randn(100, 10)


@pytest.fixture
def anomaly_data(deterministic_rng: DeterministicRNG) -> np.ndarray:
    """Generate data with known anomalies using deterministic RNG."""
    normal = deterministic_rng.randn(90, 10)
    anomalies = deterministic_rng.randn(10, 10) * 5
    return np.vstack([normal, anomalies])


@pytest.fixture
def biometric_sample(deterministic_rng: DeterministicRNG) -> dict[str, np.ndarray]:
    """Generate sample biometric data using deterministic RNG."""
    image_arr = cast("np.ndarray", deterministic_rng.randint(0, 255, (224, 224, 3)))
    return {
        "image": image_arr.astype(np.uint8),
        "face_mesh": deterministic_rng.randn(468, 3),
    }


@pytest.fixture
def univariate_data(deterministic_rng: DeterministicRNG) -> np.ndarray:
    """Generate univariate time series data for testing."""
    return deterministic_rng.randn(1000)


@pytest.fixture
def multivariate_data(deterministic_rng: DeterministicRNG) -> np.ndarray:
    """Generate multivariate time series data for testing."""
    return deterministic_rng.randn(500, 20)


@pytest.fixture
def ecg_signal(deterministic_rng: DeterministicRNG) -> np.ndarray:
    """Generate synthetic ECG-like signal for medical tests."""
    t = np.linspace(0, 10, 5000)
    # Simple ECG-like waveform
    ecg = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    ecg += deterministic_rng.randn(len(t)) * 0.1
    return ecg


@pytest.fixture
def threat_features(deterministic_rng: DeterministicRNG) -> np.ndarray:
    """Generate synthetic threat feature vectors for security tests."""
    return deterministic_rng.randn(256)


@pytest.fixture
def seismic_sequence(deterministic_rng: DeterministicRNG) -> np.ndarray:
    """Generate synthetic seismic sequence for geological tests."""
    return deterministic_rng.randn(100, 32)


@pytest.fixture
def thermal_data(deterministic_rng: DeterministicRNG) -> dict[str, Any]:
    """Generate synthetic thermal data for volcanic monitoring tests."""
    base_temp = 288.0  # 15°C in Kelvin
    return {
        "brightness_temperature_k": deterministic_rng.randn(100) * 10 + base_temp,
        "radiant_heat_mw": deterministic_rng.rand(1)[0] * 100,
    }


@pytest.fixture
def gas_emissions(deterministic_rng: DeterministicRNG) -> dict[str, float]:
    """Generate synthetic gas emission data for volcanic tests."""
    return {
        "so2_tons_per_day": deterministic_rng.rand(1)[0] * 200 + 50,
        "co2_tons_per_day": deterministic_rng.rand(1)[0] * 1000 + 200,
    }


@pytest.fixture
def schumann_resonance(deterministic_rng: DeterministicRNG) -> np.ndarray:
    """Generate synthetic Schumann resonance data."""
    t = np.linspace(0, 1, 1000)
    # 7.83 Hz fundamental frequency with noise
    signal = np.sin(2 * np.pi * 7.83 * t) + deterministic_rng.randn(len(t)) * 0.1
    return signal


# =============================================================================
# Visual Anomaly Detection Fixtures
# =============================================================================


@pytest.fixture
def sample_image(deterministic_rng: DeterministicRNG) -> Any:
    """Generate sample image tensor for visual anomaly detection tests."""
    if not HAS_TORCH:
        pytest.skip("torch not installed - skipping visual test")
    # [B, C, H, W] format - 1 batch, 3 channels, 224x224
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_image_batch(deterministic_rng: DeterministicRNG) -> Any:
    """Generate batch of sample images for visual anomaly detection tests."""
    if not HAS_TORCH:
        pytest.skip("torch not installed - skipping visual test")
    # [B, C, H, W] format - 4 batch, 3 channels, 224x224
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_video_frames(deterministic_rng: DeterministicRNG) -> list[Any]:
    """Generate sample video frames for VLM tests."""
    if not HAS_TORCH:
        pytest.skip("torch not installed - skipping VLM test")
    # List of frames [T, H, W, C]
    return [torch.randn(224, 224, 3) for _ in range(16)]


@pytest.fixture
def time_series_with_anomaly(deterministic_rng: DeterministicRNG) -> np.ndarray:
    """Generate time series with known anomaly for foundation model tests."""
    # Normal data with spike anomaly
    data = deterministic_rng.randn(200)
    # Insert anomaly spike at position 150
    data[150] = 10.0  # Clear anomaly
    return data


@pytest.fixture
def time_series_multivariate(deterministic_rng: DeterministicRNG) -> np.ndarray:
    """Generate multivariate time series for foundation model tests."""
    return deterministic_rng.randn(200, 5)


@pytest.fixture
def binary_labels(deterministic_rng: DeterministicRNG) -> np.ndarray:
    """Generate binary labels for metric testing."""
    # 90 normal (0) + 10 anomalies (1)
    labels = np.zeros(100)
    labels[90:] = 1
    return labels


@pytest.fixture
def anomaly_scores(deterministic_rng: DeterministicRNG) -> np.ndarray:
    """Generate anomaly scores corresponding to binary_labels."""
    # Lower scores for normal, higher for anomalies
    scores = deterministic_rng.rand(100)
    scores[90:] += 0.5  # Anomalies have higher scores
    return scores


@pytest.fixture
def pixel_masks(deterministic_rng: DeterministicRNG) -> np.ndarray:
    """Generate pixel-level masks for localization metrics."""
    # [N, H, W] binary masks
    masks = np.zeros((10, 64, 64))
    # Add some anomalous regions
    masks[:, 20:40, 20:40] = 1
    return masks


@pytest.fixture
def pixel_scores(deterministic_rng: DeterministicRNG) -> np.ndarray:
    """Generate pixel-level anomaly scores."""
    # [N, H, W] score maps
    scores = deterministic_rng.rand(10, 64, 64) * 0.3
    # Higher scores in anomalous regions
    scores[:, 20:40, 20:40] += 0.5
    return scores


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip ``@pytest.mark.network`` tests unless ``MERCURY_NETWORK_TESTS=1`` is set.

    Why this gate exists (and why it is a *capability* gate, not a
    defect): every test bearing ``pytest.mark.network`` calls a
    third-party live endpoint (Open-Meteo, NOAA GSOD/ERDDAP/Storm,
    NASA FIRMS, USGS Earthquake, NIST CSRC, FEMA, EIA, USDA, etc.).
    Each loader has bounded timeouts, retry-with-backoff, and a
    schema validator — verified to pass against the live source on
    demand — but external-API liveness is genuinely outside our
    control: a NOAA outage or a 30s upstream lag on a single endpoint
    would otherwise flip CI red despite zero defects on our side.

    The gate therefore keeps the *default* CI run deterministic while
    still letting an operator (or a self-hosted runner with stable
    upstream connectivity) execute the full reachability lane via
    ``MERCURY_NETWORK_TESTS=1``.  This matches the "genuine external
    dependency" branch of the project doctrine: attempt with proper
    safeguards (timeouts/retry/schema) first; gate only as the last
    resort, with this comment as the justification on record.
    """
    if os.environ.get("MERCURY_NETWORK_TESTS", "0") == "1":
        return
    skip_network = pytest.mark.skip(
        reason=(
            "network tests disabled (set MERCURY_NETWORK_TESTS=1); these hit "
            "third-party live endpoints whose availability is outside our control."
        )
    )
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


# Marker for slow tests
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "security: marks security tests")
    config.addinivalue_line("markers", "medical: marks medical domain tests")
    config.addinivalue_line("markers", "geological: marks geological domain tests")
    config.addinivalue_line("markers", "visual: marks visual anomaly detection tests")
    config.addinivalue_line("markers", "vlm: marks vision-language model tests")
    config.addinivalue_line("markers", "foundation: marks foundation model tests")
