# Contributing to Mercury Agent

## Document Information

| Property | Value |
|----------|-------|
| Document Version | 2.8 |
| Last Updated | 2026-07-11 |
| Classification | Public |
| Maintainer | Steel Security Advisors LLC |
| Applies to | Mercury Agent v2.1.x |

---

## Overview

This document provides guidelines for contributing to the Mercury Agent neuro-symbolic AI framework. Mercury Agent is released under the GNU General Public License v3.0 or later (SPDX: GPL-3.0-or-later) as free and open-source software, accessible for universal use as a knowledge vault and bridge to AI/ML frontiers.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [AI-Assisted Development](#ai-assisted-development)
3. [Real-Data Contributions](#real-data-contributions)
4. [Getting Started](#getting-started)
5. [Development Setup](#development-setup)
6. [Code Style Guidelines](#code-style-guidelines)
7. [Testing](#testing)
8. [Pull Request Process](#pull-request-process)
9. [Adding New Engines](#adding-new-engines)
10. [Security Considerations](#security-considerations)

---

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## AI-Assisted Development

We welcome AI-assisted contributions (e.g., via Claude Code, GitHub Copilot):
- All AI-assisted PRs must include tests and documentation
- Human review required for all contributions
- Clearly indicate AI-assisted work in PR description

## Real-Data Contributions

**High Priority**: We actively seek contributions for real-world data integration:
- **Medical**: MIMIC-III vital signs, clinical trial data, additional
  CGM / vitals vendor adapters (Abbott LibreView, Medtronic CareLink,
  Philips IntelliVue, GE CARESCAPE, Mindray BeneVision) — see
  [`docs/medical/SETUP.md`](docs/medical/SETUP.md) for the
  `CGMDataSource` / `VitalsDataSource` contract.
- **Cybersecurity**: Actual PCAP files, malware samples (anonymized)
- **Drone telemetry**: PX4 ULog flight logs (via `pyulog`) and
  MAVLink ingest examples — see [`docs/drone/SETUP.md`](docs/drone/SETUP.md)
  for the `DroneState` contract.
- **Compliance**: Additional OSHA sector mappings, NIST CSF 2.0
  profile contributions, TLP 2.0 watermark exporters — see
  [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md).
- **SETI**: Breakthrough Listen observation data
- **Submit via**: Issue template "Real-Data Integration Request"

> **Medical, clinical, drone, and compliance contributions** must
> satisfy the integration-ready (not pre-integrated) posture: no
> vendor credentials in tree, no synthetic fallback, real ABC adapters
> with `ConfigurationError` raised when misconfigured. See the
> per-domain SETUP docs for the exact contract.

## Getting Started

### Prerequisites

- Python 3.11 or higher (3.12 recommended)
- Git
- Basic understanding of machine learning and anomaly detection
- Familiarity with ethical AI principles
- (Optional) CUDA-capable GPU for faster training

### Ways to Contribute

We welcome contributions in the following areas:

- **Bug Reports:** Report issues with anomaly detection, ML operations, or implementation errors
- **Security Fixes:** Address security vulnerabilities (see [SECURITY.md](SECURITY.md))
- **Documentation:** Improve clarity, add examples, correct errors
- **Testing:** Add test coverage, improve test quality
- **Performance:** Optimize ML operations without compromising accuracy
- **Features:** Implement new detection capabilities (discuss first in an issue)
- **Ethical Improvements:** Enhance bias detection and fairness mechanisms

### What NOT to Contribute

Please **DO NOT** submit pull requests that:

- Weaken security in any way
- Remove or bypass ethical safeguards. In particular, the
  **Wave B dual-gate hard ethical contract** (Benevolence then
  σ_Immutable, raising `EthicalConstraintViolationError(check=…)`)
  is non-negotiable at every public boundary surface. PRs that
  re-introduce a public flag to disable a gate, restore a silent
  GOSNN fallback, expose `_GOSNN_TESTING_BYPASS` to non-test code
  paths, or drop one of the reserved `check=` codes will be
  rejected. See `ARCHITECTURE.md` §"Dual-Gate Hard Ethical
  Enforcement" and `docs/MATH_SPEC.md` §2.1.5.
- Introduce unproven or experimental algorithms without validation
- Add unnecessary dependencies
- Reintroduce `pickle` for training data or arbitrary cross-process
  serialization (PR #166 deleted that code path). For dataset and
  benchmark artefacts use `npz` / `json` / `safetensors` /
  `parquet`. Trained-model weights may continue to ship as PyTorch
  `.pt` files **provided** they are loaded via
  `torch.load(..., weights_only=True)` (the safe-tensor torch
  loader path used by `security/sigma_immutable_weights.pt`).
  Plain `pickle.load` / `torch.load(weights_only=False)` is the
  banned surface, not the `.pt` extension.
- Add a non-AMA-Cryptography PQC backend (PR #144 made AMA
  Cryptography the **sole** PQC backend; the import-time gate in
  `_pqc_gate.py` is unconditional — `AMA_REQUIRE_REAL_PQC` is retained
  only for diagnostics and no longer disables the gate. Pinned to
  `v4.0.0` via the `ama-ref` input in
  `.github/workflows/pqc-production-check.yml` and
  `pyproject.toml [project.optional-dependencies].pqc`)
- Restore the `SafeHTTPClient(..., allow_untrusted=True)` kwarg
  removed in PR #210; new private-network call sites must use
  `user_configured=True[, allow_private=True]` so the SSRF /
  DNS-rebinding gate fires explicitly
- Restore a silent `MockLLMAdapter` fallback in
  `narrative/voice.py`; `MercuryVoice(enable_llm=True)` now requires
  an explicit `llm_provider=` argument naming an implemented provider
- Move governance frameworks back into `omni_mercury_engine.security`;
  `security/` is reserved for implementation primitives (crypto, PQC,
  threat detection, audit logging) and governance frameworks (NIST
  CSF 2.0, TLP 2.0, OSHA / eCFR) live in
  `omni_mercury_engine.compliance`
- Include proprietary or non-GPL-compatible code
- Lack proper testing and documentation

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Mercury-Agent.git
cd Mercury-Agent
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies via pyproject.toml
pip install --upgrade pip
pip install -e ".[dev]"

# For full installation with all optional features
pip install -e ".[all]"
```

### 3. Verify Setup

```bash
# Run the test suite
pytest tests/ -v

# Expected: All tests should pass
```

### 4. Create Feature Branch

```bash
# Create a new branch for your contribution
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### Windows-Specific Setup

**DeepFace/dlib Installation**:

Option 1 (Recommended): Use WSL
```bash
wsl --install
wsl
pip install deepface
```

Option 2: Use pre-built wheels
```bash
pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/releases/download/v19.22.99/dlib-19.22.99-cp312-cp312-win_amd64.whl
pip install deepface
```

Option 3: Install Visual Studio Build Tools
- Download Visual Studio Build Tools
- Select "Desktop development with C++"
- Install CMake
- `pip install deepface`

## Code Style Guidelines

### File Header Policy

Every Python file under `src/`, `tests/`, `scripts/`, `tools/`, `research/`, `benchmarks/`, `examples/`, and `assets/` must use the single canonical file header below, followed immediately by a real module docstring whose first line is the module summary. The copyright line states ownership; `SPDX-License-Identifier: GPL-3.0-or-later` is the ISO/IEC 5962:2021 machine-readable license tag and keeps the compact header REUSE-compliant. Do not paste the full GPL boilerplate into individual source files; the root `LICENSE` file is the authoritative GPL text. Run `python scripts/normalize_headers.py --apply` before submitting header-touching changes, and CI enforces the same policy with `python scripts/normalize_headers.py --check`.

```python
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""One-line module summary."""
```

Rust sources under `rust_crypto/src/` carry the same canonical pair, expressed with the `//` line-comment prefix and placed **above** any `//!` crate/module doc so the header never leaks into the rendered rustdoc. The same `normalize_headers.py` tool checks and applies them (`--apply` rewrites both Python and Rust), and the same pre-commit hook and CI gate enforce them.

```rust
// Copyright (C) 2025 Steel Security Advisors LLC
// SPDX-License-Identifier: GPL-3.0-or-later
//! One-line crate/module doc.
```

### General Principles

1. **Security First:** Never compromise security for convenience or performance
2. **Ethical Alignment:** Maintain survivor-first principles and bias auditing
3. **Code Quality:** Follow PEP 8 and maintain type hints throughout
4. **Documentation:** Every change must be documented
5. **Cyclomatic Complexity**: Keep complexity <10 per function
6. **Single Responsibility**: One function/class = one job

### TODO / FIXME Discipline

Every new ``TODO`` or ``FIXME`` marker added to ``src/`` MUST include:

1. A **severity tag** — one of ``critical``, ``high``, ``medium``, or
   ``low``. The tag is the engineering signal a future reviewer uses
   to triage the marker. ``critical`` means "must be addressed before
   this surface ships to production"; ``low`` means "cosmetic / nice
   to have".
2. A **citing reference** — an audit-doc tag (``audit-YYYY-MM``), an
   issue number (``gh-1234``), or a design-doc anchor. Markers
   without a reference are unmoored: a future engineer cannot tell
   whether the issue is real or whether anyone is tracking it.

The canonical form is::

    # TODO(<reference>, severity=<level>):
    #   <one-line description of what is missing or wrong>

Example::

    # TODO(audit-2026-03, severity=critical):
    #   PreExecutionBlockingGate has an off switch — single ``False``
    #   here disables all blocking. Tracked until enable_blocking
    #   is removed.

Bare ``TODO`` / ``FIXME`` comments without a severity tag and a
reference will fail review. ``grep -rE 'TODO\(|FIXME\(' src/`` is
the canonical inventory query — markers that do not match the
``TAG(reference, severity=...)`` shape do not exist as far as the
codebase is concerned.

### Python Style

We follow PEP 8 with some modifications:

```python
# Good
def detect_anomaly(data: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Detect anomalies in data.

    Args:
        data: Input data array
        threshold: Detection threshold

    Returns:
        Dictionary with anomaly scores and metadata
    """
    scores = compute_scores(data)
    return {
        "is_anomaly": scores > threshold,
        "scores": scores,
    }

# Bad - no type hints, no docstring
def detect(data, thresh=0.5):
    scores = compute_scores(data)
    return {"is_anomaly": scores > thresh, "scores": scores}
```

### Formatting and Linting

Before pushing, run the exact CI quality-gate matrix in one command — it mirrors
the blocking Code Quality + Type Checking lanes in `.github/workflows/ci.yml`
(black, ruff, flake8, canonical headers, pydocstyle, and all **three** mypy
lanes, including the lenient `tests/` lane and the graduated strict lane that
the individual commands below do not cover):

```bash
bash scripts/run_ci_gates.sh          # full matrix (what CI will run)
bash scripts/run_ci_gates.sh --fast   # skip the mypy lanes for a quick loop
```

Individual tools, for targeted iteration:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
ruff check src/ tests/
flake8 src/ tests/

# Type check
mypy src/omni_mercury_engine/

# Security scan
bandit -r src/omni_mercury_engine/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/omni_mercury_engine --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_harmonic.py -v

# Run core tests (fast)
pytest tests/test_cli.py tests/core/ -v

# Run detector tests
pytest tests/detectors/ -v
```

### Test Coverage Requirements

Two thresholds matter and they do different things:

- **Merge gates (blocking).** CI enforces a measured floor on every
  PR:
  - `COVERAGE_THRESHOLD_CORE = 33 %` on the curated core-tests lane
    (see `.github/workflows/ci.yml` for the file selection).
  - `COVERAGE_THRESHOLD_FULL = 62 %` on the full ml-tests lane
    (which runs the entire `tests/` tree with the AMA Cryptography
    native build).
  These floors are deliberately positioned below the most recent
  measured baseline (CORE 37.94 % / FULL 67.97 % combined
  stmt+branch, re-measured 2026-07-18 on the PR #339 head with
  CI-identical invocations, both lanes green; previous baselines
  31.9 % / 59.8 %) so CI noise and dataset-availability flakes do
  not produce false PR failures while still surfacing real coverage
  regressions.  **Do not lower these floors back toward the
  historical 25/50 (or older 10/15) values to unblock unrelated
  work** — they document a non-regression guarantee.

- **Aspirational target (non-blocking).** `pyproject.toml
  [tool.coverage.report]` sets `fail_under = 85`, the long-term
  quality bar for the full suite.  PRs that move the measured
  number toward this target are welcome; PRs that move it away from
  this target without a stated reason will be questioned in review.

New code in a PR should at minimum not regress the relevant lane
floor.  PRs that add a new module under `src/omni_mercury_engine/`
should also include unit tests for the new surface — review will
flag missing tests, but the merge gate is the lane floor, not a
per-file percentage.

### Writing Tests

**Unit Test Example**:
```python
import pytest
import numpy as np
from omni_mercury_engine.ml.harmonic_encoder import SphericalHarmonicDecomposer


def test_spherical_harmonic_decomposition():
    """Test spherical harmonic decomposition."""
    decomposer = SphericalHarmonicDecomposer(l_max=5)

    # Create test data
    points = np.random.randn(100, 3)
    values = np.random.randn(100)

    # Decompose
    coeffs = decomposer.decompose_surface(points, values)

    # Assertions
    assert coeffs.shape == (36,)  # (l_max + 1)^2
    assert coeffs.dtype == np.complex128

    # Test power spectrum
    power = decomposer.compute_rotation_invariant_features(coeffs)
    assert power.shape == (6,)  # l_max + 1
    assert np.all(power >= 0)  # Power is non-negative
```

## Keeping docs and code in sync

Documentation drift (numbers in the README that no longer match the code) is
treated as a bug, not a cosmetic issue. Every quantitative or capability claim
must be **measured and gated**, never hand-typed. Checklist before you claim a
number:

- [ ] **Structural counts** (source files, LOC, packages, detector/loader
      classes, `nn.Module` subclasses, test modules, workflows) go *only* in the
      README "Codebase Scale" block between the `<!-- SCALE:START/END -->`
      markers. Regenerate with `python scripts/measure_codebase_scale.py --update
      README.md`; CI fails on drift (`--check README.md`). Do not type these
      numbers anywhere else — link to the block.
- [ ] **Benchmark headline AUC/F1** comes from the committed
      `benchmarks/mercury_benchmark_results.json` via the
      `<!-- BENCHMARK:START/END -->` block (rendered by
      `scripts/update_readme_benchmarks.py`). Cite that block, not a copied
      literal. Keep the externally-comparable subset (ADBench) clearly separated
      from the leakage-prone self-labeled loaders.
- [ ] **Performance claims** (e.g. crypto speedups) must point at a reproducible
      benchmark that writes a provenance-stamped artifact
      (`benchmarks/crypto_backend_benchmark.py`). No unbenchmarked "Nx faster".
- [ ] **Dependency posture** ("required" vs "optional extra") must match
      `pyproject.toml`. torch is the `[ml]` extra; SHAP the `[explainability]`
      extra (`lime` is deliberately excluded from the extras graph — its sole
      release cannot build on modern setuptools; the LIME adapter degrades to
      the in-repo linear surrogate); AMA/PQC is the only hard, fail-closed
      import gate.
- [ ] **Behavioural claims** (env-var toggles, fallbacks, gates) must match a
      test that pins the behaviour (e.g. `tests/test_pqc_startup_gate.py`).
- [ ] **Experimental wins** must clear their pre-registered bar *and* survive
      paired inference (`benchmarks/statistical_significance.py`) before a
      default is changed. "Directionally better" is not "better".

If you cannot measure it, do not assert it. Mark it as illustrative/target and
link to the script that would verify it.

## Pull Request Process

### Before Submitting

1. **Update from main:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks:**
   ```bash
   black src/ tests/
   isort src/ tests/
   ruff check src/ tests/
   pytest tests/ -v
   mypy src/omni_mercury_engine/
   # Docs/claims consistency gates (dependency-free):
   python scripts/measure_codebase_scale.py --check README.md
   python scripts/check_readme_lyapunov.py
   ```

3. **Update documentation:**
   - Update README.md if adding features (regenerate the Codebase Scale block;
     see "Keeping docs and code in sync" above)
   - Update SECURITY.md if affecting security
   - Add entries to CHANGELOG.md

### Commit Message Format

Follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `security`: Security vulnerability fix
- `docs`: Documentation only
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance tasks

### Squash-merge gotcha — never embed skip-directive literals in PR descriptions

When you click **Squash & merge** on a PR, GitHub copies the PR title and
the entire PR body verbatim into the squash commit message that lands on
`main`. GitHub Actions then parses every push commit message and skips
**all** workflows on that push if it contains any of the magic substrings
(shown here with backslash-escaped brackets so this file itself does not
match the scanner):

* ``\[skip ci\]``
* ``\[ci skip\]``
* ``\[no ci\]``
* ``\[skip actions\]``
* ``\[actions skip\]``

This means a PR description that *legitimately documents* an auto-commit
marker (for example, "the auto-commit step emits the skip-ci directive
so it does not re-trigger itself") will silently suppress every workflow
on the merge commit — Benchmark Pipeline, CI/CD, Security, Docker,
Auto-Format, PQC checks, all of them. Branch protection still allows
the merge, but no validation runs against `main`.

**Avoid this by:**

1. **Never write the literal substring in the PR body.** The scanner does
   a plain substring match on the raw commit message; markdown styling
   (backticks, bold, code fences) does not affect the match. Use a form
   that does not contain the literal characters in sequence:
   - Hyphenate it: ``[skip-ci]`` instead of ``[skip ci]``.
   - Escape one or both brackets: ``\[skip ci\]`` (the backslash makes
     the raw text ``\[skip ci\]``, which has no ``[skip ci]`` substring).
   - Replace the space with a non-breaking space (U+00A0): the rendered
     text reads identically but the byte sequence does not match.
2. **Audit every commit on the branch, not just the PR body.** GitHub
   Actions parses *every* commit message in a push. If you use a
   merge-commit (instead of squash) strategy, individual commits with
   skip-directives in their bodies still trip the filter on push. PR
   #182's history (`d3a2a0c` → `ffc2cc6`) is the canonical example: an
   empty trigger commit was added on top so HEAD's message did not match
   the scanner.
3. **If you only realise after merging, recover by manually dispatching**
   each suppressed workflow from the **Actions** tab → workflow → **Run
   workflow**. Every push-triggered workflow in `.github/workflows/`
   exposes ``workflow_dispatch:`` for exactly this recovery path.

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Code follows style guidelines (Black, Ruff, Flake8)
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Coverage does not regress the lane floors (CORE >= 33 %,
      FULL >= 62 %) and trends toward the 85 % aspirational target
- [ ] Security scan clean

## Testing
Describe testing performed.
```

## Adding New Engines

### Template

```python
"""
New Engine Module
"""
import numpy as np
import torch
from typing import Dict, Any, Union, Optional
from omni_mercury_engine.core.base import BaseModel


class NewEngineModel(BaseModel):
    """
    Description of new engine.

    Features:
    - Feature 1
    - Feature 2
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.param1 = self.config.get("param1", default_value)

    def predict(self, data: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """Predict anomalies."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        scores = self._compute_scores(data)

        return {
            "anomaly_scores": scores,
            "model_type": "new_engine",
        }

    def extract_features(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract features for ML fusion."""
        features = ...
        return torch.tensor(features, dtype=torch.float32)

    def _compute_scores(self, data: np.ndarray) -> np.ndarray:
        """Internal computation method."""
        pass
```

### Checklist for New Engines

- [ ] Inherits from `BaseModel` or `BaseDetector`
- [ ] Implements `predict()` method
- [ ] Implements `extract_features()` method for fusion
- [ ] Type hints on all methods
- [ ] Docstrings on all public methods
- [ ] Cyclomatic complexity <10 per function
- [ ] Unit tests covering the new surface (the merge gate is the
      core / full lane floor; per-file coverage is reviewed
      qualitatively, not enforced)
- [ ] Integration test with fusion network
- [ ] Example script in `examples/`

## Security Considerations

### Reporting Security Issues

**DO NOT** open public issues for security vulnerabilities. See [SECURITY.md](SECURITY.md) for reporting process.

### Critical Rules

**ALWAYS:**
- Validate all inputs to ML functions
- Include proper error handling
- Test against known datasets
- Document algorithmic choices
- Consider ethical implications

**NEVER:**
- Store secrets in logs, error messages, or debug output
- Ignore error conditions
- Make claims without validation
- Copy-paste code without understanding it

## Community

### Communication Channels

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** General questions, ideas
- **Email:** steel.sa.llc@gmail.com

### Recognition

Contributors will be recognized in:
- CHANGELOG.md for their contributions
- Release notes
- GitHub contributors page

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-08 | Initial release |
| 2.0.0 | 2026-01-06 | Consolidated from docs/, updated to use pyproject.toml |
| 2.1.0 | 2026-01-09 | Updated to v1.1.0 |
| 2.2.0 | 2026-02-09 | Updated to v1.4.0, aligned Python prerequisite |
| 2.3.0 | 2026-02-21 | Updated to v1.5.1, fixed test directory references, aligned with CI |
| 2.4.0 | 2026-05-05 | Updated to v1.6.x, added Wave B dual-gate hard ethics, AMA Cryptography sole PQC backend, pickle/training-data removal, TODO discipline |
| 2.5.0 | 2026-05-19 | Updated to v1.6.x / v1.7 development cycle. Added v1.7 do-not-restore items (SafeHTTPClient `allow_untrusted`, MockLLMAdapter silent fallback, `security/` vs `compliance/` boundary). Linked medical / drone / compliance integration-ready contracts. |
| 2.6.0 | 2026-05-22 | Replaced aspirational 85 / 90 / 95 % coverage claims with the actual measured-floor merge gates (CORE 25 %, FULL 50 %) plus the 85 % aspirational target.  Aligned the PR-template and new-engine checklists with the same posture. |
| 2.7.0 | 2026-06-17 | v1.7.0 released (2026-05-20); reconciled documentation with the shipped line and source tree. The "v1.7 development cycle" wording in row 2.5.0 predates the release. |
| 2.8.0 | 2026-07-11 | Date/version refresh; verified all cited paths, gates, coverage floors, and scripts against the current source tree. No content changes. |
| 2.9.0 | 2026-07-18 | Coverage floors graduated on re-measurement (CI-identical invocations, both lanes green): CORE 25 → 30 % (measured 37.94 %), FULL 50 → 55 % (measured 67.97 %). Same cushion policy that set 25/50, with extra FULL margin for measurement-environment variance. |

---

Copyright 2025-2026 Steel Security Advisors LLC. Licensed under the GNU General Public License v3.0 or later (SPDX: GPL-3.0-or-later).
