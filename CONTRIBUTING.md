# Contributing to Mercury Agent

## Document Information

| Property | Value |
|----------|-------|
| Document Version | 2.3 |
| Last Updated | 2026-02-21 |
| Classification | Public |
| Maintainer | Steel Security Advisors LLC |

---

## Overview

This document provides guidelines for contributing to the Mercury Agent neuro-symbolic AI framework. Mercury Agent is released under the GNU General Public License v3.0 as free and open-source software, accessible for universal use as a knowledge vault and bridge to AI/ML frontiers.

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
- **Medical**: MIMIC-III vital signs, clinical trial data
- **Cybersecurity**: Actual PCAP files, malware samples (anonymized)
- **SETI**: Breakthrough Listen observation data
- **Submit via**: Issue template "Real-Data Integration Request"

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
- Remove or bypass ethical safeguards
- Introduce unproven or experimental algorithms without validation
- Add unnecessary dependencies
- Include proprietary or non-GPL compatible code
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

- **Minimum coverage:** 85% for new code
- **Target coverage:** 90% overall
- **Core modules:** 95% coverage (fusion, detectors, models)

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
   ```

3. **Update documentation:**
   - Update README.md if adding features
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

### Squash-merge gotcha — never embed `[skip ci]` literals in PR descriptions

When you click **Squash & merge** on a PR, GitHub copies the PR title and
the entire PR body verbatim into the squash commit message that lands on
`main`. GitHub Actions then parses every push commit message and skips
**all** workflows on that push if it contains any of the magic substrings:

* ``[skip ci]``
* ``[ci skip]``
* ``[no ci]``
* ``[skip actions]``
* ``[actions skip]``

This means a PR description that *legitimately documents* an auto-commit
marker (for example, "the auto-commit step emits `[skip ci]` so it does
not re-trigger itself") will silently suppress every workflow on the
merge commit — Benchmark Pipeline, CI/CD, Security, Docker, Auto-Format,
PQC checks, all of them. Branch protection still allows the merge, but
no validation runs against `main`.

**Avoid this by:**

1. Replacing the literal substring in PR descriptions with a non-matching
   form, e.g. ``[skip&nbsp;ci]``, ``[skip-ci]`` (hyphen instead of space),
   or surrounding it with backticks **and** writing it as
   ``\[skip ci\]`` so the rendered text shows the directive but the
   commit-message scanner does not match.
2. If the PR genuinely needs to use the marker (rare), use a **merge
   commit** strategy instead of squash so the PR body stays out of the
   commit message.
3. If you only realise after merging, recover by manually dispatching
   each workflow from the **Actions** tab → workflow → **Run workflow**.
   Every push-triggered workflow in `.github/workflows/` also exposes
   ``workflow_dispatch:`` for exactly this recovery path.

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
- [ ] Coverage >= 85%
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
- [ ] Unit tests with >= 85% coverage
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

---

Copyright 2025-2026 Steel Security Advisors LLC. Licensed under GNU General Public License v3.0.
