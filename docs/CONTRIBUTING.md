# Contributing to Mercury Agent ♱

Thank you for your interest in contributing to the Mercury Agent ♱! This document provides guidelines for development, testing, and submitting contributions.

## AI-Assisted Development

We welcome AI-assisted contributions (e.g., via Devin, GitHub Copilot):
- All AI-assisted PRs must include tests and documentation
- Human review required for all contributions
- Clearly indicate AI-assisted work in PR description

## Real-Data Contributions

**High Priority**: We actively seek contributions for real-world data integration:
- **Medical**: MIMIC-III vital signs, clinical trial data
- **Cybersecurity**: Actual PCAP files, malware samples (anonymized)
- **SETI**: Breakthrough Listen observation data
- **Submit via**: Issue template "Real-Data Integration Request"

## Issue Templates

When opening issues, use provided templates:
- **Bug Report**: For reproducible errors
- **Feature Request**: For new functionality
- **Real-Data Integration**: For dataset contributions
- **Performance Issue**: For optimization needs

## Development Setup

### Prerequisites

- Python 3.12+
- Git
- (Optional) CUDA-capable GPU for faster training

### Installation

1. **Clone the repository** (if working from a repo):
   ```bash
   git clone https://github.com/Steel-SecAdv-LLC/Mercury Agent ♱.git
   cd Mercury Agent ♱
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Core dependencies
   pip install -r requirements.txt

   # Development dependencies
   pip install -r requirements-dev.txt

   # Optional dependencies (biometric, quantum, etc.)
   pip install -r requirements-optional.txt
   ```

4. **Install in editable mode**:
   ```bash
   pip install -e .
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

1. **Cyclomatic Complexity**: Keep complexity <10 per function
2. **Single Responsibility**: One function/class = one job
3. **Type Hints**: Always include type hints
4. **Docstrings**: Required for all public functions/classes
5. **Comments**: Minimal - code should be self-documenting
   - Exception: Important architectural decisions
   - Use `# (important-comment)` to preserve comments from stripping

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

# Bad - no type hints, no docstring, unnecessary comments
def detect(data, thresh=0.5):
    # Calculate the scores
    scores = compute_scores(data)
    # Return results
    return {"is_anomaly": scores > thresh, "scores": scores}
```

### Formatting

**Use Black** for automatic formatting:
```bash
black omni_mercury_engine/ tests/ examples/
```

**Configuration** (in `pyproject.toml`):
```toml
[tool.black]
line-length = 88
target-version = ['py312']
```

### Linting

**Use Flake8** for linting:
```bash
flake8 omni_mercury_engine/ tests/
```

**Configuration** (in `.flake8`):
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,venv
```

### Type Checking

**Use MyPy** for type checking:
```bash
mypy omni_mercury_engine/
```

**Configuration** (in `pyproject.toml`):
```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_harmonic_encoder.py

# Run with coverage
pytest tests/ --cov=omni_mercury_engine --cov-report=html --cov-report=term

# Run with verbose output
pytest tests/ -v

# Run only unit tests (fast)
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

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

**Integration Test Example**:
```python
import pytest
from omni_mercury_engine.engine import OmniMercuryEngine


def test_full_pipeline():
    """Test end-to-end anomaly detection."""
    engine = OmniMercuryEngine()

    # Normal data
    normal_data = np.random.randn(100, 10)
    result = engine.detect(normal_data)

    assert "anomaly_score" in result
    assert "component_scores" in result
    assert result["anomaly_score"] < 0.5  # Should be low for normal data

    # Anomalous data
    anomalous_data = np.random.randn(100, 10) * 10  # Much larger scale
    result = engine.detect(anomalous_data)

    assert result["anomaly_score"] > 0.5  # Should be high for anomalous data
```

### Test Coverage Requirements

- **Minimum**: 85% overall coverage
- **Target**: 90% overall coverage
- **Core modules**: 95% coverage (fusion, detectors, models)

### Mocking External Dependencies

**Example** (mocking DeepFace):
```python
from unittest.mock import patch, MagicMock

@patch('omni_mercury_engine.models.biometric.DeepFace')
def test_biometric_model_without_deepface(mock_deepface):
    """Test biometric model with mocked DeepFace."""
    # Mock DeepFace.represent
    mock_deepface.represent.return_value = [{"embedding": [0.1] * 128}]

    from omni_mercury_engine.models.biometric import BiometricAnomalyModel

    model = BiometricAnomalyModel()
    features = model.extract_features(np.zeros((224, 224, 3), dtype=np.uint8))

    assert features.shape == (1, 128)
```

## Security Scanning

**Use Bandit** for security scans:
```bash
bandit -r omni_mercury_engine/
```

**Use Safety** for dependency vulnerabilities:
```bash
safety check
```

## Benchmarking

**Create benchmark scripts** in `benchmarks/`:

```python
# benchmarks/bench_fusion.py
import time
import numpy as np
from omni_mercury_engine.engine import OmniMercuryEngine


def bench_inference_latency():
    """Benchmark inference latency."""
    engine = OmniMercuryEngine()
    data = np.random.randn(1000, 10)

    start = time.time()
    for i in range(100):
        _ = engine.detect(data[i:i+10])
    elapsed = time.time() - start

    print(f"Average latency: {elapsed / 100 * 1000:.2f}ms")


if __name__ == "__main__":
    bench_inference_latency()
```

**Run benchmarks**:
```bash
python benchmarks/bench_fusion.py
```

## Pull Request Process

### Before Submitting

1. **Run all checks**:
   ```bash
   # Format code
   black omni_mercury_engine/ tests/ examples/

   # Lint
   flake8 omni_mercury_engine/ tests/

   # Type check
   mypy omni_mercury_engine/

   # Test with coverage
   pytest tests/ --cov=omni_mercury_engine --cov-report=term

   # Security scan
   bandit -r omni_mercury_engine/
   ```

2. **Ensure tests pass**:
   - All existing tests pass
   - New tests added for new features
   - Coverage ≥85%

3. **Update documentation**:
   - Update docstrings
   - Update README.md if needed
   - Add examples if applicable

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
- [ ] Code follows style guidelines (Black, Flake8)
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Coverage ≥85%
- [ ] Security scan clean

## Testing
Describe testing performed.

## Screenshots (if applicable)
Add screenshots for UI changes.
```

### Review Process

1. **Automated checks**: CI runs automatically
2. **Code review**: Maintainer reviews code
3. **Feedback**: Address review comments
4. **Approval**: Maintainer approves
5. **Merge**: Squash and merge

## Commit Message Guidelines

Follow Conventional Commits:

```bash
# Format
<type>(<scope>): <subject>

# Types
feat: New feature
fix: Bug fix
docs: Documentation
style: Formatting
refactor: Code restructuring
test: Tests
chore: Maintenance

# Examples
feat(harmonic): add spherical harmonic decomposition
fix(biometric): resolve DeepFace import error
docs(architecture): update fusion network diagram
test(directive): add QPCP unit tests
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

        # Your logic here
        scores = self._compute_scores(data)

        return {
            "anomaly_scores": scores,
            "model_type": "new_engine",
        }

    def extract_features(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract features for ML fusion."""
        # Extract features
        features = ...

        return torch.tensor(features, dtype=torch.float32)

    def _compute_scores(self, data: np.ndarray) -> np.ndarray:
        """Internal computation method."""
        # Keep complexity <10
        pass
```

### Checklist for New Engines

- [ ] Inherits from `BaseModel` or `BaseDetector`
- [ ] Implements `predict()` method
- [ ] Implements `extract_features()` method for fusion
- [ ] Type hints on all methods
- [ ] Docstrings on all public methods
- [ ] Cyclomatic complexity <10 per function
- [ ] Unit tests with ≥85% coverage
- [ ] Integration test with fusion network
- [ ] Example script in `examples/`
- [ ] Documentation in `docs/ANALYSIS.md`

## Code Review Checklist

### For Reviewers

- [ ] Code follows style guidelines
- [ ] Logic is clear and well-documented
- [ ] No security vulnerabilities (bandit clean)
- [ ] No hardcoded secrets
- [ ] Error handling is appropriate
- [ ] Tests are comprehensive
- [ ] Performance is acceptable
- [ ] No breaking changes (or documented)

### For Authors

- [ ] Self-reviewed code
- [ ] Ran all automated checks
- [ ] Added tests
- [ ] Updated documentation
- [ ] Addressed all review comments

## Getting Help

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Email**: contact@mercury-agent.org (if applicable)

## License

By contributing, you agree that your contributions will be licensed under the GPL v3 License.

## Code of Conduct

- Be respectful and constructive
- Focus on what is best for the community
- Show empathy towards other contributors
- Accept constructive criticism gracefully

## Thank You!

Your contributions make this project better for everyone. Thank you for your time and effort!
