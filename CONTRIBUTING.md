# Contributing to Omni ♱ Ava (O♱A)

## Document Information

| Property | Value |
|----------|-------|
| Document Version | 1.0 |
| Last Updated | 2025-12-08 |
| Classification | Public |
| Maintainer | Steel Security Advisors LLC |

---

## Overview

This document provides guidelines for contributing to the Omni ♱ Ava (O♱A) multi-domain anomaly detection framework. O♱A is released under the GNU General Public License v3.0 as free and open-source software, accessible for universal use as a knowledge vault and bridge to AI/ML frontiers.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contribution Guidelines](#contribution-guidelines)
5. [Code Quality Requirements](#code-quality-requirements)
6. [Testing Requirements](#testing-requirements)
7. [Pull Request Process](#pull-request-process)
8. [Security Considerations](#security-considerations)
9. [Documentation Standards](#documentation-standards)
10. [Community](#community)

---

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git
- Basic understanding of machine learning and anomaly detection
- Familiarity with ethical AI principles

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
git clone https://github.com/YOUR_USERNAME/OMNI-AVA.git
cd OMNI-AVA
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e ".[dev]"
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

## Contribution Guidelines

### General Principles

1. **Security First:** Never compromise security for convenience or performance
2. **Ethical Alignment:** Maintain survivor-first principles and bias auditing
3. **Code Quality:** Follow PEP 8 and maintain type hints throughout
4. **Documentation:** Every change must be documented
5. **Backwards Compatibility:** Maintain compatibility unless security requires breaking changes

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

## Code Quality Requirements

### PEP 8 Compliance

All Python code must follow PEP 8 style guidelines:

```bash
# Check formatting
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/
ruff check src/ tests/
```

### Type Hints

All functions must include comprehensive type hints.

### Documentation Requirements

All functions must have docstrings including:

- **Brief description:** One-line summary
- **Args:** Type and description of each parameter
- **Returns:** Type and description of return value
- **Raises:** All possible exceptions

## Testing Requirements

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/omni_anomaly_engine --cov-report=html

# Run specific test file
pytest tests/test_specific.py
```

### Test Coverage Requirements

- **Minimum coverage:** 85% for new code
- **Error paths:** All error conditions must be tested
- **Edge cases:** Boundary conditions and corner cases

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
   flake8 src/ tests/
   pytest tests/ -v
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

## Security Considerations

### Reporting Security Issues

**DO NOT** open public issues for security vulnerabilities. See [SECURITY.md](SECURITY.md) for reporting process.

## Documentation Standards

### Academic Citations

When referencing research, include proper citations with DOI when available.

## Community

### Communication Channels

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** General questions, ideas
- **Email:** support@steelsecurityadvisors.com

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

---

Copyright 2025 Steel Security Advisors LLC. Licensed under GNU General Public License v3.0.
