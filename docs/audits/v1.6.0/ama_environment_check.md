# AMA Cryptography Environment Check (§3.6)

**Date:** 2026-05-08
**Branch:** `devin/1778041418-v1.6.0-corrective-sweep` (PR #189)
**Base:** PR #189 HEAD = `a8a231a7f14d6084967d2f66c84d692084a8caba`

## Repo reachability

```
$ git ls-remote https://github.com/Steel-SecAdv-LLC/AMA-Cryptography HEAD
24c7fc7c218a5dd16fdf41a6ad4b36d70de4c3e9	HEAD

$ git ls-remote --tags https://github.com/Steel-SecAdv-LLC/AMA-Cryptography v3.1.0
ed5397e62b648be63b17bc1486c04c3c95ce6b7c	refs/tags/v3.1.0
```

The AMA Cryptography repository is **publicly reachable over anonymous HTTPS**.
The pinned tag `v3.1.0` (used by `.github/workflows/pqc-production-check.yml`) is
present and resolves to an immutable commit.

## Toolchain

| Tool       | Path                  | Version |
|------------|-----------------------|---------|
| cmake      | /root/.local/bin/cmake | 4.3.2   |
| ninja      | /usr/bin/ninja         | 1.11.1  |
| pkg-config | /usr/bin/pkg-config    | 1.8.1   |
| gcc        | /usr/bin/gcc           | 13.3.0  |
| python3    | /usr/local/bin/python3 | 3.11.15 |

The system cmake (3.28.3) was below the floor; we upgraded to the wheel-shipped
4.3.2 in `/root/.local/bin/cmake` to match the AMA `pyproject.toml`
`[build-system].requires` constraint and the floor pinned at
`.github/workflows/pqc-production-check.yml:61-63`.

## Build-system floors (pqc-production-check.yml lines 61-63)

| Constraint           | Installed | Status |
|----------------------|-----------|--------|
| `setuptools>=78.1.1` | 82.0.1    | OK     |
| `wheel>=0.47.0`      | 0.47.0    | OK     |
| `cmake>=4.3.2`       | 4.3.2     | OK     |
| `pip>=26.1`          | 26.1.1    | OK     |

These constraints exist because of:

- **PYSEC-2025-49** — setuptools path traversal in `PackageIndex` ≤ 78.1.0
- **GHSA-cx63-2mw6-8hw5** — wheel zip-slip ≤ 0.46
- **GHSA-8rrh-rw8j-w5fx** — cmake build-system path injection ≤ 4.3.1

## Build recipe (workflow-derived)

```bash
export AMA_REF=v3.1.0
git clone --depth 1 --branch "$AMA_REF" https://github.com/Steel-SecAdv-LLC/AMA-Cryptography /tmp/ama-cryptography
cd /tmp/ama-cryptography
python -m pip install --upgrade "setuptools>=78.1.1" "wheel>=0.47.0" "cmake>=4.3.2" "pip>=26.1"
AMA_NO_CYTHON=1 pip install --no-build-isolation .
export LD_LIBRARY_PATH="/tmp/ama-cryptography/build/lib:/tmp/ama-cryptography/build:${LD_LIBRARY_PATH:-}"
```

## Verification commands

```bash
# Positive: real AMA must succeed
AMA_REQUIRE_REAL_PQC=1 python -c "import omni_mercury_engine"

# Negative: without AMA, _pqc_gate must fail closed
unset AMA_REQUIRE_REAL_PQC
python -c "import omni_mercury_engine"

# Real-AMA gate test (no --ignore)
pytest tests/security/test_pqc_gate_real_ama.py -v
```

## Result

All §3.6 prerequisites satisfied. Build/import attempts are recorded in
`final_checks.txt`.
