# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Import-time PQC gate.

AMA is mandatory for Mercury.  ``omni_mercury_engine`` package import refuses
to proceed unless the AMA Cryptography native C backend is fully loadable;
there is no development-mode or CI-mode escape hatch.  The
``AMA_REQUIRE_REAL_PQC`` / legacy ``AVA_REQUIRE_REAL_PQC`` env vars are retained
only for diagnostics and compatibility with older workflow comments — they no
longer disable the gate.

Algorithm coverage matches
``security.pqc_guards.check_pqc_production_readiness``: the gate
fails closed unless **all three** AMA algorithms are loadable —
ML-DSA-65 (``DILITHIUM_AVAILABLE``), Kyber-1024
(``KYBER_AVAILABLE``), and SPHINCS+ (``SPHINCS_AVAILABLE``).  Any
partial build is rejected because Mercury exposes all three
surfaces and a Dilithium-only install would let the process start
in a cryptographically incomplete state.

On top of algorithm availability the gate also enforces the pinned
**version** (:data:`_AMA_REQUIRED_VERSION`, ``3.2.0``): the installed
``ama_cryptography.__version__`` and, when set, the operator's
``AMA_CRYPTO_VERSION`` env var must match the pinned *release*
PEP 440-tolerantly (``v3.2.0`` / ``3.2.0.post1`` / ``3.2`` are accepted;
a different release such as ``3.1.0`` is refused), so a build of the
wrong AMA release (that happens to expose the three flags) is refused
rather than started.  See :func:`_enforce_ama_version` /
:func:`_release_matches`.

The flags are read from ``ama_cryptography.pqc_backends`` — the
canonical location matching what ``security/pqc_backends.py`` reads
(``from ama_cryptography.pqc_backends import DILITHIUM_AVAILABLE,
KYBER_AVAILABLE, SPHINCS_AVAILABLE``).  The top-level
``ama_cryptography`` package and per-algorithm submodules
(``ama_cryptography.dilithium``, etc.) are NOT reliable sources
for these flags on a real install — earlier iterations of this
gate read from those locations and produced false-positive
partial-install rejections.

The two raise paths (this gate and
``check_pqc_production_readiness``) are independent
implementations of the same contract — neither delegates to the
other.  They share the ``_PQC_BUILD_RECOVERY_HINT`` constant so
operators see identical remediation steps regardless of which
path raises.

Lives in its own module rather than inline in ``__init__.py`` so the gate
function has a stable, importable location for unit tests; ``__init__.py``
imports and calls it once at package-load time, then deletes the local
re-binding to keep the public package surface clean.
"""

from __future__ import annotations

import os
import re

#: The single AMA-Cryptography version Mercury is pinned to. Kept in lockstep
#: with the pyproject ``ama-cryptography`` git pin, ``.github/actions/build-ama-
#: cryptography`` (``AMA_REF``), and ``scripts/build_ama_native.sh``.
_AMA_REQUIRED_VERSION = "3.2.0"

#: The numeric release the pin resolves to, matched PEP 440-tolerantly: a build
#: reporting ``3.2.0``, ``v3.2.0``, ``3.2.0.post1``, ``3.2.0rc1`` or ``3.2.0+cpu``
#: all satisfy it (same release); ``3.1.0`` / ``3.3.0`` / ``9.9.9`` do not.
_AMA_REQUIRED_RELEASE = (3, 2, 0)

#: Operator-facing env var to *declare* the AMA version. When set it must match
#: the pinned *release* (:data:`_AMA_REQUIRED_RELEASE`) PEP 440-tolerantly via
#: :func:`_release_matches` -- ``v3.2.0`` / ``3.2.0.post1`` / ``3.2`` are accepted,
#: a different release is refused. A mismatch is a loud, fail-closed configuration
#: error rather than a silent downgrade.
AMA_CRYPTO_VERSION_ENV = "AMA_CRYPTO_VERSION"

_PQC_BUILD_RECOVERY_HINT = (
    "Build the AMA-Cryptography native library from a clone of the upstream\n"
    "repo (Mercury-Agent has no CMakeLists.txt of its own):\n"
    "  git clone --depth 1 --branch v3.2.0 \\\n"
    "      https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git /tmp/ama-cryptography\n"
    "  cd /tmp/ama-cryptography\n"
    "  cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build\n"
    "  AMA_NO_CYTHON=1 pip install --no-build-isolation .\n"
    "  export LD_LIBRARY_PATH=/tmp/ama-cryptography/build/lib:"
    "/tmp/ama-cryptography/build:${LD_LIBRARY_PATH:-}\n"
    "See docs/INSTALLATION.md 'Post-Quantum Cryptography backend' for the\n"
    "verified procedure (mirrors .github/workflows/pqc-production-check.yml)."
)

_PQC_HARD_REQUIRED: tuple[tuple[str, str], ...] = (
    ("DILITHIUM_AVAILABLE", "ML-DSA-65 (Dilithium)"),
    ("KYBER_AVAILABLE", "Kyber-1024"),
    ("SPHINCS_AVAILABLE", "SPHINCS+"),
)


def _enforce_pqc_production_gate() -> None:
    """Fail-closed PQC startup gate.  See module docstring for the contract.

    Reads the ``*_AVAILABLE`` flags from ``ama_cryptography.pqc_backends``
    (the canonical location), matching how
    ``security/pqc_backends.py`` already consumes them
    (``from ama_cryptography.pqc_backends import DILITHIUM_AVAILABLE,
    KYBER_AVAILABLE, SPHINCS_AVAILABLE``).  Earlier iterations of this
    gate read the flags from per-algorithm submodules
    (``ama_cryptography.dilithium.DILITHIUM_AVAILABLE``) and from the
    top-level package (``ama_cryptography.DILITHIUM_AVAILABLE``).
    Neither location is reliably populated by the real AMA v3.2.0
    install — the canonical location is the ``pqc_backends`` submodule,
    which is where Mercury's own ``security/pqc_backends.py`` reads
    them.  Aligning the gate with that reader keeps both views of AMA
    availability consistent and stops false-positive partial-install
    rejections in the verify-real-pqc CI lane.
    """
    try:
        import ama_cryptography.pqc_backends as ama_pqc_backends
    except ImportError as exc:
        raise RuntimeError(
            "AMA/PQC is mandatory for Mercury, but the AMA Cryptography Python "
            "package is not importable "
            "(import ama_cryptography.pqc_backends failed).\n"
            f"{_PQC_BUILD_RECOVERY_HINT}"
        ) from exc

    missing: list[str] = [
        friendly
        for flag_name, friendly in _PQC_HARD_REQUIRED
        if not getattr(ama_pqc_backends, flag_name, False)
    ]

    if missing:
        raise RuntimeError(
            "AMA/PQC is mandatory for Mercury, but the AMA Cryptography native C "
            f"backend is incomplete; missing or unavailable: {', '.join(missing)}.\n"
            f"{_PQC_BUILD_RECOVERY_HINT}"
        )

    _enforce_ama_version()


def _release_tuple(value: str) -> tuple[int, ...]:
    """Leading numeric release of a version string, PEP 440-tolerantly.

    ``' v3.2.0.post1 '`` / ``'3.2.0rc1'`` / ``'3.2.0+cpu'`` all yield ``(3, 2, 0)``;
    a non-numeric or empty value yields ``()``. Only the dotted numeric prefix is
    read, so pre/post/dev/local suffixes never change the matched release.
    """
    match = re.match(r"\s*[vV]?(\d+(?:\.\d+)*)", value or "")
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _release_matches(value: str) -> bool:
    """True when ``value``'s numeric release equals :data:`_AMA_REQUIRED_RELEASE`.

    Shorter releases are zero-padded (``'3.2'`` -> ``(3, 2, 0)``) and longer ones
    are truncated to the pinned width, so a patch/build suffix is accepted while a
    different minor/major (``3.1.0`` / ``3.3.0``) is refused.
    """
    release = _release_tuple(value)
    if not release:
        return False
    width = len(_AMA_REQUIRED_RELEASE)
    padded = release + (0,) * (width - len(release)) if len(release) < width else release
    return padded[:width] == _AMA_REQUIRED_RELEASE


def _enforce_ama_version() -> None:
    """Fail-closed unless AMA Cryptography is the pinned :data:`_AMA_REQUIRED_VERSION`.

    Two independent, fail-closed checks, either of which refuses:

    * ``AMA_CRYPTO_VERSION`` (:data:`AMA_CRYPTO_VERSION_ENV`) -- if the operator
      sets it, it must *declare* the pinned version. A mismatched declaration is a
      loud misconfiguration (a typo or an attempt to run an unpinned build),
      never a silent downgrade.
    * ``ama_cryptography.__version__`` -- the *installed* version, when the
      package exposes it, must equal the pinned version. A build of the wrong AMA
      release that still exposed the three backend flags would otherwise pass the
      algorithm-availability check; this closes that gap.

    Matching is PEP 440-tolerant (see :func:`_release_matches`): a post/local/dev
    build of the pinned release (``3.2.0.post1``, ``3.2.0+cpu``) is accepted, a
    different release (``3.1.0``, ``3.3.0``) is refused. Absent version metadata
    is not, on its own, fatal: the v3.2.0-only symbol imports in
    ``security/pqc_backends.py`` already floor the surface. This adds an explicit,
    operator-visible version gate on top of that structural floor.
    """
    declared = os.environ.get(AMA_CRYPTO_VERSION_ENV, "").strip()
    if declared and not _release_matches(declared):
        raise RuntimeError(
            f"AMA/PQC version mismatch: {AMA_CRYPTO_VERSION_ENV}={declared!r} but "
            f"Mercury is pinned to AMA Cryptography v{_AMA_REQUIRED_VERSION}. Unset the "
            f"variable or set {AMA_CRYPTO_VERSION_ENV}={_AMA_REQUIRED_VERSION}.\n"
            f"{_PQC_BUILD_RECOVERY_HINT}"
        )

    try:
        import ama_cryptography

        installed_raw = str(getattr(ama_cryptography, "__version__", "") or "")
    except Exception:  # pragma: no cover - top-level import already succeeded above
        installed_raw = ""

    if installed_raw and not _release_matches(installed_raw):
        raise RuntimeError(
            "AMA/PQC version mismatch: the installed ama_cryptography is "
            f"v{installed_raw}, but Mercury requires release v{_AMA_REQUIRED_VERSION} "
            "(the pyproject pin, CI AMA_REF, and the production PQC gate all agree).\n"
            f"{_PQC_BUILD_RECOVERY_HINT}"
        )
