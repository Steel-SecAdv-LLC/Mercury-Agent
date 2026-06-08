# Copyright (C) 2025 Steel Security Advisors LLC
"""Import-time PQC gate."""

from __future__ import annotations

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
