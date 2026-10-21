# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quantum-risk assessment for cryptographic inventories.

For each cryptographic asset this module answers three questions, from
published standards and caller-stated assumptions only. It never invents a
probability.

1. **What is it?** Algorithm names, OIDs and CycloneDX CBOM entries are
   classified by *exact token* lookup in a registry of standardised,
   pre-standard, quantum-vulnerable and broken algorithms. The registry covers
   FIPS 203/204/205, draft FIPS 206, SP 800-208 and SP 800-57 strength bands.
   Lookup never matches a substring, so ``ML-DSA-65`` can never be read as
   ``DSA``, and a hash such as ``SHA256`` inside ``ecdsa-with-SHA256`` never
   makes a vulnerable signature look safe. An unrecognised name is reported
   as ``unknown``, never guessed.
2. **Is it on time?** Shor-vulnerable public-key algorithms face the NIST IR
   8547 (initial public draft, November 2024) transition: 112-bit strength is
   deprecated after 2030, and every strength is disallowed after 2035. NSA
   CNSA 2.0 sets per-category "exclusively use by" dates and approves only
   ML-KEM-1024, ML-DSA-87, LMS/XMSS, AES-256 and SHA-384/512.
3. **Is its data exposed?** Mosca's inequality applies: an asset is exposed
   if ``x + y > z``, where ``x`` is how long its data must stay secret (or its
   signatures trusted), ``y`` the migration time, and ``z`` the time until a
   cryptographically relevant quantum computer (CRQC). ``z`` is the caller's
   assumption, given as a value or as a cumulative probability curve. With no
   assumption the report states the CRQC arrival time below which the asset is
   exposed, which is ``x + y``.

Confidentiality and authenticity are treated separately. Recorded ciphertext
under a Shor-vulnerable key establishment can be decrypted after a CRQC
arrives ("harvest now, decrypt later"). A signature cannot be forged
retroactively; it is at risk only if it must remain trusted after a CRQC.

``audit_mercury_posture`` applies the same assessment to Mercury's own
declared cryptography. A test pins each declaration to the source file that
implements it, so the audit cannot silently drift from the code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class QuantumStatus(StrEnum):
    """Security of an algorithm against quantum and classical attack."""

    RESISTANT = "quantum_resistant"
    VULNERABLE = "shor_vulnerable"
    BROKEN = "classically_broken"
    UNKNOWN = "unknown"


#: Higher is stronger; a robust hybrid is as strong as its strongest part.
_STATUS_RANK = {
    QuantumStatus.BROKEN: 0,
    QuantumStatus.UNKNOWN: 1,
    QuantumStatus.VULNERABLE: 2,
    QuantumStatus.RESISTANT: 3,
}

#: Algorithm kinds that protect confidentiality or authenticity.
_CONFIDENTIALITY_KINDS = frozenset({"kem", "key_agreement", "public_key_encryption", "cipher"})
_AUTHENTICITY_KINDS = frozenset({"signature", "stateful_signature", "mac"})
_PUBLIC_KEY_KINDS = frozenset({"kem", "key_agreement", "public_key_encryption", "signature"})


@dataclass(frozen=True)
class AlgorithmProfile:
    """Registry facts about one algorithm (or hybrid) as classified.

    Attributes:
        canonical: Canonical name, or the input string when ``unknown``.
        kind: ``kem``, ``key_agreement``, ``public_key_encryption``,
            ``signature``, ``stateful_signature``, ``cipher``, ``hash``,
            ``mac``, ``kdf`` or ``unknown``.
        status: Quantum/classical security status.
        classical_bits: SP 800-57 classical strength band, when known.
        nist_category: NIST post-quantum security category (1–5), when
            defined for the algorithm.
        standard: The governing standard or the reason none applies.
        cnsa2: Whether NSA CNSA 2.0 approves it.
        prestandard: A pre-FIPS name or version (for example round-3 Kyber),
            not interoperable with the standard unless the implementation
            follows it.
        notes: Caveats that matter for a migration decision.
        components: For hybrids and composites, the classified parts.
    """

    canonical: str
    kind: str
    status: QuantumStatus
    classical_bits: int | None = None
    nist_category: int | None = None
    standard: str = ""
    cnsa2: bool = False
    prestandard: bool = False
    notes: tuple[str, ...] = ()
    components: tuple[AlgorithmProfile, ...] = ()


def _normalise(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


# ----------------------------------------------------------------- registry
_REGISTRY: dict[str, AlgorithmProfile] = {}
_ALIASES: dict[str, str] = {}


def _register(profile: AlgorithmProfile, *aliases: str) -> None:
    _REGISTRY[profile.canonical] = profile
    for alias in (profile.canonical, *aliases):
        key = _normalise(alias)
        if key in _ALIASES and _ALIASES[key] != profile.canonical:
            raise RuntimeError(f"alias {alias!r} is ambiguous")
        _ALIASES[key] = profile.canonical


_R, _V, _B = QuantumStatus.RESISTANT, QuantumStatus.VULNERABLE, QuantumStatus.BROKEN
_ROUND3 = (
    "pre-standard name: the round-3 submission is not interoperable with the FIPS "
    "standard; confirm the implementation follows the standard"
)

# --- FIPS 203 ML-KEM ---------------------------------------------------------
for _size, _cat, _kyber in ((512, 1, "512"), (768, 3, "768"), (1024, 5, "1024")):
    _register(
        AlgorithmProfile(
            f"ML-KEM-{_size}",
            "kem",
            _R,
            nist_category=_cat,
            standard="FIPS 203",
            cnsa2=_size == 1024,
        ),
    )
    _register(
        AlgorithmProfile(
            f"Kyber-{_kyber}",
            "kem",
            _R,
            nist_category=_cat,
            standard=f"pre-standard (standardised as ML-KEM-{_size}, FIPS 203)",
            prestandard=True,
            notes=(_ROUND3,),
        ),
        f"Kyber{_kyber}",
    )

# --- FIPS 204 ML-DSA ---------------------------------------------------------
for _level, _cat, _dil in ((44, 2, "2"), (65, 3, "3"), (87, 5, "5")):
    _register(
        AlgorithmProfile(
            f"ML-DSA-{_level}",
            "signature",
            _R,
            nist_category=_cat,
            standard="FIPS 204",
            cnsa2=_level == 87,
        ),
    )
    _register(
        AlgorithmProfile(
            f"Dilithium{_dil}",
            "signature",
            _R,
            nist_category=_cat,
            standard=f"pre-standard (standardised as ML-DSA-{_level}, FIPS 204)",
            prestandard=True,
            notes=(_ROUND3,),
        ),
        f"Dilithium-{_dil}",
    )

# --- FIPS 205 SLH-DSA --------------------------------------------------------
for _hash in ("SHA2", "SHAKE"):
    for _bits, _cat in ((128, 1), (192, 3), (256, 5)):
        for _variant in ("s", "f"):
            _register(
                AlgorithmProfile(
                    f"SLH-DSA-{_hash}-{_bits}{_variant}",
                    "signature",
                    _R,
                    nist_category=_cat,
                    standard="FIPS 205",
                ),
            )
            _register(
                AlgorithmProfile(
                    f"SPHINCS+-{_hash}-{_bits}{_variant}",
                    "signature",
                    _R,
                    nist_category=_cat,
                    standard=f"pre-standard (standardised as SLH-DSA-{_hash}-{_bits}{_variant})",
                    prestandard=True,
                    notes=(_ROUND3,),
                ),
                f"SPHINCS+-{_hash}-{_bits}{_variant}-simple",
            )

# --- draft FIPS 206 FN-DSA and the HQC selection -------------------------------
for _size, _cat in ((512, 1), (1024, 5)):
    _register(
        AlgorithmProfile(
            f"FN-DSA-{_size}",
            "signature",
            _R,
            nist_category=_cat,
            standard="FIPS 206 (draft, not yet approved)",
            notes=("not yet a final standard; plan, do not depend on it",),
        ),
        f"Falcon-{_size}",
    )
for _bits, _cat in ((128, 1), (192, 3), (256, 5)):
    _register(
        AlgorithmProfile(
            f"HQC-{_bits}",
            "kem",
            _R,
            nist_category=_cat,
            standard="selected by NIST in March 2025; standard in draft",
            notes=("backup KEM to ML-KEM; no final standard yet",),
        ),
    )

# --- SP 800-208 stateful hash-based signatures --------------------------------
_STATEFUL = (
    "stateful: reusing a one-time key destroys security; CNSA 2.0 restricts it to "
    "software and firmware signing"
)
_register(
    AlgorithmProfile(
        "LMS",
        "stateful_signature",
        _R,
        standard="NIST SP 800-208; RFC 8554",
        cnsa2=True,
        notes=(_STATEFUL,),
    ),
    "HSS",
    "LMS-HSS",
)
_register(
    AlgorithmProfile(
        "XMSS",
        "stateful_signature",
        _R,
        standard="NIST SP 800-208; RFC 8391",
        cnsa2=True,
        notes=(_STATEFUL,),
    ),
    "XMSSMT",
    "XMSS^MT",
)

# --- broken post-quantum candidates -----------------------------------------
_register(
    AlgorithmProfile(
        "SIKE",
        "kem",
        _B,
        standard="withdrawn",
        notes=("key recovery on a laptop (Castryck–Decru, 2022)",),
    ),
    "SIKEp434",
    "SIKEp503",
    "SIKEp610",
    "SIKEp751",
    "SIDH",
)
_register(
    AlgorithmProfile(
        "Rainbow",
        "signature",
        _B,
        standard="withdrawn",
        notes=("key recovery (Beullens, 2022)",),
    ),
    "Rainbow-I",
    "Rainbow-III",
    "Rainbow-V",
)

# --- Shor-vulnerable public key (SP 800-57 strength bands) --------------------
_SHOR = "Shor's algorithm breaks it on a cryptographically relevant quantum computer"
_WEAK = "below 112-bit classical strength: already disallowed (SP 800-131A)"


def _public_key(
    canonical: str, kind: str, bits: int | None, *aliases: str, notes: tuple[str, ...] = ()
) -> None:
    status = _B if bits is not None and bits < 112 else _V
    extra = (_WEAK,) if status is _B else ()
    _register(
        AlgorithmProfile(
            canonical,
            kind,
            status,
            classical_bits=bits,
            standard="classical public key",
            notes=(_SHOR, *extra, *notes),
        ),
        *aliases,
    )


_UNSIZED = ("key size unknown: deadlines assume the 112-bit band (conservative)",)
for _size, _bits in ((1024, 80), (2048, 112), (3072, 128), (4096, 128), (7680, 192), (15360, 256)):
    _public_key(f"RSA-{_size}", "public_key_encryption", _bits, f"RSA{_size}")
_public_key(
    "RSA",
    "public_key_encryption",
    None,
    "rsaEncryption",
    "RSASSA-PSS",
    "RSAES-OAEP",
    "RSA-PSS",
    "RSA-OAEP",
    "PKCS1",
    "RSASSA-PKCS1-v1_5",
    notes=_UNSIZED,
)
for _size, _bits in ((1024, 80), (2048, 112), (3072, 128)):
    _public_key(
        f"DSA-{_size}",
        "signature",
        _bits,
        f"DSA{_size}",
        notes=("FIPS 186-5 no longer approves DSA for signature generation",),
    )
_public_key(
    "DSA",
    "signature",
    None,
    "id-dsa",
    notes=(*_UNSIZED, "FIPS 186-5 no longer approves DSA for signature generation"),
)
for _size, _bits in ((1024, 80), (2048, 112), (3072, 128), (4096, 128)):
    _public_key(f"DH-{_size}", "key_agreement", _bits, f"DH{_size}", f"FFDHE{_size}")
_public_key("DH", "key_agreement", None, "DHE", "dhpublicnumber", "Diffie-Hellman", notes=_UNSIZED)

_CURVES = {
    "P-256": (128, ("secp256r1", "prime256v1", "NIST P-256"), ()),
    "P-384": (192, ("secp384r1", "NIST P-384"), ()),
    "P-521": (256, ("secp521r1", "NIST P-521"), ()),
    "secp256k1": (128, (), ("not a NIST-approved curve",)),
}
for _curve, (_bits, _curve_aliases, _curve_notes) in _CURVES.items():
    for _scheme, _kind in (("ECDSA", "signature"), ("ECDH", "key_agreement")):
        _public_key(
            f"{_scheme}-{_curve}",
            _kind,
            _bits,
            *(f"{_scheme}-{a}" for a in _curve_aliases),
            notes=_curve_notes,
        )
_public_key("ECDSA", "signature", None, notes=_UNSIZED)
_public_key("ECDH", "key_agreement", None, "ECDHE", notes=_UNSIZED)
_public_key("EC", "public_key_encryption", None, "ecPublicKey", "id-ecPublicKey", notes=_UNSIZED)
_public_key("Ed25519", "signature", 128)
_public_key("Ed448", "signature", 192)
_public_key("EdDSA", "signature", None, notes=_UNSIZED)
_public_key("X25519", "key_agreement", 128)
_public_key("X448", "key_agreement", 192)

# --- symmetric ciphers -----------------------------------------------------------
_GROVER = "Grover's search halves the effective key length; NIST bases category 1 on AES-128"
for _bits, _cat in ((128, 1), (192, 3), (256, 5)):
    _register(
        AlgorithmProfile(
            f"AES-{_bits}",
            "cipher",
            _R,
            classical_bits=_bits,
            nist_category=_cat,
            standard="FIPS 197",
            cnsa2=_bits == 256,
            notes=(_GROVER,),
        ),
        f"AES{_bits}",
    )
_register(
    AlgorithmProfile(
        "AES",
        "cipher",
        _R,
        standard="FIPS 197",
        notes=("key size unknown; CNSA 2.0 requires AES-256",),
    ),
)
_register(
    AlgorithmProfile(
        "ChaCha20-Poly1305",
        "cipher",
        _R,
        classical_bits=256,
        standard="RFC 8439 (not FIPS)",
    ),
    "ChaCha20",
    "XChaCha20-Poly1305",
    "CHACHA20POLY1305",
)
_register(
    AlgorithmProfile(
        "3DES",
        "cipher",
        _B,
        classical_bits=112,
        standard="disallowed",
        notes=("NIST disallowed TDEA encryption after 2023 (SP 800-131A Rev. 2)",),
    ),
    "TDEA",
    "DES-EDE3",
    "TripleDES",
    "3DES-EDE",
)
_register(AlgorithmProfile("DES", "cipher", _B, classical_bits=56, standard="withdrawn"))
_register(
    AlgorithmProfile(
        "RC4",
        "cipher",
        _B,
        standard="prohibited in TLS (RFC 7465)",
        notes=("keystream biases",),
    ),
    "ARCFOUR",
)

# --- hashes, MACs and KDFs ---------------------------------------------------------
_register(
    AlgorithmProfile(
        "SHA-1",
        "hash",
        _B,
        standard="FIPS 180-4 (being withdrawn)",
        notes=("practical collisions (SHAttered, 2017); NIST plans to disallow it by 2030",),
    ),
    "SHA1",
)
_register(
    AlgorithmProfile("MD5", "hash", _B, standard="none", notes=("practical collisions",)),
)
_SHA2: tuple[tuple[str, int | None, bool], ...] = (
    ("SHA-224", None, False),
    ("SHA-256", 2, False),
    ("SHA-384", 4, True),
    ("SHA-512", 4, True),
)
for _name, _hash_cat, _cnsa in _SHA2:
    _register(
        AlgorithmProfile(
            _name, "hash", _R, nist_category=_hash_cat, standard="FIPS 180-4", cnsa2=_cnsa
        ),
        _name.replace("SHA-", "SHA2-"),
    )
for _name, _cat in (("SHA3-256", 2), ("SHA3-384", 4), ("SHA3-512", 4)):
    _register(AlgorithmProfile(_name, "hash", _R, nist_category=_cat, standard="FIPS 202"))
for _name in ("SHAKE128", "SHAKE256"):
    _register(AlgorithmProfile(_name, "hash", _R, standard="FIPS 202"))
for _name in ("BLAKE2b", "BLAKE2s", "BLAKE3"):
    _register(AlgorithmProfile(_name, "hash", _R, standard="not FIPS-approved"))
for _bits, _cnsa, _jwt in ((256, False, "HS256"), (384, True, "HS384"), (512, True, "HS512")):
    _register(
        AlgorithmProfile(
            f"HMAC-SHA-{_bits}",
            "mac",
            _R,
            standard="FIPS 198-1",
            cnsa2=_cnsa,
        ),
        f"HMAC-SHA{_bits}",
        _jwt,
    )
_KDF_NOTE = (
    "password hashing: quantum search speeds up guessing only quadratically; "
    "strength rests on password entropy and work factor"
)
for _name, _standard in (
    ("Argon2id", "RFC 9106"),
    ("scrypt", "RFC 7914"),
    ("PBKDF2", "NIST SP 800-132"),
):
    _register(AlgorithmProfile(_name, "kdf", _R, standard=_standard, notes=(_KDF_NOTE,)))

# --- hybrid key establishment (draft-ietf-tls-ecdhe-mlkem) -------------------------
_HYBRIDS = {
    "X25519MLKEM768": ("X25519", "ML-KEM-768"),
    "SecP256r1MLKEM768": ("ECDH-P-256", "ML-KEM-768"),
    "SecP384r1MLKEM1024": ("ECDH-P-384", "ML-KEM-1024"),
    "X25519Kyber768Draft00": ("X25519", "Kyber-768"),
}

#: Object identifiers (NIST CSOR, RFC 8410, RFC 3279/5480/5758) → canonical names.
OID_NAMES: Mapping[str, str] = {
    "1.2.840.113549.1.1.1": "RSA",
    "1.2.840.113549.1.1.10": "RSA",  # RSASSA-PSS
    "1.2.840.113549.1.1.11": "sha256WithRSAEncryption",
    "1.2.840.113549.1.1.12": "sha384WithRSAEncryption",
    "1.2.840.113549.1.1.13": "sha512WithRSAEncryption",
    "1.2.840.113549.1.1.5": "sha1WithRSAEncryption",
    "1.2.840.10045.2.1": "EC",
    "1.2.840.10045.4.3.2": "ecdsa-with-SHA256",
    "1.2.840.10045.4.3.3": "ecdsa-with-SHA384",
    "1.2.840.10045.4.3.4": "ecdsa-with-SHA512",
    "1.2.840.10040.4.1": "DSA",
    "1.2.840.10046.2.1": "DH",
    "1.3.101.110": "X25519",
    "1.3.101.111": "X448",
    "1.3.101.112": "Ed25519",
    "1.3.101.113": "Ed448",
    "2.16.840.1.101.3.4.4.1": "ML-KEM-512",
    "2.16.840.1.101.3.4.4.2": "ML-KEM-768",
    "2.16.840.1.101.3.4.4.3": "ML-KEM-1024",
    "2.16.840.1.101.3.4.3.17": "ML-DSA-44",
    "2.16.840.1.101.3.4.3.18": "ML-DSA-65",
    "2.16.840.1.101.3.4.3.19": "ML-DSA-87",
    **{
        f"2.16.840.1.101.3.4.3.{20 + i}": f"SLH-DSA-{h}-{b}{v}"
        for i, (h, b, v) in enumerate(
            (h, b, v) for h in ("SHA2", "SHAKE") for b in (128, 192, 256) for v in ("s", "f")
        )
    },
    "2.16.840.1.101.3.4.1.6": "AES-128",
    "2.16.840.1.101.3.4.1.26": "AES-192",
    "2.16.840.1.101.3.4.1.46": "AES-256",
    "2.16.840.1.101.3.4.2.1": "SHA-256",
    "2.16.840.1.101.3.4.2.2": "SHA-384",
    "2.16.840.1.101.3.4.2.3": "SHA-512",
    "2.16.840.1.101.3.4.2.8": "SHA3-256",
    "2.16.840.1.101.3.4.2.9": "SHA3-384",
    "2.16.840.1.101.3.4.2.10": "SHA3-512",
    "1.3.14.3.2.26": "SHA-1",
}

_AES_MODE = re.compile(
    r"^AES(128|192|256)?(GCM|CBC|CTR|CCM|XTS|CFB|OFB|KW|KWP|SIV|GCMSIV|OCB|ECB)$"
)
_OID = re.compile(r"^\d+(\.\d+)+$")


def _hybrid(name: str, parts: Sequence[str]) -> AlgorithmProfile:
    components = tuple(_REGISTRY[p] for p in parts)
    best = max(components, key=lambda c: _STATUS_RANK[c.status])
    notes = ["hybrid: secure while any component is secure (assumes a robust combiner)"]
    notes += [n for c in components if c.prestandard for n in c.notes]
    return AlgorithmProfile(
        name,
        components[0].kind,
        best.status,
        nist_category=best.nist_category,
        standard=(
            "draft-ietf-tls-ecdhe-mlkem"
            if not any(c.prestandard for c in components)
            else "pre-standard hybrid"
        ),
        cnsa2=False,
        prestandard=any(c.prestandard for c in components),
        notes=tuple(dict.fromkeys(notes)),
        components=components,
    )


def _lookup(name: str) -> AlgorithmProfile | None:
    key = _normalise(name)
    for hybrid, parts in _HYBRIDS.items():
        if key == _normalise(hybrid):
            return _hybrid(hybrid, parts)
    if key in _ALIASES:
        return _REGISTRY[_ALIASES[key]]
    mode = _AES_MODE.match(key)
    if mode:
        base = _REGISTRY[f"AES-{mode.group(1)}" if mode.group(1) else "AES"]
        if mode.group(2) == "ECB":
            return replace(base, notes=(*base.notes, "ECB mode is not semantically secure"))
        return base
    return None


def _tokens(name: str) -> list[str]:
    spaced = re.sub(r"(?i)with", " ", name)
    return [t for t in re.split(r"[\s/_\-:,;()]+", spaced) if t]


def classify_algorithm(name: str) -> AlgorithmProfile:
    """Classify one algorithm name or dotted OID by exact token lookup.

    A whole-name match wins. Otherwise the name is split on separators and
    on ``with`` (``sha256WithRSAEncryption``, ``ecdsa-with-SHA384``), and each
    span is matched exactly, longest first. Every token must resolve or the
    result is ``unknown``. A signature's strength is its public-key part's;
    an accompanying hash never upgrades it.

    Args:
        name: Algorithm name, alias or dotted OID.

    Returns:
        The classified profile, or an ``unknown`` profile naming the input.
    """
    text = name.strip()
    if _OID.match(text):
        text = OID_NAMES.get(text, text)
        if _OID.match(text):
            return AlgorithmProfile(name, "unknown", QuantumStatus.UNKNOWN, notes=("unmapped OID",))
    found = _lookup(text)
    if found is not None:
        return found

    tokens = _tokens(text)
    parts: list[AlgorithmProfile] = []
    unresolved: list[str] = []
    i = 0
    while i < len(tokens):
        for j in range(len(tokens), i, -1):
            match = _lookup("".join(tokens[i:j]))
            if match is not None:
                parts.append(match)
                i = j
                break
        else:
            unresolved.append(tokens[i])
            i += 1
    if unresolved or not parts:
        return AlgorithmProfile(
            name,
            "unknown",
            QuantumStatus.UNKNOWN,
            notes=(f"unrecognised token(s): {', '.join(unresolved) or name}",),
        )
    keyed = [p for p in parts if p.kind in _PUBLIC_KEY_KINDS]
    if len(keyed) == 1:
        primary = keyed[0]
        hashes = [p.canonical for p in parts if p is not primary]
        notes = (*primary.notes, *(f"with {h}" for h in hashes))
        # "<hash> with <public key>" names a signature scheme.
        kind = "signature" if hashes and primary.kind == "public_key_encryption" else primary.kind
        return AlgorithmProfile(
            name,
            kind,
            primary.status,
            classical_bits=primary.classical_bits,
            nist_category=primary.nist_category,
            standard=primary.standard,
            cnsa2=primary.cnsa2 and all(p.cnsa2 for p in parts),
            prestandard=primary.prestandard,
            notes=notes,
            components=tuple(parts),
        )
    if len(keyed) > 1:
        best = max(keyed, key=lambda c: _STATUS_RANK[c.status])
        return AlgorithmProfile(
            name,
            best.kind,
            best.status,
            nist_category=best.nist_category,
            standard="composite",
            prestandard=any(p.prestandard for p in keyed),
            notes=("composite: as strong as its strongest public-key component",),
            components=tuple(parts),
        )
    weakest = min(parts, key=lambda c: _STATUS_RANK[c.status])
    return AlgorithmProfile(
        name,
        weakest.kind,
        weakest.status,
        standard="composite",
        cnsa2=all(p.cnsa2 for p in parts),
        notes=("composite of non-public-key primitives",),
        components=tuple(parts),
    )


# ------------------------------------------------------------------- policy
#: NIST IR 8547 ipd (Nov 2024) for Shor-vulnerable signatures and key establishment.
NIST_IR_8547 = {"deprecated_after": 2030, "disallowed_after": 2035}

#: NSA CNSA 2.0 transition: (support and prefer by, exclusively use by).
CNSA2_DEADLINES: Mapping[str, tuple[int | None, int]] = {
    "software_firmware_signing": (2025, 2030),
    "web_browsers_servers_cloud": (2025, 2033),
    "traditional_networking": (2026, 2030),
    "operating_systems": (2027, 2033),
    "niche_equipment": (2030, 2033),
    "custom_legacy": (None, 2033),
}

_REPLACEMENT = {
    "kem": "ML-KEM-768 (FIPS 203); ML-KEM-1024 under CNSA 2.0; X25519MLKEM768 during transition",
    "key_agreement": "ML-KEM-768 (FIPS 203); ML-KEM-1024 under CNSA 2.0; X25519MLKEM768 during "
    "transition",
    "public_key_encryption": "ML-KEM-768 + AES-256-GCM (KEM/DEM); ML-KEM-1024 under CNSA 2.0",
    "signature": "ML-DSA-65 (FIPS 204); ML-DSA-87 under CNSA 2.0; LMS/XMSS for firmware signing",
    "stateful_signature": "LMS or XMSS (SP 800-208)",
    "cipher": "AES-256-GCM",
    "hash": "SHA-256 or stronger; SHA-384 under CNSA 2.0",
    "mac": "HMAC-SHA-256 or stronger; HMAC-SHA-384 under CNSA 2.0",
}
_CNSA2_REPLACEMENT = {
    "kem": "ML-KEM-1024",
    "key_agreement": "ML-KEM-1024",
    "public_key_encryption": "ML-KEM-1024",
    "signature": "ML-DSA-87 (LMS/XMSS for firmware)",
    "stateful_signature": "LMS or XMSS",
    "cipher": "AES-256",
    "hash": "SHA-384 or SHA-512",
    "mac": "HMAC-SHA-384",
}


@dataclass(frozen=True)
class CRQCAssumption:
    """The caller's assumption about when a CRQC arrives, in years from now.

    Exactly one of ``years`` (a point estimate) or ``cumulative`` (a
    non-decreasing curve of ``(years, probability CRQC exists by then)``
    points) is given. Nothing is assumed by default.
    """

    years: float | None = None
    cumulative: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        """Validate that exactly one well-formed assumption is supplied."""
        if (self.years is None) == (not self.cumulative):
            raise ValueError("give exactly one of years or cumulative")
        if self.years is not None and self.years < 0:
            raise ValueError("years to a CRQC must be non-negative")
        points = list(self.cumulative)
        if points:
            xs = [x for x, _ in points]
            ps = [p for _, p in points]
            if xs != sorted(xs) or len(set(xs)) != len(xs):
                raise ValueError("cumulative curve years must be strictly increasing")
            if ps != sorted(ps) or not all(0.0 <= p <= 1.0 for p in ps):
                raise ValueError("cumulative probabilities must be non-decreasing in [0, 1]")

    def probability_by(self, horizon: float) -> tuple[float, bool]:
        """Return ``(P(CRQC within horizon), is_lower_bound)`` from the curve.

        Linear interpolation between points; before the first point the
        curve rises linearly from ``(0, 0)``. Past the last point the last
        probability is returned and flagged as a lower bound.
        """
        xs = [0.0, *(x for x, _ in self.cumulative)]
        ps = [0.0, *(p for _, p in self.cumulative)]
        if self.cumulative and self.cumulative[0][0] == 0.0:
            xs, ps = xs[1:], ps[1:]
        if horizon >= xs[-1]:
            return ps[-1], horizon > xs[-1]
        k = bisect_left(xs, horizon)
        if xs[k] == horizon:
            return ps[k], False
        x0, x1, p0, p1 = xs[k - 1], xs[k], ps[k - 1], ps[k]
        return p0 + (p1 - p0) * (horizon - x0) / (x1 - x0), False


@dataclass(frozen=True)
class CryptoAsset:
    """A system or data flow and the cryptography that protects it.

    Attributes:
        name: Asset name.
        algorithms: Algorithm names or OIDs in use.
        protects: ``confidentiality``, ``authenticity`` or ``both``.
        shelf_life_years: ``x``: how long its data must stay secret, or its
            signatures trusted.
        migration_years: ``y``: how long migrating it will take.
        recordable: Whether an adversary can capture its ciphertext today.
        cnsa2_category: Optional CNSA 2.0 category (see ``CNSA2_DEADLINES``).
    """

    name: str
    algorithms: tuple[str, ...]
    protects: str = "both"
    shelf_life_years: float = 0.0
    migration_years: float = 0.0
    recordable: bool = True
    cnsa2_category: str | None = None

    def __post_init__(self) -> None:
        """Validate the asset's declared protection, timescales and category."""
        if self.protects not in ("confidentiality", "authenticity", "both"):
            raise ValueError(
                f"protects must be confidentiality, authenticity or both: {self.protects!r}"
            )
        if self.shelf_life_years < 0 or self.migration_years < 0:
            raise ValueError("shelf life and migration time must be non-negative")
        if self.cnsa2_category is not None and self.cnsa2_category not in CNSA2_DEADLINES:
            raise ValueError(f"unknown CNSA 2.0 category {self.cnsa2_category!r}")


@dataclass(frozen=True)
class AssetAssessment:
    """The quantum-risk assessment of one asset."""

    asset: str
    algorithms: tuple[AlgorithmProfile, ...]
    weakest: QuantumStatus
    unknown: tuple[str, ...]
    exposure_horizon_years: float | None
    crqc_exposed: bool | None
    exposure_probability: float | None
    probability_is_lower_bound: bool
    nist_deadline: int | None
    nist_late: bool | None
    cnsa2_deadline: int | None
    cnsa2_late: bool | None
    cnsa2_noncompliant: tuple[str, ...]
    findings: tuple[str, ...]
    recommendations: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready view (profiles reduced to name, kind, status, standard)."""
        out = asdict(self)
        out["weakest"] = self.weakest.value
        out["algorithms"] = [
            {
                "name": p.canonical,
                "kind": p.kind,
                "status": p.status.value,
                "standard": p.standard,
                "nist_category": p.nist_category,
                "cnsa2": p.cnsa2,
                "prestandard": p.prestandard,
                "notes": list(p.notes),
            }
            for p in self.algorithms
        ]
        return out


def _relevant(profile: AlgorithmProfile, protects: str) -> bool:
    if protects == "both":
        return True
    kinds = _CONFIDENTIALITY_KINDS if protects == "confidentiality" else _AUTHENTICITY_KINDS
    if profile.kind == "public_key_encryption":
        return True  # RSA/EC keys serve both encryption and signatures
    return profile.kind in kinds


def assess_asset(
    asset: CryptoAsset,
    *,
    current_year: int | None = None,
    crqc: CRQCAssumption | None = None,
) -> AssetAssessment:
    """Assess one asset against standards, deadlines and the CRQC assumption.

    Args:
        asset: The asset to assess.
        current_year: Calendar year of the assessment (defaults to now, UTC).
        crqc: The caller's CRQC assumption; without one, only the exposure
            horizon is reported.

    Returns:
        The assessment.
    """
    year = current_year if current_year is not None else datetime.now(UTC).year
    profiles = tuple(classify_algorithm(a) for a in asset.algorithms)
    relevant = [p for p in profiles if _relevant(p, asset.protects)] or list(profiles)
    unknown = tuple(p.canonical for p in profiles if p.status is QuantumStatus.UNKNOWN)
    weakest = min(
        (p.status for p in relevant), key=_STATUS_RANK.__getitem__, default=QuantumStatus.UNKNOWN
    )
    findings: list[str] = []
    recommendations: list[str] = []

    for p in profiles:
        if p.status is QuantumStatus.BROKEN:
            findings.append(f"{p.canonical}: classically broken now ({'; '.join(p.notes)})")
        if p.prestandard:
            findings.append(f"{p.canonical}: {_ROUND3}")
        if p.status is QuantumStatus.UNKNOWN:
            findings.append(f"{p.canonical}: not recognised; classify it before relying on it")
        if p.status in (QuantumStatus.BROKEN, QuantumStatus.VULNERABLE) and p.kind in _REPLACEMENT:
            recommendations.append(f"replace {p.canonical} with {_REPLACEMENT[p.kind]}")

    # Mosca: which vulnerable algorithms threaten this asset's protected property?
    threatened = False
    for p in relevant:
        if p.status is not QuantumStatus.VULNERABLE:
            continue
        if p.kind in ("kem", "key_agreement") or (
            p.kind == "public_key_encryption" and asset.protects != "authenticity"
        ):
            threatened |= asset.recordable  # harvest now, decrypt later
        if p.kind == "signature" or (
            p.kind == "public_key_encryption" and asset.protects != "confidentiality"
        ):
            threatened = True  # forgery once a CRQC exists, for signatures still trusted
    horizon = asset.shelf_life_years + asset.migration_years if threatened else None
    exposed: bool | None = None
    probability: float | None = None
    lower_bound = False
    if horizon is not None and crqc is not None:
        if crqc.years is not None:
            exposed = horizon > crqc.years
            if exposed:
                findings.append(
                    f"exposed: x + y = {horizon:g} years exceeds the assumed {crqc.years:g} years "
                    f"to a CRQC by {horizon - crqc.years:g}"
                )
        else:
            probability, lower_bound = crqc.probability_by(horizon)
            findings.append(
                f"P(CRQC within x + y = {horizon:g} years) = {probability:.3f}"
                + (" (lower bound: beyond the supplied curve)" if lower_bound else "")
            )
    elif horizon is not None:
        findings.append(
            f"exposed if a CRQC arrives within {horizon:g} years (x + y); no arrival assumption given"
        )

    vulnerable = [p for p in relevant if p.status is QuantumStatus.VULNERABLE]
    nist_deadline: int | None = None
    nist_late: bool | None = None
    if vulnerable:
        bands = [p.classical_bits for p in vulnerable]
        at_112 = any(b is None or b <= 112 for b in bands)
        nist_deadline = (
            NIST_IR_8547["deprecated_after"] if at_112 else NIST_IR_8547["disallowed_after"]
        )
        nist_late = year + asset.migration_years > nist_deadline
        if nist_late:
            findings.append(
                f"migration finishes in {year + asset.migration_years:g}, after the NIST IR 8547 "
                f"{'deprecation' if at_112 else 'disallowance'} date ({nist_deadline})"
            )

    cnsa2_deadline: int | None = None
    cnsa2_late: bool | None = None
    noncompliant: tuple[str, ...] = ()
    if asset.cnsa2_category is not None:
        cnsa2_deadline = CNSA2_DEADLINES[asset.cnsa2_category][1]
        noncompliant = tuple(p.canonical for p in profiles if not p.cnsa2 and p.kind != "kdf")
        cnsa2_late = bool(noncompliant) and year + asset.migration_years > cnsa2_deadline
        for p in profiles:
            if not p.cnsa2 and p.kind in _CNSA2_REPLACEMENT:
                recommendations.append(
                    f"CNSA 2.0 ({asset.cnsa2_category}, exclusively by {cnsa2_deadline}): "
                    f"replace {p.canonical} with {_CNSA2_REPLACEMENT[p.kind]}"
                )

    return AssetAssessment(
        asset=asset.name,
        algorithms=profiles,
        weakest=weakest,
        unknown=unknown,
        exposure_horizon_years=horizon,
        crqc_exposed=exposed,
        exposure_probability=probability,
        probability_is_lower_bound=lower_bound,
        nist_deadline=nist_deadline,
        nist_late=nist_late,
        cnsa2_deadline=cnsa2_deadline,
        cnsa2_late=cnsa2_late,
        cnsa2_noncompliant=noncompliant,
        findings=tuple(findings),
        recommendations=tuple(dict.fromkeys(recommendations)),
    )


def prioritize(assessments: Iterable[AssetAssessment]) -> list[AssetAssessment]:
    """Order assessments by urgency, with a stated rule rather than a score.

    Order: classically broken first; then exposed under the CRQC assumption
    (higher probability first); then late for NIST IR 8547; then late for
    CNSA 2.0; then longer exposure horizon; ties by name.
    """

    def key(a: AssetAssessment) -> tuple[Any, ...]:
        exposure = (
            a.exposure_probability
            if a.exposure_probability is not None
            else (1.0 if a.crqc_exposed else 0.0)
        )
        return (
            a.weakest is not QuantumStatus.BROKEN,
            -exposure,
            not a.nist_late,
            not a.cnsa2_late,
            -(a.exposure_horizon_years or 0.0),
            a.asset,
        )

    return sorted(assessments, key=key)


# --------------------------------------------------------------------- CBOM
@dataclass(frozen=True)
class CBOMEntry:
    """One CycloneDX cryptographic-asset algorithm, classified."""

    bom_ref: str
    declared_name: str
    profile: AlgorithmProfile
    claimed_level: int | None
    findings: tuple[str, ...]


def _walk_components(components: Iterable[Any]) -> Iterable[Mapping[str, Any]]:
    for component in components:
        if not isinstance(component, Mapping):
            continue
        yield component
        yield from _walk_components(component.get("components") or ())


def read_cbom(cbom: Mapping[str, Any]) -> list[CBOMEntry]:
    """Classify every algorithm in a CycloneDX (1.6+) CBOM.

    Each ``cryptographic-asset`` with ``assetType: algorithm`` is classified
    by OID when one is given, otherwise by name, with the
    ``parameterSetIdentifier`` appended when the name alone does not match.
    A ``nistQuantumSecurityLevel`` the inventory claims is checked against the
    registry. Claiming quantum safety for a vulnerable algorithm, or level 0
    for a resistant one, is a finding.

    Raises:
        ValueError: The document is not a CycloneDX BOM.
    """
    if not isinstance(cbom, Mapping) or cbom.get("bomFormat") != "CycloneDX":
        raise ValueError("not a CycloneDX BOM (bomFormat must be 'CycloneDX')")
    entries: list[CBOMEntry] = []
    for component in _walk_components(cbom.get("components") or ()):
        if component.get("type") != "cryptographic-asset":
            continue
        props = component.get("cryptoProperties") or {}
        if props.get("assetType") != "algorithm":
            continue
        name = str(component.get("name", ""))
        algo = props.get("algorithmProperties") or {}
        oid = props.get("oid")
        profile = classify_algorithm(str(oid)) if oid else classify_algorithm(name)
        param = algo.get("parameterSetIdentifier")
        if profile.status is QuantumStatus.UNKNOWN and param is not None:
            profile = classify_algorithm(f"{name}-{param}")
        claimed = algo.get("nistQuantumSecurityLevel")
        findings: list[str] = []
        if isinstance(claimed, int) and profile.status is not QuantumStatus.UNKNOWN:
            if claimed > 0 and profile.status is not QuantumStatus.RESISTANT:
                findings.append(
                    f"inventory claims quantum security level {claimed} for "
                    f"{profile.canonical}, which is {profile.status.value}"
                )
            elif claimed == 0 and profile.status is QuantumStatus.RESISTANT:
                findings.append(f"inventory understates {profile.canonical} as not quantum-safe")
            elif (
                claimed > 0
                and profile.nist_category is not None
                and claimed != profile.nist_category
            ):
                findings.append(
                    f"inventory claims level {claimed} for {profile.canonical}; "
                    f"the standard assigns category {profile.nist_category}"
                )
        entries.append(
            CBOMEntry(
                bom_ref=str(component.get("bom-ref", name)),
                declared_name=name,
                profile=profile,
                claimed_level=claimed if isinstance(claimed, int) else None,
                findings=tuple(findings),
            )
        )
    return entries


# --------------------------------------------------------------- self-audit
@dataclass(frozen=True)
class MercuryCryptoUse:
    """One of Mercury's own cryptographic uses, pinned to its source file."""

    use: str
    source: str
    algorithms: tuple[str, ...]
    protects: str
    shelf_life_years: float


#: Mercury's declared cryptography. ``tests`` pin every algorithm to the text
#: of its source file, so a change in the code that is not reflected here fails.
MERCURY_CRYPTO_USES: tuple[MercuryCryptoUse, ...] = (
    MercuryCryptoUse(
        "hybrid public-key encryption (KEM/DEM)",
        "security/encryption.py",
        ("ML-KEM-1024", "AES-256-GCM"),
        "confidentiality",
        10.0,
    ),
    MercuryCryptoUse(
        "sealed secrets at rest", "api/secret_sealer.py", ("AES-256-GCM",), "confidentiality", 10.0
    ),
    MercuryCryptoUse(
        "FINDΩYOU™ evidence packets",
        "integrations/findoyou_bridge.py",
        ("ML-DSA-65", "SHA3-256"),
        "authenticity",
        25.0,
    ),
    MercuryCryptoUse(
        "self-model ledger", "self_model.py", ("ML-DSA-65", "SHA3-256"), "authenticity", 10.0
    ),
    MercuryCryptoUse(
        "API tokens (JWT)",
        "security/ama_hmac.py",
        ("HMAC-SHA-256", "HMAC-SHA-384", "HMAC-SHA-512"),
        "authenticity",
        0.1,
    ),
    MercuryCryptoUse(
        "sovereign emergency tier",
        "crypto/sovereign.py",
        ("ChaCha20-Poly1305", "Ed25519"),
        "both",
        5.0,
    ),
)


def audit_mercury_posture(
    *,
    current_year: int | None = None,
    crqc: CRQCAssumption | None = None,
    migration_years: float = 2.0,
    cnsa2_category: str | None = None,
) -> list[AssetAssessment]:
    """Assess Mercury's own cryptography with the same rules as any inventory.

    Args:
        current_year: Calendar year of the assessment.
        crqc: Optional CRQC arrival assumption.
        migration_years: Assumed time to migrate any one use.
        cnsa2_category: Assess against CNSA 2.0 under this category.

    Returns:
        Assessments of each declared use, most urgent first.
    """
    assets = [
        CryptoAsset(
            name=f"{u.use} [{u.source}]",
            algorithms=u.algorithms,
            protects=u.protects,
            shelf_life_years=u.shelf_life_years,
            migration_years=migration_years,
            cnsa2_category=cnsa2_category,
        )
        for u in MERCURY_CRYPTO_USES
    ]
    return prioritize(assess_asset(a, current_year=current_year, crqc=crqc) for a in assets)


def main(argv: list[str] | None = None) -> int:
    """Assess a CBOM (``--cbom``) or Mercury itself (default) and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cbom", type=Path, help="CycloneDX CBOM (JSON) to assess")
    parser.add_argument("--crqc-years", type=float, help="assumed years to a CRQC")
    parser.add_argument(
        "--cnsa2-category", choices=sorted(CNSA2_DEADLINES), help="assess against CNSA 2.0"
    )
    parser.add_argument("--year", type=int, help="assessment year (default: current)")
    args = parser.parse_args(argv)
    crqc = CRQCAssumption(years=args.crqc_years) if args.crqc_years is not None else None
    if args.cbom is not None:
        entries = read_cbom(json.loads(args.cbom.read_text(encoding="utf-8")))
        report: Any = [
            {
                "bom_ref": e.bom_ref,
                "declared_name": e.declared_name,
                "classified_as": e.profile.canonical,
                "status": e.profile.status.value,
                "standard": e.profile.standard,
                "findings": list(e.findings),
            }
            for e in entries
        ]
    else:
        report = [
            a.to_dict()
            for a in audit_mercury_posture(
                current_year=args.year, crqc=crqc, cnsa2_category=args.cnsa2_category
            )
        ]
    json.dump(report, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


__all__ = [
    "CNSA2_DEADLINES",
    "MERCURY_CRYPTO_USES",
    "NIST_IR_8547",
    "OID_NAMES",
    "AlgorithmProfile",
    "AssetAssessment",
    "CBOMEntry",
    "CRQCAssumption",
    "CryptoAsset",
    "MercuryCryptoUse",
    "QuantumStatus",
    "assess_asset",
    "audit_mercury_posture",
    "classify_algorithm",
    "prioritize",
    "read_cbom",
]

if __name__ == "__main__":
    raise SystemExit(main())
