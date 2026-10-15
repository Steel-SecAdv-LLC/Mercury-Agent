# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury's durable internal self-model: what it is, what it can do, how well, and where it falls short.

A self-model is only worth having if it is **true** and **durable**.

*True* — every entry is derived from the running installation, never written
as prose about itself:

* **Identity** — name, version, mission (civilization-first), code system
  (Omni Codes) and the honesty contract (the three-state verdict values).
* **Capabilities** — each probed without side effects (module and dependency
  presence via import specs, no imports executed) and given a
  :class:`~omni_mercury_engine.verifiers.three_state.ThreeState`: ``GROUNDED``
  when it can run here, ``UNAVAILABLE`` when a named dependency is missing.
* **Safeguards** — whether the AMA native PQC backend is present (the gate
  that every agent surface enforces) and the digest of the σ_Immutable
  config-integrity baseline.
* **Measured performance** — read from the committed benchmark artefacts,
  each pinned by its SHA-256, so a claim about performance always points at
  the measurement behind it.
* **Limitations** — derived from those measurements and from missing
  dependencies (a weak benchmark category or an absent identity backend is a
  stated limitation, not an omission).

*Durable* — :class:`SelfModelLedger` appends snapshots to a JSON-lines file
as a SHA3-256 hash chain (each record commits to its predecessor), optionally
signed with ML-DSA-65 through AMA. :meth:`SelfModelLedger.verify` detects any
edit, deletion or reordering, and :func:`diff_snapshots` reports what changed
about Mercury between two snapshots — capabilities gained or lost, metric
drift, safeguard changes — so Mercury can account for its own evolution.

Building a snapshot imports nothing heavy and needs no AMA; signing a ledger
record does (it goes through the gated crypto surface).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omni_mercury_engine._version import get_version
from omni_mercury_engine.verifiers.three_state import ThreeState

SNAPSHOT_SCHEMA = "mercury.self_model/v1"
LEDGER_SCHEMA = "mercury.self_model.ledger/v1"
GENESIS_DIGEST = "0" * 64

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent.parent

#: Benchmark-category image AUROC below which the visual analyser's weakness
#: on that category is recorded as a limitation.
WEAK_AUROC = 0.75


@dataclass(frozen=True)
class CapabilitySpec:
    """A capability Mercury claims, and what it needs to run.

    Attributes:
        name: Stable capability identifier.
        description: One line on what it does.
        modules: In-package modules that implement it.
        requires: Third-party import names it needs to run at all.
        grounding_requires: Import names without which it runs but cannot
            reach a grounded verdict (for example, identity verification
            without a face-identity backend).
        evidence: Benchmark artefacts (repo-relative) that measure it.
    """

    name: str
    description: str
    modules: tuple[str, ...]
    requires: tuple[str, ...] = ()
    grounding_requires: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


CAPABILITIES: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        "visual_anomaly_analysis",
        "learn normal appearance; detect, localise, classify and explain image anomalies "
        "with conformal false-alarm control",
        ("omni_mercury_engine.vision.analyzer", "omni_mercury_engine.vision.features"),
        requires=("numpy", "scipy"),
        evidence=("benchmarks/results/mvtec_visual_benchmark.json",),
    ),
    CapabilitySpec(
        "image_forensics",
        "splice, JPEG-grid, copy-move and recapture (moire/halftone) screening",
        ("omni_mercury_engine.vision.forensics",),
        requires=("numpy", "scipy"),
    ),
    CapabilitySpec(
        "biometric_match",
        "reference-vs-test image matching with backend-labelled similarity",
        ("omni_mercury_engine.models.biometric",),
        requires=("numpy", "cv2"),
        grounding_requires=("deepface",),
    ),
    CapabilitySpec(
        "age_progression",
        "cardioidal craniofacial growth geometry, deterministic texture aging, audited output",
        ("omni_mercury_engine.models.biometric_multimodal",),
        requires=("numpy", "scipy", "cv2"),
    ),
    CapabilitySpec(
        "findoyou_evidence_bridge",
        "signed three-state missing-person image evidence for FINDΩYOU™",
        ("omni_mercury_engine.integrations.findoyou_bridge",),
        requires=("ama_cryptography",),
    ),
    CapabilitySpec(
        "post_quantum_signing",
        "ML-DSA-65 / SPHINCS+ signatures and Kyber-1024 key encapsulation via AMA",
        ("omni_mercury_engine.security.crypto_api",),
        requires=("ama_cryptography",),
    ),
    CapabilitySpec(
        "maritime_anomaly_detection",
        "unsupervised vessel-track anomaly detection on real AIS data",
        ("omni_mercury_engine.datasets.maritime", "omni_mercury_engine.detectors.statistical"),
        requires=("numpy",),
        evidence=("benchmarks/mercury_benchmark_results.json",),
    ),
)


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe_capability(spec: CapabilitySpec) -> dict[str, Any]:
    missing_modules = [m for m in spec.modules if not _module_present(m)]
    missing = [r for r in spec.requires if not _module_present(r)]
    missing_grounding = [r for r in spec.grounding_requires if not _module_present(r)]
    if missing_modules or missing:
        state = ThreeState.UNAVAILABLE
        reason = "missing: " + ", ".join(missing_modules + missing)
    elif missing_grounding:
        state = ThreeState.UNAVAILABLE
        reason = "runs, but cannot reach a grounded verdict without: " + ", ".join(
            missing_grounding
        )
    else:
        state = ThreeState.GROUNDED
        reason = "implementation and dependencies present"
    evidence = []
    for relative in spec.evidence:
        path = _REPO_ROOT / relative
        evidence.append({"artifact": relative, "sha256": _file_digest(path)})
    return {
        "description": spec.description,
        "state": state.value,
        "reason": reason,
        "modules": list(spec.modules),
        "evidence": evidence,
    }


def _measured_performance() -> tuple[dict[str, Any], list[str]]:
    """Read committed benchmark artefacts; return metrics and derived limitations."""
    metrics: dict[str, Any] = {}
    limitations: list[str] = []

    mvtec = _REPO_ROOT / "benchmarks" / "results" / "mvtec_visual_benchmark.json"
    if mvtec.is_file():
        data = json.loads(mvtec.read_text())
        categories = data.get("categories", {})
        metrics["mvtec_ad_textures"] = {
            "summary": data.get("summary", {}),
            "per_category_image_auroc": {k: v.get("image_auroc") for k, v in categories.items()},
            "sha256": _file_digest(mvtec),
        }
        for name, result in sorted(categories.items()):
            auroc = result.get("image_auroc")
            if auroc is not None and auroc < WEAK_AUROC:
                limitations.append(
                    f"visual anomaly analysis is weak on MVTec AD '{name}' "
                    f"(image AUROC {auroc:.3f} < {WEAK_AUROC}): measured, not assumed"
                )
        strict = data.get("summary", {}).get("mean_false_alarm_rate_at_alpha")
        alpha = data.get("protocol", {}).get("alpha")
        if strict is not None and alpha is not None and strict > alpha:
            limitations.append(
                f"conformal false-alarm control needs in-session calibration: across a "
                f"train-to-test acquisition shift the strict false-alarm rate was {strict:.3f} "
                f"at alpha={alpha} (recalibrate on session normals)"
            )

    results = _REPO_ROOT / "benchmarks" / "mercury_benchmark_results.json"
    if results.is_file():
        data = json.loads(results.read_text())
        rows = data.get("per_dataset", [])
        dtu = next(
            (r for r in rows if isinstance(r, dict) and r.get("name") == "DTU_AIS_Abnormal"), None
        )
        if dtu is not None:
            metrics["dtu_ais_abnormal"] = {
                "ensemble_auc": dtu.get("ensemble_auc"),
                "n_test": dtu.get("n_test"),
                "label_source": dtu.get("label_source"),
                "sha256": _file_digest(results),
            }
    return metrics, limitations


def _safeguards() -> dict[str, Any]:
    baseline = _PACKAGE_ROOT / "security" / "sigma_immutable_baseline.json"
    ama_present = _module_present("ama_cryptography")
    return {
        "pqc_backend_present": ama_present,
        "pqc_gate": "agent surfaces (engine, crypto, AMA-backed security) refuse to import "
        "without the AMA native backend",
        "sigma_immutable_baseline_sha256": _file_digest(baseline),
        "retired_terminology_guard": "tests/test_retired_terminology.py",
    }


@dataclass(frozen=True)
class SelfModelSnapshot:
    """One self-model snapshot: a JSON-safe body and its canonical digest."""

    body: dict[str, Any]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        """Compute the canonical SHA3-256 digest of the body."""
        object.__setattr__(self, "digest", hashlib.sha3_256(_canonical(self.body)).hexdigest())

    def to_dict(self) -> dict[str, Any]:
        """Return the body plus its digest."""
        return {**self.body, "digest": self.digest}


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def build_self_model(created_utc: str | None = None) -> SelfModelSnapshot:
    """Derive Mercury's current self-model from the installation.

    Args:
        created_utc: Timestamp to record (ISO 8601); defaults to now. Passing
            it makes the snapshot (and its digest) reproducible.

    Returns:
        The snapshot.
    """
    capabilities = {spec.name: _probe_capability(spec) for spec in CAPABILITIES}
    metrics, limitations = _measured_performance()
    for name, record in sorted(capabilities.items()):
        if record["state"] != ThreeState.GROUNDED.value:
            limitations.append(f"{name}: {record['reason']}")
    body = {
        "schema": SNAPSHOT_SCHEMA,
        "created_utc": created_utc or datetime.now(UTC).isoformat(timespec="seconds"),
        "identity": {
            "name": "Mercury Agent",
            "version": get_version(),
            "mission": "civilization-first",
            "code_system": "Omni Codes",
            "family": ["FINDΩYOU™", "Mercury Agent", "AMA Cryptography"],
            "honesty_contract": sorted(s.value for s in ThreeState),
        },
        "capabilities": capabilities,
        "safeguards": _safeguards(),
        "measured_performance": metrics,
        "limitations": limitations,
    }
    return SelfModelSnapshot(body=body)


def diff_snapshots(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Report what changed about Mercury between two snapshot bodies.

    Returns:
        ``capabilities_gained`` / ``capabilities_lost`` (state transitions to
        and from ``grounded``), ``capability_state_changes``,
        ``metric_changes`` (dotted path -> ``[old, new]`` for numeric
        values), ``safeguard_changes``, ``limitations_added`` and
        ``limitations_resolved``, and ``version_change``.
    """
    old_caps, new_caps = old.get("capabilities", {}), new.get("capabilities", {})
    gained, lost, changed = [], [], {}
    for name in sorted(set(old_caps) | set(new_caps)):
        before = old_caps.get(name, {}).get("state")
        after = new_caps.get(name, {}).get("state")
        if before == after:
            continue
        changed[name] = [before, after]
        if after == ThreeState.GROUNDED.value:
            gained.append(name)
        elif before == ThreeState.GROUNDED.value:
            lost.append(name)

    def _numbers(obj: Any, prefix: str = "") -> dict[str, float]:
        out: dict[str, float] = {}
        if isinstance(obj, dict):
            for key, value in obj.items():
                out.update(_numbers(value, f"{prefix}.{key}" if prefix else str(key)))
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            out[prefix] = float(obj)
        return out

    old_metrics = _numbers(old.get("measured_performance", {}))
    new_metrics = _numbers(new.get("measured_performance", {}))
    metric_changes = {
        key: [old_metrics.get(key), new_metrics.get(key)]
        for key in sorted(set(old_metrics) | set(new_metrics))
        if old_metrics.get(key) != new_metrics.get(key)
    }
    old_safe, new_safe = old.get("safeguards", {}), new.get("safeguards", {})
    safeguard_changes = {
        key: [old_safe.get(key), new_safe.get(key)]
        for key in sorted(set(old_safe) | set(new_safe))
        if old_safe.get(key) != new_safe.get(key)
    }
    old_lim, new_lim = set(old.get("limitations", [])), set(new.get("limitations", []))
    old_version = old.get("identity", {}).get("version")
    new_version = new.get("identity", {}).get("version")
    return {
        "capabilities_gained": gained,
        "capabilities_lost": lost,
        "capability_state_changes": changed,
        "metric_changes": metric_changes,
        "safeguard_changes": safeguard_changes,
        "limitations_added": sorted(new_lim - old_lim),
        "limitations_resolved": sorted(old_lim - new_lim),
        "version_change": None if old_version == new_version else [old_version, new_version],
    }


def default_ledger_path() -> Path:
    """Return ``$MERCURY_SELF_MODEL_PATH`` or ``~/.mercury/self_model.jsonl``."""
    configured = os.environ.get("MERCURY_SELF_MODEL_PATH")
    return Path(configured) if configured else Path.home() / ".mercury" / "self_model.jsonl"


class SelfModelLedger:
    """Append-only, hash-chained (and optionally ML-DSA-65-signed) self-model history.

    Each line is a JSON record ``{schema, index, prev_digest, snapshot,
    record_digest, signature?}`` where ``record_digest`` is the SHA3-256 of the
    canonical record without its digest and signature, and ``prev_digest`` is
    the previous record's ``record_digest`` (:data:`GENESIS_DIGEST` for the
    first). Altering, deleting or reordering any record breaks the chain.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Open (without creating) the ledger at ``path`` or the default location."""
        self.path = Path(path) if path is not None else default_ledger_path()

    def records(self) -> list[dict[str, Any]]:
        """Return every record in order (empty when the ledger does not exist yet)."""
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def latest(self) -> dict[str, Any] | None:
        """Return the most recent snapshot body, if any."""
        records = self.records()
        return records[-1]["snapshot"] if records else None

    def append(
        self, snapshot: SelfModelSnapshot, signing_keypair: tuple[bytes, bytes] | None = None
    ) -> dict[str, Any]:
        """Append ``snapshot`` to the chain, signing the record when a key pair is given.

        Args:
            snapshot: The snapshot to record.
            signing_keypair: Optional ML-DSA-65 ``(public_key, secret_key)``;
                signing requires the AMA native backend.

        Returns:
            The record written.

        Raises:
            ValueError: the existing ledger fails verification (an append never
                extends a broken chain).
        """
        ok, reason = self.verify()
        if not ok:
            raise ValueError(f"refusing to extend a broken self-model ledger: {reason}")
        records = self.records()
        record: dict[str, Any] = {
            "schema": LEDGER_SCHEMA,
            "index": len(records),
            "prev_digest": records[-1]["record_digest"] if records else GENESIS_DIGEST,
            "snapshot": snapshot.to_dict(),
        }
        record["record_digest"] = hashlib.sha3_256(_canonical(record)).hexdigest()
        if signing_keypair is not None:
            from omni_mercury_engine.security.crypto_api import MLDSAProvider

            public_key, secret_key = signing_keypair
            record["signature"] = {
                "algorithm": "ML-DSA-65",
                "public_key_hex": public_key.hex(),
                "value_hex": MLDSAProvider()
                .sign(record["record_digest"].encode(), secret_key)
                .hex(),
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def verify(self, expected_public_key_hex: str | None = None) -> tuple[bool, str]:
        """Check the whole chain (and signatures against a pinned key, when given).

        Returns:
            ``(ok, reason)`` naming the first broken record.
        """
        prev = GENESIS_DIGEST
        for position, record in enumerate(self.records()):
            if record.get("schema") != LEDGER_SCHEMA:
                return False, f"record {position}: unsupported schema {record.get('schema')!r}"
            if record.get("index") != position:
                return False, f"record {position}: index {record.get('index')} out of order"
            if record.get("prev_digest") != prev:
                return False, f"record {position}: does not chain to its predecessor"
            unsigned = {k: v for k, v in record.items() if k not in ("record_digest", "signature")}
            digest = hashlib.sha3_256(_canonical(unsigned)).hexdigest()
            if digest != record.get("record_digest"):
                return False, f"record {position}: contents altered after recording"
            snapshot = dict(record["snapshot"])
            claimed = snapshot.pop("digest", None)
            if hashlib.sha3_256(_canonical(snapshot)).hexdigest() != claimed:
                return False, f"record {position}: snapshot digest mismatch"
            if expected_public_key_hex is not None:
                signature = record.get("signature")
                if not signature or signature.get("public_key_hex") != expected_public_key_hex:
                    return False, f"record {position}: not signed by the pinned key"
                from omni_mercury_engine.security.crypto_api import MLDSAProvider

                valid = MLDSAProvider().verify(
                    digest.encode(),
                    bytes.fromhex(signature["value_hex"]),
                    bytes.fromhex(expected_public_key_hex),
                )
                if not valid:
                    return False, f"record {position}: signature does not verify"
            prev = digest
        return True, "valid"


def main(argv: list[str] | None = None) -> int:
    """Print the current self-model; ``--append`` records it, ``--verify`` checks the ledger."""
    import argparse

    parser = argparse.ArgumentParser(description="Mercury's durable internal self-model")
    parser.add_argument("--append", action="store_true", help="record this snapshot in the ledger")
    parser.add_argument("--verify", action="store_true", help="verify the ledger chain")
    parser.add_argument("--ledger", type=Path, default=None, help="ledger path")
    args = parser.parse_args(argv)
    ledger = SelfModelLedger(args.ledger)
    if args.verify:
        ok, reason = ledger.verify()
        print(json.dumps({"ledger": str(ledger.path), "valid": ok, "reason": reason}))
        return 0 if ok else 1
    snapshot = build_self_model()
    previous = ledger.latest()
    output: dict[str, Any] = {"snapshot": snapshot.to_dict()}
    if previous is not None:
        output["changes_since_last_record"] = diff_snapshots(previous, snapshot.body)
    if args.append:
        output["record_digest"] = ledger.append(snapshot)["record_digest"]
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAPABILITIES",
    "GENESIS_DIGEST",
    "LEDGER_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "CapabilitySpec",
    "SelfModelLedger",
    "SelfModelSnapshot",
    "build_self_model",
    "default_ledger_path",
    "diff_snapshots",
]
