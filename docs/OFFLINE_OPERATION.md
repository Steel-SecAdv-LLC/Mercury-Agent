<!--
Copyright (C) 2025 Steel Security Advisors LLC
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Online and Offline Operation

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-19.

Mercury Agent runs in both connectivity modes. **Detection itself is fully
local in either mode**: the engine, all detectors, fusion, the
decision/abstention layer, the multi-agent orchestrator, the ethical gates,
and the PQC backend (native AMA build) make no network calls at inference
time. Connectivity only matters for *acquiring data* and for explicitly
optional integrations.

## The two modes

| | Online (default) | Offline / air-gapped (`MERCURY_OFFLINE=1`) |
|---|---|---|
| Dataset fetches (ADBench, AD-Repository, domain loaders) | Downloaded on first use, cached under `MERCURY_DATA_DIR` | Served from the cache; **uncached fetches refuse fail-closed** (`OfflineModeError`) — never stale or fabricated data |
| Engine inference (`detect_with_fusion`, orchestrator, decision layer) | Local | Local — unchanged |
| Reasoning backend / LLM (`reasoning/`, `FallbackLLMChain`) | **Local by default** (Ollama → built-in template); reaches a cloud provider only if you explicitly configure one | **Cloud adapters are never constructed or called** — enforced in `FallbackLLMChain._initialize_chain`/`_create_cloud_adapter` and the reasoning router; local + template only |
| On-box model (Ollama at `127.0.0.1:11434`) and local sidecars (Redis) | Reachable | **Still reachable** — loopback is explicitly permitted so a local model keeps executing actions air-gapped |
| Live data-source integrations (`data_sources/`) | Reach their upstreams | Refused before the socket at the source's own transport gate (`data_sources/base.py::_http_get`/`_http_get_sync`) |
| Web/API enrichment via the shared egress client (`narrative/`, `integrations/`, medical/geological loaders) | Reach their upstreams | Refused before DNS or socket at the `SafeHTTPClient` egress gate — every non-loopback destination raises `OfflineModeError` |

The switch is enforced at **every outbound egress point**, each failing closed
before a socket is opened, so no single layer can leak past it:

- **Dataset loaders** — `datasets/base.py::http_get_with_retry`, the single
  chokepoint every dataset loader inherits.
- **Live data-source transport** — `data_sources/base.py::_http_get` /
  `_http_get_sync`, the async/sync httpx paths those integrations use directly.
- **Shared egress client** — `security/safe_http.py::SafeHTTPClient.validate_url`,
  which gates all `SafeHTTPClient` traffic (narrative retrieval, integrations,
  medical/geological loaders, the Ollama adapter). This gate refuses **before
  any DNS resolution**, so the guarantee holds even where no resolver is
  reachable, and it permits **loopback targets only** — a local Ollama model on
  `127.0.0.1` or a local sidecar keeps working while all external egress is cut.
  A **VPC-air-gap** opt-in (`MERCURY_OFFLINE_ALLOW_PRIVATE=1`, see below) extends
  this to on-prem RFC1918 hosts for callers that pass `allow_private=True`
  (SearXNG, an on-prem Ollama), while still refusing every **public** and IMDS
  destination — the air-gap from the public internet is preserved.
- **Reasoning / LLM layer** — `FallbackLLMChain` and the reasoning router never
  construct or call a cloud adapter when the switch is set; local + template only.

**Local-first is the baseline, not a consequence of the flag.** With
`MERCURY_OFFLINE` unset, the reasoning backend still runs local + template only
and reaches no network — a cloud provider is used solely when the operator
explicitly sets its key. The flag is the *hard* air-gap on top of that default.
Where multiple models are registered, `LLMModelRegistry.select()` orders
free/local ahead of paid cloud (local inference is cost-0), so the cheap, local
path is preferred by default.

## Environment variables

- `MERCURY_OFFLINE=1` — air-gapped mode (accepts `1/true/yes/on`; read at
  call time, never at import).
- `MERCURY_OFFLINE_ALLOW_PRIVATE=1` — **VPC-air-gap** opt-in (same truthy
  parsing). Only meaningful *together with* `MERCURY_OFFLINE` and a caller that
  passes `allow_private=True`. It permits reaching **on-prem RFC1918 / IPv6-ULA**
  services (an Ollama model or SearXNG inside the operator's VPC) while the
  air-gap from the **public internet** still holds: any public resolution is
  refused as egress, and the cloud metadata service (IMDS `169.254.169.254`),
  loopback abuse, multicast, reserved, and CGNAT ranges are **never** unlocked.
  On its own (without `MERCURY_OFFLINE`) it is a no-op. For an on-prem Ollama,
  set it alongside `MERCURY_OLLAMA_HOST` / `MERCURY_MODEL_ENDPOINT` pointing at
  the VPC host; a loopback host always stays loopback-only regardless.
- `MERCURY_DATA_DIR` — stable root for downloaded datasets (default
  `./data`). Set this in production so the cache survives working-directory
  changes.
- `MERCURY_CACHE_DIR` — root for processed/derived caches (default
  `./cache`).

## Priming the cache for an air-gapped deployment

While online, on the same `MERCURY_DATA_DIR` the deployment will use:

```bash
# The five benchmark datasets:
python scripts/prefetch_datasets.py --adbench cardio thyroid breastw WBC Pima

# Or the full 47-dataset ADBench Classical catalog:
python scripts/prefetch_datasets.py --adbench-all
```

Then enable offline mode:

```bash
export MERCURY_OFFLINE=1
python -m benchmarks.orchestration_validation   # runs entirely from cache
```

The prefetch script exits non-zero if any requested dataset failed to cache
— a partial prime is reported, never silently accepted.

## Contract tests

Two suites pin the whole contract:

- `tests/datasets/test_offline_mode.py` — the dataset layer: truthy-flag
  parsing, refusal-before-socket at the dataset chokepoint, cached-data service
  under `MERCURY_OFFLINE=1`, fail-closed errors (with remediation hints) for
  uncached data, and the env-aware directory defaults.
- `tests/security/test_offline_egress_gate.py` — every other egress point: the
  `SafeHTTPClient` gate refuses an external host with **zero** DNS resolution,
  permits loopback IP literals with no resolution and `localhost`, leaves the
  online path unaffected, and refuses even an allowlisted external host under
  the switch; the live data-source httpx transport (sync and async) refuses
  before the socket; and `WebSearchRetriever` honors the master switch even when
  constructed with `offline_mode=False`.
