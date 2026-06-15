<!--
Copyright (C) 2025 Steel Security Advisors LLC
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Online and Offline Operation

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
| Live data-source integrations (`data_sources/`) | Reach their upstreams | Refused at the shared HTTP chokepoint |

The switch is enforced at the dataset layer's **single network chokepoint**
(`datasets/base.py::http_get_with_retry`), before any socket is opened, so
every loader inherits the contract at once. The same `MERCURY_OFFLINE` switch
also governs the reasoning/LLM layer: when set, `FallbackLLMChain` and the
reasoning router never construct or call a cloud adapter.

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

`tests/datasets/test_offline_mode.py` pins the whole contract: truthy-flag
parsing, refusal-before-socket at the chokepoint, cached-data service under
`MERCURY_OFFLINE=1`, fail-closed errors (with remediation hints) for
uncached data, and the env-aware directory defaults.
