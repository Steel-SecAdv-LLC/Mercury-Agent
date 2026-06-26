# Mercury-Agent Deployment Guide

Applies to Mercury Agent **v2.0.x**. Last updated: 2026-06-10.

This guide covers deploying Mercury-Agent from a local Docker environment through
production Kubernetes/Helm. It documents every required configuration value, the
expected startup sequence, and the most common operational concerns.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (Docker Compose)](#quick-start-docker-compose)
3. [Production Docker](#production-docker)
4. [Kubernetes / Helm](#kubernetes--helm)
5. [Required Environment Variables](#required-environment-variables)
6. [Optional Environment Variables](#optional-environment-variables)
7. [Health Checks](#health-checks)
8. [Monitoring](#monitoring)
9. [Secrets Management](#secrets-management)
10. [Upgrade Procedure](#upgrade-procedure)
11. [Rollback Procedure](#rollback-procedure)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Tool | Minimum Version | Notes |
|------|----------------|-------|
| Python | 3.11 | 3.12 recommended |
| Docker | 24.0 | multi-stage build required |
| docker compose | 2.20 | V2 plugin (`docker compose`, not `docker-compose`) |
| kubectl | 1.28 | for Kubernetes deployments |
| Helm | 3.12 | for Helm deployments |

---

## Quick Start (Docker Compose)

For local development and evaluation use `docker-compose.yml` at the repo root:

```bash
# 1. Copy and fill in secrets
cp .env.example .env
# Edit .env — at minimum set JWT_SECRET_KEY

# 2. Start all services
docker compose up -d

# 3. Verify API is healthy
curl http://localhost:8000/health

# 4. Tail logs
docker compose logs -f mercury-agent
```

Services started by docker-compose:

| Service | Port | Purpose |
|---------|------|---------|
| `mercury-agent` | 8000 | REST API |
| `prometheus` | 9090 | Metrics scrape |
| `grafana` | 3000 | Dashboards (admin/admin) |

---

## Production Docker

### Build

```bash
docker build \
  --build-arg USERNAME=mercuryagent \
  --build-arg USER_UID=1000 \
  -t mercury-agent:latest .
```

The Dockerfile uses a multi-stage build:
- **Stage 1 (builder):** installs all dependencies into `/opt/venv`
- **Stage 2 (runtime):** copies only the venv + source, runs as non-root UID 1000

### Run

```bash
docker run -d \
  --name mercury-agent \
  -p 8000:8000 \
  --env-file .env \
  -e MERCURY_AGENT_ENV=production \
  mercury-agent:latest
```

---

## Kubernetes / Helm

### Install

```bash
helm install mercury-agent ./helm/mercury-agent \
  --namespace mercury \
  --create-namespace \
  -f helm/mercury-agent/values.yaml \
  --set config.secrets.JWT_SECRET_KEY="$(openssl rand -hex 32)" \
  --set config.secrets.API_KEY_HASH_SALT="$(openssl rand -hex 32)" \
  --set config.secrets.MERCURY_CACHE_SECRET="$(openssl rand -hex 32)"
```

### Upgrade

```bash
helm upgrade mercury-agent ./helm/mercury-agent \
  --namespace mercury \
  -f helm/mercury-agent/values.yaml
```

### Key Helm values

| Value | Default | Description |
|-------|---------|-------------|
| `replicaCount` | 3 | Number of API pods |
| `image.tag` | `latest` | Docker image tag |
| `resources.requests.cpu` | `500m` | CPU request |
| `resources.requests.memory` | `1Gi` | Memory request |
| `autoscaling.enabled` | `true` | HPA enabled |
| `autoscaling.minReplicas` | 3 | Minimum pods |
| `autoscaling.maxReplicas` | 20 | Maximum pods |
| `config.secrets.JWT_SECRET_KEY` | *(required)* | JWT signing key (or AMA-derived in production) |
| `config.secrets.API_KEY_HASH_SALT` | *(required in production)* | API-key hashing salt |
| `config.secrets.MERCURY_CACHE_SECRET` | *(recommended)* | Cache entry HMAC signing key |

The Helm chart configures liveness/readiness probes, PodDisruptionBudget, anti-affinity,
and topology spread constraints automatically.

### Deployment topology: API tier + streaming-worker (engine) tier

Mercury deploys as two cooperating tiers:

| Tier | Command | Purpose |
|------|---------|---------|
| **API** | `python -m uvicorn omni_mercury_engine.api.server:app` (the image default) | Stateless REST surface for synchronous detection, batch, model and export endpoints. Scales on request load. |
| **Engine / streaming worker** | `python -m omni_mercury_engine.cli stream` | Long-running worker that consumes data points from a stream, runs anomaly detection, and publishes results. Scales on backlog. |

The `stream` worker wires the production `StreamingAnomalyPipeline` (back-pressure
handling, circuit breaker, throughput/latency observability) to a configurable
backend and runs until it receives `SIGTERM`/`SIGINT`, then drains and shuts down
cleanly. It serves its live stats as Prometheus exposition on `:9090/metrics`
(the port the engine deployment's `prometheus.io/path` annotation scrapes).

`STREAMING_BACKEND` defaults to the in-process `memory` backend, so the base
manifests and the default Helm values run with **no external broker**. The
`k8s/overlays/distributed` overlay provisions Kafka + Redis and patches
`STREAMING_BACKEND=kafka` (plus `KAFKA_BOOTSTRAP_SERVERS` / `REDIS_URL`) onto the
engine deployment, and adds a dedicated, horizontally-autoscaled
`mercury-streaming-worker` fleet running the same `stream` command. To enable
streaming under Helm, set `STREAMING_BACKEND=kafka` in `config.app` and point
`KAFKA_BOOTSTRAP_SERVERS` / `REDIS_URL` at your broker.

```bash
# Run a streaming worker locally (in-memory backend, dev smoke test)
mercury stream

# Against a Kafka broker
mercury stream --backend kafka \
  --input-topic mercury-detections \
  --output-topic mercury-anomalies \
  --consumer-group mercury-streaming-workers
```

---

## Required Environment Variables

| Variable | Description | Generate with |
|----------|-------------|---------------|
| `JWT_SECRET_KEY` | Shared JWT signing key for API auth | `openssl rand -hex 32` |

`JWT_SECRET_KEY` resolution matches `api/auth.py::JWTAuth`: an explicit
`secret_key` argument wins, then the environment variable. Production
mode is decided by `api/auth.py::_is_production_env`, aligned with
`api/server.py`: the canonical `MERCURY_ENV` wins whenever set (unknown
values raise), and the legacy `MERCURY_AGENT_ENV` / `ENV` /
`ENVIRONMENT` aliases apply only when it is unset. In production with
no key set, the signing key is **derived via AMA HD Key Management**
(`get_auth_key_manager()`, purpose `jwt_sign`) and startup only fails
if that derivation fails; in development the insecure dev fallback key
is used with a warning.

For multi-worker / multi-replica deployments, set **either**
`JWT_SECRET_KEY` **or** `AMA_MASTER_SEED` (hex, `openssl rand -hex 64`).
The HD master seed is sourced from `AMA_MASTER_SEED` — with it, every
process derives identical `jwt_sign` material and HD-derived tokens
verify fleet-wide. Without either variable, production derives a
**per-process** key (each worker/replica/restart gets a different one;
tokens issued by one process will not verify on another) and logs a
warning naming the hazard. Locked by
`tests/security/test_jwt_auth.py::TestAMAMasterSeed`.

### Additional production-only requirements

In production mode (canonical `MERCURY_ENV=production`, or the legacy
`MERCURY_AGENT_ENV` alias when `MERCURY_ENV` is unset) the following is
also required; the application raises `ValueError` on the first API-key
hash if it is absent (`api/auth.py::APIKeyStore.hash_key`):

| Variable | Description | Generate with |
|----------|-------------|---------------|
| `API_KEY_HASH_SALT` | Salt for API key hashing (PBKDF2-HMAC-SHA256, 260,000 iterations) | `openssl rand -hex 32` |

The Helm chart additionally provisions a `MERCURY_CACHE_SECRET` pod secret
(`config.secrets.MERCURY_CACHE_SECRET`), consumed by
`integrations/stubs/cache.py::RedisCache`: when set, every Redis cache entry
is HMAC-SHA256-signed on write and verified on read, and a tampered,
unsigned, or foreign-keyed entry raises `CacheIntegrityError` instead of
being served. The same secret must be configured on every process sharing
the Redis instance. Unset, the cache stores plain JSON (no signing). Locked
by `tests/integrations/test_cache_hmac.py::TestRedisCacheHMAC`.

### v1.7 production-mode primitives

These two environment variables harden production deployments
independently of `MERCURY_AGENT_ENV` (which is the legacy API-server flag).
Set **both** in production:

| Variable | Default | Effect |
|----------|---------|--------|
| `MERCURY_ENV` | `development` | When `production`, every collaborator with a mock/stub fallback (currently `narrative.voice.MercuryVoice`, more to come) hard-fails with `MercuryProductionConfigError` rather than silently degrading. Unknown values (e.g. `prod`) also raise — typos must be loud. Locked by `tests/test_env.py`. |
| `AMA_MASTER_SEED` | unset | Hex-encoded AMA HD Key Management master seed (`openssl rand -hex 64`; ≥ 32 decoded bytes enforced, malformed values raise). When set, HD-derived keys (JWT signing, API key, audit signing) are deterministic fleet-wide. Locked by `tests/security/test_jwt_auth.py::TestAMAMasterSeed`. |
| `AMA_REQUIRE_REAL_PQC` | unset | **No-op compatibility diagnostic.** The import-time PQC gate (`omni_mercury_engine._pqc_gate._enforce_pqc_production_gate`) is unconditional: `import omni_mercury_engine` raises `RuntimeError` whenever AMA Cryptography's native library is not loadable, regardless of this variable (pinned by `tests/test_pqc_startup_gate.py`). Setting it `true` keeps legacy workflows readable. |
| `AMA_REQUIRE_CONSTANT_TIME` | unset | Recommended in production. Asserts the AMA Cryptography native library exposes its constant-time path. |

`MERCURY_ENV` is consumed via the shared
`omni_mercury_engine._env.{get_mercury_env, is_production,
require_real_component, MercuryProductionConfigError}` helpers — new
modules should adopt these rather than reading `os.environ` directly.
See [`MIGRATION-1.6-to-1.7.md`](MIGRATION-1.6-to-1.7.md) §3.

---

## Optional Environment Variables

### API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `MERCURY_AGENT_ENV` | `development` | Set to `production` for strict mode |
| `OMNI_API_HOST` | `0.0.0.0` | Bind address |
| `OMNI_API_PORT` | `8000` | Listen port |
| `OMNI_API_WORKERS` | `4` | Uvicorn worker count |

### Streaming Worker (`mercury stream`)

These configure the engine / streaming-worker tier. CLI flags
(`--backend`, `--input-topic`, `--output-topic`, `--consumer-group`,
`--metrics-port`) override the corresponding variable.

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMING_BACKEND` | `memory` | `kafka`, `redis`, or in-process `memory` |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka brokers (when `STREAMING_BACKEND=kafka`) |
| `REDIS_URL` | `redis://localhost:6379` | Redis URL (when `STREAMING_BACKEND=redis`) |
| `MERCURY_STREAM_INPUT_TOPIC` | `mercury-detections` | Topic/stream consumed for input data |
| `MERCURY_STREAM_OUTPUT_TOPIC` | `mercury-anomalies` | Topic/stream anomaly results are published to |
| `MERCURY_STREAM_CONSUMER_GROUP` | `mercury-streaming-workers` | Consumer group for load-balanced consumption |
| `MERCURY_METRICS_PORT` | `9090` | Port for the worker's Prometheus `/metrics` endpoint (`0` disables) |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `OMNI_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `OMNI_LOG_FORMAT` | `json` | `json` (structured) or `text` (human-readable) |

### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `OMNI_RATE_LIMIT_ENABLED` | `true` | Enable per-IP rate limiting |
| `OMNI_RATE_LIMIT_REQUESTS_PER_MINUTE` | `100` | Steady-state limit |
| `OMNI_RATE_LIMIT_BURST` | `20` | Burst allowance |

### ML / Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `MERCURY_CACHE_DIR` | `~/.mercury/cache` | On-disk loader cache path (`loaders/base.py`) |
| `TORCH_HOME` | `/app/models` | PyTorch model cache (consumed by torch) |
| `OMP_NUM_THREADS` | `4` | OpenMP threads for NumPy/SciPy |

ML availability is determined by the installed extras (`pip install
mercury-agent[ml]`), not by an environment toggle, and the PQC backend is
mandatory at import — there is no `OMNI_ML_ENABLED` or
`OMNI_QUANTUM_ENABLED` switch in the application.

### Database (optional persistence)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | *(none)* | e.g. `postgresql://user:pass@host:5432/db` |

---

## Health Checks

### REST endpoint

```
GET /health
```

Returns `200 OK` when the service is ready to handle requests. Used by
Kubernetes readiness and liveness probes.

### Docker HEALTHCHECK

The Dockerfile defines:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import omni_mercury_engine; print('healthy')" || exit 1
```

### CLI check

```bash
python -c "import omni_mercury_engine; print('OK')"
```

---

## Monitoring

Prometheus metrics are served on the API port at `/metrics` (the engine /
streaming-worker tier serves its own metrics on `:9090`, see the streaming
section above). The `/metrics` endpoint merges the health gauges
(`omni_mercury_up`, component status/latency) with the `prometheus_client`
default registry, so a single scrape target covers both. This requires the
`prometheus-client` dependency from the `[api]` extra (installed by `[all]`);
without it the application metrics no-op and `/metrics` serves only the health
gauges.

Emitted by the API on every request (via `CorrelationIDMiddleware`):
`http_requests_total{method,endpoint,status}` and
`http_request_duration_seconds{method,endpoint}` — the series the
`prometheus-rules.yaml` recording/alerting rules and the API HPA custom metrics
consume. The detection/model `omni_*` metrics are defined in
`core.metrics` and surface on `/metrics` as the code paths that record them run.

### Grafana dashboards

Dashboards are auto-provisioned from `monitoring/grafana/` when the Grafana
container starts. After running `docker compose up -d grafana` the
Mercury-Agent Overview dashboard is immediately available at
`http://localhost:3000` (default credentials: admin / admin).

### AlertManager

AlertManager rules are in `monitoring/alertmanager/`. Configure receivers
(Slack, PagerDuty, email) in `monitoring/alertmanager/alertmanager-config.yaml`.

### Key metrics to watch

Recording/alerting rules live in `monitoring/prometheus/prometheus-rules.yaml`; the gate-level snapshot metrics come from `python -m omni_mercury_engine.tools.prometheus_metrics_exporter`.

| Metric | Alert threshold | Notes |
|--------|----------------|-------|
| `omni_detection_requests_total` | — | Detection throughput (rules file) |
| `http_request_duration_seconds` | p99 > 2 s | API latency histogram (rules file) |
| `http_requests_total{status=~"5.."}` ratio | sustained > 0 | API error rate (rules file) |
| `mercury_gate_fires_total` | unexpected drop to 0 | Ethical-gate activity (exporter tool) |
| `mercury_benevolence_floor` / `mercury_sigma_band_*` | drift from configured values | Gate configuration drift (exporter tool) |
| `process_resident_memory_bytes` | > 4 GiB | Memory leak indicator |

---

## Secrets Management

Secrets should never be committed to source control. Recommended approaches:

### Kubernetes Secrets (built-in)

```bash
kubectl create secret generic mercury-agent-secrets \
  --namespace mercury \
  --from-literal=JWT_SECRET_KEY="$(openssl rand -hex 32)" \
  --from-literal=MERCURY_CACHE_SECRET="$(openssl rand -hex 32)" \
  --from-literal=API_KEY_HASH_SALT="$(openssl rand -hex 32)"
```

### External Secrets Operator (manual, for production)

The Helm chart renders a plain `Secret` from `config.secrets.*`; it does **not**
wire External Secrets Operator (ESO) through values. To source secrets from a
vault, apply your own `ExternalSecret` targeting the same `mercury-agent` Secret
the Deployment reads (via `envFrom`), so the operator populates `API_SECRET_KEY`
/ `JWT_SECRET_KEY`. A commented starting template ships in
[`k8s/base/secret.yaml`](../k8s/base/secret.yaml):

```yaml
apiVersion: external-secrets.io/v1
kind: ExternalSecret
metadata:
  name: mercury-agent
spec:
  secretStoreRef:
    name: cluster-secret-store
    kind: ClusterSecretStore
  target:
    name: mercury-agent          # the Secret the Deployment reads via envFrom
  data:
    - secretKey: JWT_SECRET_KEY
      remoteRef: { key: mercury-agent/prod, property: jwt_secret_key }
    - secretKey: API_SECRET_KEY
      remoteRef: { key: mercury-agent/prod, property: api_secret_key }
```

### Rotation

There is currently **no automated secret rotation**. Manual rotation steps:

1. Generate new secret values
2. Update the Kubernetes Secret or ESO reference
3. Perform a rolling restart: `kubectl rollout restart deployment/mercury-agent -n mercury`
4. Verify health: `kubectl rollout status deployment/mercury-agent -n mercury`

---

## Upgrade Procedure

```bash
# 1. Pull latest image (or build locally)
docker pull ghcr.io/steel-secadv-llc/mercury-agent:latest

# 2. For Helm deployments — upgrade in place (rolling update)
helm upgrade mercury-agent ./helm/mercury-agent \
  --namespace mercury \
  --set image.tag=<new-tag>

# 3. Monitor rollout
kubectl rollout status deployment/mercury-agent -n mercury

# 4. Smoke test
curl https://<ingress-host>/health
```

---

## Rollback Procedure

### Helm rollback

```bash
# List revision history
helm history mercury-agent --namespace mercury

# Roll back to previous revision
helm rollback mercury-agent --namespace mercury

# Roll back to a specific revision
helm rollback mercury-agent 3 --namespace mercury
```

### Docker rollback

```bash
docker stop mercury-agent
docker run -d --name mercury-agent \
  -p 8000:8000 \
  --env-file .env \
  mercury-agent:<previous-tag>
```

---

## Troubleshooting

### Service fails to start

**Symptom:** Container exits immediately with error.

1. Check `MERCURY_AGENT_ENV` is set as intended. `JWT_SECRET_KEY` is required only when AMA HD key derivation is unavailable or failing — in production `JWTAuth` derives the signing key from AMA HD Key Management when the env var is unset, and the startup error message includes the `HD derivation error:` cause when that path fails. Still set it explicitly for multi-worker / multi-replica deployments (see "Required Environment Variables" above).
2. In production mode, verify `API_KEY_HASH_SALT` is set (enforced at first API-key hash)
3. Check for import errors: `docker run --rm mercury-agent:latest python -c "import omni_mercury_engine"` — a missing AMA Cryptography native build raises `RuntimeError` here

### LLM narration unavailable

**Symptom:** `MercuryProductionConfigError` at `MercuryVoice(enable_llm=True)` in production, or a development-mode warning that the voice path fell back to template narration.

- As of v1.7.0 there is no silent `MockLLMAdapter` fallback: `enable_llm=True` requires an explicit `llm_provider=` naming an implemented provider (`huggingface`, `ollama`, `openai`, `anthropic`, `xai`, `gemini`, `cohere`, `deepseek`, `cursor`, or `template`)
- HuggingFace providers additionally require `llm_model_name=`; remote HuggingFace IDs require `llm_revision=<40-char SHA>`
- Verify optional model dependencies are installed: `pip install -e ".[llm]"`
- Contract reference: [`MIGRATION-1.6-to-1.7.md`](MIGRATION-1.6-to-1.7.md) §4; locked by `tests/narrative/test_voice_llm.py`

### High memory usage

**Symptom:** Pod OOMKilled or memory > 4 GiB.

1. Reduce `OMP_NUM_THREADS` (default 4)
2. Reduce `OMNI_API_WORKERS` (default 4)
3. Enable model offloading: set `TORCH_HOME` to a persistent volume
4. Check for detector memory leaks in `/metrics`

### Conformal prediction skipped (WARNING in logs)

**Symptom:** `Conformal prediction skipped: ... — confidence_intervals will be None`

- The conformal predictor was not fitted or encountered an error
- Call `fit()` on the `GOSNNIntegration` instance with labelled validation data
  before running inference
- confidence_intervals will be absent from detection results until fitted

### Ethics audit failures in CI

**Symptom:** CI ethics-audit job shows `FAIL`.

> The CI ethics-audit gate is **non-advisory** as of Wave B (PR #179).
> A failing run blocks merge; the dual hard gates (Benevolence,
> σ_Immutable) at every public boundary surface raise
> `EthicalConstraintViolationError(check=…)` rather than logging and
> continuing. See the top-level [`ARCHITECTURE.md`](https://github.com/Steel-SecAdv-LLC/Mercury-Agent/blob/main/ARCHITECTURE.md) §"Dual-Gate Hard Ethical
> Enforcement".

- Review the specific test(s) that failed in the CI log
- `T3` failures: the `PreExecutionBlockingGate` pattern list does not cover the flagged payload — extend the pattern list
- `T4` failures: recalibrate the `EthicalAutonomyGovernor` scoring thresholds against the failing fixtures
- `T5` failures: `ethical_compliance_threshold` immutability guard is broken — do not deploy
- `check="sigma_immutable"` failures: this code is raised by
  `SigmaImmutableGate.enforce()` for **two** distinct cases — distinguish
  them before remediating.
  1. **Corpus verification failed** (signed-weights or vocab mismatch):
     regenerate the signed σ corpus with
     `python scripts/train_sigma_immutable.py` after confirming the input
     data lineage. Look for "signature mismatch" / "vocab digest" /
     "weights load" in the error context to confirm this case.
  2. **Score below threshold** (the model is alive and signed, but the
     input scored below the immutability bar): this is a **model /
     threshold regression**, not a corpus-corruption failure. Investigate
     the prompt or detector regression in the upstream pipeline; do
     **not** regenerate the corpus, since that will not change the
     scoring behaviour and may mask the real defect.
  Never bypass either case by setting `_GOSNN_TESTING_BYPASS` outside
  tests.
- `check="gosnn_unavailable"` failures: GOSNN model artefacts are missing or
  the load path is failing — investigate the model registry; do not ship a
  fallback that returns predictions without GOSNN

### Coverage below target

The repository targets 85% test coverage (`pyproject.toml [tool.coverage.report] fail_under = 85`); CI enforces a measured floor of 50% on the full suite (`COVERAGE_THRESHOLD_FULL=50`) and 25% on the curated core lane (`COVERAGE_THRESHOLD_CORE=25`), per `.github/workflows/ci.yml`.
To run coverage locally:

```bash
pytest --cov=omni_mercury_engine --cov-report=term-missing tests/
```

To enforce the 85% target locally:

```bash
pytest --cov=omni_mercury_engine --cov-fail-under=85 tests/
```
