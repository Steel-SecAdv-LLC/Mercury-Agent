# Mercury-Agent Deployment Guide

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-24.

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

The chart declares no subchart dependencies, so this needs neither
`helm dependency build` nor network access to a chart repository.

```bash
helm install mercury-agent ./helm/mercury-agent \
  --namespace mercury \
  --create-namespace \
  -f helm/mercury-agent/values.yaml \
  --set config.secrets.JWT_SECRET_KEY="$(openssl rand -hex 32)" \
  --set config.secrets.API_KEY_HASH_SALT="$(openssl rand -hex 32)" \
  --set config.secrets.MERCURY_CACHE_SECRET="$(openssl rand -hex 32)"
```

For production use `values-production.yaml`, which sets
`config.app.MERCURY_ENV=production`. That is what arms the fail-closed
guards, and it makes two of the values above mandatory rather than advisory:

* `API_KEY_HASH_SALT` — without it `APIKeyStore.hash_key` raises instead of
  hashing every API key under the salt built into the source.
* `JWT_SECRET_KEY` — must be real key material. An empty value is treated as
  unset, and a placeholder or low-entropy value is refused outright. Omit it
  entirely to derive the signing key through AMA HD key management, in which
  case set `AMA_MASTER_SEED` (`openssl rand -hex 64`) so the derivation is
  deterministic across replicas.

The production values leave `api.ingress.enabled=false`: an externally
reachable Ingress needs a hostname the chart cannot know. Supply one to
enable it —

```bash
helm install mercury-agent ./helm/mercury-agent \
  --namespace mercury --create-namespace \
  -f helm/mercury-agent/values-production.yaml \
  --set config.secrets.JWT_SECRET_KEY="$(openssl rand -hex 32)" \
  --set config.secrets.API_KEY_HASH_SALT="$(openssl rand -hex 32)" \
  --set api.ingress.enabled=true \
  --set-json 'api.ingress.hosts=[{"host":"api.your-domain.example","paths":[{"path":"/","pathType":"Prefix"}]}]' \
  --set-json 'api.ingress.tls=[{"hosts":["api.your-domain.example"],"secretName":"mercury-agent-prod-tls"}]'
```

Enabling the Ingress without a host is a render-time error, not a silently
rule-less Ingress.

### Shared state across replicas

`config.app.MERCURY_KEYSTORE_PATH` points at `/app/data/mercury.db` on the
ReadWriteMany `persistence.data` claim, so accounts, API keys, sessions, quota
counters and rate-limit buckets are shared by every replica and survive
restarts. The claim's storage class must provide working POSIX advisory locks
and shared-memory mapping — the store runs SQLite in WAL mode. CephFS,
Longhorn RWX and AWS EFS (NFSv4.1) qualify; plain NFSv3 does not.

### Upgrade

```bash
helm upgrade mercury-agent ./helm/mercury-agent \
  --namespace mercury \
  -f helm/mercury-agent/values.yaml
```

### Key Helm values

| Value | Default | Description |
|-------|---------|-------------|
| `api.replicaCount` | 3 | Number of API pods |
| `api.image.tag` | `""` (defaults to `Chart.appVersion`) | Docker image tag |
| `api.resources.requests.cpu` | `500m` | CPU request |
| `api.resources.requests.memory` | `1Gi` | Memory request |
| `api.autoscaling.enabled` | `true` | HPA enabled |
| `api.autoscaling.minReplicas` | 3 | Minimum pods |
| `api.autoscaling.maxReplicas` | 20 | Maximum pods |
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
(the port the engine deployment's `prometheus.io/port` annotation scrapes).

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
mercury-agent stream

# Against a Kafka broker
mercury-agent stream --backend kafka \
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
| `AMA_REQUIRE_REAL_PQC` | unset | **No-op compatibility diagnostic.** The import-time PQC gate (`omni_mercury_engine._pqc_gate._enforce_pqc_production_gate`) is unconditional on the agent surfaces: importing `omni_mercury_engine.engine` / `.crypto` (or touching an AMA-backed `security` export) raises `RuntimeError` whenever AMA Cryptography's native library is not loadable, regardless of this variable (pinned by `tests/test_pqc_startup_gate.py`). Setting it `true` keeps legacy workflows readable. |
| `AMA_REQUIRE_CONSTANT_TIME` | unset | **Superseded compatibility flag with the pinned AMA v5.0.0.** AMA enforces native-only operation unconditionally (INVARIANT-7 revised), so its constant-time C primitives are always in use; setting this variable changes no cryptographic behavior on a healthy install (AMA logs a deprecation warning), and on an install without the native backend it fails closed redundantly with Mercury's mandatory import-time PQC gate. Mercury still reads it for diagnostics — `security.pqc_backends.require_constant_time()`, surfaced by `get_pqc_capabilities()` / `validate_pqc_environment()`. Leave it unset. |

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
| `MERCURY_HOST` | `127.0.0.1` | Bind address for `mercury-agent serve`; set `0.0.0.0` for all interfaces |

Port and worker count are **CLI flags**, not environment variables: `mercury-agent serve --port <n> --workers <n>` (defaults 8000 / 1). The shipped container image bypasses the CLI and serves uvicorn on a fixed `0.0.0.0:8000` via its `CMD` (single process). There are no `OMNI_API_HOST`/`OMNI_API_PORT`/`OMNI_API_WORKERS` hooks in the application.

### Streaming Worker (`mercury-agent stream`)

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
| `MERCURY_METRICS_HOST` | `127.0.0.1` | Bind address for `/metrics`. Loopback by default so bare-host runs never expose it implicitly; the in-repo k8s configmap and Helm values set `0.0.0.0` so in-cluster Prometheus can scrape cross-pod |

### Logging

Log level is a **CLI flag** on the serve command, not an environment variable:
`mercury-agent serve --log-level <debug|info|warning|error>` (default `info`).
Structured-vs-text formatting is selected by the logging setup in
`omni_mercury_engine.utils.logging`; there are no `OMNI_LOG_LEVEL`/
`OMNI_LOG_FORMAT` environment hooks in the application.

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
| `MERCURY_FUSION_CHECKPOINT` | bundled `models/default_fusion.pt` | Fusion checkpoint the API, MCP server and `detect`/`explain` serve; write one with `mercury-agent train -o`. A missing path fails like a missing bundled file, never falling back to it |
| `MERCURY_DRIFT_DETECTION` | on | `0`/`false`/`off` disables input-drift detection (see `docs/runbooks/data-drift.md`) |
| `MERCURY_ADJUDICATION_PATH` | `~/.mercury/adjudication.jsonl` | Held-out fusion-label ledger (`mercury-agent adjudication ...`, `/api/v1/assessment/adjudication/*`) |
| `MERCURY_ADJUDICATION_SIGNING_KEY` | unset | Owner-only ML-DSA-65 key file (`mercury-agent adjudication keygen --out PATH`) used to sign ledger records and exports |
| `MERCURY_ADJUDICATION_PUBLIC_KEY` | unset | Hex ML-DSA-65 public key the ledger and its exports are pinned to on read; unsigned or differently signed records are refused |
| `MERCURY_INTEL_FUSION_CHECKPOINT` | unset | All-source fusion checkpoint for the assessment surfaces; served only if the merit gate admits it against the ledger's held-out labels, else evidence scoring decides |
| `MERCURY_INTEL_LEARN_RELIABILITY` | unset | `1` learns the per-discipline reliabilities from the ledger's training split at startup (installed only when cross-validation beats the declared table; the held-out split stays for the merit gate) |
| `MERCURY_MERIT_ATTEMPT_LOG` | unset | Hash-chained log of merit-gate attempts, so alpha spending survives restarts (unset: counted per engine instance) |

### Human-subject capabilities

| Variable | Default | Description |
|----------|---------|-------------|
| `MERCURY_HUMAN_SUBJECT_CAPABILITIES` | unset (off) | Lawful bases permitted for the Concealed Information Test and contactless vital signs on every surface: `consent`, and/or `vital_interests` (vital signs of people who cannot consent, e.g. survivors under rubble). An unknown value is refused, never guessed |
| `MERCURY_ASSESSMENT_MAX_BODY_BYTES` | `67108864` | Largest declared request body the `/api/v1/assessment` routes accept (413 above it) |

With the capabilities on, every call still needs a stated purpose (refused
when it describes covert or non-consensual sensing, or harm to a person) and,
under `consent`, a `ConsentRecord` for the subject, obtained before the
recording, with the capability's scope and not withdrawn. Each admitted call is
written to the gate audit log with a fingerprint of its consent or attestation.

ML availability is determined by the installed extras (`pip install
mercury-agent-csi[ml]`), not by an environment toggle, and the PQC backend is
mandatory for every agent surface (engine, crypto, AMA-backed security
names) — there is no `OMNI_ML_ENABLED` or `OMNI_QUANTUM_ENABLED` switch in
the application.

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

Distributed tracing: the API speaks W3C Trace Context. A valid incoming
`traceparent` is continued (same trace-id, a new span-id per request; the
response carries the resulting `traceparent`, and `tracestate` is passed
through); otherwise a new trace starts. The trace and span ids are in every
request log line and on SIEM alerts (`trace_id` label), so a gateway trace, a
Mercury log line and a forwarded alert join on one id. Spans are not exported
(that needs an OpenTelemetry SDK, which is not a project dependency).

Emitted by the API on every request, including requests that end in an
unhandled exception (via `CorrelationIDMiddleware`):
`http_requests_total{method,endpoint,status}` and
`http_request_duration_seconds{method,endpoint}`. The recording and alerting
rules and the API HPA's custom metrics consume these series. The engine emits
the detection series on every component call, and the streaming worker emits
them on every message: `omni_detection_{requests,success}_total`,
`omni_detection_duration_seconds`, `omni_model_inference_errors_total` and
`omni_component_declined_total`. A component refusing input of a modality it
does not process counts as *declined*, never as an error. The streaming worker
also publishes `omni_detection_queue_depth`, its consumer-group backlog (`-1`
when the backend cannot report it). `fit_fusion` sets
`omni_model_roc_auc` / `omni_model_f1_score` from the validation split.

Wiring the series to the rules:

| Deployment | Scrape + labels | Rules | Autoscaling |
|---|---|---|---|
| docker compose | `monitoring/prometheus/prometheus.yml` (adds `app`, `service`) | `rules.yml`, rendered from the CRD by `scripts/render_prometheus_rules.py` | — |
| Kubernetes (`k8s/`) | `monitoring/prometheus/podmonitors.yaml` (adds `app`, `service`, `job`) | `prometheus-rules.yaml` (PrometheusRule) | prometheus-adapter with `monitoring/prometheus/prometheus-adapter-rules.yaml` |
| Helm chart | `templates/servicemonitor.yaml` (adds `app`, `service`, `job` from each Service's component label; the API's `metrics` Service port targets its HTTP listener, which serves `/metrics`) | `templates/prometheusrule.yaml`, rendered from the same CRD by `scripts/render_prometheus_rules.py`, plus `metrics.prometheusRule.rules` | prometheus-adapter, as for `k8s/` |

Every rule selects `{app="mercury-agent"}`. A scrape config that does not
attach that label leaves every alert permanently silent. Without the adapter
rules, the HPAs' custom metrics are unavailable and Kubernetes never scales
those deployments down. Each worker reports the backlog group-wide, so the
adapter aggregates it with `max`.

**Audit trail.** Every finished API request by an identified caller (API
key, JWT or browser session; except probes, docs, unmatched paths and 429
rejections) is recorded with:
- the principal and the route template (never the raw path);
- path parameters, status and duration;
- the client IP and the correlation id;
- query parameter *names*, never their values.

Caller-controlled fields are clipped (256 characters for the user agent, path
and resource; 64 per path parameter or query name; at most 16 query names), so
a client cannot inflate entries. Unauthenticated 4xx responses are counted (a
capped tally plus `http_requests_total`) but not written to the audit log:
anonymous traffic must not be able to rotate login and 2FA evidence out of the
retained segments. Login failures are still recorded by the auth auditor.

`GET /api/v1/export/audit-logs` serves these entries (the in-memory store
keeps the newest `MAX_AUDIT_LOG_RECORDS`, 50,000 by default). Every
`/api/v1/export/*` route requires the `export` permission: default API keys
carry `read` + `detect` only, so issue a key with `export` (or use an admin
principal) for compliance exports. CSV exports neutralise cells that a
spreadsheet would run as a formula.

With `MERCURY_AUDIT_LOG_DIR` set, the entries also go to the hash-chained
audit log together with auth events:
- The chain resumes across restarts. A crash that leaves a torn last line is
  repaired on start: only the bytes after the last newline are cut, logged
  with their digest and kept as `audit.jsonl.torn-<usec>`.
- The first process to open the directory writes `audit.jsonl`. Any further
  process, such as extra `--workers`, writes its own `audit-<pid>.jsonl` and
  holds `.audit-<pid>.lock`, so each file is one verifiable chain.
- Rotated segments get microsecond-unique names and are never overwritten.
  Resume and count-based retention only ever touch a writer's own segments;
  the time-based sweep also removes a finished worker's log and old torn
  fragments, never a live writer's active file.
- `mercury-agent platform audit verify` checks every writer's chain
  read-only (no lock taken, safe while the server runs); `--file` checks one
  file. A chain cannot reveal records cut from its very start or end without
  an external anchor.

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
| `omni_detection_success_total` / `omni_detection_requests_total` | < 0.95 for 10 m | Detection success rate; declines excluded (rules file) |
| `omni_model_inference_errors_total` | > 0.1/s for 5 m | Component, fusion-head or stream-detector failures (rules file) |
| `omni_detection_queue_depth` | HPA target 50 | Streaming backlog; `-1` = unknown (engine HPA) |
| `http_request_duration_seconds` | p99 > 2 s | API latency histogram (rules file) |
| `http_requests_total{status=~"5.."}` ratio | sustained > 0 | API error rate (rules file) |
| `mercury_gate_fires_total` | unexpected drop to 0 | Ethical-gate activity (exporter tool) |
| `mercury_benevolence_floor` / `mercury_sigma_band_*` | drift from configured values | Gate configuration drift (exporter tool) |
| `process_resident_memory_bytes` | > 4 GiB | Memory leak indicator |

### SIEM / observability alert egress

Set `MERCURY_ALERT_SINKS` to forward every detection that flagged an anomaly
(fire-and-forget; a sink failure never fails the request) to Splunk HEC,
Datadog, Elastic / OpenSearch, a Prometheus pushgateway, an OTLP collector or
a generic HTTP receiver:

```bash
export SPLUNK_HEC_TOKEN=...   # secrets come from env vars, never the JSON
export MERCURY_ALERT_SINKS='[
  {"name": "soc", "platform": "splunk", "endpoint": "https://splunk:8088",
   "auth_type": "api_key", "credentials_env": {"api_key": "SPLUNK_HEC_TOKEN"},
   "index": "security", "max_tlp": "AMBER+STRICT", "min_score": 0.5}
]'
```

Each alert carries the detection's TLP 2.0 colour (the export API's
classification) and no user identifier. A sink receives only alerts at or
below its `max_tlp` (default `AMBER+STRICT`, so `TLP:RED` is never forwarded
automatically). Under `MERCURY_OFFLINE` non-loopback sinks are refused before
any socket opens. A malformed value is logged once and disables egress.
Sinks never follow HTTP redirects (a redirect could carry the sink's API key
to another host, past the offline gate), and at most
`MERCURY_ALERT_MAX_PENDING` deliveries (default 256) are in flight: beyond
that, new alerts are dropped and counted (`dropped_backpressure` in the egress
stats) rather than queued without bound behind a slow sink.

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
vault, apply your own `ExternalSecret` targeting the same `mercury-agent-secrets`
Secret the Deployment reads (via `envFrom`), so the operator populates `API_SECRET_KEY`
/ `JWT_SECRET_KEY`. [`k8s/base/secret.yaml`](../k8s/base/secret.yaml) ships a fuller
commented ESO template wired for a Vault backend; a minimal `ClusterSecretStore`
example is:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: mercury-agent
spec:
  secretStoreRef:
    name: cluster-secret-store
    kind: ClusterSecretStore
  target:
    name: mercury-agent-secrets  # the Secret the Deployment reads via envFrom
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
  --set api.image.tag=<new-tag> \
  --set engine.image.tag=<new-tag>

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
3. Check for import errors: `docker run --rm mercury-agent:latest python -c "import omni_mercury_engine.engine"` — a missing AMA Cryptography native build raises `RuntimeError` here

### LLM narration unavailable

**Symptom:** `MercuryProductionConfigError` at `MercuryVoice(enable_llm=True)` in production, or a development-mode warning that the voice path fell back to template narration.

- As of v1.7.0 there is no silent `MockLLMAdapter` fallback: `enable_llm=True` requires an explicit `llm_provider=` naming an implemented provider (`huggingface`, `ollama`, `openai`, `anthropic`, `xai`, `gemini`, `cohere`, `deepseek`, `cursor`, or `template`)
- HuggingFace providers additionally require `llm_model_name=`; remote HuggingFace IDs require `llm_revision=<40-char SHA>`
- Verify optional model dependencies are installed: `pip install -e ".[llm]"`
- Contract reference: [`MIGRATION-1.6-to-1.7.md`](MIGRATION-1.6-to-1.7.md) §4; locked by `tests/narrative/test_voice_llm.py`

### High memory usage

**Symptom:** Pod OOMKilled or memory > 4 GiB.

1. Reduce `OMP_NUM_THREADS` (default 4)
2. Reduce the `mercury-agent serve --workers` count (default 1); the shipped
   container image already runs a single uvicorn process
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

The repository targets 85% test coverage (`pyproject.toml [tool.coverage.report] fail_under = 85`); CI enforces a measured floor of 62% on the full suite (`COVERAGE_THRESHOLD_FULL=62`) and 33% on the curated core lane (`COVERAGE_THRESHOLD_CORE=33`), per `.github/workflows/ci.yml`.
To run coverage locally:

```bash
pytest --cov=omni_mercury_engine --cov-report=term-missing tests/
```

To enforce the 85% target locally:

```bash
pytest --cov=omni_mercury_engine --cov-fail-under=85 tests/
```
