# Runbook: OmniMercuryApiRouterUnavailable

**Fires when** `max by (router) (omni_mercury_api_router_state{state="unavailable"}) == 1`
for 2 minutes.

## What the series means

`api/server.py` attaches each API surface inside a guard that tolerates an
`ImportError`, so a missing optional dependency removes one surface instead of
stopping the process. `api/router_manifest.py` records the outcome of every
attachment and `/metrics` exports it as a state set: one series per router and
state, valued `1` on the series naming the router's current state.

| `state` | Meaning | Action |
|---|---|---|
| `mounted` | The router imported and its routes are served. | None. |
| `disabled` | Configuration left it off (`frontend`, `decoys` are opt-in). | None; this is the operator's choice. |
| `unavailable` | The import failed. Its routes answer 404. | This runbook. |

`/health` is a liveness probe and reports `healthy` in all three cases; it is
pinned to carry only `status` and `version`
(`tests/api/test_health_minimal.py`). This alert is the signal for a server that
is up with part of itself missing.

## First checks

1. Which router: `omni_mercury_api_router_state{state="unavailable"} == 1`.
   The `router` label is one of `batch`, `models`, `export`, `detection`,
   `accounts`, `hazard`, `assessment`, `agent_notice`, `source`, `catalog`,
   `frontend`, `voice`, `metrics`, `decoys`.
2. Why: the reason is in the startup log, not in a metric label (it can hold a
   filesystem path). Search the pod's log from its start for
   `not available:`; the line names the surface and the exception, for example
   `Hazard visualization routes not available: No module named 'matplotlib'`.
3. Whether every replica is affected: an image built without an extra fails
   the same way on all replicas; one replica alone points at that pod's
   filesystem or a partial rollout.

## Fix

- **A missing optional dependency.** Install the extra the surface needs and
  roll the deployment. The Docker image installs `.[all]`; a bare-metal install
  with a narrower extra set is the usual cause.
- **A broken module.** An `ImportError` raised from inside the router's own
  import chain is a defect: roll back to the previous image and file it.

## What does not reach this alert

Two faults stop the process instead of dropping a router, so they show up as
`OmniMercuryServiceDown`, not here:

- **No AMA backend.** The API module runs the fail-closed PQC gate at import
  (`omni_mercury_engine/_pqc_gate.py`); a server without AMA v5.0.0 and all
  three algorithms does not start.
- **The frontend without the account API.** With
  `MERCURY_FRONTEND_ENABLED=true` and the `accounts` router unavailable, the
  server refuses to start rather than serve sign-in pages whose every request
  answers 404.
