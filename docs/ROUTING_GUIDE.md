# Routing Infrastructure Guide

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

Mercury Agent provides a flexible routing infrastructure for request handling, pattern matching, and graceful degradation through fallback chains. This guide covers the core routing components and demonstrates how to integrate them with the detection pipeline.

> **v1.7 production-mode addition.** Production deployments should
> set `MERCURY_ENV=production`. Routes themselves are unchanged, but
> downstream collaborators that previously silently substituted
> mock/stub adapters (e.g. `narrative.voice.MercuryVoice`) now raise
> `MercuryProductionConfigError` rather than degrading silently when
> `MERCURY_ENV=production`. Route handlers should let those
> exceptions propagate; the route layer is not the right place to
> mask a production-mode misconfiguration. See
> [`MIGRATION-1.6-to-1.7.md`](MIGRATION-1.6-to-1.7.md) §3.

> **Hard ethical gates run *inside* the prediction call, not at the
> route layer.** Routes never see, attenuate, or bypass the dual
> ethical contract. Every public boundary surface reached from a
> route — `OmniMercuryEngine.detect_with_fusion`,
> `detect_with_fusion_calibrated`, `CognitiveOrchestrator.analyze`,
> `NeuroSymbolicHub.predict` — runs **Benevolence then σ_Immutable**
> as mandatory hard gates (Wave B, PR #179) and raises
> `EthicalConstraintViolationError(check=…)` on failure.
>
> **Fallback chains do not mask ethical refusals.**
> `FallbackChain.execute()` (`integrations/routing/fallback.py`)
> previously caught generic `Exception` and advanced to the next
> handler unless `fail_fast=True`, which would have allowed an
> `EthicalConstraintViolationError` from a wrapped detection
> handler to be silently substituted with a "default response".
> The chain now special-cases that exception and re-raises it
> unconditionally regardless of `fail_fast`, so an ethical refusal
> propagates out of the chain even when other failure modes are
> still being absorbed. See the top-level
> [`ARCHITECTURE.md`](https://github.com/Steel-SecAdv-LLC/Mercury-Agent/blob/main/ARCHITECTURE.md) §"Dual-Gate Hard
> Ethical Enforcement" and [`MATH_SPEC.md`](MATH_SPEC.md) §2.1.5
> for the full contract.

## Overview

The routing infrastructure consists of two main systems:

1. **RequestRouter**: URL pattern matching with middleware support for HTTP-style request routing
2. **FallbackChain**: Priority-based handler chains for graceful degradation when primary services fail

Both systems are designed to work together, enabling robust request handling with automatic fallback to cached or default responses when external services are unavailable.

**Important fallback-chain semantics.** `FallbackChain.execute()` catches the **generic `Exception`** base class and advances to the next handler unless `fail_fast=True`. The only exception that is special-cased and always re-raised is `EthicalConstraintViolationError` (added in this PR). All other handler exceptions — including application bugs, `KeyError`, `ValueError`, network errors, etc. — are absorbed by the chain and treated as a fallback trigger by default. This is the intended graceful-degradation contract for *connectivity / latency / data-source* failures, but operators should be aware that it will also mask logic bugs in handlers; surface those by setting `fail_fast=True` on the chain or by adding explicit type-narrowing in your handler before raising.

## RequestRouter

The `RequestRouter` class provides URL pattern matching with support for path parameters, HTTP methods, middleware chains, and route groups.

### Basic Usage

```python
from omni_mercury_engine.integrations.routing import RequestRouter, Route, RouteMatch

router = RequestRouter()

@router.get("/api/detectors/{detector_id}")
async def get_detector(request, detector_id):
    """Retrieve detector configuration by ID."""
    return {"detector_id": detector_id, "status": "active"}

@router.post("/api/anomalies")
async def create_anomaly(request):
    """Submit new anomaly for analysis."""
    return {"status": "created", "id": "anomaly-123"}

# Match and dispatch requests
match = router.match("/api/detectors/tornado-001", method="GET")
result = await match.handler(request, **match.params)
```

### Route Parameters

Routes support dynamic path parameters using `{param}` syntax:

```python
@router.get("/api/domains/{domain}/detectors/{detector_id}/results")
async def get_detector_results(request, domain, detector_id):
    """Get results for a specific detector in a domain."""
    return {
        "domain": domain,
        "detector_id": detector_id,
        "results": []
    }

# Matches: /api/domains/geological/detectors/tornado-001/results
# Extracts: domain="geological", detector_id="tornado-001"
```

### HTTP Methods

Register routes for specific HTTP methods:

```python
@router.get("/api/health")
async def health_check(request):
    return {"status": "healthy"}

@router.post("/api/detect")
async def run_detection(request):
    return {"detection_id": "det-456"}

@router.put("/api/config/{key}")
async def update_config(request, key):
    return {"key": key, "updated": True}

@router.delete("/api/cache/{cache_id}")
async def clear_cache(request, cache_id):
    return {"cleared": cache_id}
```

### Middleware

Apply middleware to routes for cross-cutting concerns like authentication and logging:

```python
async def auth_middleware(handler, request, **kwargs):
    """Verify authentication before handler execution."""
    if not request.headers.get("Authorization"):
        raise PermissionError("Authentication required")
    return await handler(request, **kwargs)

async def logging_middleware(handler, request, **kwargs):
    """Log request details."""
    logger.info(f"Request: {request.path}")
    result = await handler(request, **kwargs)
    logger.info(f"Response: {result}")
    return result

# Apply middleware to specific routes
@router.route("/api/admin/settings", methods=["GET", "PUT"], middleware=[auth_middleware])
async def admin_settings(request):
    return {"settings": {}}

# Apply middleware to all routes via router
router = RequestRouter(middleware=[logging_middleware])
```

### Router Groups

Organize related routes with shared prefixes and middleware:

```python
from omni_mercury_engine.integrations.routing.router import RouterGroup

# Create API v1 group
api_v1 = RouterGroup("/api/v1", middleware=[auth_middleware])

@api_v1.get("/detectors")
async def list_detectors_v1(request):
    return {"version": "v1", "detectors": []}

@api_v1.post("/analyze")
async def analyze_v1(request):
    return {"version": "v1", "analysis": {}}

# Create API v2 group with different middleware
api_v2 = RouterGroup("/api/v2", middleware=[auth_middleware, rate_limit_middleware])

@api_v2.get("/detectors")
async def list_detectors_v2(request):
    return {"version": "v2", "detectors": [], "enhanced": True}
```

### Named Routes and URL Generation

Use named routes for reverse URL lookup:

```python
@router.route("/api/anomalies/{anomaly_id}", name="anomaly_detail")
async def get_anomaly(request, anomaly_id):
    return {"id": anomaly_id}

# Generate URL from route name
url = router.url_for("anomaly_detail", anomaly_id="anom-789")
# Result: "/api/anomalies/anom-789"
```

### Route Metrics

Monitor routing performance:

```python
metrics = router.get_metrics()
# Returns:
# {
#     "total_requests": 1250,
#     "routes_count": 15,
#     "route_hits": {
#         "GET:/api/health": 500,
#         "POST:/api/detect": 750
#     }
# }
```

## FallbackChain

The `FallbackChain` class provides graceful degradation by trying handlers in priority order until one succeeds.

### Basic Usage

```python
from omni_mercury_engine.integrations.routing import FallbackChain, FallbackHandler

chain = FallbackChain(name="detector_service")

@chain.handler(priority=0, timeout=5.0)
async def primary_detector(data):
    """Primary detection using live API."""
    return await external_detector_api.analyze(data)

@chain.handler(priority=1, timeout=2.0)
async def cached_result(data):
    """Fallback to cached detection results."""
    return cache.get(f"detection:{data.id}")

@chain.handler(priority=2)
async def default_response(data):
    """Final fallback with safe default."""
    return {
        "status": "degraded",
        "message": "Using default response",
        "anomaly_score": 0.5
    }

# Execute with automatic fallback
result = await chain.execute(sensor_data)

if result.degraded:
    logger.warning(f"Service degraded, used handler: {result.handler_name}")
```

### Conditional Handlers

Execute handlers only when conditions are met:

```python
def has_cache_entry(data):
    """Check if cache has entry for this data."""
    return cache.exists(f"detection:{data.id}")

@chain.handler(priority=1, condition=has_cache_entry)
async def cached_result(data):
    """Only executes if cache entry exists."""
    return cache.get(f"detection:{data.id}")
```

### Error Callbacks

Handle errors from specific handlers:

```python
def on_api_error(error):
    """Log and alert on API failures."""
    logger.error(f"API error: {error}")
    metrics.increment("api_failures")
    alerting.notify("detector_api_down")

@chain.handler(priority=0, timeout=5.0, on_error=on_api_error)
async def primary_detector(data):
    return await external_api.detect(data)
```

### Fail-Fast Mode

Stop immediately on first failure instead of trying fallbacks:

```python
critical_chain = FallbackChain(name="critical_service", fail_fast=True)

@critical_chain.handler(priority=0)
async def critical_handler(data):
    """Must succeed - no fallback allowed."""
    return await critical_service.process(data)
```

### FallbackResult

The `FallbackResult` object provides detailed execution information:

```python
result = await chain.execute(data)

print(f"Value: {result.value}")
print(f"Handler: {result.handler_name}")
print(f"Fallbacks tried: {result.fallback_count}")
print(f"Degraded: {result.degraded}")
print(f"Elapsed: {result.elapsed:.3f}s")

# Examine fallback reasons
for handler_name, reason, message in result.reasons:
    print(f"  {handler_name}: {reason.value} - {message}")
```

### Chain Metrics

Monitor fallback chain performance:

```python
metrics = chain.get_metrics()
# Returns:
# {
#     "name": "detector_service",
#     "execution_count": 1000,
#     "fallback_count": 50,
#     "fallback_rate": 0.05,
#     "handlers": [
#         {"name": "primary_detector", "call_count": 1000, "success_count": 950, ...},
#         {"name": "cached_result", "call_count": 50, "success_count": 45, ...},
#         {"name": "default_response", "call_count": 5, "success_count": 5, ...}
#     ]
# }
```

## Integration with Detection Pipeline

### Routing Detector Requests

Route anomaly detection requests to appropriate detectors based on domain:

```python
from omni_mercury_engine.integrations.routing import RequestRouter, FallbackChain
from omni_mercury_engine.detectors.geological import TornadoDetector, HurricaneDetector

router = RequestRouter(prefix="/api/v1")

# Initialize detectors
tornado_detector = TornadoDetector()
hurricane_detector = HurricaneDetector()

@router.post("/detect/geological/{detector_type}")
async def geological_detection(request, detector_type):
    """Route to appropriate geological detector."""
    data = await request.json()
    
    detectors = {
        "tornado": tornado_detector,
        "hurricane": hurricane_detector,
    }
    
    detector = detectors.get(detector_type)
    if not detector:
        raise ValueError(f"Unknown detector: {detector_type}")
    
    result = detector.predict(data["features"])
    return {
        "detector": detector_type,
        "anomaly_score": result.anomaly_score,
        "prediction": result.prediction
    }
```

### Fallback for External Data Sources

Use fallback chains for resilient data loading:

```python
import logging
from pathlib import Path

import numpy as np

from omni_mercury_engine.integrations.routing import FallbackChain
from omni_mercury_engine.validation.data_loaders import (
    USGSEarthquakeLoader,
    NOAASpaceWeatherLoader,
)

logger = logging.getLogger(__name__)

data_chain = FallbackChain(name="earthquake_data")


def _cache_path(params) -> Path:
    """Single source of truth for the cache path (read and write must agree)."""
    return Path(f"/data/cache/earthquakes_{params.get('days', 30)}d.npz")


@data_chain.handler(priority=0, timeout=30.0)
async def load_from_usgs(params):
    """Primary: load from the USGS API and warm the cache the fallback reads."""
    loader = USGSEarthquakeLoader()
    features, labels, metadata = loader.load(
        use_synthetic=False,
        days_back=params.get("days", 30),
        min_magnitude=params.get("min_mag", 2.5),
    )
    cache = _cache_path(params)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, features=features, labels=labels)
    return features, labels, metadata

@data_chain.handler(priority=1, timeout=10.0)
async def load_from_cache(params):
    """Fallback: load the cache the primary handler wrote on its last success."""
    cached = np.load(_cache_path(params))
    # Same (features, labels, metadata) shape the loaders return.
    return cached["features"], cached["labels"], {"source": "cache"}

@data_chain.handler(priority=2)
async def load_synthetic(params):
    """Final fallback: generate synthetic data."""
    loader = USGSEarthquakeLoader()
    return loader.load(use_synthetic=True, n_samples=1000)

# Use in detection pipeline
async def run_earthquake_analysis(params):
    result = await data_chain.execute(params)
    
    if result.degraded:
        logger.warning(f"Using degraded data source: {result.handler_name}")
    
    return analyze_earthquakes(result.value)
```

### Combined Router and Fallback

Build a complete detection API with routing and fallback:

```python
from omni_mercury_engine.integrations.routing import (
    RequestRouter, FallbackChain, FallbackRegistry
)

# Create fallback registry for all services
registry = FallbackRegistry()

# Register detector fallback chains
detector_chain = registry.register("detector")

@detector_chain.handler(priority=0, timeout=10.0)
async def ml_detector(data):
    """Primary ML-based detection."""
    return await ml_service.detect(data)

@detector_chain.handler(priority=1, timeout=5.0)
async def rule_detector(data):
    """Fallback to rule-based detection."""
    return rule_engine.detect(data)

@detector_chain.handler(priority=2)
async def threshold_detector(data):
    """Simple threshold-based detection."""
    return {"anomaly": data["value"] > data.get("threshold", 0.9)}

# Create router with fallback integration
router = RequestRouter(prefix="/api")

@router.post("/detect")
async def detect_anomaly(request):
    """Detect anomalies with automatic fallback."""
    data = await request.json()
    
    result = await registry.execute("detector", data)
    
    return {
        "detection": result.value,
        "handler": result.handler_name,
        "degraded": result.degraded,
        "latency_ms": result.elapsed * 1000
    }

@router.get("/metrics")
async def get_metrics(request):
    """Get routing and fallback metrics."""
    return {
        "routing": router.get_metrics(),
        "fallback": registry.get_all_metrics()
    }
```

## Error Handling

### Route Not Found

```python
from omni_mercury_engine.integrations.routing import RouteNotFoundError

try:
    match = router.match("/unknown/path", method="GET")
except RouteNotFoundError as e:
    return {"error": "Not found", "path": e.path, "method": e.method}
```

### Method Not Allowed

```python
from omni_mercury_engine.integrations.routing.router import MethodNotAllowedError

try:
    match = router.match("/api/health", method="DELETE")
except MethodNotAllowedError as e:
    return {
        "error": "Method not allowed",
        "allowed": e.allowed
    }
```

### Fallback Exhausted

```python
from omni_mercury_engine.integrations.routing import FallbackError

try:
    result = await chain.execute(data)
except FallbackError as e:
    logger.error(f"All handlers failed: {e}")
    for handler_name, error in e.errors:
        logger.error(f"  {handler_name}: {error}")
    return {"error": "Service unavailable", "degraded": True}
```

## Best Practices

1. **Set appropriate timeouts**: Always configure timeouts for handlers that call external services to prevent cascading failures.

2. **Order handlers by reliability**: Place the most reliable (but potentially slower) handlers at higher priority numbers as fallbacks.

3. **Monitor degradation**: Track the `degraded` flag and `fallback_rate` metrics to identify service health issues.

4. **Use conditional handlers**: Skip handlers that won't succeed (e.g., cache lookups when cache is empty) to reduce latency.

5. **Implement circuit breakers**: Combine fallback chains with circuit breakers for comprehensive resilience:

```python
from omni_mercury_engine.resilience import get_data_loader_breaker

@chain.handler(priority=0, timeout=10.0)
async def primary_with_circuit_breaker(data):
    breaker = get_data_loader_breaker("external_api")
    return breaker.call(lambda: external_api.fetch(data))
```

6. **Log fallback events**: Use the `FallbackResult.reasons` to understand why fallbacks occurred and improve primary handlers.

7. **Test degraded paths**: Regularly test that fallback handlers produce acceptable results when primary handlers fail.
