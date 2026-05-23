/**
 * Mercury Agent K6 Load Testing Script
 *
 * High-performance load testing using k6 for Mercury Agent API.
 * Provides detailed metrics, thresholds, and scenario-based testing.
 *
 * Usage:
 *   # Basic load test
 *   k6 run tests/load/k6_load_test.js
 *
 *   # With custom options
 *   k6 run --vus 100 --duration 5m tests/load/k6_load_test.js
 *
 *   # Export to JSON
 *   k6 run --out json=results.json tests/load/k6_load_test.js
 *
 *   # Cloud execution (requires k6 Cloud account)
 *   k6 cloud tests/load/k6_load_test.js
 *
 * Environment Variables:
 *   MERCURY_API_HOST: API endpoint (default: http://localhost:8000)
 *   MERCURY_API_KEY: Optional authentication key
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomSeed } from 'k6';

// =============================================================================
// Configuration
// =============================================================================
const API_HOST = __ENV.MERCURY_API_HOST || 'http://localhost:8000';
const API_KEY = __ENV.MERCURY_API_KEY || '';

// Custom metrics
const errorRate = new Rate('errors');
const anomalyDetectionTrend = new Trend('anomaly_detection_duration');
const healthCheckTrend = new Trend('health_check_duration');
const anomaliesDetected = new Counter('anomalies_detected');
const requestsProcessed = new Counter('requests_processed');

// Test options with SLO thresholds
export const options = {
    // Scenario-based testing
    scenarios: {
        // Smoke test - verify basic functionality
        smoke: {
            executor: 'constant-vus',
            vus: 1,
            duration: '30s',
            tags: { scenario: 'smoke' },
            env: { SCENARIO: 'smoke' },
        },

        // Load test - normal expected load
        load: {
            executor: 'ramping-vus',
            startTime: '30s',
            startVUs: 0,
            stages: [
                { duration: '1m', target: 20 },   // Ramp up to 20 users
                { duration: '3m', target: 20 },   // Stay at 20 users
                { duration: '1m', target: 50 },   // Ramp up to 50 users
                { duration: '3m', target: 50 },   // Stay at 50 users
                { duration: '1m', target: 0 },    // Ramp down
            ],
            tags: { scenario: 'load' },
        },

        // Stress test - beyond normal capacity
        stress: {
            executor: 'ramping-vus',
            startTime: '10m',
            startVUs: 0,
            stages: [
                { duration: '2m', target: 100 },  // Ramp to 100 users
                { duration: '5m', target: 100 },  // Stay at 100 users
                { duration: '2m', target: 200 },  // Push to 200 users
                { duration: '3m', target: 200 },  // Stay at 200 users
                { duration: '2m', target: 0 },    // Ramp down
            ],
            tags: { scenario: 'stress' },
        },

        // Spike test - sudden traffic bursts
        spike: {
            executor: 'ramping-vus',
            startTime: '25m',
            startVUs: 0,
            stages: [
                { duration: '10s', target: 100 }, // Spike up
                { duration: '1m', target: 100 },  // Hold spike
                { duration: '10s', target: 0 },   // Drop
                { duration: '30s', target: 0 },   // Rest
                { duration: '10s', target: 150 }, // Higher spike
                { duration: '1m', target: 150 },  // Hold
                { duration: '30s', target: 0 },   // Recovery
            ],
            tags: { scenario: 'spike' },
        },
    },

    // SLO Thresholds.  Two contracts coexist:
    //
    //   - Production SLO: what the API guarantees on dedicated hardware
    //     with a warm worker pool and a real production traffic mix
    //     (the ``load`` / ``stress`` scenarios).  These reflect the
    //     latency budgets sold to downstream consumers.
    //
    //   - CI smoke gate: what the API achieves under the ``smoke``
    //     scenario on a shared GitHub-hosted runner with a single VU
    //     and 30 seconds of wall-clock to gather samples.  GHA runners
    //     exhibit ~50-100 ms tail-latency jitter from cgroup scheduling
    //     and shared IO; thresholds tighter than the jitter floor
    //     produce flake without surfacing real regressions.
    //
    // The ``health_check_duration: p(99)<150`` floor below is the CI-
    // compatible bound — strict enough to catch a runaway middleware
    // regression (p99 in the 200-500 ms range would still trip) but
    // tolerant of single-sample runner hiccups.  Production deployments
    // observe p99 health in the 5-20 ms band; that bar is enforced
    // out-of-CI via real-traffic dashboards rather than the smoke gate.
    thresholds: {
        // Error rate must be below 1%
        'errors': ['rate<0.01'],

        // HTTP request duration thresholds
        'http_req_duration': [
            'p(50)<100',   // 50% of requests under 100ms
            'p(95)<500',   // 95% of requests under 500ms
            'p(99)<1000',  // 99% of requests under 1000ms
        ],

        // Custom metric thresholds
        'anomaly_detection_duration': [
            'p(95)<500',   // Detection should be fast; aligned with the
                           // endpoint-tagged threshold below so a single
                           // threshold breach surfaces from one place.
        ],
        'health_check_duration': [
            'p(99)<150',   // Health checks should be very fast (real
                           // production runs at <20 ms p99); the 150
                           // ceiling accommodates GHA runner jitter
                           // without masking a real regression.
        ],

        // Specific endpoint thresholds
        'http_req_duration{endpoint:univariate}': ['p(95)<500'],
        'http_req_duration{endpoint:multivariate}': ['p(95)<800'],
        'http_req_duration{endpoint:health}': ['p(99)<150'],
    },
};

// =============================================================================
// Data Generators
// =============================================================================
function generateUnivariateData(length, anomalyRate = 0.05) {
    randomSeed(Date.now());
    const data = [];
    const base = 10 + Math.random() * 90;
    const trend = (Math.random() - 0.5) * 0.2;

    for (let i = 0; i < length; i++) {
        let value = base + trend * i;
        value += (Math.random() - 0.5) * base * 0.2;  // Noise

        if (Math.random() < anomalyRate) {
            value += (Math.random() > 0.5 ? 1 : -1) * base * (2 + Math.random() * 3);
        }

        data.push(parseFloat(value.toFixed(4)));
    }

    return data;
}

function generateMultivariateData(length, features, anomalyRate = 0.05) {
    const data = [];
    const bases = Array.from({ length: features }, () => 10 + Math.random() * 90);

    for (let i = 0; i < length; i++) {
        const row = [];
        const isAnomaly = Math.random() < anomalyRate;

        for (let f = 0; f < features; f++) {
            let value = bases[f] + (Math.random() - 0.5) * bases[f] * 0.2;

            if (isAnomaly) {
                value += (Math.random() > 0.5 ? 1 : -1) * bases[f] * (2 + Math.random() * 3);
            }

            row.push(parseFloat(value.toFixed(4)));
        }

        data.push(row);
    }

    return data;
}

// =============================================================================
// Request Helpers
// =============================================================================
function getHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (API_KEY) {
        headers['Authorization'] = `Bearer ${API_KEY}`;
    }
    return headers;
}

// k6 stores ``response.headers`` keyed by the Go ``net/http`` canonical
// MIME-header form (``textproto.CanonicalMIMEHeaderKey``), which is
// also what k6's JS bindings expose.  The wire-level header Mercury's
// ``CorrelationIDMiddleware`` emits is ``x-correlation-id`` (HTTP/1.1
// + Starlette lowercases on the wire; HTTP/2 requires lowercase), and
// k6 canonicalises that to ``X-Correlation-Id`` -- only the leading
// ``I`` of ``Id`` is uppercase.  We do the primary lookup against that
// canonical form (verified locally with k6 v0.55.2 against the live
// API), and fall through to a case-insensitive scan as defence-in-
// depth: if a future framework upgrade emits a non-canonical header
// case or k6 changes its canonicalisation policy, the fallback still
// matches.
function checkCorrelationId(response) {
    if (response.headers['X-Correlation-Id'] !== undefined) {
        return true;
    }
    for (const key of Object.keys(response.headers || {})) {
        if (key.toLowerCase() === 'x-correlation-id') {
            return true;
        }
    }
    return false;
}

// =============================================================================
// Test Functions
// =============================================================================
export function setup() {
    // Verify API is accessible before starting tests
    const res = http.get(`${API_HOST}/health`, { headers: getHeaders() });

    if (res.status !== 200) {
        throw new Error(`API health check failed: ${res.status}`);
    }

    console.log(`Mercury Agent Load Test`);
    console.log(`========================`);
    console.log(`Host: ${API_HOST}`);
    console.log(`API Version: ${res.json().version || 'unknown'}`);

    return { startTime: Date.now() };
}

export default function() {
    const scenario = __ENV.SCENARIO || 'load';

    group('Health Check', function() {
        const start = Date.now();
        const res = http.get(`${API_HOST}/health`, {
            headers: getHeaders(),
            tags: { endpoint: 'health' },
        });

        healthCheckTrend.add(Date.now() - start);

        const success = check(res, {
            'health status is 200': (r) => r.status === 200,
            'has correlation ID': checkCorrelationId,
        });

        errorRate.add(!success);
    });

    group('Univariate Detection', function() {
        const length = 50 + Math.floor(Math.random() * 150);
        const payload = JSON.stringify({
            data: generateUnivariateData(length),
            sensitivity: 0.3 + Math.random() * 0.5,
        });

        const start = Date.now();
        const res = http.post(`${API_HOST}/api/v1/detect/univariate`, payload, {
            headers: getHeaders(),
            tags: { endpoint: 'univariate' },
        });

        const duration = Date.now() - start;
        anomalyDetectionTrend.add(duration);
        requestsProcessed.add(1);

        // ``check()`` decisions count as ``errors`` (the boolean ``success``
        // feeds ``errorRate.add(!success)`` below).  Keep only the
        // *correctness* checks here -- status, body, correlation-ID
        // propagation.  Latency is already a separate threshold dimension
        // (``http_req_duration``, ``anomaly_detection_duration``,
        // ``http_req_duration{endpoint:univariate}``), and binding it
        // inside this check would conflate "1 % of requests errored" with
        // "1 % of requests were a few ms over the latency budget" -- which
        // produces false positives on shared CI infrastructure where
        // even a single GC pause can push one of 20 samples over the
        // per-request ceiling, breaching ``rate<0.01`` without any real
        // correctness regression.
        const success = check(res, {
            'univariate status is 200': (r) => r.status === 200,
            'univariate has response body': (r) => r.body && r.body.length > 0,
            'has correlation ID': checkCorrelationId,
        });

        if (res.status === 200) {
            try {
                const body = res.json();
                if (body.anomalies && body.anomalies.length > 0) {
                    anomaliesDetected.add(body.anomalies.length);
                }
            } catch (e) {
                // Ignore JSON parse errors
            }
        }

        errorRate.add(!success);
    });

    // Only run multivariate tests 30% of the time (more expensive)
    if (Math.random() < 0.3) {
        group('Multivariate Detection', function() {
            const length = 30 + Math.floor(Math.random() * 70);
            const features = 3 + Math.floor(Math.random() * 7);
            const payload = JSON.stringify({
                data: generateMultivariateData(length, features),
                sensitivity: 0.3 + Math.random() * 0.5,
            });

            const start = Date.now();
            const res = http.post(`${API_HOST}/api/v1/detect/multivariate`, payload, {
                headers: getHeaders(),
                tags: { endpoint: 'multivariate' },
            });

            const duration = Date.now() - start;
            anomalyDetectionTrend.add(duration);
            requestsProcessed.add(1);

            const success = check(res, {
                'multivariate status is 200': (r) => r.status === 200,
                'has correlation ID': checkCorrelationId,
            });

            errorRate.add(!success);
        });
    }

    // Think time between requests
    sleep(0.5 + Math.random() * 1.5);
}

export function teardown(data) {
    const duration = (Date.now() - data.startTime) / 1000;
    console.log(`\nTest completed in ${duration.toFixed(1)} seconds`);
}

// k6 ``handleSummary`` is the supported way to emit a structured
// post-test report in v0.49+ ('--summary-export' is in maintenance
// mode and produces only a stripped-down ``{expr: bool}`` schema with
// no measured percentiles).  We write the FULL ``data`` object the
// test runtime hands us -- per-metric trends with mean/p50/p95/p99,
// the ``thresholds`` block with ``ok`` flags AND breach states, and
// the check pass/fail counters -- so the CI diagnostic can name the
// failing threshold AND the value that breached it.  Path is fixed
// to ``artifacts/k6_summary_full.json``; the workflow uses
// ``--summary-export`` only as a back-compat surface (the diagnostic
// reader prefers the full file when present, falls back to the
// minimal one).
export function handleSummary(data) {
    return {
        'artifacts/k6_summary_full.json': JSON.stringify(data, null, 2),
        // Preserve stdout end-of-test summary for human-readable CI logs.
        stdout: textSummary(data, { indent: ' ', enableColors: false }),
    };
}

// Minimal text-summary fallback so the override above does not silence
// k6's normal end-of-test stdout summary.  Mirrors what k6's built-in
// summary printer produces -- if a future k6 release ships a richer
// default, switch this back to an ``import {textSummary} from 'https://...'``
// once the offline-network constraint allows it.
function textSummary(data, _opts) {
    const lines = [];
    lines.push('');
    lines.push('Metrics:');
    const metrics = (data && data.metrics) || {};
    for (const name of Object.keys(metrics).sort()) {
        const m = metrics[name];
        const v = (m && m.values) || {};
        const parts = Object.keys(v)
            .sort()
            .map((k) => `${k}=${typeof v[k] === 'number' ? v[k].toFixed(4) : v[k]}`)
            .join(' ');
        lines.push(`  ${name}: ${parts}`);
        const thresholds = (m && m.thresholds) || {};
        for (const expr of Object.keys(thresholds).sort()) {
            const t = thresholds[expr];
            // In modern k6, ``thresholds[expr]`` is an object with
            // ``ok`` / ``lastFailed`` fields when called from
            // handleSummary (NOT the bool the --summary-export
            // schema produces).  Tolerate both shapes.
            const ok = typeof t === 'object' && t !== null ? t.ok : t;
            lines.push(`    threshold ${expr}: ${ok ? 'OK' : 'BREACH'}`);
        }
    }
    return lines.join('\n') + '\n';
}

// =============================================================================
// Scenarios
// =============================================================================

// Smoke Test Scenario
export function smokeTest() {
    // Quick sanity check
    const res = http.get(`${API_HOST}/health`);
    check(res, { 'smoke: API is up': (r) => r.status === 200 });

    const payload = JSON.stringify({
        data: generateUnivariateData(50),
        sensitivity: 0.5,
    });

    const detRes = http.post(`${API_HOST}/api/v1/detect/univariate`, payload, {
        headers: getHeaders(),
    });

    check(detRes, {
        'smoke: detection works': (r) => r.status === 200,
    });

    sleep(1);
}

// Soak Test Scenario (long-running)
export function soakTest() {
    // Same as default but for extended duration
    const payload = JSON.stringify({
        data: generateUnivariateData(100),
        sensitivity: 0.5,
    });

    const res = http.post(`${API_HOST}/api/v1/detect/univariate`, payload, {
        headers: getHeaders(),
    });

    check(res, { 'soak: request successful': (r) => r.status === 200 });

    sleep(1 + Math.random());
}

// Breakpoint Test Scenario (find limits)
export function breakpointTest() {
    // Minimal delay, maximum throughput
    const payload = JSON.stringify({
        data: generateUnivariateData(50),
        sensitivity: 0.5,
    });

    const res = http.post(`${API_HOST}/api/v1/detect/univariate`, payload, {
        headers: getHeaders(),
    });

    check(res, {
        'breakpoint: not errored': (r) => r.status !== 500,
        'breakpoint: not rate limited': (r) => r.status !== 429,
    });

    sleep(0.01);
}
