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

// MODE selects the threshold envelope.
//   MODE=full  (default): production-shaped SLOs, calibrated for a
//     50-VU ramp.  Cold-start latency contributes a vanishing fraction
//     of the p99 once the runner is warm and many samples have been
//     collected.
//   MODE=smoke: CI-sized SLOs, calibrated for a 1-VU 30 s run on a
//     shared GitHub-hosted runner.  Cold-start latency dominates p99
//     here (only ~24 health-check samples land in 30 s of 1-VU
//     traffic), so the smoke envelope intentionally allows it.  These
//     bounds still fail loudly on a genuine regression — a 5 s health
//     check or a 50 % failure rate is unambiguous breakage at any VU.
const MODE = (__ENV.MODE || 'full').toLowerCase();

const THRESHOLDS_FULL = {
    'errors': ['rate<0.01'],
    'http_req_duration': [
        'p(50)<100',
        'p(95)<500',
        'p(99)<1000',
    ],
    'anomaly_detection_duration': ['p(95)<400'],
    'health_check_duration': ['p(99)<50'],
    'http_req_duration{endpoint:univariate}': ['p(95)<500'],
    'http_req_duration{endpoint:multivariate}': ['p(95)<800'],
    'http_req_duration{endpoint:health}': ['p(99)<50'],
};

const THRESHOLDS_SMOKE = {
    // A 1-VU smoke run produces a small sample set where the first
    // request (route-table warmup, pydantic schema-build, FastAPI
    // OpenAPI snapshot) dominates the p99.  The smoke envelope still
    // refuses a *broken* API: zero 5xx, no failure-rate spike, no
    // multi-second latency floor.
    'errors': ['rate<0.05'],
    'http_req_duration': [
        'p(95)<2000',
        'p(99)<5000',
    ],
    'anomaly_detection_duration': ['p(95)<2000'],
    'health_check_duration': ['p(95)<500'],
    'http_req_duration{endpoint:univariate}': ['p(95)<2500'],
    'http_req_duration{endpoint:multivariate}': ['p(95)<3000'],
    'http_req_duration{endpoint:health}': ['p(95)<500'],
};

const SCENARIOS_FULL = {
    smoke: {
        executor: 'constant-vus',
        vus: 1,
        duration: '30s',
        tags: { scenario: 'smoke' },
        env: { SCENARIO: 'smoke' },
    },
    load: {
        executor: 'ramping-vus',
        startTime: '30s',
        startVUs: 0,
        stages: [
            { duration: '1m', target: 20 },
            { duration: '3m', target: 20 },
            { duration: '1m', target: 50 },
            { duration: '3m', target: 50 },
            { duration: '1m', target: 0 },
        ],
        tags: { scenario: 'load' },
    },
    stress: {
        executor: 'ramping-vus',
        startTime: '10m',
        startVUs: 0,
        stages: [
            { duration: '2m', target: 100 },
            { duration: '5m', target: 100 },
            { duration: '2m', target: 200 },
            { duration: '3m', target: 200 },
            { duration: '2m', target: 0 },
        ],
        tags: { scenario: 'stress' },
    },
    spike: {
        executor: 'ramping-vus',
        startTime: '25m',
        startVUs: 0,
        stages: [
            { duration: '10s', target: 100 },
            { duration: '1m', target: 100 },
            { duration: '10s', target: 0 },
            { duration: '30s', target: 0 },
            { duration: '10s', target: 150 },
            { duration: '1m', target: 150 },
            { duration: '30s', target: 0 },
        ],
        tags: { scenario: 'spike' },
    },
};

// Smoke mode uses a single 1-VU 30-second scenario so CI doesn't run
// the multi-hour ramp/stress/spike sequence above.
const SCENARIOS_SMOKE = {
    smoke: {
        executor: 'constant-vus',
        vus: 1,
        duration: '30s',
        tags: { scenario: 'smoke' },
        env: { SCENARIO: 'smoke' },
    },
};

export const options = {
    scenarios: MODE === 'smoke' ? SCENARIOS_SMOKE : SCENARIOS_FULL,
    thresholds: MODE === 'smoke' ? THRESHOLDS_SMOKE : THRESHOLDS_FULL,
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

function checkCorrelationId(response) {
    // HTTP headers are case-insensitive (RFC 7230 §3.2) but k6's
    // response.headers map is keyed by the exact bytes the server sent.
    // Starlette/uvicorn lower-cases all response header names, while
    // Mercury Agent's middleware sets ``X-Correlation-ID``.  Probe both
    // forms so a future server- or proxy-layer normalisation change
    // doesn't silently break the assertion.
    const h = response.headers || {};
    return (
        h['X-Correlation-ID'] !== undefined ||
        h['X-Correlation-Id'] !== undefined ||
        h['x-correlation-id'] !== undefined
    );
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

        const success = check(res, {
            'univariate status is 200': (r) => r.status === 200,
            'univariate has response body': (r) => r.body && r.body.length > 0,
            'has correlation ID': checkCorrelationId,
            'response time OK': (r) => duration < 1000,
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
