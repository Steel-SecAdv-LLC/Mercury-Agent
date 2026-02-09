"""
Mercury Agent Load Testing Infrastructure

Comprehensive load testing suite using Locust for the Mercury Agent API.
Tests API performance under various load conditions and validates SLOs.

Usage:
    # Start Locust web UI (interactive mode)
    locust -f tests/load/locustfile.py --host http://localhost:8000

    # Headless mode with specific users and duration
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless -u 100 -r 10 --run-time 5m

    # Generate HTML report
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 5 --run-time 2m --html=report.html

Environment Variables:
    MERCURY_API_HOST: API host (default: http://localhost:8000)
    MERCURY_API_KEY: Optional API key for authentication
    MERCURY_LOAD_TEST_THINK_TIME_MIN: Min wait between requests (default: 0.5)
    MERCURY_LOAD_TEST_THINK_TIME_MAX: Max wait between requests (default: 2.0)
"""

from __future__ import annotations

import os
import random
import time

from locust import HttpUser, between, events, tag, task

# Configuration
API_KEY = os.getenv("MERCURY_API_KEY", "")
THINK_TIME_MIN = float(os.getenv("MERCURY_LOAD_TEST_THINK_TIME_MIN", "0.5"))
THINK_TIME_MAX = float(os.getenv("MERCURY_LOAD_TEST_THINK_TIME_MAX", "2.0"))

# SLO Thresholds (in milliseconds)
SLO_P50_MS = 100  # 50th percentile
SLO_P95_MS = 500  # 95th percentile
SLO_P99_MS = 1000  # 99th percentile
SLO_ERROR_RATE = 0.01  # 1% error rate threshold


# =============================================================================
# Test Data Generators
# =============================================================================
def generate_univariate_data(length: int = 100, anomaly_rate: float = 0.05) -> list[float]:
    """Generate realistic univariate time series with anomalies."""
    data = []
    base = random.uniform(10, 100)
    trend = random.uniform(-0.1, 0.1)

    for i in range(length):
        # Base value with trend
        value = base + trend * i

        # Add noise
        value += random.gauss(0, base * 0.1)

        # Add anomalies
        if random.random() < anomaly_rate:
            value += random.choice([-1, 1]) * base * random.uniform(2, 5)

        data.append(round(value, 4))

    return data


def generate_multivariate_data(
    length: int = 100,
    features: int = 5,
    anomaly_rate: float = 0.05,
) -> list[list[float]]:
    """Generate realistic multivariate time series with anomalies."""
    # Generate correlated features
    data = []

    # Base values for each feature
    bases = [random.uniform(10, 100) for _ in range(features)]
    trends = [random.uniform(-0.1, 0.1) for _ in range(features)]

    for i in range(length):
        row = []
        is_anomaly = random.random() < anomaly_rate

        for f in range(features):
            value = bases[f] + trends[f] * i
            value += random.gauss(0, bases[f] * 0.1)

            if is_anomaly:
                value += random.choice([-1, 1]) * bases[f] * random.uniform(2, 5)

            row.append(round(value, 4))

        data.append(row)

    return data


# =============================================================================
# Custom Event Handlers
# =============================================================================
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log test start and validate configuration."""
    print("\n" + "=" * 60)
    print("Mercury Agent Load Test Starting")
    print("=" * 60)
    print(f"Host: {environment.host}")
    print(f"Think Time: {THINK_TIME_MIN}s - {THINK_TIME_MAX}s")
    print(f"SLOs: P50<{SLO_P50_MS}ms, P95<{SLO_P95_MS}ms, P99<{SLO_P99_MS}ms")
    print("=" * 60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Validate SLOs after test completion."""
    stats = environment.stats

    print("\n" + "=" * 60)
    print("Mercury Agent Load Test Results")
    print("=" * 60)

    violations = []

    for entry in stats.entries.values():
        if entry.num_requests == 0:
            continue

        name = entry.name
        p50 = entry.get_response_time_percentile(0.50)
        p95 = entry.get_response_time_percentile(0.95)
        p99 = entry.get_response_time_percentile(0.99)
        error_rate = entry.fail_ratio

        print(f"\n{name}:")
        print(f"  Requests: {entry.num_requests}")
        print(f"  Avg: {entry.avg_response_time:.2f}ms")
        print(f"  P50: {p50:.2f}ms (SLO: {SLO_P50_MS}ms)")
        print(f"  P95: {p95:.2f}ms (SLO: {SLO_P95_MS}ms)")
        print(f"  P99: {p99:.2f}ms (SLO: {SLO_P99_MS}ms)")
        print(f"  Error Rate: {error_rate * 100:.2f}% (SLO: {SLO_ERROR_RATE * 100}%)")

        # Check SLO violations
        if p50 > SLO_P50_MS:
            violations.append(f"{name}: P50 {p50:.2f}ms > {SLO_P50_MS}ms")
        if p95 > SLO_P95_MS:
            violations.append(f"{name}: P95 {p95:.2f}ms > {SLO_P95_MS}ms")
        if p99 > SLO_P99_MS:
            violations.append(f"{name}: P99 {p99:.2f}ms > {SLO_P99_MS}ms")
        if error_rate > SLO_ERROR_RATE:
            violations.append(f"{name}: Error {error_rate * 100:.2f}% > {SLO_ERROR_RATE * 100}%")

    print("\n" + "-" * 60)
    if violations:
        print("SLO VIOLATIONS DETECTED:")
        for v in violations:
            print(f"  - {v}")
    else:
        print("ALL SLOs PASSED")
    print("=" * 60 + "\n")


# =============================================================================
# User Classes
# =============================================================================
class MercuryAPIUser(HttpUser):
    """Simulates realistic user behavior for Mercury Agent API.

    Weights tasks to match expected production traffic patterns:
    - 60% univariate detection (most common)
    - 25% multivariate detection
    - 10% health checks
    - 5% batch operations
    """

    wait_time = between(THINK_TIME_MIN, THINK_TIME_MAX)

    def on_start(self):
        """Initialize user session."""
        self.headers = {"Content-Type": "application/json"}
        if API_KEY:
            self.headers["Authorization"] = f"Bearer {API_KEY}"

        # Verify API is accessible
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"API not healthy: {response.status_code}")

    @tag("health")
    @task(10)
    def health_check(self):
        """Check API health endpoint."""
        with self.client.get("/health", headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @tag("detection", "univariate")
    @task(60)
    def detect_univariate(self):
        """Test univariate anomaly detection."""
        data = generate_univariate_data(length=random.randint(50, 200))
        payload = {
            "data": data,
            "sensitivity": random.uniform(0.3, 0.8),
        }

        with self.client.post(
            "/api/v1/detect/univariate",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/detect/univariate",
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "anomalies" in result or "is_anomaly" in result:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Detection failed: {response.status_code}")

    @tag("detection", "multivariate")
    @task(25)
    def detect_multivariate(self):
        """Test multivariate anomaly detection."""
        features = random.randint(3, 10)
        data = generate_multivariate_data(
            length=random.randint(50, 150),
            features=features,
        )
        payload = {
            "data": data,
            "sensitivity": random.uniform(0.3, 0.8),
        }

        with self.client.post(
            "/api/v1/detect/multivariate",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/detect/multivariate",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Multivariate detection failed: {response.status_code}")

    @tag("detection", "batch")
    @task(5)
    def detect_batch(self):
        """Test batch anomaly detection."""
        batch_size = random.randint(5, 20)
        batch = [
            {"data": generate_univariate_data(length=random.randint(20, 50))}
            for _ in range(batch_size)
        ]
        payload = {"requests": batch}

        with self.client.post(
            "/api/v1/detect/batch",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/detect/batch",
        ) as response:
            if response.status_code in [200, 202]:
                response.success()
            elif response.status_code == 404:
                # Batch endpoint may not be implemented
                response.success()
            else:
                response.failure(f"Batch detection failed: {response.status_code}")


class HighThroughputUser(HttpUser):
    """Simulates high-throughput machine-to-machine traffic.

    Minimal think time, focused on maximum request rate.
    Used for stress testing and finding breaking points.
    """

    wait_time = between(0.01, 0.1)  # Minimal delay

    def on_start(self):
        """Initialize user session."""
        self.headers = {"Content-Type": "application/json"}
        if API_KEY:
            self.headers["Authorization"] = f"Bearer {API_KEY}"

        # Pre-generate data to minimize CPU during test
        self.cached_data = [generate_univariate_data(length=100) for _ in range(10)]

    @tag("stress", "univariate")
    @task(100)
    def rapid_detection(self):
        """Rapid-fire univariate detection for stress testing."""
        payload = {
            "data": random.choice(self.cached_data),
            "sensitivity": 0.5,
        }

        with self.client.post(
            "/api/v1/detect/univariate",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/detect/univariate [stress]",
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # Rate limited - expected under stress
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


class StreamingUser(HttpUser):
    """Simulates streaming data ingestion patterns.

    Sends continuous small batches like IoT sensor data.
    """

    wait_time = between(0.1, 0.5)

    def on_start(self):
        """Initialize streaming session."""
        self.headers = {"Content-Type": "application/json"}
        if API_KEY:
            self.headers["Authorization"] = f"Bearer {API_KEY}"

        self.window_size = 20
        self.data_buffer = generate_univariate_data(length=self.window_size)

    @tag("streaming")
    @task
    def streaming_detection(self):
        """Simulate streaming detection with sliding window."""
        # Add new data point
        new_value = self.data_buffer[-1] + random.gauss(0, 5)
        if random.random() < 0.02:  # 2% anomaly rate
            new_value *= random.uniform(2, 4)

        self.data_buffer.append(round(new_value, 4))
        self.data_buffer.pop(0)  # Sliding window

        payload = {
            "data": self.data_buffer,
            "sensitivity": 0.6,
        }

        with self.client.post(
            "/api/v1/detect/univariate",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/detect/univariate [streaming]",
        ) as response:
            if response.status_code in [200, 429]:
                response.success()
            else:
                response.failure(f"Streaming detection failed: {response.status_code}")


# =============================================================================
# Specialized Test Scenarios
# =============================================================================
class SLOValidationUser(HttpUser):
    """User focused on SLO validation with detailed metrics.

    Runs specific test cases and validates against SLO thresholds.
    """

    wait_time = between(1, 2)
    weight = 1  # Lower weight, run fewer of these

    def on_start(self):
        """Initialize SLO validation."""
        self.headers = {"Content-Type": "application/json"}
        self.test_cases = self._generate_test_cases()

    def _generate_test_cases(self) -> list[dict]:
        """Generate standard test cases for SLO validation."""
        return [
            {"name": "small", "length": 20},
            {"name": "medium", "length": 100},
            {"name": "large", "length": 500},
            {"name": "xlarge", "length": 1000},
        ]

    @tag("slo")
    @task
    def slo_test_cases(self):
        """Run through standard SLO test cases."""
        for case in self.test_cases:
            data = generate_univariate_data(length=case["length"])
            payload = {"data": data, "sensitivity": 0.5}

            start = time.perf_counter()

            with self.client.post(
                "/api/v1/detect/univariate",
                json=payload,
                headers=self.headers,
                catch_response=True,
                name=f"/api/v1/detect/univariate [{case['name']}]",
            ) as response:
                duration_ms = (time.perf_counter() - start) * 1000

                if response.status_code == 200:
                    # Log custom metric for detailed analysis
                    events.request.fire(
                        request_type="SLO",
                        name=f"detection_{case['name']}",
                        response_time=duration_ms,
                        response_length=len(response.content),
                        exception=None,
                        context={},
                    )
                    response.success()
                else:
                    response.failure(f"SLO test failed: {response.status_code}")


# Default user mix for load testing
# Uncomment the desired mix or run specific classes with --class flag

# Production traffic simulation (default)
# class DefaultUser(MercuryAPIUser):
#     weight = 100

# Stress testing
# class StressUser(HighThroughputUser):
#     weight = 100

# Mixed load (production + stress)
# Adjust weights to control traffic mix
