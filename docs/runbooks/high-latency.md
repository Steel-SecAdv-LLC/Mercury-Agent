# OMNI-AVA Runbook: High Latency

## Alert: OmniAvaHighLatency

### Overview
This runbook provides guidance for responding to high latency alerts in the OMNI-AVA platform, where response times exceed acceptable thresholds.

### Alert Threshold
- **Warning**: P95 latency > 500ms for 10 minutes

### Impact
- Degraded user experience
- Potential timeout errors for clients
- May indicate underlying performance issues
- Could lead to cascading failures

---

## Diagnosis Steps

### 1. Check Current Latency Metrics
```promql
# Prometheus queries for latency analysis
# P50 latency
histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{app="omni-ava"}[5m])) by (le))

# P95 latency
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{app="omni-ava"}[5m])) by (le))

# P99 latency
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{app="omni-ava"}[5m])) by (le))
```

### 2. Identify Slow Endpoints
```bash
# Check logs for slow requests
kubectl logs -n omni-ava -l app.kubernetes.io/component=api --tail=200 | grep -E "duration=[0-9]+\.[0-9]+" | sort -t= -k2 -rn | head -20

# Check by endpoint (Prometheus)
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{app="omni-ava"}[5m])) by (le, path))
```

### 3. Check Resource Utilization
```bash
# View CPU and memory usage
kubectl top pods -n omni-ava

# Check for CPU throttling
kubectl get pods -n omni-ava -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].resources.limits.cpu}{"\n"}{end}'
```

### 4. Check Dependencies
```bash
# Check engine response times
kubectl logs -n omni-ava -l app.kubernetes.io/component=engine --tail=100 | grep -i "duration\|time"

# Check network latency between pods
kubectl exec -n omni-ava deployment/omni-ava-api -- curl -w "@/dev/stdin" -o /dev/null -s http://omni-ava-engine:8080/health <<< "time_total: %{time_total}s\n"
```

---

## Resolution Steps

### Scenario 1: CPU Saturation

**Symptoms:**
- High CPU usage across pods
- CPU throttling observed
- Latency correlates with CPU usage

**Actions:**
1. Scale horizontally:
   ```bash
   kubectl scale deployment omni-ava-api -n omni-ava --replicas=6
   ```
2. Increase CPU limits:
   ```bash
   kubectl patch deployment omni-ava-api -n omni-ava -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"cpu":"4"}}}]}}}}'
   ```
3. Check HPA configuration:
   ```bash
   kubectl get hpa -n omni-ava -o yaml
   ```

### Scenario 2: Slow Detection Processing

**Symptoms:**
- Detection endpoints are slow
- Engine pods show high latency
- Model inference taking too long

**Actions:**
1. Check model inference times:
   ```bash
   kubectl logs -n omni-ava -l app.kubernetes.io/component=engine --tail=100 | grep -i "inference"
   ```
2. Enable model caching:
   ```bash
   kubectl set env deployment/omni-ava-engine -n omni-ava ENABLE_MODEL_CACHE=true
   ```
3. Reduce batch sizes for faster response:
   ```bash
   kubectl set env deployment/omni-ava-engine -n omni-ava MAX_BATCH_SIZE=16
   ```

### Scenario 3: Network Issues

**Symptoms:**
- Inter-pod communication is slow
- Network timeouts in logs
- Latency varies significantly

**Actions:**
1. Check network policies:
   ```bash
   kubectl get networkpolicies -n omni-ava
   ```
2. Verify service endpoints:
   ```bash
   kubectl get endpoints -n omni-ava
   ```
3. Check for DNS resolution issues:
   ```bash
   kubectl exec -n omni-ava deployment/omni-ava-api -- nslookup omni-ava-engine
   ```

### Scenario 4: Database/Storage Latency

**Symptoms:**
- Latency on data-heavy operations
- PVC access is slow
- Storage metrics show high latency

**Actions:**
1. Check PVC performance:
   ```bash
   kubectl exec -n omni-ava deployment/omni-ava-engine -- dd if=/dev/zero of=/data/test bs=1M count=100 oflag=direct
   ```
2. Consider using faster storage class
3. Implement caching layer for frequently accessed data

---

## Escalation

If latency doesn't improve after following these steps:

1. **Notify team** via Slack #omni-ava-alerts
2. **Check for broader infrastructure issues**
3. **Engage infrastructure team** if cluster-wide

### Escalation Contacts
- **Platform Team**: platform-oncall@example.com
- **Infrastructure Team**: infra-oncall@example.com

---

## Prevention

1. **Set appropriate resource limits** based on load testing
2. **Implement request timeouts** to fail fast
3. **Use caching** for expensive operations
4. **Monitor latency percentiles** continuously
5. **Regular performance testing** and optimization

---

## Related Runbooks
- [High Error Rate](./high-error-rate.md)
- [High CPU](./high-cpu.md)
- [Slow Detection](./slow-detection.md)
