# OMNI-AVA Runbook: High Memory Usage

## Alert: OmniAvaHighMemory

### Overview
This runbook provides guidance for responding to high memory utilization in OMNI-AVA pods, which can lead to OOMKills and service disruption.

### Alert Threshold
- **Warning**: Memory utilization > 80% for 15 minutes

### Impact
- Risk of OOMKill if usage continues to grow
- Potential garbage collection pressure
- May affect application performance
- Could lead to service disruption

---

## Diagnosis Steps

### 1. Check Memory Usage
```bash
# View current memory usage
kubectl top pods -n omni-ava

# Check memory limits
kubectl get pods -n omni-ava -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].resources.limits.memory}{"\n"}{end}'

# Check for recent OOMKills
kubectl get pods -n omni-ava -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[*].lastState.terminated.reason}{"\n"}{end}'
```

### 2. Analyze Memory Trends
```promql
# Prometheus query for memory usage over time
container_memory_usage_bytes{namespace="omni-ava",container="api"}

# Memory usage percentage
container_memory_usage_bytes{namespace="omni-ava"} / container_spec_memory_limit_bytes{namespace="omni-ava"}
```

### 3. Check Application Memory
```bash
# Check for memory-related logs
kubectl logs -n omni-ava -l app.kubernetes.io/component=api --tail=200 | grep -i "memory\|heap\|gc"

# Check model loading status
kubectl logs -n omni-ava -l app.kubernetes.io/component=engine --tail=100 | grep -i "model\|load"
```

### 4. Check Workload Patterns
```bash
# View recent activity
kubectl logs -n omni-ava -l app.kubernetes.io/component=api --tail=100 | grep -i "request\|detect"

# Check batch sizes
kubectl logs -n omni-ava -l app.kubernetes.io/component=engine --tail=50 | grep -i "batch"
```

---

## Resolution Steps

### Scenario 1: Gradual Memory Growth (Potential Leak)

**Symptoms:**
- Memory grows steadily over time
- No correlation with traffic
- Restarts temporarily fix issue

**Actions:**
1. Schedule a rolling restart:
   ```bash
   kubectl rollout restart deployment/omni-ava-api -n omni-ava
   ```
2. Enable memory profiling:
   ```bash
   kubectl set env deployment/omni-ava-api -n omni-ava ENABLE_MEMORY_PROFILING=true
   ```
3. File bug report with memory profile data

### Scenario 2: Large Data Processing

**Symptoms:**
- Memory spikes during operations
- Correlates with large requests
- Batch processing visible in logs

**Actions:**
1. Reduce batch sizes:
   ```bash
   kubectl set env deployment/omni-ava-engine -n omni-ava MAX_BATCH_SIZE=32
   ```
2. Enable streaming for large datasets:
   ```bash
   kubectl set env deployment/omni-ava-api -n omni-ava ENABLE_STREAMING=true
   ```
3. Implement request size limits

### Scenario 3: Model Memory Usage

**Symptoms:**
- High memory on engine pods
- Memory spike at startup
- Multiple models loaded

**Actions:**
1. Enable lazy model loading:
   ```bash
   kubectl set env deployment/omni-ava-engine -n omni-ava LAZY_MODEL_LOADING=true
   ```
2. Reduce preloaded models:
   ```bash
   kubectl set env deployment/omni-ava-engine -n omni-ava PRELOAD_MODELS="statistical"
   ```
3. Consider model quantization for smaller footprint

### Scenario 4: Insufficient Memory Limits

**Symptoms:**
- Memory consistently near limit
- Normal workload patterns
- No leak pattern

**Actions:**
1. Increase memory limits:
   ```bash
   kubectl patch deployment omni-ava-api -n omni-ava -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"8Gi"},"requests":{"memory":"4Gi"}}}]}}}}'
   ```
2. Scale horizontally to distribute load:
   ```bash
   kubectl scale deployment omni-ava-api -n omni-ava --replicas=5
   ```

---

## Escalation

If memory issues persist:

1. **Monitor closely** for escalation to critical
2. **Notify team** via Slack #omni-ava-alerts
3. **Engage development team** for memory optimization

### Escalation Contacts
- **Platform Team**: platform-oncall@example.com
- **Development Team**: dev-oncall@example.com

---

## Prevention

1. **Set appropriate memory limits** based on profiling
2. **Implement memory monitoring** with early warnings
3. **Use streaming** for large data processing
4. **Regular memory profiling** during development
5. **Load testing** with realistic data sizes

---

## Related Runbooks
- [Memory Exhaustion](./memory-exhaustion.md)
- [High CPU](./high-cpu.md)
- [High Error Rate](./high-error-rate.md)
