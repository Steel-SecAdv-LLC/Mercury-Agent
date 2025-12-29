# Mercury-Agent Runbook: Memory Exhaustion

## Alert: OmniMercuryMemoryExhaustion

### Overview
This runbook provides guidance for responding when Mercury-Agent pods are experiencing memory exhaustion, which can lead to OOMKilled pods and service degradation.

### Alert Threshold
- **Critical**: Memory usage > 95% of limit for 5 minutes

### Impact
- Pods may be OOMKilled and restarted
- Detection processing may fail mid-operation
- Service latency increases as memory pressure builds
- Potential data loss from interrupted operations

---

## Diagnosis Steps

### 1. Check Memory Usage
```bash
# View current memory usage
kubectl top pods -n mercury-agent

# Check memory limits vs usage
kubectl get pods -n mercury-agent -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].resources.limits.memory}{"\n"}{end}'

# Check for OOMKilled events
kubectl get pods -n mercury-agent -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[*].lastState.terminated.reason}{"\n"}{end}'
```

### 2. Analyze Memory Patterns
```bash
# Check pod restart counts
kubectl get pods -n mercury-agent -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[*].restartCount}{"\n"}{end}'

# View memory metrics over time (Prometheus query)
container_memory_usage_bytes{namespace="mercury-agent",container="api"}
```

### 3. Check Application Logs
```bash
# Look for memory-related errors
kubectl logs -n mercury-agent -l app.kubernetes.io/component=api --tail=200 | grep -i "memory\|oom\|heap"

# Check for large model loading
kubectl logs -n mercury-agent -l app.kubernetes.io/component=engine --tail=100 | grep -i "loading\|model"
```

### 4. Check Workload Patterns
```bash
# View recent detection requests
kubectl logs -n mercury-agent -l app.kubernetes.io/component=api --tail=100 | grep -i "detect"

# Check batch sizes being processed
kubectl logs -n mercury-agent -l app.kubernetes.io/component=engine --tail=100 | grep -i "batch"
```

---

## Resolution Steps

### Scenario 1: Memory Leak

**Symptoms:**
- Memory usage grows continuously over time
- No correlation with request volume
- Restarts temporarily fix the issue

**Actions:**
1. Restart affected pods to recover immediately:
   ```bash
   kubectl rollout restart deployment/mercury-agent-api -n mercury-agent
   ```
2. Enable memory profiling for investigation:
   ```bash
   kubectl set env deployment/mercury-agent-api -n mercury-agent ENABLE_MEMORY_PROFILING=true
   ```
3. Collect heap dump before next restart:
   ```bash
   kubectl exec -n mercury-agent <pod-name> -- python -c "import tracemalloc; tracemalloc.start()"
   ```
4. File bug report with memory profile data

### Scenario 2: Large Batch Processing

**Symptoms:**
- Memory spikes during detection operations
- Correlation with large input data
- Batch processing logs show large sizes

**Actions:**
1. Reduce batch size configuration:
   ```bash
   kubectl set env deployment/mercury-agent-engine -n mercury-agent MAX_BATCH_SIZE=32
   ```
2. Increase memory limits temporarily:
   ```bash
   kubectl patch deployment mercury-agent-engine -n mercury-agent -p '{"spec":{"template":{"spec":{"containers":[{"name":"engine","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
   ```
3. Consider implementing streaming processing for large datasets

### Scenario 3: Model Loading

**Symptoms:**
- Memory spike at pod startup
- Multiple models loaded simultaneously
- Engine pods affected more than API pods

**Actions:**
1. Enable lazy model loading:
   ```bash
   kubectl set env deployment/mercury-agent-engine -n mercury-agent LAZY_MODEL_LOADING=true
   ```
2. Reduce number of preloaded models:
   ```bash
   kubectl set env deployment/mercury-agent-engine -n mercury-agent PRELOAD_MODELS="statistical,temporal"
   ```
3. Increase memory limits for engine pods:
   ```bash
   kubectl patch deployment mercury-agent-engine -n mercury-agent -p '{"spec":{"template":{"spec":{"containers":[{"name":"engine","resources":{"limits":{"memory":"16Gi"}}}]}}}}'
   ```

### Scenario 4: Insufficient Resources

**Symptoms:**
- Memory limits too low for workload
- Consistent OOMKills under normal load
- No memory leak pattern

**Actions:**
1. Increase memory limits:
   ```bash
   kubectl patch deployment mercury-agent-api -n mercury-agent -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"8Gi"},"requests":{"memory":"4Gi"}}}]}}}}'
   ```
2. Scale horizontally to distribute load:
   ```bash
   kubectl scale deployment mercury-agent-api -n mercury-agent --replicas=5
   ```
3. Review and update resource quotas if needed

---

## Escalation

If memory issues persist after following these steps:

1. **Page on-call engineer** via PagerDuty
2. **Create incident** in incident management system
3. **Notify ML team** if model-related issues suspected
4. **Collect diagnostics** before any destructive actions

### Escalation Contacts
- **Platform Team**: platform-oncall@example.com
- **ML Team**: ml-oncall@example.com
- **Management**: ops-manager@example.com

---

## Prevention

1. **Set appropriate memory limits** based on profiling data
2. **Implement memory monitoring** with early warning thresholds
3. **Use streaming processing** for large datasets
4. **Regular memory profiling** during development
5. **Load testing** with realistic data sizes

---

## Related Runbooks
- [High Memory](./high-memory.md)
- [High Error Rate](./high-error-rate.md)
- [Service Down](./service-down.md)
