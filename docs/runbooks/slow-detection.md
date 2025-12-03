# OMNI-AVA Runbook: Slow Detection Processing

## Alert: OmniAvaSlowDetection

### Overview
This runbook provides guidance for responding when anomaly detection processing times exceed acceptable thresholds, affecting user experience and system throughput.

### Alert Threshold
- **Warning**: P95 detection time > 10 seconds for 10 minutes

### Impact
- Degraded user experience
- Potential request timeouts
- Reduced system throughput
- May affect downstream systems waiting for results

---

## Diagnosis Steps

### 1. Check Detection Latency
```promql
# P95 detection latency
histogram_quantile(0.95, sum(rate(omni_detection_duration_seconds_bucket{app="omni-ava"}[5m])) by (le, detector_type))

# Detection latency by detector type
histogram_quantile(0.95, sum(rate(omni_detection_duration_seconds_bucket{app="omni-ava"}[5m])) by (le, detector_type))
```

### 2. Check Engine Performance
```bash
# View engine logs for slow operations
kubectl logs -n omni-ava -l app.kubernetes.io/component=engine --tail=200 | grep -i "duration\|slow\|timeout"

# Check engine resource usage
kubectl top pods -n omni-ava -l app.kubernetes.io/component=engine
```

### 3. Check Model Inference Times
```bash
# Check inference metrics
kubectl exec -n omni-ava deployment/omni-ava-engine -- curl -s localhost:8080/metrics | grep inference

# View model-specific timing
kubectl logs -n omni-ava -l app.kubernetes.io/component=engine --tail=100 | grep -i "inference\|predict"
```

### 4. Check Input Data Characteristics
```bash
# Look for large input processing
kubectl logs -n omni-ava -l app.kubernetes.io/component=engine --tail=100 | grep -i "batch\|size\|shape"

# Check for complex detection requests
kubectl logs -n omni-ava -l app.kubernetes.io/component=api --tail=100 | grep -i "detect"
```

---

## Resolution Steps

### Scenario 1: Large Batch Sizes

**Symptoms:**
- Latency correlates with input size
- Batch processing logs show large sizes
- Memory usage spikes during processing

**Actions:**
1. Reduce maximum batch size:
   ```bash
   kubectl set env deployment/omni-ava-engine -n omni-ava MAX_BATCH_SIZE=32
   ```
2. Enable batch splitting:
   ```bash
   kubectl set env deployment/omni-ava-engine -n omni-ava ENABLE_BATCH_SPLITTING=true
   ```
3. Implement streaming for large datasets

### Scenario 2: Model Complexity

**Symptoms:**
- Specific detector types are slow
- Fusion detection slower than individual
- Model inference dominates latency

**Actions:**
1. Enable model optimization:
   ```bash
   kubectl set env deployment/omni-ava-engine -n omni-ava ENABLE_MODEL_OPTIMIZATION=true
   ```
2. Use lighter model variants:
   ```bash
   kubectl set env deployment/omni-ava-engine -n omni-ava MODEL_VARIANT=lite
   ```
3. Consider model quantization for faster inference

### Scenario 3: Resource Contention

**Symptoms:**
- High CPU usage on engine pods
- Multiple concurrent requests
- Latency varies with load

**Actions:**
1. Scale engine pods:
   ```bash
   kubectl scale deployment omni-ava-engine -n omni-ava --replicas=5
   ```
2. Increase CPU allocation:
   ```bash
   kubectl patch deployment omni-ava-engine -n omni-ava -p '{"spec":{"template":{"spec":{"containers":[{"name":"engine","resources":{"limits":{"cpu":"4"},"requests":{"cpu":"2"}}}]}}}}'
   ```
3. Implement request queuing:
   ```bash
   kubectl set env deployment/omni-ava-engine -n omni-ava MAX_CONCURRENT_INFERENCES=4
   ```

### Scenario 4: Cold Start Issues

**Symptoms:**
- First requests after idle are slow
- Latency improves after warmup
- Model loading on demand

**Actions:**
1. Enable model preloading:
   ```bash
   kubectl set env deployment/omni-ava-engine -n omni-ava PRELOAD_ALL_MODELS=true
   ```
2. Implement warmup requests:
   ```bash
   kubectl set env deployment/omni-ava-engine -n omni-ava ENABLE_WARMUP=true WARMUP_REQUESTS=10
   ```
3. Configure readiness probe to wait for warmup:
   ```bash
   kubectl patch deployment omni-ava-engine -n omni-ava -p '{"spec":{"template":{"spec":{"containers":[{"name":"engine","readinessProbe":{"initialDelaySeconds":60}}]}}}}'
   ```

---

## Escalation

If detection latency doesn't improve:

1. **Notify ML team** for model optimization
2. **Engage platform team** for infrastructure scaling
3. **Consider temporary SLO adjustment** if needed

### Escalation Contacts
- **ML Team**: ml-oncall@example.com
- **Platform Team**: platform-oncall@example.com

---

## Prevention

1. **Set appropriate timeouts** for detection requests
2. **Implement request size limits**
3. **Regular performance testing** with realistic data
4. **Model optimization** as part of deployment
5. **Capacity planning** based on expected load

---

## Related Runbooks
- [Detection Success](./detection-success.md)
- [High Latency](./high-latency.md)
- [High CPU](./high-cpu.md)
