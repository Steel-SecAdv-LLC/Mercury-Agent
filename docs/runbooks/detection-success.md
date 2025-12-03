# OMNI-AVA Runbook: Detection Success Rate Drop

## Alert: OmniAvaDetectionSuccessRateDrop

### Overview
This runbook provides guidance for responding when the anomaly detection success rate drops below acceptable thresholds, indicating issues with the ML detection pipeline.

### Alert Threshold
- **Warning**: Detection success rate < 95% for 10 minutes

### Impact
- Anomaly detection may be unreliable
- Users may receive incorrect results
- Critical anomalies may be missed
- Business decisions based on detection may be affected

---

## Diagnosis Steps

### 1. Check Detection Metrics
```promql
# Detection success rate
sum(rate(omni_detection_success_total{app="omni-ava"}[5m])) / sum(rate(omni_detection_requests_total{app="omni-ava"}[5m]))

# Detection errors by type
sum(rate(omni_detection_errors_total{app="omni-ava"}[5m])) by (error_type)
```

### 2. Check Engine Logs
```bash
# View engine logs for errors
kubectl logs -n omni-ava -l app.kubernetes.io/component=engine --tail=200 | grep -i "error\|fail\|exception"

# Check for model loading issues
kubectl logs -n omni-ava -l app.kubernetes.io/component=engine --tail=100 | grep -i "model\|load"
```

### 3. Check Model Status
```bash
# Verify models are loaded
kubectl exec -n omni-ava deployment/omni-ava-engine -- curl -s localhost:8080/models/status

# Check model health
kubectl exec -n omni-ava deployment/omni-ava-engine -- curl -s localhost:8080/health
```

### 4. Check Input Data Patterns
```bash
# Look for invalid input errors
kubectl logs -n omni-ava -l app.kubernetes.io/component=api --tail=200 | grep -i "invalid\|validation"

# Check request patterns
kubectl logs -n omni-ava -l app.kubernetes.io/component=api --tail=100 | grep -i "detect"
```

---

## Resolution Steps

### Scenario 1: Model Loading Failure

**Symptoms:**
- Model not found errors
- Model loading timeouts
- Engine pods restarting

**Actions:**
1. Check model availability:
   ```bash
   kubectl exec -n omni-ava deployment/omni-ava-engine -- ls -la /data/models/
   ```
2. Restart engine to reload models:
   ```bash
   kubectl rollout restart deployment/omni-ava-engine -n omni-ava
   ```
3. Verify model files are not corrupted:
   ```bash
   kubectl exec -n omni-ava deployment/omni-ava-engine -- python -c "import torch; torch.load('/data/models/fusion_model.pt')"
   ```

### Scenario 2: Resource Exhaustion

**Symptoms:**
- OOMKilled engine pods
- Inference timeouts
- High memory/CPU usage

**Actions:**
1. Check resource usage:
   ```bash
   kubectl top pods -n omni-ava -l app.kubernetes.io/component=engine
   ```
2. Increase resources:
   ```bash
   kubectl patch deployment omni-ava-engine -n omni-ava -p '{"spec":{"template":{"spec":{"containers":[{"name":"engine","resources":{"limits":{"memory":"16Gi","cpu":"4"}}}]}}}}'
   ```
3. Scale engine pods:
   ```bash
   kubectl scale deployment omni-ava-engine -n omni-ava --replicas=3
   ```

### Scenario 3: Invalid Input Data

**Symptoms:**
- Validation errors in logs
- Specific endpoints failing
- Client sending malformed data

**Actions:**
1. Identify problematic requests:
   ```bash
   kubectl logs -n omni-ava -l app.kubernetes.io/component=api --tail=200 | grep -B5 "validation error"
   ```
2. Add input validation:
   ```bash
   kubectl set env deployment/omni-ava-api -n omni-ava STRICT_INPUT_VALIDATION=true
   ```
3. Return clear error messages to clients

### Scenario 4: Model Degradation

**Symptoms:**
- Model producing incorrect results
- Confidence scores abnormally low
- No obvious errors but poor quality

**Actions:**
1. Check model metrics:
   ```bash
   kubectl exec -n omni-ava deployment/omni-ava-engine -- curl -s localhost:8080/models/metrics
   ```
2. Compare with baseline performance
3. Consider model retraining or rollback:
   ```bash
   kubectl set env deployment/omni-ava-engine -n omni-ava MODEL_VERSION=v1.2.0
   ```
4. Engage ML team for investigation

---

## Escalation

If detection success rate doesn't improve:

1. **Notify ML team** immediately
2. **Create incident** in incident management system
3. **Consider fallback** to simpler detection methods

### Escalation Contacts
- **ML Team**: ml-oncall@example.com
- **Platform Team**: platform-oncall@example.com
- **Data Science Lead**: ds-lead@example.com

---

## Prevention

1. **Monitor model performance** continuously
2. **Implement model validation** before deployment
3. **A/B testing** for model updates
4. **Regular model retraining** schedules
5. **Input validation** at API layer

---

## Related Runbooks
- [Slow Detection](./slow-detection.md)
- [Inference Errors](./inference-errors.md)
- [High Error Rate](./high-error-rate.md)
