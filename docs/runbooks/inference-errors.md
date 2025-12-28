# Mercury-Agent Runbook: Model Inference Errors

## Alert: OmniMercuryModelInferenceErrors

### Overview
This runbook provides guidance for responding when model inference errors are detected in the Mercury-Agent platform, indicating issues with the ML prediction pipeline.

### Alert Threshold
- **Warning**: Inference errors > 0.1 per second for 5 minutes

### Impact
- Detection requests may fail
- Users receive error responses instead of results
- Anomaly detection reliability is compromised
- May indicate model or infrastructure issues

---

## Diagnosis Steps

### 1. Check Inference Error Metrics
```promql
# Inference error rate
sum(rate(omni_model_inference_errors_total{app="mercury-agent"}[5m])) by (model, error_type)

# Inference success vs failure
sum(rate(omni_model_inference_total{app="mercury-agent",status="success"}[5m])) / sum(rate(omni_model_inference_total{app="mercury-agent"}[5m]))
```

### 2. Check Engine Logs
```bash
# View inference errors
kubectl logs -n mercury-agent -l app.kubernetes.io/component=engine --tail=300 | grep -i "inference\|error\|exception"

# Check for specific error types
kubectl logs -n mercury-agent -l app.kubernetes.io/component=engine --tail=200 | grep -E "RuntimeError|ValueError|TypeError"
```

### 3. Check Model Status
```bash
# Verify model health
kubectl exec -n mercury-agent deployment/mercury-agent-engine -- curl -s localhost:8080/models/health

# Check loaded models
kubectl exec -n mercury-agent deployment/mercury-agent-engine -- curl -s localhost:8080/models/list
```

### 4. Check Input Patterns
```bash
# Look for input-related errors
kubectl logs -n mercury-agent -l app.kubernetes.io/component=engine --tail=200 | grep -i "input\|shape\|dimension"

# Check for NaN or invalid values
kubectl logs -n mercury-agent -l app.kubernetes.io/component=engine --tail=100 | grep -i "nan\|inf\|invalid"
```

---

## Resolution Steps

### Scenario 1: Model File Corruption

**Symptoms:**
- Model loading errors
- Checksum failures
- Deserialization errors

**Actions:**
1. Check model file integrity:
   ```bash
   kubectl exec -n mercury-agent deployment/mercury-agent-engine -- md5sum /data/models/*.pt
   ```
2. Re-download model files:
   ```bash
   kubectl exec -n mercury-agent deployment/mercury-agent-engine -- python -c "from omni_mercury_engine.ml import download_models; download_models()"
   ```
3. Restart engine to reload:
   ```bash
   kubectl rollout restart deployment/mercury-agent-engine -n mercury-agent
   ```

### Scenario 2: Input Data Issues

**Symptoms:**
- Shape mismatch errors
- NaN/Inf in input data
- Type conversion errors

**Actions:**
1. Enable strict input validation:
   ```bash
   kubectl set env deployment/mercury-agent-api -n mercury-agent STRICT_INPUT_VALIDATION=true
   ```
2. Add input sanitization:
   ```bash
   kubectl set env deployment/mercury-agent-engine -n mercury-agent SANITIZE_INPUT=true REPLACE_NAN=true
   ```
3. Return clear error messages to clients

### Scenario 3: Resource Exhaustion

**Symptoms:**
- OOM errors during inference
- CUDA out of memory (if GPU)
- Timeout errors

**Actions:**
1. Reduce batch size:
   ```bash
   kubectl set env deployment/mercury-agent-engine -n mercury-agent MAX_BATCH_SIZE=16
   ```
2. Increase memory limits:
   ```bash
   kubectl patch deployment mercury-agent-engine -n mercury-agent -p '{"spec":{"template":{"spec":{"containers":[{"name":"engine","resources":{"limits":{"memory":"16Gi"}}}]}}}}'
   ```
3. Enable memory-efficient inference:
   ```bash
   kubectl set env deployment/mercury-agent-engine -n mercury-agent MEMORY_EFFICIENT_INFERENCE=true
   ```

### Scenario 4: Model Version Mismatch

**Symptoms:**
- Errors after deployment
- API/model version incompatibility
- Missing model features

**Actions:**
1. Check model version:
   ```bash
   kubectl exec -n mercury-agent deployment/mercury-agent-engine -- cat /data/models/version.txt
   ```
2. Rollback to previous model version:
   ```bash
   kubectl set env deployment/mercury-agent-engine -n mercury-agent MODEL_VERSION=v1.2.0
   kubectl rollout restart deployment/mercury-agent-engine -n mercury-agent
   ```
3. Verify API compatibility with model version

### Scenario 5: Numerical Instability

**Symptoms:**
- NaN in model outputs
- Overflow/underflow errors
- Inconsistent results

**Actions:**
1. Enable numerical stability checks:
   ```bash
   kubectl set env deployment/mercury-agent-engine -n mercury-agent CHECK_NUMERICAL_STABILITY=true
   ```
2. Use mixed precision carefully:
   ```bash
   kubectl set env deployment/mercury-agent-engine -n mercury-agent USE_MIXED_PRECISION=false
   ```
3. Engage ML team for model investigation

---

## Escalation

If inference errors persist:

1. **Notify ML team** immediately
2. **Create incident** for tracking
3. **Consider disabling affected models** temporarily

### Escalation Contacts
- **ML Team**: ml-oncall@example.com
- **Platform Team**: platform-oncall@example.com
- **Data Science Lead**: ds-lead@example.com

---

## Prevention

1. **Model validation** before deployment
2. **Input validation** at API layer
3. **Comprehensive testing** with edge cases
4. **Monitoring inference metrics** continuously
5. **Canary deployments** for model updates

---

## Related Runbooks
- [Detection Success](./detection-success.md)
- [Slow Detection](./slow-detection.md)
- [Memory Exhaustion](./memory-exhaustion.md)
