# OMNI-AVA Runbook: High Error Rate

## Alert: OmniAvaHighErrorRate / OmniAvaElevatedErrorRate

### Overview
This runbook provides guidance for responding to high error rate alerts in the OMNI-AVA platform.

### Alert Thresholds
- **Critical (OmniAvaHighErrorRate)**: Error rate > 5% for 5 minutes
- **Warning (OmniAvaElevatedErrorRate)**: Error rate > 1% for 10 minutes

### Impact
- Users may experience failed API requests
- Detection processing may fail or return incorrect results
- Service reliability is degraded

---

## Diagnosis Steps

### 1. Check Overall Service Health
```bash
# View pod status
kubectl get pods -n omni-ava -l app.kubernetes.io/name=omni-ava

# Check recent events
kubectl get events -n omni-ava --sort-by='.lastTimestamp' | head -20

# View deployment status
kubectl describe deployment omni-ava-api -n omni-ava
```

### 2. Analyze Error Patterns
```bash
# Check API pod logs for errors
kubectl logs -n omni-ava -l app.kubernetes.io/component=api --tail=100 | grep -i error

# Get error rate by endpoint
# (Prometheus query)
sum(rate(http_requests_total{app="omni-ava",status=~"5.."}[5m])) by (path)
```

### 3. Check Resource Utilization
```bash
# View resource usage
kubectl top pods -n omni-ava

# Check if pods are being OOMKilled
kubectl get pods -n omni-ava -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[*].lastState.terminated.reason}{"\n"}{end}'
```

### 4. Check Dependencies
```bash
# Verify engine pods are running
kubectl get pods -n omni-ava -l app.kubernetes.io/component=engine

# Check PVC status
kubectl get pvc -n omni-ava

# Verify network connectivity
kubectl exec -n omni-ava deployment/omni-ava-api -- curl -s localhost:8000/health
```

---

## Resolution Steps

### Scenario 1: Application Errors (500s)

**Symptoms:**
- HTTP 500 errors in logs
- Stack traces in application logs

**Actions:**
1. Identify the failing endpoint from logs
2. Check for recent deployments:
   ```bash
   kubectl rollout history deployment/omni-ava-api -n omni-ava
   ```
3. If recent deployment caused issue, rollback:
   ```bash
   kubectl rollout undo deployment/omni-ava-api -n omni-ava
   ```
4. Investigate root cause in application code

### Scenario 2: Resource Exhaustion

**Symptoms:**
- OOMKilled pods
- High CPU/memory usage
- Slow response times before errors

**Actions:**
1. Scale up replicas:
   ```bash
   kubectl scale deployment omni-ava-api -n omni-ava --replicas=5
   ```
2. Increase resource limits (if appropriate):
   ```bash
   kubectl patch deployment omni-ava-api -n omni-ava -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
   ```
3. Check for memory leaks in application

### Scenario 3: Dependency Failures

**Symptoms:**
- Connection errors in logs
- Timeouts to external services

**Actions:**
1. Verify engine pods are healthy:
   ```bash
   kubectl logs -n omni-ava -l app.kubernetes.io/component=engine --tail=50
   ```
2. Check network policies aren't blocking traffic:
   ```bash
   kubectl get networkpolicies -n omni-ava
   ```
3. Verify PVCs are accessible:
   ```bash
   kubectl describe pvc -n omni-ava
   ```

### Scenario 4: Load Spike

**Symptoms:**
- Sudden increase in request rate
- HPA scaling events
- Request queuing

**Actions:**
1. Check current HPA status:
   ```bash
   kubectl get hpa -n omni-ava
   ```
2. Manually scale if HPA is at max:
   ```bash
   kubectl scale deployment omni-ava-api -n omni-ava --replicas=10
   ```
3. Consider rate limiting if traffic is malicious:
   ```bash
   # Update ingress rate limiting
   kubectl annotate ingress omni-ava -n omni-ava nginx.ingress.kubernetes.io/limit-rps="50" --overwrite
   ```

---

## Escalation

If the issue persists after following these steps:

1. **Page on-call engineer** via PagerDuty
2. **Create incident** in incident management system
3. **Notify stakeholders** via Slack #omni-ava-incidents

### Escalation Contacts
- **Platform Team**: steel.sa.llc@gmail.com
- **ML Team**: steel.sa.llc@gmail.com
- **Management**: steel.sa.llc@gmail.com

---

## Prevention

1. **Implement circuit breakers** for external dependencies
2. **Add request validation** to fail fast on invalid input
3. **Improve logging** for better error diagnosis
4. **Add integration tests** for critical paths
5. **Review resource limits** regularly

---

## Related Runbooks
- [Service Down](./service-down.md)
- [High Latency](./high-latency.md)
- [Memory Exhaustion](./memory-exhaustion.md)
