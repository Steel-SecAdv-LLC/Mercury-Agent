# Mercury-Agent Runbook: Elevated Error Rate

## Alert: OmniMercuryElevatedErrorRate

### Overview
This runbook provides guidance for responding to elevated (but not critical) error rates in the Mercury-Agent platform. This is a warning-level alert indicating degraded service quality.

### Alert Threshold
- **Warning**: Error rate > 1% for 10 minutes

### Impact
- Some users may experience failed requests
- Service reliability is slightly degraded
- May indicate an emerging issue that could escalate

---

## Diagnosis Steps

### 1. Identify Error Patterns
```bash
# Check error distribution by status code
kubectl logs -n mercury-agent -l app.kubernetes.io/component=api --tail=500 | grep -E "HTTP/[0-9.]+ [45][0-9]{2}" | sort | uniq -c

# View recent errors
kubectl logs -n mercury-agent -l app.kubernetes.io/component=api --tail=100 | grep -i error
```

### 2. Check Error Rate by Endpoint
```promql
# Prometheus query for error rate by path
sum(rate(http_requests_total{app="mercury-agent",status=~"[45].."}[5m])) by (path)
/
sum(rate(http_requests_total{app="mercury-agent"}[5m])) by (path)
```

### 3. Check Recent Changes
```bash
# View deployment history
kubectl rollout history deployment/mercury-agent-api -n mercury-agent

# Check for recent config changes
kubectl get configmap -n mercury-agent -o yaml | grep -A5 "metadata:"
```

### 4. Check Dependencies
```bash
# Verify engine health
kubectl exec -n mercury-agent deployment/mercury-agent-api -- curl -s localhost:8000/health

# Check engine pod status
kubectl get pods -n mercury-agent -l app.kubernetes.io/component=engine
```

---

## Resolution Steps

### Scenario 1: Specific Endpoint Failing

**Symptoms:**
- Errors concentrated on one endpoint
- Other endpoints working normally

**Actions:**
1. Identify the failing endpoint from logs
2. Check if endpoint has specific dependencies
3. Review recent changes to that endpoint's code
4. Consider disabling the endpoint temporarily if non-critical:
   ```bash
   kubectl set env deployment/mercury-agent-api -n mercury-agent DISABLE_ENDPOINTS="/problematic/endpoint"
   ```

### Scenario 2: Intermittent Failures

**Symptoms:**
- Errors spread across endpoints
- No clear pattern
- Some requests succeed, some fail

**Actions:**
1. Check for resource contention:
   ```bash
   kubectl top pods -n mercury-agent
   ```
2. Look for timeout patterns:
   ```bash
   kubectl logs -n mercury-agent -l app.kubernetes.io/component=api --tail=200 | grep -i timeout
   ```
3. Scale up to reduce load per pod:
   ```bash
   kubectl scale deployment mercury-agent-api -n mercury-agent --replicas=5
   ```

### Scenario 3: Client Errors (4xx)

**Symptoms:**
- High rate of 400/401/403/404 errors
- May indicate client misconfiguration or attack

**Actions:**
1. Identify source of bad requests:
   ```bash
   kubectl logs -n mercury-agent -l app.kubernetes.io/component=api --tail=200 | grep "HTTP/1.1 4"
   ```
2. Check for authentication issues:
   ```bash
   kubectl logs -n mercury-agent -l app.kubernetes.io/component=api --tail=100 | grep -i "auth\|token\|unauthorized"
   ```
3. Review rate limiting if potential abuse:
   ```bash
   kubectl get ingress mercury-agent -n mercury-agent -o yaml | grep -i limit
   ```

### Scenario 4: Gradual Degradation

**Symptoms:**
- Error rate slowly increasing over time
- May indicate resource exhaustion or leak

**Actions:**
1. Check memory and CPU trends
2. Review connection pool status
3. Consider proactive restart:
   ```bash
   kubectl rollout restart deployment/mercury-agent-api -n mercury-agent
   ```

---

## Escalation

If error rate continues to rise or doesn't improve:

1. **Monitor closely** for escalation to critical threshold
2. **Notify team** via Slack #mercury-agent-alerts
3. **Prepare for incident** if rate exceeds 5%

### Escalation Contacts
- **Platform Team**: platform-oncall@example.com
- **Development Team**: dev-oncall@example.com

---

## Prevention

1. **Implement comprehensive error handling** in application code
2. **Add circuit breakers** for external dependencies
3. **Monitor error rates** with lower warning thresholds
4. **Regular load testing** to identify breaking points
5. **Canary deployments** to catch issues early

---

## Related Runbooks
- [High Error Rate](./high-error-rate.md)
- [High Latency](./high-latency.md)
- [Service Down](./service-down.md)
