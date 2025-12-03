# OMNI-AVA Runbook: HPA at Maximum Replicas

## Alert: OmniAvaHPAAtMax

### Overview
This runbook provides guidance for responding when the Horizontal Pod Autoscaler (HPA) has scaled to its maximum replica count and may not be able to handle additional load.

### Alert Threshold
- **Warning**: HPA at maximum replicas for 30 minutes

### Impact
- Cannot scale further to handle load
- May experience degraded performance
- Risk of service degradation under increased load
- SLO breach risk if load continues to grow

---

## Diagnosis Steps

### 1. Check HPA Status
```bash
# View HPA details
kubectl get hpa -n omni-ava

# Check HPA events and conditions
kubectl describe hpa omni-ava-api -n omni-ava

# View current vs desired replicas
kubectl get hpa -n omni-ava -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.currentReplicas}/{.spec.maxReplicas}{"\n"}{end}'
```

### 2. Check Current Load
```bash
# View pod resource usage
kubectl top pods -n omni-ava

# Check request rate (Prometheus)
sum(rate(http_requests_total{app="omni-ava"}[5m]))
```

### 3. Analyze Traffic Patterns
```promql
# Request rate over time
sum(rate(http_requests_total{app="omni-ava"}[5m]))

# Compare to historical baseline
sum(rate(http_requests_total{app="omni-ava"}[5m])) / sum(rate(http_requests_total{app="omni-ava"}[5m] offset 1d))
```

### 4. Check Cluster Capacity
```bash
# View node resources
kubectl top nodes

# Check for pending pods
kubectl get pods -n omni-ava --field-selector=status.phase=Pending
```

---

## Resolution Steps

### Scenario 1: Legitimate Traffic Growth

**Symptoms:**
- Traffic has genuinely increased
- Business event or expected growth
- Performance is acceptable

**Actions:**
1. Increase HPA maximum:
   ```bash
   kubectl patch hpa omni-ava-api -n omni-ava -p '{"spec":{"maxReplicas":30}}'
   ```
2. Verify cluster has capacity:
   ```bash
   kubectl describe nodes | grep -A5 "Allocated resources:"
   ```
3. Request additional cluster capacity if needed

### Scenario 2: Traffic Spike (Temporary)

**Symptoms:**
- Sudden traffic increase
- May be time-limited event
- Performance degrading

**Actions:**
1. Temporarily increase max replicas:
   ```bash
   kubectl patch hpa omni-ava-api -n omni-ava -p '{"spec":{"maxReplicas":25}}'
   ```
2. Enable rate limiting to protect service:
   ```bash
   kubectl annotate ingress omni-ava -n omni-ava nginx.ingress.kubernetes.io/limit-rps="100" --overwrite
   ```
3. Monitor and reduce max after spike passes

### Scenario 3: Inefficient Resource Usage

**Symptoms:**
- High replica count but low per-pod efficiency
- Resource usage not matching traffic
- Possible application issues

**Actions:**
1. Check per-pod metrics:
   ```bash
   kubectl top pods -n omni-ava
   ```
2. Investigate application performance:
   ```bash
   kubectl logs -n omni-ava -l app.kubernetes.io/component=api --tail=100 | grep -i "slow\|timeout"
   ```
3. Optimize application before scaling further
4. Consider vertical scaling (more resources per pod)

### Scenario 4: Attack or Abuse

**Symptoms:**
- Unusual traffic patterns
- Traffic from suspicious sources
- High error rates alongside high traffic

**Actions:**
1. Analyze traffic sources:
   ```bash
   kubectl logs -n omni-ava -l app.kubernetes.io/component=api --tail=500 | grep -oE "[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" | sort | uniq -c | sort -rn | head -20
   ```
2. Implement rate limiting:
   ```bash
   kubectl annotate ingress omni-ava -n omni-ava nginx.ingress.kubernetes.io/limit-rps="50" --overwrite
   ```
3. Block suspicious IPs if needed:
   ```bash
   kubectl annotate ingress omni-ava -n omni-ava nginx.ingress.kubernetes.io/whitelist-source-range="10.0.0.0/8" --overwrite
   ```
4. Engage security team if attack confirmed

---

## Escalation

If HPA remains at max and service is degraded:

1. **Notify team** via Slack #omni-ava-alerts
2. **Request cluster capacity** from infrastructure team
3. **Engage business stakeholders** for traffic management decisions

### Escalation Contacts
- **Platform Team**: platform-oncall@example.com
- **Infrastructure Team**: infra-oncall@example.com
- **Security Team**: security-oncall@example.com (if attack suspected)

---

## Prevention

1. **Set appropriate HPA limits** based on capacity planning
2. **Monitor HPA utilization** trends
3. **Implement rate limiting** as a safety net
4. **Regular capacity planning** reviews
5. **Load testing** to understand scaling limits

---

## Related Runbooks
- [Low Replicas](./low-replicas.md)
- [High CPU](./high-cpu.md)
- [High Latency](./high-latency.md)
