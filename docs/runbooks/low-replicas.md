# OMNI-AVA Runbook: Low Replica Count

## Alert: OmniAvaLowReplicaCount

### Overview
This runbook provides guidance for responding when the OMNI-AVA deployment has fewer replicas than the recommended minimum, reducing service resilience.

### Alert Threshold
- **Warning**: Available replicas < 2 for 5 minutes

### Impact
- Reduced fault tolerance
- Single point of failure risk
- May not handle traffic spikes
- Rolling updates may cause downtime

---

## Diagnosis Steps

### 1. Check Current Replica Status
```bash
# View deployment status
kubectl get deployment -n omni-ava

# Check replica details
kubectl describe deployment omni-ava-api -n omni-ava | grep -A5 "Replicas:"

# View pod status
kubectl get pods -n omni-ava -l app.kubernetes.io/component=api
```

### 2. Check for Scheduling Issues
```bash
# Check for pending pods
kubectl get pods -n omni-ava --field-selector=status.phase=Pending

# View events for scheduling failures
kubectl get events -n omni-ava --field-selector=reason=FailedScheduling

# Check node availability
kubectl get nodes -o wide
```

### 3. Check HPA Status
```bash
# View HPA configuration and status
kubectl get hpa -n omni-ava

# Check HPA events
kubectl describe hpa omni-ava-api -n omni-ava
```

### 4. Check Resource Availability
```bash
# Check namespace resource quotas
kubectl describe resourcequota -n omni-ava

# Check node resources
kubectl describe nodes | grep -A10 "Allocated resources:"
```

---

## Resolution Steps

### Scenario 1: Pod Failures

**Symptoms:**
- Pods in CrashLoopBackOff or Error state
- Restart count increasing
- Application errors in logs

**Actions:**
1. Check pod logs for errors:
   ```bash
   kubectl logs -n omni-ava -l app.kubernetes.io/component=api --tail=100
   ```
2. Check for OOMKilled:
   ```bash
   kubectl get pods -n omni-ava -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[*].lastState.terminated.reason}{"\n"}{end}'
   ```
3. If OOMKilled, increase memory limits:
   ```bash
   kubectl patch deployment omni-ava-api -n omni-ava -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
   ```

### Scenario 2: Scheduling Failures

**Symptoms:**
- Pods stuck in Pending state
- FailedScheduling events
- Insufficient resources messages

**Actions:**
1. Check what resources are needed:
   ```bash
   kubectl describe pod <pending-pod> -n omni-ava | grep -A10 "Events:"
   ```
2. Scale down other workloads if possible
3. Request additional cluster capacity
4. Temporarily reduce resource requests:
   ```bash
   kubectl patch deployment omni-ava-api -n omni-ava -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"requests":{"cpu":"250m","memory":"512Mi"}}}]}}}}'
   ```

### Scenario 3: HPA Scaled Down

**Symptoms:**
- HPA shows low target utilization
- Replicas at minimum
- Low traffic period

**Actions:**
1. Verify this is expected behavior:
   ```bash
   kubectl get hpa -n omni-ava -o yaml
   ```
2. If minimum should be higher, update HPA:
   ```bash
   kubectl patch hpa omni-ava-api -n omni-ava -p '{"spec":{"minReplicas":3}}'
   ```
3. Consider time-based scaling for predictable patterns

### Scenario 4: Manual Scale Down

**Symptoms:**
- Deployment shows desired replicas < minimum
- No HPA or HPA disabled
- Recent manual intervention

**Actions:**
1. Check deployment history:
   ```bash
   kubectl rollout history deployment/omni-ava-api -n omni-ava
   ```
2. Scale back up:
   ```bash
   kubectl scale deployment omni-ava-api -n omni-ava --replicas=3
   ```
3. Review who made the change and why

---

## Escalation

If replica count cannot be restored:

1. **Notify team** via Slack #omni-ava-alerts
2. **Document the limitation** for stakeholders
3. **Request cluster resources** if capacity issue

### Escalation Contacts
- **Platform Team**: platform-oncall@example.com
- **Infrastructure Team**: infra-oncall@example.com

---

## Prevention

1. **Set appropriate PDB** to prevent excessive scale-down
2. **Configure HPA minimum replicas** appropriately
3. **Monitor cluster capacity** proactively
4. **Use pod anti-affinity** to spread across nodes
5. **Regular capacity planning** reviews

---

## Related Runbooks
- [All Replicas Down](./all-replicas-down.md)
- [Service Down](./service-down.md)
- [HPA at Max](./hpa-max.md)
