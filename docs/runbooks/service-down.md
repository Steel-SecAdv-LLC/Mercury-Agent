# OMNI-AVA Runbook: Service Down

## Alert: OmniAvaServiceDown / OmniAvaAllReplicasDown

### Overview
This runbook provides guidance for responding when the OMNI-AVA service is down or unavailable.

### Alert Thresholds
- **OmniAvaServiceDown**: Service endpoint returns down for > 1 minute
- **OmniAvaAllReplicasDown**: Zero available replicas for > 1 minute

### Impact
- **Critical**: Complete service outage
- All API requests will fail
- No anomaly detection processing

---

## Immediate Actions

### 1. Verify the Alert
```bash
# Check pod status
kubectl get pods -n omni-ava

# Check endpoint directly
kubectl exec -n omni-ava deployment/omni-ava-api -- curl -s localhost:8000/health || echo "Health check failed"

# Check service endpoints
kubectl get endpoints omni-ava-api -n omni-ava
```

### 2. Check for Recent Changes
```bash
# View recent deployments
kubectl rollout history deployment/omni-ava-api -n omni-ava

# Check for recent config changes
kubectl get configmap omni-ava-config -n omni-ava -o yaml

# View recent events
kubectl get events -n omni-ava --sort-by='.lastTimestamp' | head -30
```

---

## Diagnosis Steps

### Scenario 1: Pods in CrashLoopBackOff

**Symptoms:**
```bash
$ kubectl get pods -n omni-ava
NAME                            READY   STATUS             RESTARTS   AGE
omni-ava-api-xxx-yyy           0/1     CrashLoopBackOff   5          10m
```

**Actions:**
1. Check pod logs:
   ```bash
   kubectl logs -n omni-ava -l app.kubernetes.io/component=api --previous
   ```
2. Check for configuration errors:
   ```bash
   kubectl describe pod -n omni-ava -l app.kubernetes.io/component=api
   ```
3. If config issue, fix and redeploy:
   ```bash
   kubectl rollout restart deployment/omni-ava-api -n omni-ava
   ```

### Scenario 2: Pods in Pending State

**Symptoms:**
```bash
$ kubectl get pods -n omni-ava
NAME                            READY   STATUS    RESTARTS   AGE
omni-ava-api-xxx-yyy           0/1     Pending   0          10m
```

**Actions:**
1. Check resource availability:
   ```bash
   kubectl describe pod -n omni-ava -l app.kubernetes.io/component=api | grep -A 10 Events
   ```
2. Check node capacity:
   ```bash
   kubectl top nodes
   kubectl describe nodes | grep -A 5 "Allocated resources"
   ```
3. Check PVC status:
   ```bash
   kubectl get pvc -n omni-ava
   kubectl describe pvc -n omni-ava
   ```

### Scenario 3: Pods Running but Not Ready

**Symptoms:**
```bash
$ kubectl get pods -n omni-ava
NAME                            READY   STATUS    RESTARTS   AGE
omni-ava-api-xxx-yyy           0/1     Running   0          10m
```

**Actions:**
1. Check readiness probe:
   ```bash
   kubectl describe pod -n omni-ava -l app.kubernetes.io/component=api | grep -A 10 "Readiness"
   ```
2. Check application startup:
   ```bash
   kubectl logs -n omni-ava -l app.kubernetes.io/component=api --tail=100
   ```
3. Verify dependencies are available:
   ```bash
   kubectl exec -n omni-ava deployment/omni-ava-api -- python -c "from omni_anomaly_engine import OmniAnomalyEngine; print('OK')"
   ```

### Scenario 4: No Pods Scheduled

**Symptoms:**
```bash
$ kubectl get pods -n omni-ava
No resources found in omni-ava namespace.
```

**Actions:**
1. Check deployment status:
   ```bash
   kubectl get deployment -n omni-ava
   kubectl describe deployment omni-ava-api -n omni-ava
   ```
2. Check if deployment was scaled to zero:
   ```bash
   kubectl get deployment omni-ava-api -n omni-ava -o jsonpath='{.spec.replicas}'
   ```
3. Scale up if needed:
   ```bash
   kubectl scale deployment omni-ava-api -n omni-ava --replicas=3
   ```

---

## Recovery Steps

### Quick Recovery: Rollback
If a recent deployment caused the issue:
```bash
# Rollback to previous version
kubectl rollout undo deployment/omni-ava-api -n omni-ava

# Verify rollback
kubectl rollout status deployment/omni-ava-api -n omni-ava
```

### Quick Recovery: Force Restart
If pods are stuck:
```bash
# Force delete stuck pods
kubectl delete pods -n omni-ava -l app.kubernetes.io/component=api --force --grace-period=0

# Trigger new deployment
kubectl rollout restart deployment/omni-ava-api -n omni-ava
```

### Recovery: Fix Configuration
If configuration is broken:
```bash
# Edit configmap
kubectl edit configmap omni-ava-config -n omni-ava

# Restart pods to pick up changes
kubectl rollout restart deployment/omni-ava-api -n omni-ava
```

---

## Post-Incident Actions

1. **Update status page** when service is restored
2. **Notify stakeholders** of resolution
3. **Schedule post-mortem** within 48 hours
4. **Document root cause** and corrective actions
5. **Update runbooks** if new scenarios identified

---

## Escalation

### Escalation Criteria
- Issue not resolved within 15 minutes
- Root cause not identified
- Multiple services affected

### Contacts
- **Platform Team Lead**: platform-lead@example.com
- **SRE Manager**: sre-manager@example.com
- **VP Engineering**: vp-eng@example.com (for P1 incidents)

---

## Related Runbooks
- [High Error Rate](./high-error-rate.md)
- [All Replicas Down](./all-replicas-down.md)
- [Memory Exhaustion](./memory-exhaustion.md)
