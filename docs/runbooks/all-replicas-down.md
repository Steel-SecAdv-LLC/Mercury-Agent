# OMNI-AVA Runbook: All Replicas Down

## Alert: OmniAvaAllReplicasDown

### Overview
This runbook provides guidance for responding when all OMNI-AVA replicas are unavailable, resulting in a complete service outage.

### Alert Threshold
- **Critical**: All replicas unavailable for 1 minute

### Impact
- Complete service outage
- All API requests will fail
- Detection processing is halted
- Potential data loss if in-flight requests are not handled

---

## Diagnosis Steps

### 1. Check Pod Status
```bash
# View all pods and their status
kubectl get pods -n omni-ava -o wide

# Check for pending or failed pods
kubectl get pods -n omni-ava --field-selector=status.phase!=Running

# View recent events
kubectl get events -n omni-ava --sort-by='.lastTimestamp' | head -30
```

### 2. Check Node Health
```bash
# Verify nodes are ready
kubectl get nodes

# Check node conditions
kubectl describe nodes | grep -A5 "Conditions:"

# Check if pods are unschedulable
kubectl get pods -n omni-ava -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="PodScheduled")].status}{"\n"}{end}'
```

### 3. Check Deployment Status
```bash
# View deployment details
kubectl describe deployment omni-ava-api -n omni-ava
kubectl describe deployment omni-ava-engine -n omni-ava

# Check replica sets
kubectl get replicasets -n omni-ava
```

### 4. Check Resource Availability
```bash
# Check namespace resource quotas
kubectl describe resourcequota -n omni-ava

# Check PVC status
kubectl get pvc -n omni-ava

# Check if storage is available
kubectl describe pvc -n omni-ava
```

---

## Resolution Steps

### Scenario 1: Node Failure

**Symptoms:**
- Nodes in NotReady state
- Pods stuck in Pending

**Actions:**
1. Identify failed nodes:
   ```bash
   kubectl get nodes | grep NotReady
   ```
2. Cordon affected nodes:
   ```bash
   kubectl cordon <node-name>
   ```
3. Force reschedule pods:
   ```bash
   kubectl delete pods -n omni-ava --all --grace-period=0 --force
   ```
4. Contact infrastructure team for node recovery

### Scenario 2: Image Pull Failure

**Symptoms:**
- ImagePullBackOff or ErrImagePull status
- Registry authentication errors

**Actions:**
1. Check image pull secrets:
   ```bash
   kubectl get secrets -n omni-ava | grep docker
   kubectl describe secret <secret-name> -n omni-ava
   ```
2. Verify image exists:
   ```bash
   kubectl describe pod <pod-name> -n omni-ava | grep -A5 "Events:"
   ```
3. Update image pull secret if expired:
   ```bash
   kubectl create secret docker-registry regcred \
     --docker-server=ghcr.io \
     --docker-username=<username> \
     --docker-password=<token> \
     -n omni-ava --dry-run=client -o yaml | kubectl apply -f -
   ```

### Scenario 3: Resource Quota Exceeded

**Symptoms:**
- Pods stuck in Pending
- "exceeded quota" in events

**Actions:**
1. Check current quota usage:
   ```bash
   kubectl describe resourcequota -n omni-ava
   ```
2. Temporarily increase quota or reduce replica count:
   ```bash
   kubectl scale deployment omni-ava-api -n omni-ava --replicas=2
   ```
3. Request quota increase from cluster admin

### Scenario 4: PVC Issues

**Symptoms:**
- Pods waiting for volume attachment
- PVC in Pending state

**Actions:**
1. Check PVC status:
   ```bash
   kubectl describe pvc -n omni-ava
   ```
2. Check storage class availability:
   ```bash
   kubectl get storageclass
   ```
3. If PVC is stuck, delete and recreate (data loss warning):
   ```bash
   kubectl delete pvc <pvc-name> -n omni-ava
   kubectl apply -f k8s/base/pvc.yaml
   ```

---

## Escalation

This is a **critical incident** requiring immediate escalation:

1. **Page on-call engineer immediately** via PagerDuty
2. **Create P1 incident** in incident management system
3. **Notify stakeholders** via Slack #omni-ava-incidents
4. **Start incident bridge call** if not resolved in 5 minutes

### Escalation Contacts
- **Platform Team**: platform-oncall@example.com
- **Infrastructure Team**: infra-oncall@example.com
- **Management**: ops-manager@example.com

---

## Prevention

1. **Implement Pod Disruption Budgets** to prevent simultaneous pod termination
2. **Use pod anti-affinity** to spread replicas across nodes
3. **Monitor node health** proactively
4. **Maintain spare capacity** in the cluster
5. **Regular disaster recovery drills**

---

## Related Runbooks
- [Service Down](./service-down.md)
- [High Error Rate](./high-error-rate.md)
- [Low Replicas](./low-replicas.md)
