# Mercury-Agent Runbook: PVC Near Capacity

## Alert: OmniMercuryPVCNearCapacity

### Overview
This runbook provides guidance for responding when Persistent Volume Claims (PVCs) in the Mercury-Agent namespace are approaching storage capacity limits.

### Alert Threshold
- **Warning**: PVC usage > 85% for 15 minutes

### Impact
- Risk of storage exhaustion
- Write operations may fail
- Model storage may be affected
- Log files may fill up storage

---

## Diagnosis Steps

### 1. Check PVC Usage
```bash
# View PVC status
kubectl get pvc -n mercury-agent

# Check PVC capacity and usage
kubectl exec -n mercury-agent deployment/mercury-agent-engine -- df -h /data

# View PVC details
kubectl describe pvc -n mercury-agent
```

### 2. Identify Large Files
```bash
# Find large files in the data volume
kubectl exec -n mercury-agent deployment/mercury-agent-engine -- du -sh /data/*

# Find largest files
kubectl exec -n mercury-agent deployment/mercury-agent-engine -- find /data -type f -size +100M -exec ls -lh {} \;
```

### 3. Check Log Files
```bash
# Check log file sizes
kubectl exec -n mercury-agent deployment/mercury-agent-api -- du -sh /var/log/*

# Check for log rotation
kubectl exec -n mercury-agent deployment/mercury-agent-api -- ls -la /var/log/
```

### 4. Check Model Storage
```bash
# Check model directory size
kubectl exec -n mercury-agent deployment/mercury-agent-engine -- du -sh /data/models/*

# List cached models
kubectl exec -n mercury-agent deployment/mercury-agent-engine -- ls -la /data/models/
```

---

## Resolution Steps

### Scenario 1: Log File Accumulation

**Symptoms:**
- Log files consuming significant space
- No log rotation configured
- Old logs not being cleaned

**Actions:**
1. Clean old log files:
   ```bash
   kubectl exec -n mercury-agent deployment/mercury-agent-api -- find /var/log -name "*.log" -mtime +7 -delete
   ```
2. Enable log rotation:
   ```bash
   kubectl set env deployment/mercury-agent-api -n mercury-agent LOG_ROTATION_ENABLED=true LOG_MAX_SIZE=100M LOG_MAX_FILES=5
   ```
3. Consider external log aggregation (e.g., Loki, ELK)

### Scenario 2: Model Cache Growth

**Symptoms:**
- Model cache directory is large
- Multiple model versions stored
- Unused models not cleaned

**Actions:**
1. Clean old model versions:
   ```bash
   kubectl exec -n mercury-agent deployment/mercury-agent-engine -- find /data/models -name "*.old" -delete
   ```
2. Enable model cache cleanup:
   ```bash
   kubectl set env deployment/mercury-agent-engine -n mercury-agent MODEL_CACHE_MAX_SIZE=10G MODEL_CACHE_CLEANUP_ENABLED=true
   ```
3. Remove unused models:
   ```bash
   kubectl exec -n mercury-agent deployment/mercury-agent-engine -- rm -rf /data/models/unused_model
   ```

### Scenario 3: Temporary Files

**Symptoms:**
- Temp directory consuming space
- Processing artifacts not cleaned
- Crash dumps accumulated

**Actions:**
1. Clean temporary files:
   ```bash
   kubectl exec -n mercury-agent deployment/mercury-agent-engine -- rm -rf /tmp/*
   kubectl exec -n mercury-agent deployment/mercury-agent-engine -- rm -rf /data/tmp/*
   ```
2. Enable automatic temp cleanup:
   ```bash
   kubectl set env deployment/mercury-agent-engine -n mercury-agent TEMP_CLEANUP_INTERVAL=1h
   ```

### Scenario 4: Need More Storage

**Symptoms:**
- Legitimate data growth
- All cleanup done but still near capacity
- Business requirements need more space

**Actions:**
1. Expand PVC (if storage class supports it):
   ```bash
   kubectl patch pvc mercury-agent-data -n mercury-agent -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'
   ```
2. If expansion not supported, migrate to larger PVC:
   ```bash
   # Create new larger PVC
   kubectl apply -f - <<EOF
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: mercury-agent-data-new
     namespace: mercury-agent
   spec:
     accessModes: ["ReadWriteOnce"]
     resources:
       requests:
         storage: 100Gi
   EOF
   ```
3. Migrate data and update deployment

---

## Escalation

If storage issues cannot be resolved:

1. **Notify team** via Slack #mercury-agent-alerts
2. **Request storage expansion** from infrastructure team
3. **Plan data migration** if needed

### Escalation Contacts
- **Platform Team**: platform-oncall@example.com
- **Infrastructure Team**: infra-oncall@example.com

---

## Prevention

1. **Implement log rotation** and retention policies
2. **Set up storage monitoring** with early warnings
3. **Regular cleanup jobs** for temporary files
4. **Capacity planning** based on growth trends
5. **Use external storage** for large datasets

---

## Related Runbooks
- [Service Down](./service-down.md)
- [High Error Rate](./high-error-rate.md)
