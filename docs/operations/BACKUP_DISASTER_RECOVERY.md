# OMNI-AVA Backup and Disaster Recovery Guide

## Overview

This document outlines the backup strategy, disaster recovery procedures, and business continuity planning for the OMNI-AVA platform.

---

## 1. Backup Strategy

### 1.1 Backup Scope

| Data Type | Frequency | Retention | Method |
|-----------|-----------|-----------|--------|
| Application Data | Daily | 30 days | Volume Snapshots |
| ML Models | Weekly | 90 days | Object Storage |
| Configuration | On Change | 365 days | Git + Vault |
| Database (if applicable) | Every 4 hours | 14 days | pg_dump / mongodump |
| Logs | Real-time | 30 days | Log Aggregation |

### 1.2 Backup Implementation

#### Volume Snapshot CronJob
```yaml
# k8s/base/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: omni-ava-backup
  namespace: omni-ava
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: omni-ava-backup
          containers:
            - name: backup
              image: bitnami/kubectl:latest
              command:
                - /bin/bash
                - -c
                - |
                  set -e

                  # Create volume snapshot for data PVC
                  cat <<EOF | kubectl apply -f -
                  apiVersion: snapshot.storage.k8s.io/v1
                  kind: VolumeSnapshot
                  metadata:
                    name: omni-ava-data-$(date +%Y%m%d-%H%M%S)
                    namespace: omni-ava
                  spec:
                    volumeSnapshotClassName: omni-ava-snapshot
                    source:
                      persistentVolumeClaimName: omni-ava-data
                  EOF

                  # Create volume snapshot for models PVC
                  cat <<EOF | kubectl apply -f -
                  apiVersion: snapshot.storage.k8s.io/v1
                  kind: VolumeSnapshot
                  metadata:
                    name: omni-ava-models-$(date +%Y%m%d-%H%M%S)
                    namespace: omni-ava
                  spec:
                    volumeSnapshotClassName: omni-ava-snapshot
                    source:
                      persistentVolumeClaimName: omni-ava-models
                  EOF

                  # Clean up old snapshots (keep last 30)
                  kubectl get volumesnapshots -n omni-ava --sort-by=.metadata.creationTimestamp -o name | \
                    head -n -30 | xargs -r kubectl delete -n omni-ava

                  echo "Backup completed successfully"
          restartPolicy: OnFailure
```

#### Backup Verification Script
```bash
#!/bin/bash
# scripts/verify-backup.sh

set -e

NAMESPACE="${NAMESPACE:-omni-ava}"
MIN_SNAPSHOTS=7  # At least 7 daily snapshots

echo "Verifying OMNI-AVA backups..."

# Check volume snapshots
SNAPSHOT_COUNT=$(kubectl get volumesnapshots -n $NAMESPACE --no-headers | wc -l)
if [ $SNAPSHOT_COUNT -lt $MIN_SNAPSHOTS ]; then
    echo "ERROR: Only $SNAPSHOT_COUNT snapshots found (minimum: $MIN_SNAPSHOTS)"
    exit 1
fi

# Check snapshot readiness
READY_SNAPSHOTS=$(kubectl get volumesnapshots -n $NAMESPACE -o jsonpath='{.items[?(@.status.readyToUse==true)].metadata.name}' | wc -w)
if [ $READY_SNAPSHOTS -lt $MIN_SNAPSHOTS ]; then
    echo "ERROR: Only $READY_SNAPSHOTS ready snapshots (minimum: $MIN_SNAPSHOTS)"
    exit 1
fi

# Verify most recent snapshot age
LATEST_SNAPSHOT=$(kubectl get volumesnapshots -n $NAMESPACE --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.creationTimestamp}')
LATEST_TIMESTAMP=$(date -d "$LATEST_SNAPSHOT" +%s)
NOW=$(date +%s)
AGE_HOURS=$(( ($NOW - $LATEST_TIMESTAMP) / 3600 ))

if [ $AGE_HOURS -gt 26 ]; then
    echo "ERROR: Latest backup is $AGE_HOURS hours old (maximum: 26 hours)"
    exit 1
fi

echo "Backup verification passed:"
echo "  - Total snapshots: $SNAPSHOT_COUNT"
echo "  - Ready snapshots: $READY_SNAPSHOTS"
echo "  - Latest backup age: $AGE_HOURS hours"
```

### 1.3 Backup Encryption

All backups are encrypted using:
- **At-rest encryption**: Cloud provider KMS keys
- **In-transit encryption**: TLS 1.3

```yaml
# AWS KMS encryption for volume snapshots
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: omni-ava-encrypted
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  encrypted: "true"
  kmsKeyId: "arn:aws:kms:region:account:key/key-id"
```

---

## 2. Disaster Recovery

### 2.1 Recovery Objectives

| Metric | Target | Description |
|--------|--------|-------------|
| **RTO** (Recovery Time Objective) | 4 hours | Maximum downtime |
| **RPO** (Recovery Point Objective) | 4 hours | Maximum data loss |
| **MTTR** (Mean Time to Recovery) | 2 hours | Average recovery time |

### 2.2 Disaster Scenarios

#### Scenario 1: Single Pod Failure
- **Impact**: Minimal (handled by Kubernetes)
- **Detection**: Automatic via health checks
- **Recovery**: Automatic pod replacement
- **RTO**: < 5 minutes

#### Scenario 2: Node Failure
- **Impact**: Temporary degradation
- **Detection**: Node status monitoring
- **Recovery**: Pod rescheduling to healthy nodes
- **RTO**: < 10 minutes

#### Scenario 3: Availability Zone Failure
- **Impact**: Potential service degradation
- **Detection**: Multi-AZ monitoring
- **Recovery**: Failover to healthy AZs
- **RTO**: < 30 minutes

#### Scenario 4: Region Failure
- **Impact**: Complete regional outage
- **Detection**: Global health monitoring
- **Recovery**: Multi-region failover
- **RTO**: < 4 hours

### 2.3 Multi-Region Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Global Load Balancer         │
                    │        (CloudFlare/Route53)         │
                    └─────────────┬───────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
    ┌───────▼───────┐     ┌───────▼───────┐     ┌───────▼───────┐
    │   Region A    │     │   Region B    │     │   Region C    │
    │   (Primary)   │     │  (Secondary)  │     │    (DR)       │
    │               │     │               │     │               │
    │  ┌─────────┐  │     │  ┌─────────┐  │     │  ┌─────────┐  │
    │  │ K8s     │  │     │  │ K8s     │  │     │  │ K8s     │  │
    │  │ Cluster │  │     │  │ Cluster │  │     │  │ Cluster │  │
    │  └────┬────┘  │     │  └────┬────┘  │     │  └────┬────┘  │
    │       │       │     │       │       │     │       │       │
    │  ┌────▼────┐  │     │  ┌────▼────┐  │     │  ┌────▼────┐  │
    │  │ Storage │◄─┼─────┼──┤ Storage │◄─┼─────┼──┤ Storage │  │
    │  │ (RW)    │  │     │  │ (Replica)│  │     │  │ (Replica)│  │
    │  └─────────┘  │     │  └─────────┘  │     │  └─────────┘  │
    └───────────────┘     └───────────────┘     └───────────────┘
```

### 2.4 Recovery Procedures

#### Procedure 1: Restore from Volume Snapshot
```bash
#!/bin/bash
# scripts/restore-from-snapshot.sh

SNAPSHOT_NAME="${1:-latest}"
NAMESPACE="${NAMESPACE:-omni-ava}"

# Get the snapshot to restore
if [ "$SNAPSHOT_NAME" == "latest" ]; then
    SNAPSHOT_NAME=$(kubectl get volumesnapshots -n $NAMESPACE --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}')
fi

echo "Restoring from snapshot: $SNAPSHOT_NAME"

# Scale down deployments
kubectl scale deployment omni-ava-api -n $NAMESPACE --replicas=0
kubectl scale deployment omni-ava-engine -n $NAMESPACE --replicas=0

# Wait for pods to terminate
kubectl wait --for=delete pod -l app.kubernetes.io/name=omni-ava -n $NAMESPACE --timeout=300s || true

# Delete existing PVC
kubectl delete pvc omni-ava-data -n $NAMESPACE --wait=true

# Create new PVC from snapshot
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: omni-ava-data
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: standard
  resources:
    requests:
      storage: 50Gi
  dataSource:
    name: $SNAPSHOT_NAME
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
EOF

# Wait for PVC to be bound
kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/omni-ava-data -n $NAMESPACE --timeout=600s

# Scale up deployments
kubectl scale deployment omni-ava-api -n $NAMESPACE --replicas=3
kubectl scale deployment omni-ava-engine -n $NAMESPACE --replicas=2

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=omni-ava -n $NAMESPACE --timeout=600s

echo "Restore completed successfully"
```

#### Procedure 2: Regional Failover
```bash
#!/bin/bash
# scripts/regional-failover.sh

PRIMARY_REGION="${PRIMARY_REGION:-us-west-2}"
DR_REGION="${DR_REGION:-us-east-1}"

echo "Initiating failover from $PRIMARY_REGION to $DR_REGION"

# Update DNS to point to DR region
# (Example using AWS Route53)
aws route53 change-resource-record-sets \
    --hosted-zone-id $HOSTED_ZONE_ID \
    --change-batch '{
        "Changes": [{
            "Action": "UPSERT",
            "ResourceRecordSet": {
                "Name": "api.omni-ava.example.com",
                "Type": "A",
                "AliasTarget": {
                    "HostedZoneId": "'$DR_ALB_ZONE'",
                    "DNSName": "'$DR_ALB_DNS'",
                    "EvaluateTargetHealth": true
                }
            }
        }]
    }'

# Switch kubectl context to DR region
kubectl config use-context omni-ava-$DR_REGION

# Scale up DR region
kubectl scale deployment omni-ava-api -n omni-ava --replicas=5
kubectl scale deployment omni-ava-engine -n omni-ava --replicas=3

# Wait for DR pods to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=omni-ava -n omni-ava --timeout=600s

echo "Failover completed. DR region is now active."
```

---

## 3. Business Continuity

### 3.1 Communication Plan

| Event | Internal Notification | External Notification |
|-------|----------------------|----------------------|
| Planned Maintenance | 72 hours prior | 48 hours prior |
| Minor Incident | 15 minutes | 30 minutes |
| Major Incident | Immediate | 15 minutes |
| Disaster | Immediate | 30 minutes |

### 3.2 Incident Response Team

| Role | Primary | Backup |
|------|---------|--------|
| Incident Commander | On-call SRE | SRE Manager |
| Technical Lead | Platform Lead | Senior Engineer |
| Communications | DevRel | Customer Success |
| Executive Sponsor | VP Engineering | CTO |

### 3.3 Recovery Checklist

#### Phase 1: Assessment (0-15 minutes)
- [ ] Confirm incident scope and impact
- [ ] Activate incident response team
- [ ] Update status page to "Investigating"
- [ ] Begin customer communication

#### Phase 2: Containment (15-60 minutes)
- [ ] Isolate affected components
- [ ] Implement temporary mitigations
- [ ] Update status page with details
- [ ] Notify executive stakeholders

#### Phase 3: Recovery (1-4 hours)
- [ ] Execute recovery procedures
- [ ] Verify data integrity
- [ ] Run validation tests
- [ ] Gradual traffic restoration

#### Phase 4: Post-Incident (4-48 hours)
- [ ] Complete service restoration
- [ ] Update status page to "Resolved"
- [ ] Conduct post-mortem
- [ ] Implement preventive measures

---

## 4. Testing and Validation

### 4.1 Disaster Recovery Testing Schedule

| Test Type | Frequency | Scope | Duration |
|-----------|-----------|-------|----------|
| Backup Verification | Daily | Automated | 5 min |
| Restore Test | Monthly | Single component | 2 hours |
| Failover Test | Quarterly | Regional | 4 hours |
| Full DR Exercise | Annually | Complete system | 8 hours |

### 4.2 Chaos Engineering

```yaml
# chaos/pod-failure.yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: omni-ava-pod-failure
  namespace: chaos-testing
spec:
  action: pod-failure
  mode: one
  duration: "60s"
  selector:
    namespaces:
      - omni-ava
    labelSelectors:
      app.kubernetes.io/name: omni-ava
  scheduler:
    cron: "@hourly"
```

### 4.3 Runbook Validation

All runbooks should be:
1. Tested quarterly
2. Updated after each incident
3. Reviewed by multiple team members
4. Automated where possible

---

## 5. Compliance and Audit

### 5.1 Backup Audit Trail

All backup operations are logged:
- Backup start/completion times
- Backup sizes
- Verification results
- Retention policy compliance

### 5.2 Compliance Requirements

| Standard | Requirement | Implementation |
|----------|-------------|----------------|
| SOC 2 | Data backup and recovery | Automated daily backups |
| ISO 27001 | Business continuity | DR plan and testing |
| GDPR | Data retention | 30-day backup retention |
| HIPAA | Data protection | Encrypted backups |

---

## 6. Contact Information

### Emergency Contacts

| Role | Name | Phone | Email |
|------|------|-------|-------|
| Primary On-Call | Rotating | PagerDuty | steel.sa.llc@gmail.com |
| Platform Lead | TBD | TBD | steel.sa.llc@gmail.com |
| SRE Manager | TBD | TBD | steel.sa.llc@gmail.com |

### Vendor Contacts

| Service | Support Portal | Phone |
|---------|---------------|-------|
| AWS | support.aws.amazon.com | 1-800-xxx-xxxx |
| GCP | cloud.google.com/support | 1-800-xxx-xxxx |
| PagerDuty | support.pagerduty.com | 1-800-xxx-xxxx |

---

## Appendix A: Quick Reference Commands

```bash
# List all backups
kubectl get volumesnapshots -n omni-ava

# Check backup status
kubectl get volumesnapshots -n omni-ava -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.readyToUse}{"\n"}{end}'

# Manual backup trigger
kubectl create job --from=cronjob/omni-ava-backup manual-backup-$(date +%s) -n omni-ava

# Verify data integrity
kubectl exec -n omni-ava deployment/omni-ava-api -- python -c "from omni_anomaly_engine import OmniAnomalyEngine; e = OmniAnomalyEngine(); print('Data integrity: OK')"

# Check replication status (if applicable)
kubectl get volumereplication -n omni-ava
```
