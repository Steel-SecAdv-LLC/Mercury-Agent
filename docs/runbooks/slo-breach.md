# Mercury-Agent Runbook: SLO Breach

## Alert: OmniMercurySLOBreach

### Overview
This runbook provides guidance when the Mercury-Agent service is approaching or has breached its Service Level Objectives (SLOs).

### Alert Threshold
- **Critical**: Error budget remaining < 10%

### SLO Definitions
| SLI | Target | Measurement Window |
|-----|--------|-------------------|
| Availability | 99.9% | 30-day rolling |
| Latency (p95) | < 500ms | 30-day rolling |

### Impact
- SLA commitments at risk
- Customer experience degraded
- Potential contractual penalties

---

## Understanding Error Budgets

### Error Budget Calculation
```
Monthly Error Budget = (1 - SLO Target) × Time Window
For 99.9% availability over 30 days:
Error Budget = 0.1% × 30 days × 24 hours × 60 minutes = 43.2 minutes/month
```

### Current Status Query
```promql
# Remaining error budget percentage
omni_mercury:error_budget:remaining

# Error budget consumption rate (days to exhaust)
omni_mercury:error_budget:consumption_rate
```

---

## Diagnosis Steps

### 1. Identify Contributing Factors
```bash
# Check error rate trend
# Prometheus: sum(rate(http_requests_total{app="mercury-agent",status=~"5.."}[1h])) / sum(rate(http_requests_total{app="mercury-agent"}[1h]))

# Check latency trend
# Prometheus: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{app="mercury-agent"}[1h])) by (le))

# View historical incidents
kubectl get events -n mercury-agent --field-selector reason=Unhealthy
```

### 2. Review Recent Changes
```bash
# Deployment history
kubectl rollout history deployment/mercury-agent-api -n mercury-agent

# Recent config changes
git log --oneline -20 k8s/ helm/

# Check for infrastructure changes
kubectl get nodes -o wide
```

### 3. Analyze Traffic Patterns
```bash
# Request volume changes
# Prometheus: sum(rate(http_requests_total{app="mercury-agent"}[1h]))

# Geographic distribution (if available)
# Check Cloudflare/CDN analytics
```

---

## Mitigation Strategies

### Immediate Actions (< 1 hour)

#### 1. Increase Redundancy
```bash
# Scale up API replicas
kubectl scale deployment mercury-agent-api -n mercury-agent --replicas=5

# Verify HPA limits allow scaling
kubectl get hpa mercury-agent-api -n mercury-agent -o yaml
```

#### 2. Enable Feature Flags
```bash
# Disable non-critical features
kubectl set env deployment/mercury-agent-api -n mercury-agent \
  OMNI_ENABLE_ADVANCED_DETECTION=false \
  OMNI_ENABLE_PROFILING=false
```

#### 3. Implement Traffic Management
```bash
# Add rate limiting
kubectl annotate ingress mercury-agent -n mercury-agent \
  nginx.ingress.kubernetes.io/limit-rps="50" --overwrite

# Enable circuit breaker (if using service mesh)
kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: mercury-agent-circuit-breaker
  namespace: mercury-agent
spec:
  host: mercury-agent-api
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        h2UpgradePolicy: UPGRADE
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 60s
EOF
```

### Short-term Actions (1-24 hours)

#### 1. Performance Optimization
- Review and optimize slow endpoints
- Add caching for frequently accessed data
- Optimize database queries

#### 2. Infrastructure Improvements
```bash
# Upgrade node pool (cloud provider specific)
# AWS EKS example:
eksctl scale nodegroup --cluster mercury-agent --name standard-workers --nodes 5

# Increase resource limits
kubectl patch deployment mercury-agent-api -n mercury-agent -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "api",
          "resources": {
            "requests": {"cpu": "1", "memory": "2Gi"},
            "limits": {"cpu": "4", "memory": "8Gi"}
          }
        }]
      }
    }
  }
}'
```

#### 3. Review Architecture
- Identify single points of failure
- Add redundancy to critical paths
- Consider multi-region deployment

### Long-term Actions (> 24 hours)

1. **Conduct thorough post-mortem**
2. **Implement reliability improvements**
3. **Review and adjust SLO targets if needed**
4. **Add proactive monitoring and alerting**
5. **Update capacity planning**

---

## Communication

### Internal Communication
1. Update #mercury-agent-incidents Slack channel
2. Send status update to engineering leadership
3. Schedule war room if SLO breach imminent

### External Communication
1. Update status page (if public-facing)
2. Prepare customer communication (if SLA affected)
3. Notify customer success team

---

## Error Budget Policy

### When Budget > 50%
- Normal operations
- Feature development continues
- Regular monitoring

### When Budget 25-50%
- Increase monitoring frequency
- Prioritize reliability work
- Review upcoming changes

### When Budget 10-25%
- Freeze non-critical changes
- Focus on reliability improvements
- Daily status updates

### When Budget < 10% (THIS ALERT)
- **Freeze all changes** except reliability fixes
- **24/7 monitoring** by on-call team
- **Executive escalation**
- **Customer communication preparation**

---

## Escalation

| Budget Level | Escalation |
|-------------|------------|
| < 25% | Platform Team Lead |
| < 10% | Engineering Director |
| < 5% | VP Engineering |
| Breach | CTO + Customer Communication |

---

## Related Runbooks
- [High Error Rate](./high-error-rate.md)
- [High Latency](./high-latency.md)
- [Service Down](./service-down.md)
