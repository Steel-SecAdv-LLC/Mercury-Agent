# Mercury-Agent Runbook: High CPU Usage

## Alert: OmniMercuryHighCPU

### Overview
This runbook provides guidance for responding to high CPU utilization in Mercury-Agent pods, which can lead to performance degradation and increased latency.

### Alert Threshold
- **Warning**: CPU utilization > 80% for 15 minutes

### Impact
- Increased response latency
- Potential request timeouts
- CPU throttling may occur
- May trigger HPA scaling

---

## Diagnosis Steps

### 1. Check CPU Usage
```bash
# View current CPU usage
kubectl top pods -n mercury-agent

# Check CPU limits
kubectl get pods -n mercury-agent -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].resources.limits.cpu}{"\n"}{end}'

# Check for CPU throttling (requires metrics-server)
kubectl get --raw /apis/metrics.k8s.io/v1beta1/namespaces/mercury-agent/pods
```

### 2. Identify CPU-Intensive Operations
```bash
# Check application logs for processing activity
kubectl logs -n mercury-agent -l app.kubernetes.io/component=api --tail=200 | grep -i "processing\|detect\|inference"

# Check request rates
kubectl logs -n mercury-agent -l app.kubernetes.io/component=api --tail=100 | grep -c "HTTP"
```

### 3. Check Request Patterns
```promql
# Prometheus query for request rate
sum(rate(http_requests_total{app="mercury-agent"}[5m])) by (path)

# Check detection request rate
sum(rate(omni_detection_requests_total{app="mercury-agent"}[5m])) by (detector_type)
```

### 4. Check HPA Status
```bash
# View HPA scaling status
kubectl get hpa -n mercury-agent

# Check if HPA is responding
kubectl describe hpa mercury-agent-api -n mercury-agent
```

---

## Resolution Steps

### Scenario 1: Traffic Spike

**Symptoms:**
- Sudden increase in request rate
- CPU correlates with traffic
- HPA may be scaling

**Actions:**
1. Verify HPA is scaling:
   ```bash
   kubectl get hpa -n mercury-agent -w
   ```
2. Manually scale if HPA is slow:
   ```bash
   kubectl scale deployment mercury-agent-api -n mercury-agent --replicas=6
   ```
3. Consider rate limiting if traffic is excessive:
   ```bash
   kubectl annotate ingress mercury-agent -n mercury-agent nginx.ingress.kubernetes.io/limit-rps="100" --overwrite
   ```

### Scenario 2: Expensive Operations

**Symptoms:**
- CPU spikes during specific operations
- Detection or inference operations taking long
- Batch processing in progress

**Actions:**
1. Reduce batch sizes:
   ```bash
   kubectl set env deployment/mercury-agent-engine -n mercury-agent MAX_BATCH_SIZE=16
   ```
2. Enable request queuing:
   ```bash
   kubectl set env deployment/mercury-agent-api -n mercury-agent ENABLE_REQUEST_QUEUE=true MAX_CONCURRENT_REQUESTS=10
   ```
3. Consider async processing for heavy operations

### Scenario 3: Inefficient Code

**Symptoms:**
- High CPU without corresponding traffic increase
- CPU usage doesn't correlate with requests
- Recent deployment may have introduced issue

**Actions:**
1. Check recent deployments:
   ```bash
   kubectl rollout history deployment/mercury-agent-api -n mercury-agent
   ```
2. Rollback if recent change caused issue:
   ```bash
   kubectl rollout undo deployment/mercury-agent-api -n mercury-agent
   ```
3. Enable CPU profiling for investigation:
   ```bash
   kubectl set env deployment/mercury-agent-api -n mercury-agent ENABLE_CPU_PROFILING=true
   ```

### Scenario 4: Insufficient Resources

**Symptoms:**
- CPU consistently at limit
- Normal traffic levels
- Throttling observed

**Actions:**
1. Increase CPU limits:
   ```bash
   kubectl patch deployment mercury-agent-api -n mercury-agent -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"cpu":"4"},"requests":{"cpu":"2"}}}]}}}}'
   ```
2. Scale horizontally:
   ```bash
   kubectl scale deployment mercury-agent-api -n mercury-agent --replicas=5
   ```
3. Review and optimize resource allocation

---

## Escalation

If CPU issues persist after following these steps:

1. **Notify team** via Slack #mercury-agent-alerts
2. **Engage development team** for code optimization
3. **Request capacity increase** if cluster-wide

### Escalation Contacts
- **Platform Team**: platform-oncall@example.com
- **Development Team**: dev-oncall@example.com

---

## Prevention

1. **Set appropriate CPU limits** based on profiling
2. **Implement request throttling** for expensive operations
3. **Use async processing** for CPU-intensive tasks
4. **Regular performance testing** and optimization
5. **Monitor CPU trends** for capacity planning

---

## Related Runbooks
- [High Latency](./high-latency.md)
- [High Memory](./high-memory.md)
- [HPA at Max](./hpa-max.md)
