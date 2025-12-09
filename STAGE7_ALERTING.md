# Stage 7 — Alert Rules & Auto-Notification

## Overview
Configure AlertManager for automated notifications via Slack, email, and console logging.

## What's Added

### 1. Prometheus AlertManager
- Alert routing and grouping
- Notification templates
- Retry and throttling logic

### 2. Notification Channels
- **Slack**: Webhook integration for team channels
- **Email**: SMTP configuration for email alerts
- **Console**: Fallback logging for testing

### 3. Alert Routing Rules
- **Critical**: Immediate notification to all channels
- **Warning**: Throttled, grouped by service
- **Info**: Log only, no active notification

## Architecture

```
Prometheus Alerts → AlertManager → Routing Rules → Notification Channels
                                                    ├─ Slack Webhook
                                                    ├─ Email (SMTP)
                                                    └─ Console Log
```

## Configuration Files

### alertmanager.yml
Location: `prometheus/alertmanager.yml`

Defines:
- Global configuration
- Route tree for alert routing
- Receiver definitions (Slack, email, console)
- Inhibition rules

### Alert Templates
Location: `prometheus/templates/`

Custom message templates for:
- Slack notifications (rich formatting)
- Email notifications (HTML)
- Console output

## Slack Integration

### Setup
1. Create Slack incoming webhook:
   - Go to https://api.slack.com/apps
   - Create app → Incoming Webhooks → Activate
   - Add webhook to your channel
   - Copy webhook URL

2. Add to `.env`:
   ```bash
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
   ```

3. Restart AlertManager:
   ```bash
   docker compose restart alertmanager
   ```

### Slack Message Format
```
🔴 [CRITICAL] MLModelDriftHigh
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Service: ml-api
Severity: critical

ML model experiencing significant data drift

Details:
• Drift Score: 0.72
• Threshold: 0.5
• Started: 2025-12-08 10:30:00

Runbook: https://docs.company.com/runbooks/ml-drift
```

## Email Integration

### SMTP Configuration
Add to `.env`:
```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_FROM=alerts@company.com
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL=team@company.com
```

### Email Format
- **Subject**: `[CRITICAL] MLModelDriftHigh - ml-api`
- **Body**: HTML formatted with:
  - Alert summary
  - Current metric values
  - Grafana dashboard links
  - Runbook links

## Console Logging (Fallback)

When Slack/email not configured, alerts log to console:

```bash
# View AlertManager logs
docker compose logs -f alertmanager

# Example output:
[2025-12-08 10:30:00] CRITICAL: MLModelDriftHigh
  Service: ml-api
  Drift Score: 0.72
  Description: ML model experiencing significant data drift
```

## Alert Routing Rules

### Route Tree
```yaml
Root (default: console)
├─ Critical Alerts → Slack + Email
│  ├─ MLModelDriftHigh
│  ├─ LLMHallucinationCritical
│  ├─ PipelineLatencyHigh
│  └─ APIDown alerts
│
├─ Warning Alerts → Slack (throttled)
│  ├─ MLLatencyHigh
│  ├─ LLMRelevanceLow
│  └─ LowRequestRate
│
└─ Info Alerts → Console only
   └─ MLLLMCorrelationLow
```

### Grouping
- Alerts grouped by: `[service, alertname]`
- Group wait: 10s
- Group interval: 5m
- Repeat interval: 4h

### Inhibition
- Critical alerts inhibit warning alerts for same service
- Prevents alert spam during incidents

## Commands

### Test Alert Notifications

#### 1. Test Slack Webhook
```bash
curl -X POST $SLACK_WEBHOOK_URL \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "🧪 Test alert from ML+LLM Observability Lab",
    "blocks": [
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "*Test Alert*\nThis is a test notification from AlertManager"
        }
      }
    ]
  }'
```

#### 2. Trigger Test Alert
```bash
# Manually fire an alert via Prometheus API
curl -X POST http://localhost:9090/api/v1/alerts \
  -H 'Content-Type: application/json' \
  -d '[{
    "labels": {
      "alertname": "TestAlert",
      "severity": "warning",
      "service": "test"
    },
    "annotations": {
      "summary": "This is a test alert",
      "description": "Testing AlertManager notification system"
    }
  }]'
```

#### 3. Check AlertManager Status
```bash
# View current alerts
curl http://localhost:9093/api/v2/alerts | jq

# View AlertManager config
curl http://localhost:9093/api/v2/status | jq

# Silence an alert
curl -X POST http://localhost:9093/api/v2/silences \
  -H 'Content-Type: application/json' \
  -d '{
    "matchers": [{"name": "alertname", "value": "MLModelDriftHigh", "isRegex": false}],
    "startsAt": "2025-12-08T10:00:00Z",
    "endsAt": "2025-12-08T12:00:00Z",
    "comment": "Maintenance window",
    "createdBy": "admin"
  }'
```

### View AlertManager UI
```bash
# Open AlertManager web interface
open http://localhost:9093

# Features:
# - View active alerts
# - Create silences
# - Check notification status
# - View routing tree
```

## Testing Workflow

### End-to-End Test
```bash
# 1. Trigger drift to fire alert
python3 simulate_drift.py --duration 60 --drift-amount 0.9

# 2. Wait for alert to fire (5 minutes)
# Check Prometheus: http://localhost:9090/alerts

# 3. Verify notification received
# - Check Slack channel
# - Check email inbox
# - Check AlertManager logs: docker compose logs alertmanager

# 4. Silence the alert (optional)
# Use AlertManager UI: http://localhost:9093

# 5. Verify silence works
# New alerts should not notify during silence period
```

## Alert Response Runbooks

### MLModelDriftHigh
1. Check drift status: `curl http://localhost:8000/drift/status`
2. Review recent changes to input data
3. Compare current vs reference distributions
4. Options:
   - Retrain model with recent data
   - Update reference distribution
   - Investigate data pipeline changes

### LLMHallucinationCritical
1. Check hallucination metrics: Grafana LLM Quality dashboard
2. Review recent queries: TruLens logs
3. Verify knowledge base integrity
4. Options:
   - Update knowledge base
   - Adjust LLM temperature
   - Review prompt engineering
   - Check context retrieval quality

### PipelineLatencyHigh
1. Check component latencies: Grafana Combined Pipeline dashboard
2. Identify bottleneck (ML or LLM)
3. Check resource utilization
4. Options:
   - Scale services horizontally
   - Optimize slow operations
   - Add caching layer
   - Review database performance

## Next Stage
Proceed to Stage 8 to create comprehensive end-to-end documentation.
