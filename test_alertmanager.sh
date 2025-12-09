#!/bin/bash
# Test AlertManager Configuration

echo "========================================="
echo "AlertManager Configuration Test"
echo "========================================="

ALERTMANAGER_URL="http://localhost:9093"

echo -e "\n1. Checking AlertManager status..."
curl -s "$ALERTMANAGER_URL/api/v2/status" | python3 -m json.tool

echo -e "\n2. Checking current configuration..."
curl -s "$ALERTMANAGER_URL/api/v2/status" | python3 -m json.tool | grep -A5 "configYAML"

echo -e "\n3. Checking active alerts..."
curl -s "$ALERTMANAGER_URL/api/v2/alerts" | python3 -m json.tool

echo -e "\n4. Testing Slack webhook (if configured)..."
if [ -n "$SLACK_WEBHOOK_URL" ]; then
    curl -X POST "$SLACK_WEBHOOK_URL" \
      -H 'Content-Type: application/json' \
      -d '{
        "text": "🧪 Test Alert from ML+LLM Observability Lab",
        "blocks": [{
          "type": "section",
          "text": {
            "type": "mrkdwn",
            "text": "*Test Alert*\nAlertManager is configured and working!"
          }
        }]
      }'
    echo "✓ Slack webhook test sent"
else
    echo "⚠ SLACK_WEBHOOK_URL not configured"
fi

echo -e "\n5. Creating test silence..."
SILENCE_ID=$(curl -s -X POST "$ALERTMANAGER_URL/api/v2/silences" \
  -H 'Content-Type: application/json' \
  -d '{
    "matchers": [{"name": "alertname", "value": "TestAlert", "isRegex": false}],
    "startsAt": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",
    "endsAt": "'$(date -u -d '+1 hour' +%Y-%m-%dT%H:%M:%SZ)'",
    "comment": "Test silence",
    "createdBy": "test-script"
  }' | python3 -c "import sys, json; print(json.load(sys.stdin)['silenceID'])" 2>/dev/null)

if [ -n "$SILENCE_ID" ]; then
    echo "✓ Silence created: $SILENCE_ID"
    
    echo -e "\n6. Listing silences..."
    curl -s "$ALERTMANAGER_URL/api/v2/silences" | python3 -m json.tool
    
    echo -e "\n7. Deleting test silence..."
    curl -s -X DELETE "$ALERTMANAGER_URL/api/v2/silence/$SILENCE_ID"
    echo "✓ Silence deleted"
else
    echo "⚠ Could not create test silence"
fi

echo -e "\n========================================="
echo "Testing complete!"
echo "========================================="
echo ""
echo "Access Points:"
echo "  • AlertManager UI: http://localhost:9093"
echo "  • Prometheus Alerts: http://localhost:9090/alerts"
echo ""
echo "To trigger real alerts, run:"
echo "  python3 simulate_drift.py --duration 60 --drift-amount 0.9"
echo ""
