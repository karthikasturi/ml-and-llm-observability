#!/bin/bash
# Test script for Combined ML + LLM Pipeline (Stage 4)

echo "========================================="
echo "Testing Combined Pipeline - Stage 4"
echo "========================================="

ML_API_URL="http://localhost:8000"

echo -e "\n1. Testing HIGH severity triage (Production Outage)..."
curl -s -X POST "$ML_API_URL/triage" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "URGENT: Production database is down affecting all customers! Critical systems offline!",
    "ticket_length": 200,
    "urgency_keywords": 5,
    "business_impact": 10,
    "customer_tier": 5
  }' | python3 -m json.tool

echo -e "\n2. Testing MEDIUM severity triage (Bug Report)..."
curl -s -X POST "$ML_API_URL/triage" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "Found a bug in the reporting feature affecting premium users. Export function not working.",
    "ticket_length": 120,
    "urgency_keywords": 2,
    "business_impact": 6,
    "customer_tier": 4
  }' | python3 -m json.tool

echo -e "\n3. Testing LOW severity triage (Feature Request)..."
curl -s -X POST "$ML_API_URL/triage" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "Would be nice to have dark mode option in the UI for better user experience.",
    "ticket_length": 50,
    "urgency_keywords": 0,
    "business_impact": 2,
    "customer_tier": 2
  }' | python3 -m json.tool

echo -e "\n4. Testing triage with network issue..."
curl -s -X POST "$ML_API_URL/triage" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "Cannot connect to VPN. Tried multiple times but connection keeps failing.",
    "ticket_length": 85,
    "urgency_keywords": 3,
    "business_impact": 5,
    "customer_tier": 3
  }' | python3 -m json.tool

echo -e "\n5. Testing triage with account access issue..."
curl -s -X POST "$ML_API_URL/triage" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "My account has been locked after multiple failed login attempts. Need urgent help!",
    "ticket_length": 110,
    "urgency_keywords": 4,
    "business_impact": 7,
    "customer_tier": 5
  }' | python3 -m json.tool

echo -e "\n6. Checking combined pipeline metrics..."
echo "Triage request counts:"
curl -s "$ML_API_URL/metrics" | grep "triage_requests_total"

echo -e "\nTriage latency:"
curl -s "$ML_API_URL/metrics" | grep "triage_latency" | head -5

echo -e "\nML-LLM correlation:"
curl -s "$ML_API_URL/metrics" | grep "triage_ml_llm_correlation"

echo -e "\n========================================="
echo "Testing complete!"
echo "View Prometheus: http://localhost:9090"
echo "Query examples:"
echo "  - rate(triage_requests_total[1m])"
echo "  - histogram_quantile(0.95, rate(triage_latency_seconds_bucket[5m]))"
echo "========================================="
