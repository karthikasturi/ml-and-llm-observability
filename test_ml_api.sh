#!/bin/bash
# Test script for ML API (Stage 2)

echo "========================================="
echo "Testing ML Inference API - Stage 2"
echo "========================================="

API_URL="http://localhost:8000"

echo -e "\n1. Testing health endpoint..."
curl -s "$API_URL/health" | python3 -m json.tool

echo -e "\n2. Testing model info..."
curl -s "$API_URL/model/info" | python3 -m json.tool

echo -e "\n3. Testing LOW severity prediction..."
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_length": 50,
    "urgency_keywords": 0,
    "business_impact": 2,
    "customer_tier": 1
  }' | python3 -m json.tool

echo -e "\n4. Testing MEDIUM severity prediction..."
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_length": 120,
    "urgency_keywords": 2,
    "business_impact": 5,
    "customer_tier": 3
  }' | python3 -m json.tool

echo -e "\n5. Testing HIGH severity prediction..."
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_length": 200,
    "urgency_keywords": 5,
    "business_impact": 9,
    "customer_tier": 5
  }' | python3 -m json.tool

echo -e "\n6. Checking Prometheus metrics..."
echo "Request counts:"
curl -s "$API_URL/metrics" | grep "ml_predictions_total"

echo -e "\nLatency metrics:"
curl -s "$API_URL/metrics" | grep "ml_prediction_latency"

echo -e "\n========================================="
echo "Testing complete!"
echo "========================================="
