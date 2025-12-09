#!/bin/bash
# Test script for LLM RAG API (Stage 3)

echo "========================================="
echo "Testing LLM RAG API - Stage 3"
echo "========================================="

API_URL="http://localhost:8001"

echo -e "\n1. Testing health endpoint..."
curl -s "$API_URL/health" | python3 -m json.tool

echo -e "\n2. Loading knowledge base..."
curl -s -X POST "$API_URL/load-knowledge" | python3 -m json.tool

echo -e "\n3. Checking knowledge base stats..."
curl -s "$API_URL/kb/stats" | python3 -m json.tool

echo -e "\n4. Testing RAG - Password Reset Query..."
curl -s -X POST "$API_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I reset my password?",
    "session_id": "test-session-1",
    "num_context_docs": 3
  }' | python3 -m json.tool

echo -e "\n5. Testing RAG - Account Access Query..."
curl -s -X POST "$API_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I cannot access my account, what should I do?",
    "session_id": "test-session-2",
    "num_context_docs": 3
  }' | python3 -m json.tool

echo -e "\n6. Testing RAG - Network Issues..."
curl -s -X POST "$API_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "My network connection is not working",
    "session_id": "test-session-3",
    "num_context_docs": 3
  }' | python3 -m json.tool

echo -e "\n7. Testing out-of-domain query (should have high hallucination)..."
curl -s -X POST "$API_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "session_id": "test-session-4",
    "num_context_docs": 3
  }' | python3 -m json.tool

echo -e "\n8. Checking Prometheus metrics..."
echo "Request counts:"
curl -s "$API_URL/metrics" | grep "llm_requests_total"

echo -e "\nLatency metrics:"
curl -s "$API_URL/metrics" | grep "llm_request_latency"

echo -e "\nQuality metrics:"
curl -s "$API_URL/metrics" | grep "llm_relevance_score" | head -5
curl -s "$API_URL/metrics" | grep "llm_hallucination_score" | head -5
curl -s "$API_URL/metrics" | grep "llm_groundedness_score" | head -5

echo -e "\n========================================="
echo "Testing complete!"
echo "View TruLens logs: docker compose exec llm-rag-api cat /logs/trulens/evaluations.log"
echo "========================================="
