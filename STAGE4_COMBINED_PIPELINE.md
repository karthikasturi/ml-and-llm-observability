# Stage 4 — Combined ML + LLM Pipeline

## Overview
Create a unified `/triage` endpoint that combines ML severity prediction with LLM-generated explanations.

## What's Included
- New `/triage` endpoint in ML API
- Integration between ML and LLM services
- Error handling and fallback mechanisms
- Structured logging
- End-to-end metrics

## Flow
```
User Ticket → ML API (/triage)
                 ↓
            1. ML Prediction (severity)
                 ↓
            2. LLM RAG Call (explanation)
                 ↓
            3. Combined Response
                 ↓
         {severity, reason, context}
```

## New Endpoint

### POST /triage
Combines ML prediction with LLM explanation

**Request**:
```json
{
  "ticket_text": "URGENT: Production database is down, all users affected!",
  "ticket_length": 250,
  "urgency_keywords": 5,
  "business_impact": 9,
  "customer_tier": 5
}
```

**Response**:
```json
{
  "severity": "High",
  "severity_code": 2,
  "confidence": 0.92,
  "explanation": "This is a high-severity issue requiring immediate attention...",
  "recommended_actions": ["Contact on-call engineer", "Check system status"],
  "context_sources": ["Incident Response Guide", "SLA Policy"],
  "processing_time_ms": 1456.3,
  "ml_processing_ms": 15.2,
  "llm_processing_ms": 1441.1
}
```

## Test Scenarios

### 1. High Severity Production Outage
```json
{
  "ticket_text": "URGENT: Production database is down affecting all customers",
  "ticket_length": 200,
  "urgency_keywords": 5,
  "business_impact": 10,
  "customer_tier": 5
}
```
Expected: High severity with critical response plan

### 2. Medium Severity Bug Report
```json
{
  "ticket_text": "Found a bug in the reporting feature for premium users",
  "ticket_length": 120,
  "urgency_keywords": 2,
  "business_impact": 6,
  "customer_tier": 4
}
```
Expected: Medium severity with investigation steps

### 3. Low Severity Feature Request
```json
{
  "ticket_text": "Would be nice to have dark mode option",
  "ticket_length": 45,
  "urgency_keywords": 0,
  "business_impact": 1,
  "customer_tier": 2
}
```
Expected: Low severity with feature backlog guidance

## Error Handling

The triage endpoint handles:
- ML API failures → fallback to rule-based classification
- LLM API failures → return ML prediction only
- Timeout scenarios → partial results with warnings
- Invalid inputs → validation errors with details

## Metrics

New combined metrics:
- `triage_request_total`: Total triage requests
- `triage_latency_seconds`: End-to-end triage time
- `triage_ml_llm_correlation`: Correlation between ML confidence and LLM quality

## Commands

### Test triage endpoint
```bash
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "URGENT: Cannot access critical systems",
    "ticket_length": 150,
    "urgency_keywords": 4,
    "business_impact": 8,
    "customer_tier": 5
  }'
```

### Run comprehensive tests
```bash
./test_combined_pipeline.sh
```

### Check end-to-end metrics
```bash
curl http://localhost:8000/metrics | grep triage_
```

## Architecture Notes

**Service Communication**:
- ML API calls LLM API via HTTP
- Services communicate over Docker network `observability-net`
- Timeouts configured: ML prediction (5s), LLM call (30s)

**Fallback Strategy**:
1. Attempt ML prediction
2. If successful, attempt LLM explanation
3. If LLM fails, return ML prediction only
4. If ML fails, return error

## Next Stage
Proceed to Stage 5 to add drift detection and comprehensive monitoring dashboards.
