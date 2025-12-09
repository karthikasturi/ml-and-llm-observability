# Stage 3 — LLM RAG API with TruLens Monitoring

## Overview
Build a RAG (Retrieval-Augmented Generation) pipeline with quality monitoring using TruLens.

## What's Included
- OpenAI embeddings and chat completion
- ChromaDB vector store
- Knowledge base loader
- RAG retrieval pipeline
- TruLens instrumentation for:
  - Relevance evaluation
  - Hallucination detection
  - Groundedness scoring
- Prometheus metrics

## Architecture
```
User Query → Embedding → Vector Search (Chroma) → Context Retrieval
                                                          ↓
                                           LLM (OpenAI) + Context → Response
                                                          ↓
                                              TruLens Evaluation
                                             (Relevance, Hallucination, Groundedness)
```

## Endpoints

### 1. Health Check
```bash
curl http://localhost:8001/health
```

### 2. Load Knowledge Base
```bash
curl -X POST http://localhost:8001/load-knowledge
```

### 3. Chat with RAG
```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I reset my password?",
    "session_id": "test-session-1"
  }'
```

**Response**:
```json
{
  "response": "To reset your password, go to Settings > Security...",
  "context_used": ["Password reset can be done via..."],
  "relevance_score": 0.92,
  "hallucination_score": 0.05,
  "groundedness_score": 0.89,
  "num_context_docs": 3,
  "processing_time_ms": 1234.5
}
```

### 4. Prometheus Metrics
```bash
curl http://localhost:8001/metrics
```

## Test Inputs

### Query 1: Password Reset
```json
{
  "query": "How do I reset my password?",
  "session_id": "session-1"
}
```
Expected: High relevance, low hallucination

### Query 2: Account Access
```json
{
  "query": "I can't access my account, what should I do?",
  "session_id": "session-2"
}
```
Expected: Good context retrieval

### Query 3: Out of Domain (tests hallucination)
```json
{
  "query": "What is the capital of France?",
  "session_id": "session-3"
}
```
Expected: Higher hallucination score, lower groundedness

## TruLens Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Relevance | How relevant retrieved context is | > 0.7 |
| Hallucination | Fabricated information not in context | < 0.3 |
| Groundedness | Response supported by context | > 0.6 |

## Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `llm_request_latency_seconds` | Histogram | RAG pipeline latency |
| `llm_requests_total` | Counter | Total RAG requests |
| `llm_relevance_score` | Histogram | TruLens relevance scores |
| `llm_hallucination_score` | Histogram | TruLens hallucination scores |
| `llm_groundedness_score` | Histogram | TruLens groundedness scores |

## Commands

### Start services
```bash
cd ml-and-llm-observability
docker compose up -d --build
```

### Load knowledge base
```bash
curl -X POST http://localhost:8001/load-knowledge
```

### Test RAG pipeline
```bash
./test_llm_api.sh
```

### View TruLens logs
```bash
docker compose exec llm-rag-api cat /logs/trulens/evaluations.log
```

### Check in Prometheus
1. Open http://localhost:9090
2. Query: `rate(llm_requests_total[1m])`
3. Query: `llm_hallucination_score`

## Knowledge Base
Sample IT support knowledge articles in `data/knowledge_base/`:
- Password resets
- Account access issues
- Software installation
- Network connectivity
- Security policies

## Next Stage
Proceed to Stage 4 to combine ML and LLM APIs into a unified triage endpoint.
