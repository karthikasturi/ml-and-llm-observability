# Stage 6 — Drift & Hallucination Simulation Code

## Overview
Scripts and tools to simulate various failure modes for testing monitoring and alerting.

## Simulations Included

### 1. ML Data Drift
Simulate distribution shifts in input features to trigger drift alerts.

### 2. LLM Hallucination
Generate queries that lead to hallucinations by:
- Missing context in knowledge base
- Out-of-domain queries
- Ambiguous questions

### 3. RAG Failure Modes
- Disable vector database
- Empty knowledge base
- Corrupted embeddings

### 4. Latency Spikes
Inject artificial delays to test latency alerts.

## Simulation Scripts

### simulate_drift.py
```bash
# Simulate ML data drift for 2 minutes
python simulate_drift.py --duration 120 --drift-amount 0.8

# Options:
#   --duration: Simulation duration in seconds
#   --drift-amount: Drift magnitude (0.0-1.0)
#   --interval: Request interval in seconds
```

### simulate_hallucination.py
```bash
# Simulate LLM hallucinations
python simulate_hallucination.py --num-queries 50 --hallucination-rate 0.7

# Options:
#   --num-queries: Number of queries to send
#   --hallucination-rate: Target hallucination rate
```

### simulate_latency_spike.py
```bash
# Inject latency spikes
python simulate_latency_spike.py --duration 60 --spike-ms 5000

# Options:
#   --duration: Spike duration in seconds
#   --spike-ms: Additional latency in milliseconds
```

### simulate_rag_failure.py
```bash
# Simulate RAG failures
python simulate_rag_failure.py --mode vector-db-down

# Modes:
#   vector-db-down: Disconnect from vector DB
#   empty-kb: Empty knowledge base
#   no-context: Queries with no relevant context
```

## Expected Outcomes

### ML Drift Simulation
- **Metrics**: `ml_data_drift_score` increases above 0.5
- **Alert**: `MLModelDriftHigh` fires after 5 minutes
- **Dashboard**: Drift score graph shows spike
- **Duration**: Alert clears after normal traffic resumes

### Hallucination Simulation
- **Metrics**: `llm_hallucination_score` increases above 0.7
- **Alert**: `LLMHallucinationCritical` fires
- **Dashboard**: Hallucination trend shows spike
- **Quality**: Relevance and groundedness scores drop

### Latency Spike Simulation
- **Metrics**: Latency percentiles spike dramatically
- **Alerts**: `PipelineLatencyHigh` and `LLMLatencyHigh` fire
- **Dashboard**: Latency graphs show clear spike
- **Recovery**: Metrics return to normal after simulation ends

### RAG Failure Simulation
- **Metrics**: Error rate increases, success rate drops
- **Alerts**: `LLMAPIDown` or high error rate alerts
- **Dashboard**: Error visualization shows failures
- **Logs**: Clear error messages in TruLens logs

## Combined Scenario Testing

### test_all_failure_modes.sh
Runs all simulations in sequence:
```bash
./test_all_failure_modes.sh
```

Sequence:
1. Normal baseline (2 min)
2. ML drift (3 min)
3. Recovery period (2 min)
4. LLM hallucination (3 min)
5. Recovery period (2 min)
6. Latency spike (2 min)
7. Final recovery (2 min)

Total duration: ~16 minutes

## Verification Commands

### Check Prometheus Alerts
```bash
# View active alerts
curl http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | {alert: .labels.alertname, state: .state}'

# Check specific metric
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=ml_data_drift_score'
```

### Check Grafana
```bash
# Open dashboards
open http://localhost:3000/d/ml-service
open http://localhost:3000/d/llm-quality
open http://localhost:3000/d/pipeline-health
```

### View Application Logs
```bash
# ML API logs
docker compose logs -f ml-api

# LLM API logs
docker compose logs -f llm-rag-api

# TruLens evaluation logs
docker compose exec llm-rag-api tail -f /logs/trulens/evaluations.log
```

## Safety Notes

- Simulations run against localhost only
- No production impact (all containerized)
- Services automatically recover after simulation
- Use `Ctrl+C` to stop any simulation early
- Reset drift detectors with `/drift/reset` endpoints if needed

## Next Stage
Proceed to Stage 7 to configure AlertManager for Slack and email notifications.
