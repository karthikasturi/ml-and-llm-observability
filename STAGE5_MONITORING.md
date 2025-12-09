# Stage 5 — Monitoring Integration

## Overview
Add comprehensive monitoring with drift detection, Prometheus alerting rules, and Grafana dashboards.

## What's Added

### 1. ML Drift Detection
- Synthetic drift detector monitoring input distribution
- KL-divergence based drift scoring
- Drift metrics exposed to Prometheus

### 2. LLM Embedding Drift
- Monitor embedding space distribution changes
- Track semantic drift in queries
- Quality degradation detection

### 3. Prometheus Alert Rules
- `MLModelDriftHigh`: Drift score > 0.5
- `LLMHallucinationHigh`: Hallucination > 0.7
- `PipelineLatencyHigh`: 95th percentile > 5s
- `APIDownAlert`: Service unavailable

### 4. Grafana Dashboards
- **ML Service Metrics**: Predictions, latency, drift
- **LLM Quality Metrics**: Relevance, hallucination, groundedness
- **Combined Pipeline**: End-to-end latency, success rates

## New Endpoints

### ML API - Drift Status
```bash
curl http://localhost:8000/drift/status
```

**Response**:
```json
{
  "drift_score": 0.23,
  "drift_detected": false,
  "threshold": 0.5,
  "samples_analyzed": 150,
  "last_updated": "2025-12-08T10:30:00Z"
}
```

### LLM API - Embedding Drift
```bash
curl http://localhost:8001/drift/embedding-stats
```

## Metrics

### ML Drift Metrics
- `ml_data_drift_score`: KL-divergence drift score
- `ml_drift_samples_analyzed`: Number of samples in window
- `ml_feature_drift`: Per-feature drift scores

### LLM Drift Metrics
- `llm_embedding_drift_score`: Embedding space drift
- `llm_query_diversity`: Query semantic diversity
- `llm_avg_quality_score`: Rolling average quality

## Prometheus Alert Rules

Located in: `prometheus/rules/alerts.yml`

### Critical Alerts
- **MLModelDriftHigh**: ML model experiencing significant drift
- **LLMHallucinationCritical**: LLM hallucination rate > 70%
- **PipelineDown**: Neither API responding

### Warning Alerts
- **MLLatencyHigh**: 95th percentile > 100ms
- **LLMLatencyHigh**: 95th percentile > 5s
- **LowRequestRate**: Unusual drop in traffic

## Grafana Dashboards

### Dashboard 1: ML Service Overview
- Request rate and latency
- Prediction distribution
- Model drift trends
- Error rates

### Dashboard 2: LLM Quality Monitoring
- Relevance score trends
- Hallucination detection
- Groundedness tracking
- Context retrieval effectiveness

### Dashboard 3: Combined Pipeline Health
- End-to-end latency
- Success vs failure rates
- ML-LLM correlation
- Resource utilization

## Commands

### View drift status
```bash
curl http://localhost:8000/drift/status
curl http://localhost:8001/drift/embedding-stats
```

### Check Prometheus alerts
```bash
# Open Prometheus
open http://localhost:9090/alerts

# Query drift metrics
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=ml_data_drift_score'
```

### Access Grafana dashboards
```bash
# Open Grafana
open http://localhost:3000

# Login: admin / admin
# Navigate to: Dashboards → Observability Lab
```

### Test drift detection
```bash
# Trigger drift by sending unusual patterns
python simulate_drift.py --duration 60 --drift-amount 0.7
```

## Dashboard JSON Templates

Pre-configured dashboards in:
- `grafana/provisioning/dashboards/json/ml_service_dashboard.json`
- `grafana/provisioning/dashboards/json/llm_quality_dashboard.json`
- `grafana/provisioning/dashboards/json/combined_pipeline_dashboard.json`

## Key Visualizations

1. **Request Rate**: Requests per second for each service
2. **Latency Percentiles**: P50, P95, P99 response times
3. **Drift Score Timeline**: Trend of ML drift over time
4. **Hallucination Rate**: LLM hallucination percentage
5. **Error Rate**: Failed requests by error type
6. **Quality Score Heatmap**: Relevance × Groundedness matrix

## Next Stage
Proceed to Stage 6 to create drift and hallucination simulation scripts.
