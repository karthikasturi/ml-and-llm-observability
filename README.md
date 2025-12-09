# Deploy and Monitor ML + LLM Pipelines Together
## ML + LLM Observability Lab

Complete hands-on lab for deploying and monitoring ML inference and LLM RAG pipelines with comprehensive observability.

## 🎯 Lab Overview

This lab demonstrates end-to-end observability for modern AI systems combining:
- **ML Inference**: Scikit-learn model for ticket severity prediction
- **LLM RAG**: OpenAI-powered retrieval-augmented generation
- **Monitoring**: Prometheus metrics + Grafana dashboards
- **Quality**: TruLens evaluation (relevance, hallucination, groundedness)
- **Alerting**: Prometheus AlertManager with Slack/email notifications
- **Drift Detection**: Automated detection of ML data drift and LLM quality degradation

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Stage-by-Stage Guide](#stage-by-stage-guide)
4. [Testing the System](#testing-the-system)
5. [Monitoring & Dashboards](#monitoring--dashboards)
6. [Alerting](#alerting)
7. [Failure Mode Simulation](#failure-mode-simulation)
8. [Troubleshooting](#troubleshooting)
9. [Production Considerations](#production-considerations)

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- 8GB RAM minimum
- Ports available: 8000, 8001, 8002, 9090, 9093, 3000

### 1. Clone and Setup
```bash
cd ml-and-llm-observability

# Copy environment template
cp .env.example .env

# (Optional) Add OpenAI API key to .env
# OPENAI_API_KEY=your_key_here

# (Optional) Configure Slack webhook
# SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### 2. Start All Services
```bash
# Start infrastructure
docker compose up -d

# Wait for services to be ready (~30 seconds)
docker compose ps

# Check health
curl http://localhost:8000/health
curl http://localhost:8001/health
```

### 3. Load Knowledge Base
```bash
curl -X POST http://localhost:8001/load-knowledge
```

### 4. Run Quick Tests
```bash
# Test ML API
./test_ml_api.sh

# Test LLM RAG API
./test_llm_api.sh

# Test combined pipeline
./test_combined_pipeline.sh
```

### 5. Access Dashboards
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **AlertManager**: http://localhost:9093
- **ML API**: http://localhost:8000
- **LLM API**: http://localhost:8001

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User / Application                           │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
    ┌────────────────┐
    │   ML API       │◄────┐
    │  (Port 8000)   │     │
    │                │     │  Combined /triage
    │ • Predict      │     │  Endpoint
    │ • Drift        │     │
    └────────┬───────┘     │
             │             │
             │    ┌────────┴────────┐
             │    │   LLM RAG API   │
             │    │  (Port 8001)    │
             │    │                 │
             │    │ • Chat          │
             │    │ • RAG Pipeline  │
             │    │ • TruLens       │
             │    └────────┬────────┘
             │             │
             │    ┌────────▼────────┐
             │    │   Chroma DB     │
             │    │  Vector Store   │
             │    └─────────────────┘
             │
    ┌────────▼──────────────────────────────────────────┐
    │              Prometheus                            │
    │         (Metrics Collection)                       │
    │                                                    │
    │  • Scrapes ML & LLM APIs every 10s                │
    │  • Evaluates alert rules                          │
    │  • Sends alerts to AlertManager                   │
    └────────┬──────────────────┬───────────────────────┘
             │                  │
    ┌────────▼────────┐  ┌──────▼──────────┐
    │    Grafana      │  │  AlertManager   │
    │  (Dashboards)   │  │ (Notifications) │
    │                 │  │                 │
    │ • ML Metrics    │  │ • Slack         │
    │ • LLM Quality   │  │ • Email         │
    │ • Pipeline      │  │ • Console       │
    └─────────────────┘  └─────────────────┘
```

### Service Components

| Service | Purpose | Port | Technology |
|---------|---------|------|------------|
| ML API | ML inference with drift detection | 8000 | FastAPI, Scikit-learn |
| LLM RAG API | RAG pipeline with quality monitoring | 8001 | FastAPI, OpenAI, Chroma |
| Chroma DB | Vector database for embeddings | 8002 | ChromaDB |
| Prometheus | Metrics collection and alerting | 9090 | Prometheus |
| AlertManager | Alert routing and notifications | 9093 | AlertManager |
| Grafana | Metrics visualization | 3000 | Grafana |

## 📚 Stage-by-Stage Guide

### Stage 1: Environment Setup
See [STAGE1_SETUP.md](STAGE1_SETUP.md)
- Docker Compose configuration
- Network setup
- Prometheus & Grafana provisioning

### Stage 2: ML Inference API
See [STAGE2_ML_API.md](STAGE2_ML_API.md)
- Logistic Regression model
- `/predict` endpoint
- Prometheus metrics integration
- Model training script

### Stage 3: LLM RAG API
See [STAGE3_LLM_RAG.md](STAGE3_LLM_RAG.md)
- RAG pipeline implementation
- ChromaDB vector store
- TruLens quality monitoring
- Knowledge base management

### Stage 4: Combined Pipeline
See [STAGE4_COMBINED_PIPELINE.md](STAGE4_COMBINED_PIPELINE.md)
- `/triage` endpoint
- ML + LLM integration
- Error handling
- End-to-end metrics

### Stage 5: Monitoring Integration
See [STAGE5_MONITORING.md](STAGE5_MONITORING.md)
- Drift detection (ML & LLM)
- Prometheus alert rules
- Grafana dashboards
- Quality metrics

### Stage 6: Failure Simulation
See [STAGE6_SIMULATION.md](STAGE6_SIMULATION.md)
- Data drift simulation
- Hallucination testing
- Latency spike generation
- RAG failure modes

### Stage 7: Alerting
See [STAGE7_ALERTING.md](STAGE7_ALERTING.md)
- AlertManager configuration
- Slack integration
- Email notifications
- Alert routing rules

## 🧪 Testing the System

### Test ML API

```bash
# Basic prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_length": 150,
    "urgency_keywords": 3,
    "business_impact": 7,
    "customer_tier": 4
  }'

# Expected response:
# {
#   "severity": "High",
#   "severity_code": 2,
#   "confidence": 0.87,
#   "processing_time_ms": 12.5
# }

# Check drift status
curl http://localhost:8000/drift/status

# View metrics
curl http://localhost:8000/metrics | grep ml_
```

### Test LLM RAG API

```bash
# Load knowledge base
curl -X POST http://localhost:8001/load-knowledge

# Chat query
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I reset my password?",
    "session_id": "test-1"
  }'

# Expected response includes:
# - response: Generated answer
# - relevance_score: ~0.8-0.9 for good retrieval
# - hallucination_score: <0.3 for factual answers
# - groundedness_score: >0.7 for context-based answers

# Check embedding drift
curl http://localhost:8001/drift/embedding-stats

# View metrics
curl http://localhost:8001/metrics | grep llm_
```

### Test Combined Pipeline

```bash
# Triage endpoint
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "URGENT: Production database is down!",
    "ticket_length": 200,
    "urgency_keywords": 5,
    "business_impact": 10,
    "customer_tier": 5
  }'

# Response includes:
# - severity: ML prediction
# - explanation: LLM-generated context
# - recommended_actions: Actionable steps
# - processing_time_ms: End-to-end latency
# - llm_quality_scores: Quality metrics
```

### Automated Test Scripts

```bash
# Test all ML endpoints
./test_ml_api.sh

# Test all LLM endpoints
./test_llm_api.sh

# Test combined pipeline
./test_combined_pipeline.sh

# Test AlertManager config
./test_alertmanager.sh
```

## 📊 Monitoring & Dashboards

### Prometheus Queries

Access Prometheus at http://localhost:9090

**Request Rate**:
```promql
rate(ml_predictions_total[1m])
rate(llm_requests_total[1m])
rate(triage_requests_total[1m])
```

**Latency (95th percentile)**:
```promql
histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m]))
histogram_quantile(0.95, rate(llm_request_latency_seconds_bucket[5m]))
histogram_quantile(0.95, rate(triage_latency_seconds_bucket[5m]))
```

**Drift Scores**:
```promql
ml_data_drift_score
llm_embedding_drift_score
```

**Quality Metrics**:
```promql
histogram_quantile(0.90, rate(llm_hallucination_score_bucket[5m]))
histogram_quantile(0.50, rate(llm_relevance_score_bucket[5m]))
histogram_quantile(0.50, rate(llm_groundedness_score_bucket[5m]))
```

### Grafana Dashboards

Access Grafana at http://localhost:3000 (admin/admin)

**Dashboard 1: ML Service Metrics**
- Prediction rate and distribution
- Latency percentiles
- Data drift trends
- Per-feature drift scores
- Error rates

**Dashboard 2: LLM Quality Metrics**
- Request rate
- Response latency
- Relevance scores (trend)
- Hallucination detection (trend)
- Groundedness scores (trend)
- Embedding drift
- Average quality score

**Dashboard 3: Combined Pipeline Health**
- End-to-end triage latency
- Success vs error rates
- ML-LLM correlation
- Severity distribution
- Component latency breakdown
- 24h statistics

### Viewing Dashboards

```bash
# Open all dashboards
open http://localhost:3000

# Navigate to:
# Dashboards → Observability Lab → [Select Dashboard]

# Or direct URLs:
# http://localhost:3000/d/ml-service
# http://localhost:3000/d/llm-quality
# http://localhost:3000/d/pipeline-health
```

## 🚨 Alerting

### Configured Alerts

| Alert | Condition | Severity | Channels |
|-------|-----------|----------|----------|
| MLModelDriftHigh | drift_score > 0.5 for 5m | Critical | Slack + Email |
| LLMHallucinationCritical | p90_hallucination > 0.7 for 5m | Critical | Slack + Email |
| PipelineLatencyHigh | p95_latency > 10s for 5m | Critical | Slack + Email |
| MLLatencyHigh | p95_latency > 100ms for 5m | Warning | Slack |
| LLMRelevanceLow | median_relevance < 0.5 for 10m | Warning | Slack |
| APIDown | service unreachable for 1m | Critical | Slack + Email |

### Viewing Alerts

```bash
# Prometheus alerts
open http://localhost:9090/alerts

# AlertManager UI
open http://localhost:9093

# Check via API
curl http://localhost:9090/api/v1/alerts | jq '.data.alerts[]'
```

### Slack Setup

1. Create Slack webhook at https://api.slack.com/apps
2. Add to `.env`:
   ```bash
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
   ```
3. Restart AlertManager:
   ```bash
   docker compose restart alertmanager
   ```

### Email Setup

1. Add SMTP config to `.env`:
   ```bash
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_FROM=alerts@company.com
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=your-app-password
   ALERT_EMAIL=team@company.com
   ```
2. Restart AlertManager:
   ```bash
   docker compose restart alertmanager
   ```

### Testing Alerts

```bash
# Test configuration
./test_alertmanager.sh

# Trigger ML drift alert
python3 simulate_drift.py --duration 60 --drift-amount 0.9

# Trigger hallucination alert
python3 simulate_hallucination.py --num-queries 30 --hallucination-rate 0.9

# Trigger latency alert
python3 simulate_latency_spike.py --duration 60 --burst-size 30
```

## 🔥 Failure Mode Simulation

### Simulate ML Data Drift

```bash
# High drift for 2 minutes
python3 simulate_drift.py --duration 120 --drift-amount 0.8

# Expected:
# - ml_data_drift_score increases above 0.5
# - MLModelDriftHigh alert fires after 5 minutes
# - Dashboard shows spike in drift graph
```

### Simulate LLM Hallucinations

```bash
# Send out-of-domain queries
python3 simulate_hallucination.py --num-queries 50 --hallucination-rate 0.7

# Expected:
# - llm_hallucination_score increases above 0.7
# - LLMHallucinationCritical alert fires
# - Quality dashboard shows degradation
```

### Simulate Latency Spikes

```bash
# Burst traffic causing latency
python3 simulate_latency_spike.py --duration 60 --burst-size 20

# Expected:
# - Latency percentiles spike
# - PipelineLatencyHigh alert fires
# - Latency graphs show clear spike
```

### Run All Simulations

```bash
# Comprehensive test (~16 minutes)
./test_all_failure_modes.sh

# Runs through all failure modes sequentially
# with recovery periods between each test
```

## 🔧 Troubleshooting

### Services Not Starting

```bash
# Check service status
docker compose ps

# View logs
docker compose logs ml-api
docker compose logs llm-rag-api
docker compose logs chroma-db

# Restart specific service
docker compose restart ml-api

# Rebuild and restart
docker compose up -d --build ml-api
```

### ML API Issues

```bash
# Check health
curl http://localhost:8000/health

# View logs
docker compose logs -f ml-api

# Common issues:
# - Model not trained: Container will train on startup
# - Port conflict: Change port in docker-compose.yml
```

### LLM API Issues

```bash
# Check health
curl http://localhost:8001/health

# Load knowledge base manually
curl -X POST http://localhost:8001/load-knowledge

# Check ChromaDB connection
docker compose logs chroma-db

# Common issues:
# - ChromaDB not ready: Wait 10-20 seconds after startup
# - Knowledge base empty: Run load-knowledge endpoint
# - OpenAI API key missing: Set in .env (optional, will use mock)
```

### Prometheus Not Scraping

```bash
# Check Prometheus targets
open http://localhost:9090/targets

# Should show:
# - ml-api: UP
# - llm-rag-api: UP
# - prometheus: UP

# If DOWN:
# 1. Check service is running: docker compose ps
# 2. Check /metrics endpoint: curl http://localhost:8000/metrics
# 3. Check prometheus config: docker compose logs prometheus
```

### Grafana Dashboards Not Loading

```bash
# Check Grafana logs
docker compose logs grafana

# Verify datasource
open http://localhost:3000/datasources

# Reimport dashboards if needed
# 1. Go to Dashboards → Import
# 2. Upload JSON from grafana/provisioning/dashboards/json/
```

### AlertManager Not Sending Notifications

```bash
# Check AlertManager status
curl http://localhost:9093/api/v2/status

# Test Slack webhook manually
curl -X POST $SLACK_WEBHOOK_URL \
  -H 'Content-Type: application/json' \
  -d '{"text": "Test alert"}'

# View AlertManager logs
docker compose logs alertmanager

# Common issues:
# - SLACK_WEBHOOK_URL not set
# - Wrong SMTP credentials
# - Alerts not firing: Check Prometheus alerts first
```

## 🏭 Production Considerations

### Scaling

```yaml
# Horizontal scaling example
services:
  ml-api:
    deploy:
      replicas: 3
    # Add load balancer (nginx/traefik)

  llm-rag-api:
    deploy:
      replicas: 2
```

### Security

- **API Authentication**: Add JWT/OAuth
- **TLS**: Enable HTTPS with certificates
- **Secrets Management**: Use Docker secrets or Vault
- **Network Policies**: Restrict inter-service communication

### Persistence

```yaml
# Persistent volumes for production
volumes:
  prometheus-data:
    driver: local
    driver_opts:
      type: none
      device: /data/prometheus
      o: bind
```

### High Availability

- Run multiple Prometheus instances
- Use Prometheus federation
- Deploy redundant AlertManagers
- Use external storage (Thanos, Cortex)

### Model Management

- Version ML models
- A/B testing framework
- Automated retraining pipelines
- Model registry (MLflow)

### LLM Considerations

- Rate limiting for API calls
- Caching frequent queries
- Fallback to smaller models
- Cost monitoring and budgets

### Monitoring Production

- Set up external monitoring (uptime checks)
- Implement distributed tracing (Jaeger, Tempo)
- Log aggregation (ELK, Loki)
- APM integration (Datadog, New Relic)

## 📖 Additional Resources

### Documentation Files
- [STAGE1_SETUP.md](STAGE1_SETUP.md) - Initial setup
- [STAGE2_ML_API.md](STAGE2_ML_API.md) - ML service details
- [STAGE3_LLM_RAG.md](STAGE3_LLM_RAG.md) - RAG implementation
- [STAGE4_COMBINED_PIPELINE.md](STAGE4_COMBINED_PIPELINE.md) - Integrated pipeline
- [STAGE5_MONITORING.md](STAGE5_MONITORING.md) - Monitoring setup
- [STAGE6_SIMULATION.md](STAGE6_SIMULATION.md) - Testing failure modes
- [STAGE7_ALERTING.md](STAGE7_ALERTING.md) - Alert configuration

### Test Scripts
- `test_ml_api.sh` - Test ML endpoints
- `test_llm_api.sh` - Test LLM endpoints
- `test_combined_pipeline.sh` - Test triage endpoint
- `test_alertmanager.sh` - Test alert configuration
- `test_all_failure_modes.sh` - Comprehensive testing

### Simulation Scripts
- `simulate_drift.py` - ML data drift
- `simulate_hallucination.py` - LLM quality issues
- `simulate_latency_spike.py` - Performance degradation

## 🤝 Contributing

This is a learning lab. Feel free to:
- Add new monitoring metrics
- Implement additional ML models
- Enhance RAG pipeline
- Add more failure simulations
- Improve dashboards

## 📄 License

MIT License - See LICENSE.md for details

## 🙏 Acknowledgments

- Prometheus & Grafana communities
- TruLens for LLM evaluation
- ChromaDB for vector storage
- OpenAI for LLM capabilities
- FastAPI for excellent API framework

---

**Questions or Issues?**
- Check the troubleshooting section
- Review stage-specific documentation
- Examine service logs with `docker compose logs`

**Happy Monitoring! 🎉**
