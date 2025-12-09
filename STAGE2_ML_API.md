# Stage 2 — ML Inference API

## Overview
Implement a FastAPI service that serves ML predictions with Prometheus metrics for observability.

## What's Included
- Logistic Regression model trained on synthetic customer support ticket data
- `/predict` endpoint for severity classification
- Prometheus metrics: latency, request counter, error counter
- Model training script
- Health check endpoint

## Model Details
**Task**: Classify customer support ticket severity (Low, Medium, High)

**Features**:
- `ticket_length`: Number of words in ticket
- `urgency_keywords`: Count of urgency indicators
- `business_impact`: Impact score (0-10)
- `customer_tier`: Customer importance (1-5)

**Output**: Severity class (0=Low, 1=Medium, 2=High)

## Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_length": 150,
    "urgency_keywords": 3,
    "business_impact": 7,
    "customer_tier": 4
  }'
```

**Response**:
```json
{
  "severity": "High",
  "severity_code": 2,
  "confidence": 0.87,
  "processing_time_ms": 12.5
}
```

### 3. Prometheus Metrics
```bash
curl http://localhost:8000/metrics
```

## Test Inputs

### Low Severity
```json
{
  "ticket_length": 50,
  "urgency_keywords": 0,
  "business_impact": 2,
  "customer_tier": 1
}
```
Expected: `severity: "Low"`

### Medium Severity
```json
{
  "ticket_length": 120,
  "urgency_keywords": 2,
  "business_impact": 5,
  "customer_tier": 3
}
```
Expected: `severity: "Medium"`

### High Severity
```json
{
  "ticket_length": 200,
  "urgency_keywords": 5,
  "business_impact": 9,
  "customer_tier": 5
}
```
Expected: `severity: "High"`

## Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `ml_prediction_latency_seconds` | Histogram | Time taken for predictions |
| `ml_predictions_total` | Counter | Total number of predictions |
| `ml_prediction_errors_total` | Counter | Total prediction errors |
| `ml_model_info` | Gauge | Model metadata (version, accuracy) |

## Commands

### Train the model (optional - model is pre-trained)
```bash
docker compose exec ml-api python train_model.py
```

### Test the API
```bash
# Start services
cd ml-and-llm-observability
docker compose up -d --build

# Wait for startup
sleep 10

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticket_length": 150, "urgency_keywords": 3, "business_impact": 7, "customer_tier": 4}'

# Check metrics
curl http://localhost:8000/metrics | grep ml_
```

### View in Prometheus
1. Open http://localhost:9090
2. Query: `rate(ml_predictions_total[1m])`
3. Query: `histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m]))`

## Next Stage
Proceed to Stage 3 to implement the LLM RAG API with TruLens monitoring.
