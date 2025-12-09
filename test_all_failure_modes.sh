#!/bin/bash
# Test All Failure Modes - Comprehensive Simulation Script

echo "========================================================================"
echo "ML + LLM Observability Lab - Complete Failure Mode Testing"
echo "========================================================================"
echo ""
echo "This script will run all simulations in sequence to test monitoring:"
echo "  1. Normal baseline (2 min)"
echo "  2. ML data drift (3 min)"
echo "  3. Recovery period (2 min)"
echo "  4. LLM hallucinations (3 min)"
echo "  5. Recovery period (2 min)"
echo "  6. Latency spikes (2 min)"
echo "  7. Final recovery (2 min)"
echo ""
echo "Total duration: ~16 minutes"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."

# Function to check if services are healthy
check_services() {
    echo "Checking service health..."
    ml_health=$(curl -s http://localhost:8000/health | grep -o '"status":"healthy"' || echo "")
    llm_health=$(curl -s http://localhost:8001/health | grep -o '"status":"healthy"' || echo "")
    
    if [ -z "$ml_health" ] || [ -z "$llm_health" ]; then
        echo "❌ Services not healthy. Please start with: docker compose up -d"
        exit 1
    fi
    echo "✓ Services are healthy"
}

# Function to show current metrics
show_metrics() {
    echo ""
    echo "=== Current Metrics ==="
    echo "ML Drift Score:"
    curl -s http://localhost:8000/drift/status | python3 -m json.tool | grep drift_score || echo "N/A"
    echo ""
    echo "LLM Avg Quality:"
    curl -s http://localhost:8001/drift/embedding-stats | python3 -m json.tool | grep avg_quality || echo "N/A"
    echo ""
}

# Check services before starting
check_services

echo ""
echo "========================================================================"
echo "PHASE 1: Normal Baseline Traffic (2 minutes)"
echo "========================================================================"
echo "Establishing normal traffic patterns..."

python3 simulate_drift.py --duration 120 --drift-amount 0.0 --interval 1.0

show_metrics

echo ""
echo "========================================================================"
echo "PHASE 2: ML Data Drift Simulation (3 minutes)"
echo "========================================================================"
echo "Injecting data drift to trigger MLModelDriftHigh alert..."

python3 simulate_drift.py --duration 180 --drift-amount 0.8 --interval 0.5

show_metrics

echo ""
echo "Check Prometheus alerts: http://localhost:9090/alerts"
echo "Expected: MLModelDriftHigh alert should be firing"
read -p "Press Enter to continue to recovery phase..."

echo ""
echo "========================================================================"
echo "PHASE 3: Recovery Period (2 minutes)"
echo "========================================================================"
echo "Sending normal traffic to allow drift to stabilize..."

python3 simulate_drift.py --duration 120 --drift-amount 0.0 --interval 1.0

show_metrics

echo ""
echo "========================================================================"
echo "PHASE 4: LLM Hallucination Simulation (30 queries)"
echo "========================================================================"
echo "Sending out-of-domain queries to trigger hallucinations..."

python3 simulate_hallucination.py --num-queries 30 --hallucination-rate 0.8

show_metrics

echo ""
echo "Check Prometheus alerts: http://localhost:9090/alerts"
echo "Expected: LLMHallucinationCritical alert may be firing"
read -p "Press Enter to continue to recovery phase..."

echo ""
echo "========================================================================"
echo "PHASE 5: Recovery Period (20 normal queries)"
echo "========================================================================"
echo "Sending normal LLM queries..."

python3 simulate_hallucination.py --num-queries 20 --hallucination-rate 0.2

show_metrics

echo ""
echo "========================================================================"
echo "PHASE 6: Latency Spike Simulation (2 minutes)"
echo "========================================================================"
echo "Sending burst traffic to create latency spikes..."

python3 simulate_latency_spike.py --duration 120 --burst-size 15 --burst-interval 10

show_metrics

echo ""
echo "Check Prometheus alerts: http://localhost:9090/alerts"
echo "Expected: PipelineLatencyHigh alert may have fired"
read -p "Press Enter to continue to final recovery..."

echo ""
echo "========================================================================"
echo "PHASE 7: Final Recovery (2 minutes)"
echo "========================================================================"
echo "Returning to normal traffic patterns..."

python3 simulate_drift.py --duration 120 --drift-amount 0.0 --interval 1.0

show_metrics

echo ""
echo "========================================================================"
echo "TESTING COMPLETE!"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  ✓ Baseline established"
echo "  ✓ ML drift triggered and recovered"
echo "  ✓ LLM hallucinations triggered and recovered"
echo "  ✓ Latency spikes generated"
echo "  ✓ System returned to normal"
echo ""
echo "Review Results:"
echo "  • Prometheus Alerts: http://localhost:9090/alerts"
echo "  • Grafana Dashboards: http://localhost:3000"
echo "  • ML Service Dashboard: http://localhost:3000/d/ml-service"
echo "  • LLM Quality Dashboard: http://localhost:3000/d/llm-quality"
echo "  • Pipeline Health: http://localhost:3000/d/pipeline-health"
echo ""
echo "View Logs:"
echo "  docker compose logs ml-api"
echo "  docker compose logs llm-rag-api"
echo "  docker compose exec llm-rag-api cat /logs/trulens/evaluations.log"
echo ""
echo "========================================================================"
