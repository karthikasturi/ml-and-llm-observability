# ⚡ Quick Start Guide

Get the ML + LLM Observability Lab running in 5 minutes!

## Prerequisites Checklist

- [ ] Docker installed (version 20.10+)
- [ ] Docker Compose installed (version 2.0+)
- [ ] 8GB RAM available
- [ ] Ports free: 8000, 8001, 8002, 9090, 9093, 3000
- [ ] Python 3.10+ (for test scripts)
- [ ] curl installed
- [ ] (Optional) OpenAI API key

## 🚀 3-Step Quick Start

### Step 1: Start the Stack (60 seconds)

```bash
cd ml-and-llm-observability

# Copy environment config
cp .env.example .env

# Start all services
docker compose up -d

# Wait for services to be healthy
sleep 30
docker compose ps
```

Expected output:
```
NAME                  STATUS          PORTS
ml-api                Up (healthy)    0.0.0.0:8000->8000/tcp
llm-rag-api           Up (healthy)    0.0.0.0:8001->8001/tcp
chroma-db             Up (healthy)    0.0.0.0:8002->8000/tcp
prometheus            Up              0.0.0.0:9090->9090/tcp
alertmanager          Up              0.0.0.0:9093->9093/tcp
grafana               Up              0.0.0.0:3000->3000/tcp
```

### Step 2: Load Knowledge Base (10 seconds)

```bash
# Initialize LLM RAG knowledge base
curl -X POST http://localhost:8001/load-knowledge

# Verify
curl http://localhost:8001/kb-stats
```

Expected output:
```json
{
  "kb_size": 5,
  "last_updated": "2025-01-21T10:30:00Z"
}
```

### Step 3: Test the System (30 seconds)

```bash
# Test ML prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticket_length": 200, "urgency_keywords": 5, "business_impact": 10, "customer_tier": 5}'

# Test LLM chat
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?", "session_id": "test-1"}'

# Test combined pipeline
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "URGENT: Cannot access production database", "ticket_length": 200, "urgency_keywords": 4, "business_impact": 9, "customer_tier": 5}'
```

## 🎛️ Access Dashboards

Open these URLs in your browser:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana** | http://localhost:3000 | admin / admin |
| **Prometheus** | http://localhost:9090 | - |
| **AlertManager** | http://localhost:9093 | - |
| **ML API Docs** | http://localhost:8000/docs | - |
| **LLM API Docs** | http://localhost:8001/docs | - |

### Quick Dashboard Navigation

1. Go to http://localhost:3000
2. Login with `admin` / `admin`
3. Navigate: **Dashboards** → **Observability Lab**
4. View:
   - **ML Service Metrics** - Model predictions, drift
   - **LLM Quality Metrics** - RAG performance, hallucination
   - **Combined Pipeline Health** - End-to-end latency, errors

## 🧪 Run Test Suites

```bash
# Make scripts executable
chmod +x *.sh

# Test ML API (6 tests, ~10 seconds)
./test_ml_api.sh

# Test LLM API (8 tests, ~30 seconds)
./test_llm_api.sh

# Test combined pipeline (6 tests, ~20 seconds)
./test_combined_pipeline.sh

# Test AlertManager (7 tests, ~10 seconds)
./test_alertmanager.sh
```

## 🔥 Trigger Failure Modes

### Simulate Data Drift
```bash
# 2-minute drift simulation
python3 simulate_drift.py --duration 120 --drift-amount 0.8

# Watch in Grafana:
# - ML Service Dashboard → Data Drift Score graph
# - Alert should fire after 5 minutes
```

### Simulate Hallucinations
```bash
# 50 out-of-domain queries
python3 simulate_hallucination.py --num-queries 50 --hallucination-rate 0.7

# Watch in Grafana:
# - LLM Quality Dashboard → Hallucination Score graph
# - Alert fires when p90 > 0.7
```

### Simulate Latency Spike
```bash
# 60-second burst of traffic
python3 simulate_latency_spike.py --duration 60 --burst-size 20

# Watch in Grafana:
# - Combined Pipeline Dashboard → Latency graph
# - Alert fires when p95 > 10s
```

### Run All Tests
```bash
# Comprehensive test (~16 minutes)
./test_all_failure_modes.sh
```

## 🚨 Setup Alerting

### Slack Notifications

1. Create Slack webhook:
   - Go to https://api.slack.com/apps
   - Create app → Incoming Webhooks → Add to channel
   - Copy webhook URL

2. Update `.env`:
   ```bash
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
   ```

3. Restart AlertManager:
   ```bash
   docker compose restart alertmanager
   ```

4. Test:
   ```bash
   ./test_alertmanager.sh
   ```

### Email Notifications

1. Update `.env` with SMTP settings:
   ```bash
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_FROM=alerts@company.com
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=your-app-password
   ALERT_EMAIL=team@company.com
   ```

2. Restart:
   ```bash
   docker compose restart alertmanager
   ```

## 🛑 Stop the Lab

```bash
# Stop all services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v

# Stop and remove images
docker compose down --rmi all
```

## ❓ Quick Troubleshooting

### Services won't start
```bash
# Check what's running
docker compose ps

# View logs
docker compose logs [service-name]

# Restart specific service
docker compose restart ml-api
```

### Can't access dashboards
```bash
# Check port conflicts
netstat -tuln | grep -E '8000|8001|9090|3000'

# View all service URLs
docker compose ps
```

### Knowledge base empty
```bash
# Reload knowledge base
curl -X POST http://localhost:8001/load-knowledge

# Verify
curl http://localhost:8001/kb-stats
```

### Metrics not showing in Grafana
```bash
# Check Prometheus targets
open http://localhost:9090/targets

# Should show all services UP
# If DOWN, check service logs
docker compose logs prometheus
```

## 📚 Next Steps

1. **Read Full Documentation**: [README.md](README.md)
2. **Understand Architecture**: See architecture diagram in README
3. **Explore Stages**: Read STAGE1 through STAGE7 documentation
4. **Customize Alerts**: Edit `prometheus/rules/alerts.yml`
5. **Add Metrics**: Extend ML/LLM APIs with custom metrics
6. **Production Setup**: Review production considerations in README

## 🎯 Learning Path

1. ✅ **Start Here** - Run Quick Start (you are here!)
2. 📖 [STAGE1_SETUP.md](STAGE1_SETUP.md) - Understand infrastructure
3. 🤖 [STAGE2_ML_API.md](STAGE2_ML_API.md) - Learn ML monitoring
4. 🧠 [STAGE3_LLM_RAG.md](STAGE3_LLM_RAG.md) - Understand RAG quality
5. 🔗 [STAGE4_COMBINED_PIPELINE.md](STAGE4_COMBINED_PIPELINE.md) - Integration patterns
6. 📊 [STAGE5_MONITORING.md](STAGE5_MONITORING.md) - Deep dive on metrics
7. 💥 [STAGE6_SIMULATION.md](STAGE6_SIMULATION.md) - Failure mode testing
8. 🚨 [STAGE7_ALERTING.md](STAGE7_ALERTING.md) - Alert configuration
9. 📘 [README.md](README.md) - Complete reference

## 💡 Pro Tips

- **Leave it running**: Services use minimal resources, keep them up for demos
- **Watch metrics live**: Keep Grafana open while running simulations
- **Test alerts early**: Configure Slack webhook first to see alerts in action
- **Use mock mode**: LLM API works without OpenAI key for learning
- **Check logs often**: `docker compose logs -f` is your friend

---

**Ready to dive deeper?** See [README.md](README.md) for complete documentation!

**Having issues?** Check the troubleshooting section above or view service logs.

**Want to learn?** Follow the stage-by-stage documentation for detailed explanations.
