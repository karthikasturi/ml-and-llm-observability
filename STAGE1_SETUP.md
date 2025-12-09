# Stage 1 — Environment Setup (Docker Compose)

## Overview
Set up the complete infrastructure for ML + LLM observability using Docker Compose.

## Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   ML API    │────▶│  Prometheus  │────▶│   Grafana   │
│  (FastAPI)  │     │  (Scraping)  │     │ (Dashboard) │
└─────────────┘     └──────────────┘     └─────────────┘
       │                                         │
       │            ┌──────────────┐             │
       └───────────▶│  LLM RAG API │◀────────────┘
                    │  (FastAPI)   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Chroma DB   │
                    │ (Vector Store)│
                    └──────────────┘
```

## Folder Structure
```
ml-and-llm-observability/
├── docker-compose.yml
├── .env.example
├── ml-api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py (placeholder)
├── llm-rag-api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py (placeholder)
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── prometheus.yml
│       └── dashboards/
│           └── dashboard.yml
├── data/
│   ├── chroma/
│   └── knowledge_base/
└── logs/
    └── trulens/
```

## Services
- **ml-api**: Port 8000 - ML inference with Prometheus metrics
- **llm-rag-api**: Port 8001 - RAG pipeline with TruLens monitoring
- **chroma-db**: Port 8002 - Vector database
- **prometheus**: Port 9090 - Metrics collection
- **grafana**: Port 3000 - Visualization dashboards

## Commands

### Start all services
```bash
cd ml-and-llm-observability
docker compose up -d
```

### Check service health
```bash
docker compose ps
```

### View logs
```bash
docker compose logs -f ml-api
docker compose logs -f llm-rag-api
```

### Stop all services
```bash
docker compose down
```

### Rebuild after code changes
```bash
docker compose up -d --build
```

## Access Points
- ML API: http://localhost:8000
- LLM RAG API: http://localhost:8001
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Next Stage
Proceed to Stage 2 to implement the ML Inference API.
