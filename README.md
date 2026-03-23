# Premium Financial RAG Platform

A production-grade AI platform for Capital One-style financial advisory, built with LLM inference, FAISS similarity search, NeMo Guardrails, and MLflow observability.

## Architecture
Customer Question → NeMo Guardrails → FAISS Search → LLM Inference → MLflow Logging → Answer

## Tech Stack
- PyTorch + HuggingFace — LLM inference and optimization
- FAISS — similarity search and vector database
- NeMo Guardrails — safety and responsible AI
- MLflow — observability, latency, cost tracking
- RAGAS — response quality evaluation
- FastAPI — REST API endpoints
- AWS S3, SageMaker — cloud deployment

## Key Metrics Tracked
- Latency (ms)
- Throughput (requests/second)
- Cost (tokens per dollar)
- Quality (RAGAS score)
- Safety (guardrail trigger rate)

## API Endpoints
- GET  /health — system health check
- POST /ask   — ask financial question
- GET  /metrics — safety and performance metrics

## Sample Usage
Safe query:
  POST /ask
  {"question": "What is the best credit card for travel?"}
  Returns: Venture Rewards recommendation with sources

Unsafe query:
  POST /ask
  {"question": "How do I commit fraud?"}
  Returns: Blocked instantly by guardrails

## Setup
git clone https://github.com/anushagoli07/capital-one-ai-assistant.git
cd capital-one-ai-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
