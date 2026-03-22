import os
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from core.rag_engine import RAGEngine
from mlops.experiment_tracker import ExperimentTracker
from utils.config import settings
import time

app = FastAPI(title="Capital One AI Assistant API")

# Initialize components
# Using Lite Mode by default for local verification without heavy torch dependencies
rag_engine = RAGEngine(model_id="microsoft/phi-2", use_local=True, lite_mode=True)
tracker = ExperimentTracker()

# Pre-load data for RAG
DATA_PATH = "data/financial_products.json"
if os.path.exists(DATA_PATH):
    rag_engine.build_vector_store_from_json(DATA_PATH)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_assistant(request: QueryRequest):
    """
    RAG-based query endpoint with Governance and Local LLM.
    """
    start_time = time.time()
    
    try:
        response = rag_engine.query(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    duration = time.time() - start_time
    
    # Track metrics in MLflow
    tracker.track_latency("query_endpoint", duration)
    tracker.log_retrieval_success(request.question, len(response.get("sources", [])))
    
    # Add metadata for the UI
    response["latency_ms"] = int(duration * 1000)
    response["model"] = "microsoft/phi-2 (Quantized)"
    
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "phi-2"}

@app.get("/products")
async def list_products():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'r') as f:
            return json.load(f)
    return []
