from fastapi import FastAPI
from pydantic import BaseModel
import time
from core.rag_engine import RAGEngine
from core.guardrails.guardrails_engine import FinancialGuardrails
from mlops.experiment_tracker import ExperimentTracker
from data.data_loader import load_financial_products

app = FastAPI(
    title="Capital One Financial RAG Platform",
    description="AI-powered financial advisor with guardrails and observability",
    version="1.0.0"
)

# Initialize all components
print("Initializing platform...")
documents = load_financial_products()
rag = RAGEngine()
rag.build_knowledge_base(documents)
guardrails = FinancialGuardrails()
tracker = ExperimentTracker()
print("Platform ready!")

# Input schema
class Question(BaseModel):
    question: str

# Output schema
class Answer(BaseModel):
    question: str
    answer: str
    sources: list
    latency_ms: float
    is_safe: bool
    safety_message: str

@app.get("/")
def home():
    return {
        "message": "Capital One Financial RAG Platform",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "components": {
            "rag_engine": "ready",
            "guardrails": "active",
            "mlflow": "tracking"
        }
    }

@app.post("/ask", response_model=Answer)
def ask(question: Question):
    # Step 1: Check safety with guardrails
    safety_check = guardrails.check_input(
        question.question
    )

    # If unsafe - block and log
    if not safety_check["safe"]:
        tracker.log_query({
            "question": question.question,
            "answer": safety_check["message"],
            "latency_ms": 0,
            "is_safe": False
        })
        return Answer(
            question=question.question,
            answer=safety_check["message"],
            sources=[],
            latency_ms=0,
            is_safe=False,
            safety_message=safety_check["reason"]
        )

    # Step 2: Get answer from RAG engine
    result = rag.query(question.question)

    # Step 3: Log metrics to MLflow
    tracker.log_query({
        "question": question.question,
        "answer": result["answer"],
        "latency_ms": result["latency_ms"],
        "retrieval_count": result["retrieval_count"],
        "is_safe": True
    })

    return Answer(
        question=question.question,
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=result["latency_ms"],
        is_safe=True,
        safety_message="approved"
    )

@app.get("/metrics")
def metrics():
    safety_metrics = guardrails.get_safety_metrics()
    return {
        "safety": safety_metrics,
        "message": "Check MLflow UI for detailed metrics"
  }
