import mlflow
from core.config import settings

class ExperimentTracker:
    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("capital-one-financial-rag")
        print("MLflow tracker initialized!")

    def log_query(self, query_data):
        with mlflow.start_run():
            mlflow.log_param("question", query_data["question"])
            mlflow.log_metric("latency_ms", query_data["latency_ms"])
            mlflow.log_metric("retrieval_count", query_data.get("retrieval_count", 0))
            mlflow.log_metric("is_safe", 1 if query_data.get("is_safe", True) else 0)

            if "quality_score" in query_data:
                mlflow.log_metric("quality_score", query_data["quality_score"])
            if "faithfulness" in query_data:
                mlflow.log_metric("faithfulness", query_data["faithfulness"])
            if "answer_relevancy" in query_data:
                mlflow.log_metric("answer_relevancy", query_data["answer_relevancy"])
            if "context_precision" in query_data:
                mlflow.log_metric("context_precision", query_data["context_precision"])

            answer_length = len(query_data.get("answer", "").split())
            mlflow.log_metric("answer_word_count", answer_length)

            print(f"Logged query metrics to MLflow!")
            print(f"  Latency: {query_data['latency_ms']}ms")
            print(f"  Safe: {query_data.get('is_safe', True)}")
            if "quality_score" in query_data:
                print(f"  Quality: {query_data['quality_score']}")

    def log_safety_metrics(self, safety_data):
        with mlflow.start_run(run_name="safety_metrics"):
            mlflow.log_metric("total_requests", safety_data["total_requests"])
            mlflow.log_metric("blocked_requests", safety_data["blocked_requests"])
            mlflow.log_metric("trigger_rate_percent", safety_data["trigger_rate_percent"])
            print("Safety metrics logged!")

if __name__ == "__main__":
    tracker = ExperimentTracker()
    test_query = {
        "question": "What is the best credit card for travel?",
        "answer": "The Capital One Venture Rewards card is best.",
        "latency_ms": 55000,
        "retrieval_count": 1,
        "is_safe": True,
        "quality_score": 0.778,
        "faithfulness": 0.667,
        "answer_relevancy": 0.667,
        "context_precision": 1.0
    }
    tracker.log_query(test_query)
    print("All metrics logged!")
