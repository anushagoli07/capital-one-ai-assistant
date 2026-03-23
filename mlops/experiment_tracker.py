import mlflow
import time
from core.config import settings

class ExperimentTracker:
    def __init__(self):
        # Set MLflow tracking location
        mlflow.set_tracking_uri(
            settings.mlflow_tracking_uri
        )
        # Create experiment
        mlflow.set_experiment(
            "capital-one-financial-rag"
        )
        print("MLflow tracker initialized!")

    def log_query(self, query_data):
        # Start a new MLflow run for each query
        with mlflow.start_run():
            # Log the question
            mlflow.log_param(
                "question", query_data["question"]
            )

            # Log performance metrics
            mlflow.log_metric(
                "latency_ms",
                query_data["latency_ms"]
            )
            mlflow.log_metric(
                "retrieval_count",
                query_data.get("retrieval_count", 0)
            )

            # Log safety metrics
            mlflow.log_metric(
                "is_safe",
                1 if query_data.get("is_safe", True) else 0
            )

            # Log quality score if available
            if "quality_score" in query_data:
                mlflow.log_metric(
                    "quality_score",
                    query_data["quality_score"]
                )

            # Log token estimate (cost tracking)
            answer_length = len(
                query_data.get("answer", "").split()
            )
            mlflow.log_metric(
                "answer_word_count",
                answer_length
            )

            print(f"Logged query metrics to MLflow!")
            print(f"  Latency: {query_data['latency_ms']}ms")
            print(f"  Safe: {query_data.get('is_safe', True)}")

    def log_safety_metrics(self, safety_data):
        with mlflow.start_run(run_name="safety_metrics"):
            mlflow.log_metric(
                "total_requests",
                safety_data["total_requests"]
            )
            mlflow.log_metric(
                "blocked_requests",
                safety_data["blocked_requests"]
            )
            mlflow.log_metric(
                "trigger_rate_percent",
                safety_data["trigger_rate_percent"]
            )
            print("Safety metrics logged!")

if __name__ == "__main__":
    tracker = ExperimentTracker()

    # Test logging a query
    test_query = {
        "question": "What is the best credit card for travel?",
        "answer": "The Capital One Venture Rewards card is best for travel.",
        "latency_ms": 55000,
        "retrieval_count": 1,
        "is_safe": True,
        "quality_score": 0.85
    }

    tracker.log_query(test_query)

    # Test logging safety metrics
    safety_metrics = {
        "total_requests": 6,
        "blocked_requests": 3,
        "trigger_rate_percent": 50.0
    }

    tracker.log_safety_metrics(safety_metrics)
    print("\nAll metrics logged to MLflow!")
    print("Run 'mlflow ui' to see dashboard!")



