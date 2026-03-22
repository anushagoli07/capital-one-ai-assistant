import mlflow

class ABTestTracker:
    def __init__(self, experiment_name="RAG_AB_Testing"):
        mlflow.set_experiment(experiment_name)

    def log_trial(self, model_version, query, response, latency, accuracy, safety_score):
        """
        Log a single trial for A/B testing comparison.
        """
        with mlflow.start_run(run_name=f"Trial_{model_version}"):
            mlflow.log_param("model_version", model_version)
            mlflow.log_metric("latency_ms", latency)
            mlflow.log_metric("accuracy_score", accuracy)
            mlflow.log_metric("safety_score", safety_score)
            
            # Log as artifact if needed
            mlflow.log_text(f"Query: {query}\nResponse: {response}", f"trial_details.txt")

    def get_summary(self):
        """
        In a real app, this would query MLflow for comparison stats.
        """
        return "A/B Test Summary: Comparison metrics available in MLflow UI."
