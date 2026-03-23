import boto3
import json
import time
from core.config import settings

class SageMakerDeployer:
    def __init__(self):
        # Connect to AWS SageMaker
        self.sm_client = boto3.client(
            "sagemaker",
            region_name=settings.aws_region
        )
        self.runtime_client = boto3.client(
            "sagemaker-runtime",
            region_name=settings.aws_region
        )
        self.bucket = settings.aws_bucket_name
        print("SageMaker client initialized!")

    def create_model_config(self):
        # Define model configuration
        # This tells SageMaker what model to deploy
        model_config = {
            "model_name": "capital-one-rag-model",
            "model_type": "HuggingFace",
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "instance_type": "ml.m5.xlarge",
            "num_instances": 1,
            "environment": {
                "HF_MODEL_ID": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "HF_TASK": "text-generation",
                "MAX_NEW_TOKENS": "150"
            }
        }
        return model_config

    def estimate_deployment_cost(self):
        # Estimate hourly cost for deployment
        # ml.m5.xlarge = $0.23/hour on AWS
        costs = {
            "instance_type": "ml.m5.xlarge",
            "hourly_cost_usd": 0.23,
            "daily_cost_usd": 0.23 * 24,
            "monthly_cost_usd": 0.23 * 24 * 30,
            "currency": "USD"
        }
        return costs

    def simulate_endpoint_call(self, question):
        # Simulate what a real SageMaker endpoint call looks like
        # In production this would call:
        # self.runtime_client.invoke_endpoint(...)

        print(f"Simulating SageMaker endpoint call...")
        print(f"Endpoint: capital-one-rag-endpoint")
        print(f"Question: {question}")

        # Simulate latency of a real endpoint
        start = time.time()
        time.sleep(0.1)
        latency = (time.time() - start) * 1000

        # Simulate response
        response = {
            "endpoint": "capital-one-rag-endpoint",
            "model": "TinyLlama-1.1B-Chat",
            "question": question,
            "answer": "Based on Capital One products, I recommend the Venture Rewards card for travel.",
            "latency_ms": round(latency, 2),
            "status": "success"
        }
        return response

    def get_deployment_summary(self):
        # Summary of what would be deployed
        config = self.create_model_config()
        costs = self.estimate_deployment_cost()

        summary = {
            "deployment_config": config,
            "cost_estimate": costs,
            "endpoints": {
                "inference": "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/capital-one-rag-endpoint/invocations",
                "health": "https://runtime.sagemaker.us-east-1.amazonaws.com/ping"
            },
            "monitoring": {
                "cloudwatch_metrics": [
                    "Invocations",
                    "InvocationErrors",
                    "ModelLatency",
                    "OverheadLatency"
                ]
            }
        }
        return summary

if __name__ == "__main__":
    deployer = SageMakerDeployer()

    print("\n=== SageMaker Deployment Config ===")
    config = deployer.create_model_config()
    print(f"Model: {config['model_id']}")
    print(f"Instance: {config['instance_type']}")
    print(f"Task: {config['environment']['HF_TASK']}")

    print("\n=== Cost Estimate ===")
    costs = deployer.estimate_deployment_cost()
    print(f"Instance Type: {costs['instance_type']}")
    print(f"Hourly Cost:   ${costs['hourly_cost_usd']}")
    print(f"Daily Cost:    ${costs['daily_cost_usd']}")
    print(f"Monthly Cost:  ${costs['monthly_cost_usd']}")

    print("\n=== Simulating Endpoint Call ===")
    response = deployer.simulate_endpoint_call(
        "What is the best credit card for travel?"
    )
    print(f"Status:   {response['status']}")
    print(f"Answer:   {response['answer']}")
    print(f"Latency:  {response['latency_ms']}ms")

    print("\n=== Full Deployment Summary ===")
    summary = deployer.get_deployment_summary()
    print(f"Inference Endpoint: {summary['endpoints']['inference']}")
    print(f"CloudWatch Metrics: {summary['monitoring']['cloudwatch_metrics']}")
