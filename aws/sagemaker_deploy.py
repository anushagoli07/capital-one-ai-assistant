import boto3
import time
import uuid

class SageMakerDeployer:
    def __init__(self, role_arn, region_name="us-east-1"):
        self.role = role_arn
        self.region = region_name
        self.sm_client = boto3.client('sagemaker', region_name=region_name)

    def deploy_huggingface_model(self, model_id, task="text-generation", instance_type="ml.g5.2xlarge"):
        """
        Deploy a HuggingFace model using direct boto3 calls (Lightweight).
        """
        unique_id = str(uuid.uuid4())[:8]
        model_name = f"phi-2-model-{unique_id}"
        container = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-pt2.0.0-cpu-py310-ubuntu20.04"
        
        # Use GPU container if G5 instance is used
        if "g5" in instance_type or "g4" in instance_type:
            container = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-pt2.0.0-gpu-py310-cu118-ubuntu20.04"

        print(f"Creating Model: {model_name}...")
        self.sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': container,
                'Environment': {
                    'HF_MODEL_ID': model_id,
                    'HF_TASK': task
                }
            },
            ExecutionRoleArn=self.role
        )

        config_name = f"{model_name}-config"
        print(f"Creating Endpoint Config: {config_name}...")
        self.sm_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1
                }
            ]
        )

        endpoint_name = f"phi-2-endpoint-{unique_id}"
        print(f"Creating Endpoint: {endpoint_name}...")
        self.sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        
        return endpoint_name

    def cleanup(self, endpoint_name):
        self.sm_client.delete_endpoint(EndpointName=endpoint_name)
