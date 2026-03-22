import boto3
import os
import time
import uuid
from dotenv import load_dotenv

load_dotenv()

def final_deploy():
    sm = boto3.client('sagemaker', region_name=os.getenv("AWS_REGION", "us-east-1"))
    iam = boto3.client('iam')
    
    try:
        # Dynamically reconstruct ARN to be safe
        role_name = "CapitalOne-SageMaker-Execution-Role"
        role_info = iam.get_role(RoleName=role_name)
        role_arn = role_info['Role']['Arn'].strip()
        
        unique_id = str(uuid.uuid4())[:8]
        model_name = f"phi-2-model-{unique_id}"
        container = f"763104351884.dkr.ecr.{os.getenv('AWS_REGION', 'us-east-1')}.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-pt2.0.0-gpu-py310-cu118-ubuntu20.04"

        print(f"Creating Model: {model_name}")
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': container,
                'Environment': {'HF_MODEL_ID': 'microsoft/phi-2', 'HF_TASK': 'text-generation'}
            },
            ExecutionRoleArn=role_arn
        )

        config_name = f"{model_name}-config"
        print(f"Creating Config: {config_name}")
        sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.g5.2xlarge',
                'InitialVariantWeight': 1
            }]
        )

        endpoint_name = f"phi-2-endpoint-{unique_id}"
        print(f"Creating Endpoint: {endpoint_name}")
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        
        print(f"SUCCESS! Endpoint {endpoint_name} is being created.")
        
    except Exception as e:
        print(f"FAILED: {str(e)}")

if __name__ == "__main__":
    final_deploy()
