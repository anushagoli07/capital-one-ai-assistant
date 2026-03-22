import json
import boto3
import requests
import os

def handler(event, context):
    """
    AWS Lambda handler that triggers on S3 object creation.
    Sends a request to the FastAPI application to process the new document.
    """
    s3 = boto3.client('s3')
    
    # API endpoint (FastAPI app)
    api_url = os.environ.get('FASTAPI_API_URL', 'http://your-fastapi-app-url:8000/process')
    
    try:
        # Get the bucket and object key from the event
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        print(f"Triggered processing for s3://{bucket}/{key}")
        
        # Notify the backend to process the file
        payload = {
            "bucket": bucket,
            "key": key
        }
        
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        
        return {
            'statusCode': 200,
            'body': json.dumps(f"Successfully triggered processing for {key}")
        }
        
    except Exception as e:
        print(f"Error triggering processing: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(str(e))
        }
