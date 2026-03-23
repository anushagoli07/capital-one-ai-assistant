from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # HuggingFace Model
    hf_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # MLflow
    mlflow_tracking_uri: str = "./mlruns"

    # AWS
    aws_bucket_name: str = "anusha-rag-22629"
    aws_region: str = "us-east-1"

    # App settings
    max_new_tokens: int = 150
    chunk_size: int = 500
    top_k_results: int = 1

    class Config:
        env_file = ".env"

settings = Settings()
