from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None

    # AWS Credentials
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: Optional[str] = None

    # App Settings
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    
    # Storage
    KPI_STORAGE_PATH: str = "database/kpi_records.json"
    VECTOR_DB_PATH: str = "database/faiss_index"

    # MLOps
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

settings = Settings()
