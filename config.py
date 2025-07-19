import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    DEBUG: bool = False
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Celery Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # GPU Configuration
    GPU_DEVICE: str = "cuda:0"
    HIGH_VRAM_THRESHOLD: float = 60.0  # GB
    GPU_MEMORY_PRESERVATION: float = 6.0  # GB
    
    # Model Configuration
    HUNYUAN_MODEL_PATH: str = "hunyuanvideo-community/HunyuanVideo"
    FRAMEPACK_MODEL_PATH: str = "lllyasviel/FramePackI2V_HY"
    FRAMEPACK_F1_MODEL_PATH: str = "lllyasviel/FramePack_F1_I2V_HY_20250503"
    FLUX_REDUX_MODEL_PATH: str = "lllyasviel/flux_redux_bfl"
    
    # File Storage
    UPLOAD_DIR: str = "./uploads"
    OUTPUT_DIR: str = "./outputs"
    TEMP_DIR: str = "./temp"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # Video Generation Defaults
    DEFAULT_DURATION: float = 5.0
    DEFAULT_STEPS: int = 25
    DEFAULT_CFG_SCALE: float = 1.0
    DEFAULT_DISTILLED_CFG_SCALE: float = 10.0
    DEFAULT_CFG_RESCALE: float = 0.0
    DEFAULT_SEED: int = 31337
    DEFAULT_MP4_CRF: int = 16
    DEFAULT_LATENT_WINDOW_SIZE: int = 9
    
    # Job Management
    MAX_CONCURRENT_JOBS: int = 2
    JOB_TIMEOUT: int = 3600  # 1 hour
    CLEANUP_INTERVAL: int = 300  # 5 minutes
    FILE_RETENTION_HOURS: int = 24
    
    # Security
    API_KEY: Optional[str] = None
    ALLOWED_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)