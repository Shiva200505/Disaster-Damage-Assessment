import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    postgres_url: str = "postgresql+asyncpg://appuser:changeme_in_production@postgres:5432/disaster_damage"
    redis_url: str = "redis://redis:6379/0"
    secret_key: str = "changeme_32_char_random_string"
    siamese_ckpt: Path = Path("checkpoints/siamese_best.pth")
    device: str = "cpu"
    max_upload_size_mb: int = 50
    task_expiry_hours: int = 24

    class Config:
        env_file = ".env"

settings = Settings()
