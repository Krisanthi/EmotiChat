import os
from pydantic_settings import BaseSettings
from pathlib import Path


def _find_env_file():
    for path in [
        Path(".env"),
        Path("../.env"),
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]:
        if path.exists():
            return str(path.resolve())
    return ".env"


class Settings(BaseSettings):

    APP_NAME: str = "EmotiChat"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://localhost:8502"

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_MAX_TOKENS: int = 1024
    GROQ_TEMPERATURE: float = 0.7

    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_S3_BUCKET: str = ""

    DYNAMODB_TABLE_NAME: str = "emotichat-emotions"

    FACE_EMOTION_MODEL: str = "deepface"
    VOICE_EMOTION_MODEL: str = "librosa"
    TEXT_EMOTION_MODEL: str = "distilbert-base-uncased-finetuned-sst-2-english"

    WEIGHT_FACE: float = 0.35
    WEIGHT_VOICE: float = 0.30
    WEIGHT_TEXT: float = 0.35

    MAX_AUDIO_SIZE_MB: int = 10
    MAX_IMAGE_SIZE_MB: int = 5

    class Config:
        env_file = _find_env_file()
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()
