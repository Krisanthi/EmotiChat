"""
Application configuration — loads from environment variables.
"""

import os
from pydantic_settings import BaseSettings
from pathlib import Path


def _find_env_file():
    """Find .env file — check current dir, then parent dir, then project root."""
    for path in [
        Path(".env"),
        Path("../.env"),
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]:
        if path.exists():
            return str(path.resolve())
    return ".env"


class Settings(BaseSettings):
    """Central configuration loaded from .env or environment."""

    # ── App ──
    APP_NAME: str = "Emotion-Aware Chatbot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # ── Groq LLM ──
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_MAX_TOKENS: int = 1024
    GROQ_TEMPERATURE: float = 0.7

    # ── AWS ──
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_S3_BUCKET: str = ""
    AWS_REKOGNITION_ENABLED: bool = False

    # ── Model settings ──
    FACE_EMOTION_MODEL: str = "deepface"  # opencv | deepface
    VOICE_EMOTION_MODEL: str = "librosa"  # librosa | wav2vec2
    TEXT_EMOTION_MODEL: str = "distilbert-base-uncased-finetuned-sst-2-english"

    # ── Fusion weights ──
    WEIGHT_FACE: float = 0.35
    WEIGHT_VOICE: float = 0.30
    WEIGHT_TEXT: float = 0.35

    # ── Upload limits ──
    MAX_AUDIO_SIZE_MB: int = 10
    MAX_IMAGE_SIZE_MB: int = 5

    class Config:
        env_file = _find_env_file()
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Get settings - NO caching so .env changes are picked up on restart."""
    return Settings()
