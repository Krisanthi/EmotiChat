"""
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import chat, emotion

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"   Model: {settings.GROQ_MODEL}")
    logger.info(f"   Face backend: {settings.FACE_EMOTION_MODEL}")
    logger.info(f"   Voice backend: {settings.VOICE_EMOTION_MODEL}")
    logger.info(f"   Fusion weights: face={settings.WEIGHT_FACE}, "
                f"voice={settings.WEIGHT_VOICE}, text={settings.WEIGHT_TEXT}")

    if not settings.GROQ_API_KEY:
        logger.warning("⚠️  GROQ_API_KEY not set — LLM calls will fail")

    yield

    logger.info("👋 Shutting down")


# ── App Factory ──

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Multimodal Emotion-Aware Chatbot — analyzes facial expressions, "
            "voice tone, and text sentiment to generate empathetic AI responses."
        ),
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # ── CORS ──
    origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──
    app.include_router(chat.router)
    app.include_router(emotion.router)

    # ── Health check ──
    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "version": settings.APP_VERSION,
            "model": settings.GROQ_MODEL,
            "groq_configured": bool(settings.GROQ_API_KEY),
        }

    @app.get("/")
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/api/docs",
        }

    return app


app = create_app()
