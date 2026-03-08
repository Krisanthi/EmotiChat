"""
Chat router — main conversational endpoint with multimodal emotion analysis.
"""

import logging
from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.services.face_emotion import detect_face_emotion
from app.services.voice_emotion import detect_voice_emotion
from app.services.text_emotion import detect_text_emotion
from app.services.emotion_fusion import weighted_fusion
from app.services.llm_service import generate_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Accepts text message + optional base64-encoded webcam frame and audio clip.
    Performs multimodal emotion detection, fuses emotions, and generates
    an emotion-aware LLM response.
    """
    try:
        # ── 1. Multimodal emotion detection (parallel in production) ──
        face_emotion = await detect_face_emotion(request.face_image_b64)
        voice_emotion = await detect_voice_emotion(
            request.audio_b64, request.audio_format
        )
        text_emotion = await detect_text_emotion(request.message)

        # ── 2. Fuse emotions ──
        fused = weighted_fusion(face_emotion, voice_emotion, text_emotion)

        # ── 3. Generate emotion-aware response ──
        result = await generate_response(
            message=request.message,
            fused_emotion=fused,
            conversation_id=request.conversation_id,
            history=request.history,
        )

        return ChatResponse(
            reply=result["reply"],
            conversation_id=result["conversation_id"],
            emotion=fused,
            system_prompt_used=result["system_prompt"][:200] + "...",
            model=result["model"],
            tokens_used=result.get("tokens_used"),
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/text-only")
async def chat_text_only(request: ChatRequest):
    """
    Text-only chat endpoint (no face/voice analysis).
    Faster for text-only interactions.
    """
    try:
        text_emotion = await detect_text_emotion(request.message)
        fused = weighted_fusion(None, None, text_emotion)

        result = await generate_response(
            message=request.message,
            fused_emotion=fused,
            conversation_id=request.conversation_id,
            history=request.history,
        )

        return ChatResponse(
            reply=result["reply"],
            conversation_id=result["conversation_id"],
            emotion=fused,
            system_prompt_used=result["system_prompt"][:200] + "...",
            model=result["model"],
            tokens_used=result.get("tokens_used"),
        )

    except Exception as e:
        logger.error(f"Text-only chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
