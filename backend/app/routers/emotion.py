"""
Emotion analysis router — standalone endpoints for individual modality testing.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.schemas import EmotionVector, FusedEmotion
from app.services.face_emotion import detect_face_emotion
from app.services.voice_emotion import detect_voice_emotion
from app.services.text_emotion import detect_text_emotion
from app.services.emotion_fusion import weighted_fusion, get_emotion_summary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/emotion", tags=["Emotion Analysis"])


class FaceRequest(BaseModel):
    image_b64: str


class VoiceRequest(BaseModel):
    audio_b64: str
    audio_format: str = "wav"


class TextRequest(BaseModel):
    text: str


class MultimodalRequest(BaseModel):
    text: Optional[str] = None
    image_b64: Optional[str] = None
    audio_b64: Optional[str] = None
    audio_format: str = "wav"


class EmotionResponse(BaseModel):
    emotion: EmotionVector
    summary: str


class FusedEmotionResponse(BaseModel):
    fused_emotion: FusedEmotion
    summary: str


# ── Individual modality endpoints ──

@router.post("/face", response_model=EmotionResponse)
async def analyze_face(request: FaceRequest):
    """Analyze facial emotion from a base64-encoded image."""
    try:
        emotion = await detect_face_emotion(request.image_b64)
        return EmotionResponse(
            emotion=emotion,
            summary=f"Detected facial emotion: {emotion.dominant} "
                     f"(confidence: {emotion.confidence:.0%})",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice", response_model=EmotionResponse)
async def analyze_voice(request: VoiceRequest):
    """Analyze voice emotion from base64-encoded audio."""
    try:
        emotion = await detect_voice_emotion(request.audio_b64, request.audio_format)
        return EmotionResponse(
            emotion=emotion,
            summary=f"Detected voice emotion: {emotion.dominant} "
                     f"(confidence: {emotion.confidence:.0%})",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text", response_model=EmotionResponse)
async def analyze_text(request: TextRequest):
    """Analyze text sentiment/emotion."""
    try:
        emotion = await detect_text_emotion(request.text)
        return EmotionResponse(
            emotion=emotion,
            summary=f"Detected text emotion: {emotion.dominant} "
                     f"(confidence: {emotion.confidence:.0%})",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Multimodal fusion endpoint ──

@router.post("/analyze", response_model=FusedEmotionResponse)
async def analyze_multimodal(request: MultimodalRequest):
    """
    Analyze emotion from multiple modalities and return fused result.
    Send any combination of text, image, and audio.
    """
    try:
        face = await detect_face_emotion(request.image_b64)
        voice = await detect_voice_emotion(request.audio_b64, request.audio_format)
        text = await detect_text_emotion(request.text)

        fused = weighted_fusion(face, voice, text)
        summary = get_emotion_summary(fused)

        return FusedEmotionResponse(fused_emotion=fused, summary=summary)

    except Exception as e:
        logger.error(f"Multimodal analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
