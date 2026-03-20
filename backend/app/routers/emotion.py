import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import base64
import logging
import os
import tempfile
import time
from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.schemas import EmotionVector, FusedEmotion
from app.models.emotion import empty_emotion_vector, dominant_emotion
from app.services.face_emotion import detect_face_emotion
from app.services.voice_emotion import detect_voice_emotion
from app.services.text_emotion import detect_text_emotion
from app.services.emotion_fusion import weighted_fusion, get_emotion_summary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/emotion", tags=["Emotion Analysis"])

_frame_buffer: Dict[str, List[Dict[str, float]]] = defaultdict(list)
_audio_buffer: Dict[str, List[Dict]] = defaultdict(list)
_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model, first run may take a few minutes...")
        import whisper
        _whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully")
    return _whisper_model


def _convert_to_wav(input_path: str, output_path: str) -> bool:
    try:
        import subprocess
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1",
                "-f", "wav", output_path
            ],
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            logger.info(f"ffmpeg converted to WAV: {os.path.getsize(output_path)} bytes")
            return True
        logger.warning(f"ffmpeg failed rc={result.returncode}: {result.stderr.decode()[:300]}")
        return False
    except FileNotFoundError:
        logger.warning("ffmpeg not found — install with: brew install ffmpeg (mac) or apt install ffmpeg (linux)")
        return False
    except Exception as e:
        logger.warning(f"ffmpeg conversion error: {e}")
        return False


class FrameRequest(BaseModel):
    image_b64: str
    user_id: str


class AudioChunkRequest(BaseModel):
    audio_b64: str
    user_id: str
    audio_format: str = "webm"


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


class AverageResponse(BaseModel):
    face_vector: Dict[str, float]
    face_dominant: str
    face_confidence: float
    face_frame_count: int
    transcript: str
    voice_vector: Dict[str, float]
    voice_dominant: str
    voice_confidence: float
    audio_chunk_count: int


@router.post("/frame")
async def receive_frame(request: FrameRequest):
    try:
        face_result = await detect_face_emotion(request.image_b64)
        vector = {
            "happy": face_result.happy,
            "sad": face_result.sad,
            "angry": face_result.angry,
            "fearful": face_result.fearful,
            "disgusted": face_result.disgusted,
            "surprised": face_result.surprised,
            "neutral": face_result.neutral,
            "contempt": face_result.contempt,
        }
        _frame_buffer[request.user_id].append(vector)
        dom = face_result.dominant
        conf = face_result.confidence
        logger.info(
            f"Frame buffered for {request.user_id[:8]}... "
            f"{dom}({conf:.2f}) total={len(_frame_buffer[request.user_id])}"
        )
        return {
            "status": "ok",
            "dominant": dom,
            "confidence": conf,
            "buffered_frames": len(_frame_buffer[request.user_id]),
        }
    except Exception as e:
        logger.error(f"Frame processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio")
async def receive_audio(request: AudioChunkRequest):
    input_path = None
    wav_path = None
    try:
        raw_b64 = request.audio_b64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]

        rem = len(raw_b64) % 4
        if rem:
            raw_b64 += "=" * (4 - rem)

        audio_bytes = base64.b64decode(raw_b64)
        logger.info(f"Audio received: {len(audio_bytes)} bytes format={request.audio_format}")

        if len(audio_bytes) < 1000:
            logger.warning(f"Audio chunk too small ({len(audio_bytes)} bytes), skipping")
            return {
                "status": "skipped",
                "reason": "audio too small",
                "size_bytes": len(audio_bytes)
            }

        fmt = request.audio_format.lower().replace("audio/", "").split(";")[0].strip()
        suffix = f".{fmt}" if fmt in ("webm", "ogg", "opus", "mp4", "m4a", "wav") else ".webm"

        tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_in.write(audio_bytes)
        tmp_in.flush()
        tmp_in.close()
        input_path = tmp_in.name
        logger.info(f"Saved audio input: {input_path} ({os.path.getsize(input_path)} bytes)")
        import shutil
        shutil.copy(input_path, f"/tmp/debug_audio_{request.user_id[:4]}.webm")
        logger.info(f"DEBUG: saved raw audio to /tmp/debug_audio_{request.user_id[:4]}.webm")
        wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_tmp.close()
        wav_path = wav_tmp.name

        converted = _convert_to_wav(input_path, wav_path)
        transcribe_path = wav_path if converted else input_path
        logger.info(f"Transcribe path: {transcribe_path} ({os.path.getsize(transcribe_path)} bytes)")

        transcript = ""
        try:
            model = _load_whisper()
            result = model.transcribe(
                transcribe_path,
                language="en",
                fp16=False,
                verbose=False,
                condition_on_previous_text=False,
            )
            logger.info(f"Whisper raw result keys: {list(result.keys())}")
            logger.info(f"Whisper segments: {result.get('segments', [])[:3]}")
            transcript = result.get("text", "").strip()
            if not transcript:
                logger.warning(
                    f"Whisper returned empty transcript — "
                    f"audio size={os.path.getsize(transcribe_path)} bytes, "
                    f"format={fmt}"
                )
            else:
                logger.info(f"Whisper transcript ({len(transcript)} chars): '{transcript[:200]}'")
        except Exception as we:
            logger.error(f"Whisper transcription failed: {we}", exc_info=True)

        voice_b64_clean = base64.b64encode(audio_bytes).decode("utf-8")
        voice_fmt = "wav" if converted else fmt
        voice_result = await detect_voice_emotion(voice_b64_clean, voice_fmt)

        voice_vector = {
            "happy": voice_result.happy,
            "sad": voice_result.sad,
            "angry": voice_result.angry,
            "fearful": voice_result.fearful,
            "disgusted": voice_result.disgusted,
            "surprised": voice_result.surprised,
            "neutral": voice_result.neutral,
            "contempt": voice_result.contempt,
        }

        _audio_buffer[request.user_id].append({
            "transcript": transcript,
            "voice_vector": voice_vector,
            "voice_dominant": voice_result.dominant,
            "voice_confidence": voice_result.confidence,
        })

        logger.info(
            f"Audio buffered for {request.user_id[:8]}... "
            f"voice={voice_result.dominant}({voice_result.confidence:.2f}) "
            f"chunks={len(_audio_buffer[request.user_id])}"
        )
        return {
            "status": "ok",
            "transcript": transcript,
            "voice_dominant": voice_result.dominant,
            "voice_confidence": voice_result.confidence,
            "buffered_chunks": len(_audio_buffer[request.user_id]),
            "audio_size_bytes": len(audio_bytes),
        }

    except Exception as e:
        logger.error(f"Audio processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [input_path, wav_path]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


@router.get("/average")
async def get_average(user_id: str):
    frames = _frame_buffer.get(user_id, [])
    audio_chunks = _audio_buffer.get(user_id, [])

    if frames:
        avg_vector = empty_emotion_vector()
        for vec in frames:
            for key in avg_vector:
                avg_vector[key] += vec.get(key, 0.0)
        for key in avg_vector:
            avg_vector[key] /= len(frames)
        face_dom = dominant_emotion(avg_vector)
        face_conf = avg_vector.get(face_dom, 0.0)
    else:
        avg_vector = empty_emotion_vector()
        face_dom = "neutral"
        face_conf = 0.0

    all_transcripts = []
    combined_voice = empty_emotion_vector()
    for chunk in audio_chunks:
        t = chunk.get("transcript", "").strip()
        if t:
            all_transcripts.append(t)
        vv = chunk.get("voice_vector", {})
        for key in combined_voice:
            combined_voice[key] += vv.get(key, 0.0)
    if audio_chunks:
        for key in combined_voice:
            combined_voice[key] /= len(audio_chunks)

    voice_dom = dominant_emotion(combined_voice)
    voice_conf = combined_voice.get(voice_dom, 0.0)
    full_transcript = " ".join(all_transcripts).strip()

    logger.info(
        f"Average for {user_id[:8]}...: "
        f"frames={len(frames)} face={face_dom}({face_conf:.2f}) "
        f"audio_chunks={len(audio_chunks)} voice={voice_dom}({voice_conf:.2f}) "
        f"transcript='{full_transcript[:100]}'"
    )

    return AverageResponse(
        face_vector=avg_vector,
        face_dominant=face_dom,
        face_confidence=face_conf,
        face_frame_count=len(frames),
        transcript=full_transcript,
        voice_vector=combined_voice,
        voice_dominant=voice_dom,
        voice_confidence=voice_conf,
        audio_chunk_count=len(audio_chunks),
    )


@router.delete("/buffer")
async def clear_buffer(user_id: str):
    cleared_frames = len(_frame_buffer.pop(user_id, []))
    cleared_audio = len(_audio_buffer.pop(user_id, []))
    logger.info(f"Buffer cleared for {user_id[:8]}... frames={cleared_frames} audio={cleared_audio}")
    return {
        "status": "cleared",
        "frames_cleared": cleared_frames,
        "audio_cleared": cleared_audio
    }


@router.post("/face", response_model=EmotionResponse)
async def analyze_face(request: FaceRequest):
    try:
        emotion = await detect_face_emotion(request.image_b64)
        return EmotionResponse(
            emotion=emotion,
            summary=f"Detected facial emotion: {emotion.dominant} (confidence: {emotion.confidence:.0%})",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice", response_model=EmotionResponse)
async def analyze_voice(request: VoiceRequest):
    try:
        emotion = await detect_voice_emotion(request.audio_b64, request.audio_format)
        return EmotionResponse(
            emotion=emotion,
            summary=f"Detected voice emotion: {emotion.dominant} (confidence: {emotion.confidence:.0%})",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text", response_model=EmotionResponse)
async def analyze_text(request: TextRequest):
    try:
        emotion = await detect_text_emotion(request.text)
        return EmotionResponse(
            emotion=emotion,
            summary=f"Detected text emotion: {emotion.dominant} (confidence: {emotion.confidence:.0%})",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=FusedEmotionResponse)
async def analyze_multimodal(request: MultimodalRequest):
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