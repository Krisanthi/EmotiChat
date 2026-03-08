"""
Voice / audio emotion detection service.
Uses librosa for feature extraction and a simple classifier,
with optional Wav2Vec2 from HuggingFace for higher accuracy.
"""

import base64
import io
import os
import tempfile
import time
import logging
from typing import Dict, Optional

import numpy as np

from app.config import get_settings
from app.models.emotion import (
    EmotionLabel,
    VOICE_EMOTION_MAP,
    empty_emotion_vector,
    dominant_emotion,
)
from app.models.schemas import EmotionVector

logger = logging.getLogger(__name__)

# ── Lazy-loaded models ──
_wav2vec_pipeline = None


def _load_wav2vec_pipeline():
    """Lazy-load the Wav2Vec2 emotion classification pipeline."""
    global _wav2vec_pipeline
    if _wav2vec_pipeline is None:
        try:
            from transformers import pipeline
            _wav2vec_pipeline = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                top_k=None,
            )
            logger.info("Wav2Vec2 emotion pipeline loaded")
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 pipeline: {e}")
    return _wav2vec_pipeline


def decode_audio(audio_b64: str, fmt: str = "wav") -> str:
    """Decode base64 audio to a temp file path."""
    if "," in audio_b64:
        audio_b64 = audio_b64.split(",", 1)[1]

    audio_bytes = base64.b64decode(audio_b64)
    tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def extract_librosa_features(audio_path: str) -> Dict[str, float]:
    """
    Extract audio features with librosa and map to emotion probabilities
    using energy, pitch, tempo, and spectral features.
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=16000, duration=30)

    if len(y) == 0:
        return empty_emotion_vector()

    # Feature extraction
    energy = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = float(np.mean(mfccs))

    # Tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0])

    # Pitch (fundamental frequency estimate)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    avg_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0

    # ── Rule-based emotion mapping from audio features ──
    vector = empty_emotion_vector()

    # Normalize features to 0–1 range (approximate)
    energy_norm = min(energy / 0.1, 1.0)
    tempo_norm = min(tempo / 200.0, 1.0)
    pitch_norm = min(avg_pitch / 500.0, 1.0)
    zcr_norm = min(zcr / 0.15, 1.0)

    # High energy + high tempo + high pitch → angry or happy
    if energy_norm > 0.6 and tempo_norm > 0.6:
        if pitch_norm > 0.5:
            vector["angry"] = 0.4
            vector["happy"] = 0.3
        else:
            vector["angry"] = 0.5
            vector["fearful"] = 0.2

    # Low energy + slow tempo → sad
    elif energy_norm < 0.3 and tempo_norm < 0.4:
        vector["sad"] = 0.5
        vector["neutral"] = 0.3

    # High pitch + moderate energy → surprised or happy
    elif pitch_norm > 0.6 and energy_norm > 0.4:
        vector["surprised"] = 0.4
        vector["happy"] = 0.3

    # Default → neutral
    else:
        vector["neutral"] = 0.5
        vector["happy"] = 0.15
        vector["sad"] = 0.15

    # Fill remaining probability
    total = sum(vector.values())
    if total < 1.0:
        vector["neutral"] += 1.0 - total

    return vector


def analyze_voice_wav2vec(audio_path: str) -> Dict[str, float]:
    """Use Wav2Vec2 fine-tuned for speech emotion recognition."""
    pipe = _load_wav2vec_pipeline()
    if pipe is None:
        return extract_librosa_features(audio_path)

    try:
        results = pipe(audio_path)

        vector = empty_emotion_vector()
        for item in results:
            label = item["label"].lower()
            score = item["score"]
            canonical = VOICE_EMOTION_MAP.get(label)
            if canonical:
                vector[canonical.value] = max(vector[canonical.value], score)

        # Normalize
        total = sum(vector.values())
        if total > 0:
            vector = {k: v / total for k, v in vector.items()}

        return vector

    except Exception as e:
        logger.error(f"Wav2Vec2 analysis failed, falling back to librosa: {e}")
        return extract_librosa_features(audio_path)


async def detect_voice_emotion(
    audio_b64: Optional[str], audio_format: str = "wav"
) -> EmotionVector:
    """
    Main entry point for voice emotion detection.
    """
    if not audio_b64:
        return EmotionVector(dominant="neutral", confidence=0.0)

    settings = get_settings()
    start = time.time()
    audio_path = None

    try:
        audio_path = decode_audio(audio_b64, audio_format)

        if settings.VOICE_EMOTION_MODEL == "wav2vec2":
            vector = analyze_voice_wav2vec(audio_path)
        else:
            vector = extract_librosa_features(audio_path)

        dom = dominant_emotion(vector)
        conf = vector.get(dom, 0.0)

        elapsed = (time.time() - start) * 1000
        logger.info(f"Voice emotion: {dom} ({conf:.2f}) in {elapsed:.0f}ms")

        return EmotionVector(
            **vector,
            dominant=dom,
            confidence=conf,
        )

    except Exception as e:
        logger.error(f"Voice emotion detection failed: {e}")
        return EmotionVector(dominant="neutral", confidence=0.0)

    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
