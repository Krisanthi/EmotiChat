"""
Text sentiment / emotion detection service.
Uses HuggingFace transformers for text classification.
"""

import time
import logging
from typing import Dict, Optional

from app.config import get_settings
from app.models.emotion import (
    EmotionLabel,
    TEXT_SENTIMENT_MAP,
    empty_emotion_vector,
    dominant_emotion,
)
from app.models.schemas import EmotionVector

logger = logging.getLogger(__name__)

# ── Lazy-loaded pipelines ──
_sentiment_pipeline = None
_emotion_pipeline = None


def _load_sentiment_pipeline():
    """Load the default sentiment analysis pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            from transformers import pipeline

            settings = get_settings()
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=settings.TEXT_EMOTION_MODEL,
                top_k=None,
            )
            logger.info(f"Sentiment pipeline loaded: {settings.TEXT_EMOTION_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load sentiment pipeline: {e}")
    return _sentiment_pipeline


def _load_emotion_pipeline():
    """Load a dedicated emotion classification pipeline (more granular)."""
    global _emotion_pipeline
    if _emotion_pipeline is None:
        try:
            from transformers import pipeline

            _emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
            )
            logger.info("Emotion pipeline loaded: j-hartmann/emotion-english-distilroberta-base")
        except Exception as e:
            logger.warning(f"Emotion pipeline unavailable, using sentiment only: {e}")
    return _emotion_pipeline


def analyze_text_sentiment(text: str) -> Dict[str, float]:
    """Analyze text with the basic sentiment model (POSITIVE/NEGATIVE)."""
    pipe = _load_sentiment_pipeline()
    if pipe is None:
        return empty_emotion_vector()

    results = pipe(text[:512])  # Truncate to model max length

    vector = empty_emotion_vector()
    if results and isinstance(results[0], list):
        results = results[0]

    for item in results:
        label = item["label"].upper()
        score = item["score"]
        canonical = TEXT_SENTIMENT_MAP.get(label)
        if canonical:
            vector[canonical.value] = score

    # Fill neutral for remaining probability
    total = sum(vector.values())
    if total < 1.0:
        vector["neutral"] += (1.0 - total) * 0.5

    return vector


def analyze_text_emotion(text: str) -> Dict[str, float]:
    """
    Analyze text with the granular emotion model.
    Maps labels: anger, disgust, fear, joy, neutral, sadness, surprise
    """
    pipe = _load_emotion_pipeline()
    if pipe is None:
        # Fallback to basic sentiment
        return analyze_text_sentiment(text)

    results = pipe(text[:512])

    vector = empty_emotion_vector()
    if results and isinstance(results[0], list):
        results = results[0]

    label_map = {
        "anger": "angry",
        "disgust": "disgusted",
        "fear": "fearful",
        "joy": "happy",
        "neutral": "neutral",
        "sadness": "sad",
        "surprise": "surprised",
    }

    for item in results:
        label = item["label"].lower()
        score = item["score"]
        mapped = label_map.get(label, label)
        if mapped in vector:
            vector[mapped] = score

    return vector


async def detect_text_emotion(text: Optional[str]) -> EmotionVector:
    """
    Main entry point for text emotion detection.
    Tries the granular emotion model first, falls back to sentiment.
    """
    if not text or not text.strip():
        return EmotionVector(dominant="neutral", confidence=0.0)

    start = time.time()

    try:
        # Try granular emotion first
        vector = analyze_text_emotion(text)

        dom = dominant_emotion(vector)
        conf = vector.get(dom, 0.0)

        elapsed = (time.time() - start) * 1000
        logger.info(f"Text emotion: {dom} ({conf:.2f}) in {elapsed:.0f}ms")

        return EmotionVector(
            **vector,
            dominant=dom,
            confidence=conf,
        )

    except Exception as e:
        logger.error(f"Text emotion detection failed: {e}")
        return EmotionVector(dominant="neutral", confidence=0.0)
