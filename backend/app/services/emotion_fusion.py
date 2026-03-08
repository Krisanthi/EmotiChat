"""
Emotion fusion service.
Combines face, voice, and text emotion vectors into a single fused emotion state
using configurable weighted averaging with confidence-based adjustments.
"""

import logging
from typing import Dict, Optional

from app.config import get_settings
from app.models.emotion import EmotionLabel, empty_emotion_vector, dominant_emotion
from app.models.schemas import EmotionVector, FusedEmotion

logger = logging.getLogger(__name__)


def weighted_fusion(
    face: Optional[EmotionVector],
    voice: Optional[EmotionVector],
    text: Optional[EmotionVector],
) -> FusedEmotion:
    """
    Fuse emotion vectors using weighted averaging.

    Strategy:
    1. Start with configured weights (face=0.35, voice=0.30, text=0.35)
    2. Adjust weights based on confidence of each modality
    3. Re-normalize weights to sum to 1.0
    4. Compute weighted average across all emotion dimensions
    5. Apply softmax-like normalization to produce valid probabilities
    """
    settings = get_settings()

    # Base weights
    weights = {
        "face": settings.WEIGHT_FACE,
        "voice": settings.WEIGHT_VOICE,
        "text": settings.WEIGHT_TEXT,
    }

    # Available modalities and their vectors
    modalities: Dict[str, EmotionVector] = {}
    if face and face.confidence > 0:
        modalities["face"] = face
    if voice and voice.confidence > 0:
        modalities["voice"] = voice
    if text and text.confidence > 0:
        modalities["text"] = text

    # If no modalities available, return neutral
    if not modalities:
        neutral_vector = EmotionVector(neutral=1.0, dominant="neutral", confidence=0.0)
        return FusedEmotion(
            face=face,
            voice=voice,
            text=text,
            fused=neutral_vector,
            weights_used=weights,
        )

    # Adjust weights by confidence and filter to available modalities
    adjusted_weights = {}
    for mod_name, mod_vector in modalities.items():
        # Scale weight by confidence (higher confidence → more influence)
        adjusted_weights[mod_name] = weights[mod_name] * (0.5 + 0.5 * mod_vector.confidence)

    # Normalize weights to sum to 1.0
    total_weight = sum(adjusted_weights.values())
    if total_weight > 0:
        adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

    # Compute fused vector
    fused = empty_emotion_vector()
    emotion_labels = list(fused.keys())

    for label in emotion_labels:
        weighted_sum = 0.0
        for mod_name, mod_vector in modalities.items():
            mod_dict = mod_vector.model_dump()
            value = mod_dict.get(label, 0.0)
            weighted_sum += value * adjusted_weights.get(mod_name, 0.0)
        fused[label] = weighted_sum

    # Normalize fused vector to sum to 1.0
    total = sum(fused.values())
    if total > 0:
        fused = {k: v / total for k, v in fused.items()}

    # Determine dominant emotion and confidence
    dom = dominant_emotion(fused)
    conf = fused.get(dom, 0.0)

    # Compute overall confidence as weighted average of modality confidences
    overall_confidence = sum(
        mod.confidence * adjusted_weights.get(name, 0.0)
        for name, mod in modalities.items()
    )

    fused_vector = EmotionVector(
        **fused,
        dominant=dom,
        confidence=min(overall_confidence, 1.0),
    )

    logger.info(
        f"Fused emotion: {dom} (conf={conf:.2f}) from "
        f"{list(modalities.keys())} with weights {adjusted_weights}"
    )

    return FusedEmotion(
        face=face,
        voice=voice,
        text=text,
        fused=fused_vector,
        weights_used=adjusted_weights,
    )


def get_emotion_summary(fused: FusedEmotion) -> str:
    """Generate a human-readable summary of the emotional state."""
    dom = fused.fused.dominant
    conf = fused.fused.confidence

    # Identify contributing modalities
    sources = []
    if fused.face and fused.face.confidence > 0.1:
        sources.append(f"facial expression ({fused.face.dominant})")
    if fused.voice and fused.voice.confidence > 0.1:
        sources.append(f"voice tone ({fused.voice.dominant})")
    if fused.text and fused.text.confidence > 0.1:
        sources.append(f"text sentiment ({fused.text.dominant})")

    source_str = ", ".join(sources) if sources else "no clear signals"

    intensity = "strongly" if conf > 0.6 else "moderately" if conf > 0.3 else "slightly"

    return (
        f"The user appears {intensity} {dom} "
        f"(confidence: {conf:.0%}), based on {source_str}."
    )
