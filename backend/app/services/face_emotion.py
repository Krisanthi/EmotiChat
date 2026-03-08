"""
Face emotion detection service.
Uses DeepFace with OpenCV backend for real-time facial emotion analysis.
Falls back to AWS Rekognition when configured.
"""

import base64
import io
import time
import logging
from typing import Dict, Optional

import cv2
import numpy as np

from app.config import get_settings
from app.models.emotion import (
    EmotionLabel,
    DEEPFACE_MAP,
    empty_emotion_vector,
    dominant_emotion,
)
from app.models.schemas import EmotionVector

logger = logging.getLogger(__name__)

# Lazy-loaded model
_deepface_loaded = False


def _ensure_deepface():
    """Lazy-load DeepFace to avoid heavy import at startup."""
    global _deepface_loaded
    if not _deepface_loaded:
        try:
            import deepface  # noqa: F401
            _deepface_loaded = True
            logger.info("DeepFace models loaded successfully")
        except ImportError:
            logger.warning("DeepFace not installed — face emotion will use fallback")


def decode_image(image_b64: str) -> np.ndarray:
    """Decode a base64 image string to an OpenCV BGR numpy array."""
    # Strip data URI prefix if present
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    img_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image from base64 data")
    return img


def analyze_face_deepface(img: np.ndarray) -> Dict[str, float]:
    """Run DeepFace emotion analysis on an image."""
    from deepface import DeepFace

    results = DeepFace.analyze(
        img_path=img,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="opencv",
        silent=True,
    )

    # DeepFace returns a list when multiple faces detected
    if isinstance(results, list):
        result = results[0] if results else {}
    else:
        result = results

    raw_emotions = result.get("emotion", {})

    # Map to canonical vector
    vector = empty_emotion_vector()
    for key, score in raw_emotions.items():
        canonical = DEEPFACE_MAP.get(key.lower())
        if canonical:
            vector[canonical.value] = score / 100.0  # DeepFace gives 0–100

    return vector


def analyze_face_opencv(img: np.ndarray) -> Dict[str, float]:
    """
    Lightweight fallback: detect face presence with OpenCV Haar cascades.
    Returns a basic emotion vector (neutral if face found, empty if not).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    vector = empty_emotion_vector()
    if len(faces) > 0:
        vector["neutral"] = 0.6
        vector["happy"] = 0.2
        vector["surprised"] = 0.1
        vector["sad"] = 0.1
    return vector


def analyze_face_rekognition(image_b64: str) -> Dict[str, float]:
    """Use AWS Rekognition for face emotion (requires AWS credentials)."""
    try:
        import boto3

        settings = get_settings()

        # Build boto3 client with explicit credentials from .env
        client_kwargs = {"region_name": settings.AWS_REGION}
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            client_kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
            client_kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY

        client = boto3.client("rekognition", **client_kwargs)

        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        response = client.detect_faces(
            Image={"Bytes": base64.b64decode(image_b64)},
            Attributes=["ALL"],
        )

        vector = empty_emotion_vector()
        if response.get("FaceDetails"):
            face = response["FaceDetails"][0]
            for emotion in face.get("Emotions", []):
                etype = emotion["Type"].lower()
                conf = emotion["Confidence"] / 100.0
                if etype in [e.value for e in EmotionLabel]:
                    vector[etype] = conf
                elif etype == "calm":
                    vector["neutral"] = conf
                elif etype == "confused":
                    vector["surprised"] = conf
                elif etype == "fear":
                    vector["fearful"] = conf
            logger.info(f"Rekognition detected face emotions: {vector}")
        else:
            logger.warning("Rekognition: no face detected in image")

        return vector

    except Exception as e:
        logger.error(f"Rekognition failed: {e}, falling back to DeepFace")
        return None  # Signal to caller to try fallback


async def detect_face_emotion(image_b64: Optional[str]) -> EmotionVector:
    """
    Main entry point for face emotion detection.
    Tries Rekognition first (if enabled), then DeepFace, then OpenCV.
    """
    if not image_b64:
        return EmotionVector(dominant="neutral", confidence=0.0)

    settings = get_settings()
    start = time.time()

    try:
        vector = None

        # Try Rekognition first
        if settings.AWS_REKOGNITION_ENABLED:
            vector = analyze_face_rekognition(image_b64)

        # Fallback to DeepFace
        if vector is None or all(v == 0.0 for v in vector.values()):
            try:
                _ensure_deepface()
                img = decode_image(image_b64)
                vector = analyze_face_deepface(img)
                logger.info("Used DeepFace for face emotion")
            except Exception as df_err:
                logger.warning(f"DeepFace failed: {df_err}, using OpenCV")
                img = decode_image(image_b64)
                vector = analyze_face_opencv(img)

        dom = dominant_emotion(vector)
        conf = vector.get(dom, 0.0)

        elapsed = (time.time() - start) * 1000
        logger.info(f"Face emotion: {dom} ({conf:.2f}) in {elapsed:.0f}ms")

        return EmotionVector(
            **vector,
            dominant=dom,
            confidence=conf,
        )

    except Exception as e:
        logger.error(f"Face emotion detection failed: {e}", exc_info=True)
        return EmotionVector(dominant="neutral", confidence=0.0)
