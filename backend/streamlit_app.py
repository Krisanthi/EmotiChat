"""
EmotiChat — Streamlit Frontend
Multimodal Emotion-Aware Chatbot

Run with:  streamlit run streamlit_app.py
"""

import streamlit as st
import asyncio
import base64
import time
import os
import sys
import logging

# ── Setup ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmotiChat")

# Load .env
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
load_dotenv(override=True)

from app.services.face_emotion import detect_face_emotion
from app.services.voice_emotion import detect_voice_emotion
from app.services.text_emotion import detect_text_emotion
from app.services.emotion_fusion import weighted_fusion, get_emotion_summary
from app.services.llm_service import generate_response
from app.models.schemas import EmotionVector, FusedEmotion, ChatMessage


# ══════════════════════════════════════════
#  Page Config
# ══════════════════════════════════════════

st.set_page_config(
    page_title="EmotiChat — Emotion-Aware AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 0.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #a78bfa, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .main-header p {
        color: #9898b0;
        font-size: 0.85rem;
    }
    .emotion-bar-container {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }
    .emotion-bar-label {
        width: 90px;
        font-size: 0.8rem;
        color: #ccc;
    }
    .emotion-bar-track {
        flex: 1;
        height: 10px;
        background: rgba(255,255,255,0.08);
        border-radius: 99px;
        overflow: hidden;
    }
    .emotion-bar-fill {
        height: 100%;
        border-radius: 99px;
    }
    .emotion-bar-value {
        width: 45px;
        font-size: 0.8rem;
        color: #888;
        text-align: right;
    }
    .big-emoji {
        text-align: center;
        font-size: 4rem;
        padding: 0.5rem;
    }
    .dominant-label {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        text-transform: capitalize;
    }
    .confidence-label {
        text-align: center;
        font-size: 0.8rem;
        color: #888;
    }
    .result-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════

EMOTION_EMOJI = {
    "happy": "😊", "sad": "😢", "angry": "😠", "fearful": "😰",
    "disgusted": "🤢", "surprised": "😲", "neutral": "😐", "contempt": "😏",
}
EMOTION_COLORS = {
    "happy": "#fbbf24", "sad": "#60a5fa", "angry": "#f87171", "fearful": "#a78bfa",
    "disgusted": "#34d399", "surprised": "#fb923c", "neutral": "#94a3b8", "contempt": "#e879f9",
}
EMOTION_LABELS = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral", "contempt"]


# ══════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════

def run_async(coro):
    """Run async function from sync Streamlit."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def render_emotion_bars(vector: EmotionVector) -> str:
    """Render emotion bars as HTML."""
    html = ""
    for label in EMOTION_LABELS:
        val = getattr(vector, label, 0.0)
        color = EMOTION_COLORS.get(label, "#94a3b8")
        pct = max(val * 100, 0.5)
        emoji = EMOTION_EMOJI.get(label, "")
        html += f"""
        <div class="emotion-bar-container">
            <span class="emotion-bar-label">{emoji} {label}</span>
            <div class="emotion-bar-track">
                <div class="emotion-bar-fill" style="width:{pct}%;background:{color};"></div>
            </div>
            <span class="emotion-bar-value">{val*100:.0f}%</span>
        </div>"""
    return html


def show_emotion_result(fused: FusedEmotion):
    """Display emotion analysis result in the sidebar."""
    st.session_state.emotion = fused

    dom = fused.fused.dominant
    conf = fused.fused.confidence
    emoji = EMOTION_EMOJI.get(dom, "🤖")
    color = EMOTION_COLORS.get(dom, "#94a3b8")

    st.session_state.sidebar_emoji = emoji
    st.session_state.sidebar_dominant = dom
    st.session_state.sidebar_confidence = conf
    st.session_state.sidebar_vector = fused.fused
    st.session_state.sidebar_face = fused.face
    st.session_state.sidebar_voice = fused.voice
    st.session_state.sidebar_text = fused.text


# ══════════════════════════════════════════
#  Session State
# ══════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "emotion" not in st.session_state:
    st.session_state.emotion = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "sidebar_emoji" not in st.session_state:
    st.session_state.sidebar_emoji = "🤖"
if "sidebar_dominant" not in st.session_state:
    st.session_state.sidebar_dominant = "waiting"
if "sidebar_confidence" not in st.session_state:
    st.session_state.sidebar_confidence = 0.0
if "sidebar_vector" not in st.session_state:
    st.session_state.sidebar_vector = None
if "sidebar_face" not in st.session_state:
    st.session_state.sidebar_face = None
if "sidebar_voice" not in st.session_state:
    st.session_state.sidebar_voice = None
if "sidebar_text" not in st.session_state:
    st.session_state.sidebar_text = None


# ══════════════════════════════════════════
#  Sidebar — Emotion Display
# ══════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🎯 Detected Emotion")

    emoji = st.session_state.sidebar_emoji
    dom = st.session_state.sidebar_dominant
    conf = st.session_state.sidebar_confidence
    color = EMOTION_COLORS.get(dom, "#94a3b8")

    st.markdown(f'<div class="big-emoji">{emoji}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="dominant-label" style="color:{color}">{dom}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence-label">{conf*100:.0f}% confidence</div>', unsafe_allow_html=True)

    st.divider()

    # ── Emotion Vector Bars ──
    st.markdown("### 📊 Emotion Vector")
    if st.session_state.sidebar_vector:
        bars_html = render_emotion_bars(st.session_state.sidebar_vector)
        st.markdown(f'<div class="result-box">{bars_html}</div>', unsafe_allow_html=True)
    else:
        empty_vec = EmotionVector()
        bars_html = render_emotion_bars(empty_vec)
        st.markdown(f'<div class="result-box">{bars_html}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Per-Modality ──
    st.markdown("### 🔍 Per-Modality")
    face = st.session_state.sidebar_face
    voice = st.session_state.sidebar_voice
    text = st.session_state.sidebar_text

    if face and face.confidence > 0:
        st.markdown(f"📸 Face: **{face.dominant}** ({face.confidence*100:.0f}%)")
    else:
        st.markdown("📸 Face: *no data*")

    if voice and voice.confidence > 0:
        st.markdown(f"🎤 Voice: **{voice.dominant}** ({voice.confidence*100:.0f}%)")
    else:
        st.markdown("🎤 Voice: *no data*")

    if text and text.confidence > 0:
        st.markdown(f"💬 Text: **{text.dominant}** ({text.confidence*100:.0f}%)")
    else:
        st.markdown("💬 Text: *no data*")

    st.divider()

    # ── Settings ──
    st.markdown("### ⚙️ Settings")
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key or groq_key == "your_groq_api_key_here":
        st.warning("⚠️ GROQ_API_KEY not set")
        key_input = st.text_input("Enter Groq API Key:", type="password")
        if key_input:
            os.environ["GROQ_API_KEY"] = key_input
            st.success("Key set!")
            st.rerun()
    else:
        st.success("✅ Groq API Key set")

    aws_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    if aws_key:
        st.success("✅ AWS credentials set")
    else:
        st.info("ℹ️ AWS not configured")

    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.messages = []
        st.session_state.emotion = None
        st.session_state.conversation_id = None
        st.session_state.sidebar_emoji = "🤖"
        st.session_state.sidebar_dominant = "waiting"
        st.session_state.sidebar_confidence = 0.0
        st.session_state.sidebar_vector = None
        st.session_state.sidebar_face = None
        st.session_state.sidebar_voice = None
        st.session_state.sidebar_text = None
        st.rerun()


# ══════════════════════════════════════════
#  Main Area
# ══════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🧠 EmotiChat</h1>
    <p>Detects your emotions from face, voice, or text — and responds accordingly</p>
</div>
""", unsafe_allow_html=True)

# ── Three Input Modes ──
tab_photo, tab_audio, tab_chat = st.tabs(["📸 Photo Emotion", "🎤 Voice Emotion", "💬 Chat"])


# ══════════════════════════════════════════
#  TAB 1: Photo Emotion Detection
# ══════════════════════════════════════════

with tab_photo:
    st.markdown("### Detect emotion from your face")
    st.markdown("Take a photo with your webcam. Click **Analyze Face** to detect your emotion.")

    camera_image = st.camera_input("Take a photo", key="photo_tab_cam")

    if st.button("🔍 Analyze Face", key="analyze_face_btn", type="primary", use_container_width=True):
        if camera_image is None:
            st.warning("Please take a photo first!")
        else:
            with st.spinner("Analyzing facial expression..."):
                face_b64 = base64.b64encode(camera_image.getvalue()).decode("utf-8")
                face_emotion = run_async(detect_face_emotion(face_b64))
                text_emotion = EmotionVector(dominant="neutral", confidence=0.0)
                voice_emotion = EmotionVector(dominant="neutral", confidence=0.0)

                fused = weighted_fusion(face_emotion, voice_emotion, text_emotion)
                show_emotion_result(fused)

                dom = fused.fused.dominant
                conf = fused.fused.confidence
                emoji = EMOTION_EMOJI.get(dom, "🤖")

                st.success(f"**Detected: {emoji} {dom.upper()}** ({conf*100:.0f}% confidence)")

                summary = get_emotion_summary(fused)
                st.info(summary)

                # Auto-generate AI response about the detected emotion
                try:
                    history = [ChatMessage(role=m["role"], content=m["content"]) for m in st.session_state.messages[-10:]]
                    result = run_async(generate_response(
                        message=f"[User sent a photo. Their facial expression shows: {dom} with {conf*100:.0f}% confidence. React to their emotion and describe what you detected.]",
                        fused_emotion=fused,
                        conversation_id=st.session_state.conversation_id,
                        history=history,
                    ))
                    st.session_state.conversation_id = result["conversation_id"]
                    st.markdown("**AI Response:**")
                    st.write(result["reply"])

                    st.session_state.messages.append({"role": "user", "content": f"[Sent photo: {dom}]"})
                    st.session_state.messages.append({"role": "assistant", "content": result["reply"]})
                except Exception as e:
                    st.error(f"AI response error: {e}")

                st.rerun()


# ══════════════════════════════════════════
#  TAB 2: Voice Emotion Detection
# ══════════════════════════════════════════

with tab_audio:
    st.markdown("### Detect emotion from your voice")
    st.markdown("Record yourself speaking. Click **Analyze Voice** to detect your emotional tone.")

    audio_input = st.audio_input("Record audio", key="audio_tab_rec")

    if st.button("🔍 Analyze Voice", key="analyze_voice_btn", type="primary", use_container_width=True):
        if audio_input is None:
            st.warning("Please record audio first!")
        else:
            with st.spinner("Analyzing voice tone..."):
                audio_b64 = base64.b64encode(audio_input.getvalue()).decode("utf-8")
                voice_emotion = run_async(detect_voice_emotion(audio_b64, "wav"))
                face_emotion = EmotionVector(dominant="neutral", confidence=0.0)
                text_emotion = EmotionVector(dominant="neutral", confidence=0.0)

                fused = weighted_fusion(face_emotion, voice_emotion, text_emotion)
                show_emotion_result(fused)

                dom = fused.fused.dominant
                conf = fused.fused.confidence
                emoji = EMOTION_EMOJI.get(dom, "🤖")

                st.success(f"**Detected: {emoji} {dom.upper()}** ({conf*100:.0f}% confidence)")

                summary = get_emotion_summary(fused)
                st.info(summary)

                # Auto-generate AI response
                try:
                    history = [ChatMessage(role=m["role"], content=m["content"]) for m in st.session_state.messages[-10:]]
                    result = run_async(generate_response(
                        message=f"[User sent a voice recording. Their voice tone shows: {dom} with {conf*100:.0f}% confidence. React to their emotion and describe what you heard in their voice.]",
                        fused_emotion=fused,
                        conversation_id=st.session_state.conversation_id,
                        history=history,
                    ))
                    st.session_state.conversation_id = result["conversation_id"]
                    st.markdown("**AI Response:**")
                    st.write(result["reply"])

                    st.session_state.messages.append({"role": "user", "content": f"[Sent audio: {dom}]"})
                    st.session_state.messages.append({"role": "assistant", "content": result["reply"]})
                except Exception as e:
                    st.error(f"AI response error: {e}")

                st.rerun()


# ══════════════════════════════════════════
#  TAB 3: Chat
# ══════════════════════════════════════════

with tab_chat:
    st.markdown("### Chat with EmotiChat")
    st.markdown("Type a message. Your text sentiment will be analyzed automatically.")

    # Optional: attach photo/audio to chat
    with st.expander("📎 Attach photo or audio to your message (optional)"):
        chat_camera = st.camera_input("Photo for chat", key="chat_cam", label_visibility="collapsed")
        chat_audio = st.audio_input("Audio for chat", key="chat_audio", label_visibility="collapsed")

    # Show message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.write(msg["content"])

    # Chat input
    user_input = st.chat_input("Type a message...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user", avatar="👤"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing emotions and responding..."):

                # Face emotion
                face_b64 = None
                if chat_camera is not None:
                    face_b64 = base64.b64encode(chat_camera.getvalue()).decode("utf-8")
                face_emotion = run_async(detect_face_emotion(face_b64))

                # Voice emotion
                audio_b64 = None
                if chat_audio is not None:
                    audio_b64 = base64.b64encode(chat_audio.getvalue()).decode("utf-8")
                voice_emotion = run_async(detect_voice_emotion(audio_b64, "wav"))

                # Text emotion
                text_emotion = run_async(detect_text_emotion(user_input))

                # Fuse
                fused = weighted_fusion(face_emotion, voice_emotion, text_emotion)
                show_emotion_result(fused)

                # Generate response
                try:
                    history = [ChatMessage(role=m["role"], content=m["content"]) for m in st.session_state.messages[-10:]]
                    result = run_async(generate_response(
                        message=user_input,
                        fused_emotion=fused,
                        conversation_id=st.session_state.conversation_id,
                        history=history,
                    ))
                    reply = result["reply"]
                    st.session_state.conversation_id = result["conversation_id"]
                except Exception as e:
                    reply = f"❌ Error: {str(e)}"

                st.write(reply)

                dom = fused.fused.dominant
                emoji = EMOTION_EMOJI.get(dom, "")
                summary = get_emotion_summary(fused)
                st.caption(f"{emoji} {summary}")

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()
