import streamlit as st
import streamlit.components.v1 as components
import asyncio
import base64
import time
import os
import sys
import uuid
import logging
import random
import json
import threading
import socket
from datetime import datetime, timedelta, timezone
from collections import Counter
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmotiChat")

from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
load_dotenv(override=True)


def _port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _start_fastapi_background():
    if _port_in_use(8000):
        logger.info("FastAPI already running on port 8000")
        return
    logger.info("Starting FastAPI server on port 8000 (background thread)...")
    import uvicorn
    from app.main import app as fastapi_app
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")


if not _port_in_use(8000):
    _api_thread = threading.Thread(target=_start_fastapi_background, daemon=True)
    _api_thread.start()
    time.sleep(2)


from app.services.text_emotion import detect_text_emotion
from app.services.emotion_fusion import weighted_fusion, get_emotion_summary
from app.services.llm_service import generate_response
from app.services.dynamo_service import save_interaction, fetch_emotion_history
from app.models.schemas import EmotionVector, FusedEmotion, ChatMessage

API_BASE = "http://localhost:8000/api"

st.set_page_config(
    page_title="EmotiChat | Purrbot",
    page_icon="😸",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMOTION_EMOJI = {
    "happy": "😊", "sad": "😢", "angry": "😠", "fearful": "😰",
    "disgusted": "🤢", "surprised": "😲", "neutral": "😐", "contempt": "😏",
}
EMOTION_COLORS = {
    "happy": "#fbbf24", "sad": "#60a5fa", "angry": "#f87171", "fearful": "#a78bfa",
    "disgusted": "#34d399", "surprised": "#fb923c", "neutral": "#94a3b8", "contempt": "#e879f9",
}
EMOTION_LABELS = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral", "contempt"]

PURRBOT_TIPS = [
    "Did you know cats spend 70% of their lives sleeping? Maybe you need a catnap too! 😸",
    "A cat's purr vibrates at 25-150 Hz — frequencies that promote healing. Let Purrbot heal your mood! 🐾",
    "Cats have over 20 vocalizations. Purrbot has 8 emotions to read yours! 😸",
    "When a cat slow-blinks at you, it means they trust you. Purrbot slow-blinks at you right now. 🐾",
    "Cats can rotate their ears 180 degrees. Purrbot is always listening to your feelings! 😸",
    "A group of cats is called a clowder. You are part of the EmotiChat clowder now! 🐾",
    "Cats dream just like humans. What emotions will you share with Purrbot today? 😸",
    "The oldest known cat lived to 38 years. Purrbot plans to be here even longer! 🐾",
]

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp { background-color: #1a1a18 !important; font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] {
        background-color: #111110 !important;
        border-right: 1px solid #2a2a26 !important;
    }
    [data-testid="stSidebar"] * { color: #e8a87c !important; }

    .main-header { text-align: center; padding: 0.8rem 0 0.3rem 0; }
    .main-header h1 {
        background: linear-gradient(135deg, #c4692a, #e8a87c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem; font-weight: 800; margin-bottom: 0;
    }
    .main-header p { color: #f0c9a8; font-size: 0.85rem; margin-top: 4px; }

    .emotion-pill {
        display: inline-block; padding: 3px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin-bottom: 4px;
    }

    .emotion-bar-container { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
    .emotion-bar-label { width: 90px; font-size: 0.78rem; color: #e8a87c; }
    .emotion-bar-track { flex: 1; height: 10px; background: rgba(196,105,42,0.15); border-radius: 99px; overflow: hidden; }
    .emotion-bar-fill { height: 100%; border-radius: 99px; }
    .emotion-bar-value { width: 42px; font-size: 0.78rem; color: #f0c9a8; text-align: right; }

    .big-emoji { text-align: center; font-size: 3.5rem; padding: 0.3rem; }
    .dominant-label { text-align: center; font-size: 1.1rem; font-weight: 700; text-transform: capitalize; color: #c4692a; }
    .confidence-label { text-align: center; font-size: 0.78rem; color: #f0c9a8; }

    .purrbot-card {
        background: linear-gradient(135deg, rgba(196,105,42,0.15), rgba(232,168,124,0.08));
        border: 1px solid rgba(196,105,42,0.3); border-radius: 12px; padding: 0.8rem; margin: 0.5rem 0;
    }
    .purrbot-card-title { font-size: 0.85rem; font-weight: 700; color: #c4692a; margin-bottom: 4px; }
    .purrbot-card-body { font-size: 0.78rem; color: #f0c9a8; line-height: 1.4; }

    .result-box {
        background: rgba(196,105,42,0.06); border: 1px solid rgba(196,105,42,0.15);
        border-radius: 12px; padding: 0.8rem; margin: 0.5rem 0;
    }

    .user-badge {
        text-align: center; font-size: 1rem; font-weight: 600; color: #c4692a;
        padding: 0.3rem; background: rgba(196,105,42,0.1); border-radius: 8px; margin-bottom: 0.5rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background-color: rgba(196,105,42,0.08); border-radius: 12px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] { color: #e8a87c !important; border-radius: 8px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: rgba(196,105,42,0.25) !important; color: #c4692a !important; }

    [data-testid="stChatMessage"] {
        background: rgba(196,105,42,0.05) !important;
        border: 1px solid rgba(196,105,42,0.1) !important;
        border-radius: 12px !important;
    }

    .stChatInput > div { border-color: rgba(196,105,42,0.3) !important; }

    .stButton > button { border-color: #c4692a !important; color: #e8a87c !important; }
    .stButton > button:hover { background-color: rgba(196,105,42,0.2) !important; }
    .stButton > button[kind="primary"] { background-color: #c4692a !important; color: #1a1a18 !important; }

    .week-card {
        background: rgba(196,105,42,0.08); border: 1px solid rgba(196,105,42,0.15);
        border-radius: 12px; padding: 0.6rem; text-align: center; min-height: 90px;
    }
    .week-card-today {
        background: rgba(196,105,42,0.20); border: 2px solid #c4692a;
        border-radius: 12px; padding: 0.6rem; text-align: center; min-height: 90px;
    }
    .week-card-emoji { font-size: 1.8rem; }
    .week-card-label { font-size: 0.7rem; color: #f0c9a8; text-transform: capitalize; margin-top: 2px; }
    .week-card-day { font-size: 0.65rem; color: #e8a87c; font-weight: 600; }

    .month-cell {
        background: rgba(196,105,42,0.06); border: 1px solid rgba(196,105,42,0.1);
        border-radius: 8px; padding: 4px; text-align: center; min-height: 50px;
        font-size: 0.65rem; color: #f0c9a8;
    }
    .month-cell-emoji { font-size: 1.2rem; }

    .stat-box {
        background: rgba(196,105,42,0.10); border: 1px solid rgba(196,105,42,0.2);
        border-radius: 12px; padding: 0.8rem; text-align: center;
    }
    .stat-value { font-size: 1.4rem; font-weight: 700; color: #c4692a; }
    .stat-label { font-size: 0.7rem; color: #f0c9a8; }

    h1, h2, h3, h4, h5, h6, p, span, div, label { color: #e8a87c !important; }
    .stMarkdown { color: #e8a87c !important; }
</style>
""", unsafe_allow_html=True)


def run_async(coro):
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


def build_media_html(user_id: str, camera_on: bool, mic_on: bool) -> str:
    if not camera_on and not mic_on:
        return ""

    show_video = "true" if camera_on else "false"
    show_audio = "true" if mic_on else "false"

    html = f"""
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#1a1a18; font-family:'Inter',sans-serif; overflow:hidden; }}
    #media-row {{ display:flex; gap:12px; width:100%; padding:8px; }}
    .media-box {{
        flex:1; background:#000; border:2px solid #c4692a;
        border-radius:12px; overflow:hidden; position:relative;
    }}
    .media-box video {{ width:100%; height:180px; object-fit:cover; display:block; }}
    .media-box canvas {{ width:100%; height:180px; display:block; }}
    .media-status {{
        position:absolute; bottom:0; left:0; right:0;
        background:rgba(0,0,0,0.75); color:#f0c9a8;
        font-size:11px; padding:4px 8px; text-align:center;
    }}
</style>

<div id="media-row"></div>

<script>
(function() {{
    const API      = "{API_BASE}";
    const UID      = "{user_id}";
    const wantVideo = {show_video};
    const wantAudio = {show_audio};
    const row = document.getElementById('media-row');

    if (wantVideo) {{
        const box = document.createElement('div');
        box.className = 'media-box';
        box.innerHTML = '<video id="cam-preview" autoplay playsinline muted></video>'
                      + '<div id="face-status" class="media-status">Starting camera...</div>';
        row.appendChild(box);
    }}

    if (wantAudio) {{
        const box = document.createElement('div');
        box.className = 'media-box';
        box.innerHTML = '<canvas id="audio-viz"></canvas>'
                      + '<div id="mic-status" class="media-status">Starting mic...</div>';
        row.appendChild(box);
    }}

    async function startMedia() {{
        const constraints = {{}};
        if (wantVideo) constraints.video = {{ width:640, height:480, facingMode:"user" }};
        if (wantAudio) constraints.audio = {{ echoCancellation:true, noiseSuppression:true, autoGainControl:true }};

        let stream;
        try {{
            stream = await navigator.mediaDevices.getUserMedia(constraints);
        }} catch(err) {{
            if (document.getElementById('face-status'))
                document.getElementById('face-status').textContent = 'Camera denied: ' + err.message;
            if (document.getElementById('mic-status'))
                document.getElementById('mic-status').textContent = 'Mic denied: ' + err.message;
            return;
        }}

        if (wantVideo) {{
            const video = document.getElementById('cam-preview');
            video.srcObject = stream;
            await video.play();

            const capCanvas = document.createElement('canvas');
            capCanvas.width = 640; capCanvas.height = 480;
            const capCtx = capCanvas.getContext('2d');
            const faceStatus = document.getElementById('face-status');
            let fc = 0;

            function captureFrame() {{
                if (video.readyState >= 2 && video.videoWidth > 0) {{
                    capCtx.drawImage(video, 0, 0, 640, 480);
                    const dataUrl = capCanvas.toDataURL('image/jpeg', 0.8);
                    fc++;
                    faceStatus.textContent = '📸 Capturing frame ' + fc + '...';
                    fetch(API + '/emotion/frame', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{ image_b64: dataUrl, user_id: UID }})
                    }})
                    .then(r => r.json())
                    .then(d => {{
                        if (d.dominant)
                            faceStatus.textContent = '📸 ' + d.dominant
                                + ' (' + Math.round(d.confidence * 100) + '%) | ' + fc + ' frames';
                    }})
                    .catch(() => {{ faceStatus.textContent = '📸 Frame ' + fc + ' | connecting...'; }});
                }}
            }}
            setInterval(captureFrame, 3000);
            setTimeout(captureFrame, 1000);
        }}

        if (wantAudio) {{
            const micStatus = document.getElementById('mic-status');
            const vizCanvas = document.getElementById('audio-viz');

            let audioCtx;
            try {{
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }} catch(e) {{
                micStatus.textContent = 'AudioContext not supported';
                return;
            }}
            if (audioCtx.state === 'suspended') await audioCtx.resume();

            const audioStream = new MediaStream(stream.getAudioTracks());

            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;
            const source = audioCtx.createMediaStreamSource(audioStream);
            source.connect(analyser);
            const freqData = new Uint8Array(analyser.frequencyBinCount);

            vizCanvas.width  = vizCanvas.offsetWidth  || 400;
            vizCanvas.height = 180;
            const vizCtx = vizCanvas.getContext('2d');

            function drawViz() {{
                requestAnimationFrame(drawViz);
                analyser.getByteFrequencyData(freqData);
                vizCtx.fillStyle = '#000';
                vizCtx.fillRect(0, 0, vizCanvas.width, vizCanvas.height);
                const bars = 50;
                const step = Math.ceil(freqData.length / bars);
                const bw   = vizCanvas.width / bars;
                for (let i = 0; i < bars; i++) {{
                    const val = freqData[i * step] || 0;
                    const bh  = (val / 255) * (vizCanvas.height - 10);
                    const hue = 20 + (val / 255) * 15;
                    vizCtx.fillStyle = 'hsl(' + hue + ',75%,' + (35 + val/255*35) + '%)';
                    vizCtx.fillRect(i * bw + 1, vizCanvas.height - bh, bw - 2, bh);
                }}
            }}
            drawViz();

            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : MediaRecorder.isTypeSupported('audio/webm')
                    ? 'audio/webm'
                    : MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')
                        ? 'audio/ogg;codecs=opus'
                        : '';

            if (!mimeType) {{
                micStatus.textContent = '🎤 MediaRecorder not supported in this browser';
                return;
            }}

            micStatus.textContent = '🎤 Listening (' + mimeType + ')...';
            let chunkNum = 0;

            function startRecordingCycle() {{
                let chunks = [];
                let recorder;

                try {{
                    recorder = new MediaRecorder(audioStream, {{ mimeType: mimeType }});
                }} catch(e) {{
                    micStatus.textContent = '🎤 MediaRecorder error: ' + e.message;
                    return;
                }}

                recorder.ondataavailable = function(e) {{
                    if (e.data && e.data.size > 0) chunks.push(e.data);
                }};

                recorder.onstop = function() {{
                    const blob = new Blob(chunks, {{ type: mimeType }});
                    chunkNum++;
                    const cn = chunkNum;
                    micStatus.textContent = '🎤 Sending chunk ' + cn + ' (' + Math.round(blob.size/1024) + ' KB)...';

                    const reader = new FileReader();
                    reader.onloadend = function() {{
                        const b64 = reader.result;
                        fetch(API + '/emotion/audio', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{
                                audio_b64:    b64,
                                user_id:      UID,
                                audio_format: mimeType
                            }})
                        }})
                        .then(r => r.json())
                        .then(d => {{
                            let info = '🎤 Chunk ' + cn;
                            if (d.status === 'skipped') {{
                                info += ' | too short, skipped';
                            }} else {{
                                if (d.voice_dominant) info += ' | voice: ' + d.voice_dominant;
                                if (d.transcript && d.transcript.length > 0)
                                    info += ' | "' + d.transcript.substring(0, 50) + '"';
                                else
                                    info += ' | (no speech detected)';
                            }}
                            micStatus.textContent = info;
                        }})
                        .catch(e => {{
                            micStatus.textContent = '🎤 Chunk ' + cn + ' | server error: ' + e.message;
                        }});
                    }};
                    reader.readAsDataURL(blob);

                    startRecordingCycle();
                }};

                recorder.start();
                setTimeout(() => {{
                    if (recorder.state === 'recording') recorder.stop();
                }}, 30000);
            }}

            startRecordingCycle();
        }}
    }}

    startMedia();
}})();
</script>
"""
    return html


def process_media_send(user_id: str, user_message: str = ""):
    face_vector = None
    voice_vector = None
    whisper_transcript = ""

    try:
        avg_resp = requests.get(
            f"{API_BASE}/emotion/average",
            params={"user_id": user_id},
            timeout=15,
        )
        if avg_resp.status_code == 200:
            avg_data = avg_resp.json()
            if avg_data.get("face_frame_count", 0) > 0:
                fv = avg_data["face_vector"]
                face_vector = EmotionVector(
                    happy=fv.get("happy", 0), sad=fv.get("sad", 0),
                    angry=fv.get("angry", 0), fearful=fv.get("fearful", 0),
                    disgusted=fv.get("disgusted", 0), surprised=fv.get("surprised", 0),
                    neutral=fv.get("neutral", 0), contempt=fv.get("contempt", 0),
                    dominant=avg_data["face_dominant"],
                    confidence=avg_data["face_confidence"],
                )
            if avg_data.get("audio_chunk_count", 0) > 0:
                vv = avg_data["voice_vector"]
                voice_vector = EmotionVector(
                    happy=vv.get("happy", 0), sad=vv.get("sad", 0),
                    angry=vv.get("angry", 0), fearful=vv.get("fearful", 0),
                    disgusted=vv.get("disgusted", 0), surprised=vv.get("surprised", 0),
                    neutral=vv.get("neutral", 0), contempt=vv.get("contempt", 0),
                    dominant=avg_data["voice_dominant"],
                    confidence=avg_data["voice_confidence"],
                )
            whisper_transcript = avg_data.get("transcript", "")
    except Exception as e:
        logger.warning(f"Could not fetch average: {e}")

    if face_vector is None:
        face_vector = EmotionVector(dominant="neutral", confidence=0.0)
    if voice_vector is None:
        voice_vector = EmotionVector(dominant="neutral", confidence=0.0)

    combined_text = user_message
    if whisper_transcript:
        combined_text = f"{user_message} {whisper_transcript}".strip() if user_message else whisper_transcript
    if not combined_text:
        combined_text = "Hello"

    text_emotion = run_async(detect_text_emotion(combined_text))
    fused = weighted_fusion(face_vector, voice_vector, text_emotion)

    return fused, whisper_transcript, combined_text


if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "emotion" not in st.session_state:
    st.session_state.emotion = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "mic_on" not in st.session_state:
    st.session_state.mic_on = False
if "sidebar_emoji" not in st.session_state:
    st.session_state.sidebar_emoji = "😸"
if "sidebar_dominant" not in st.session_state:
    st.session_state.sidebar_dominant = "waiting"
if "sidebar_confidence" not in st.session_state:
    st.session_state.sidebar_confidence = 0.0
if "sidebar_vector" not in st.session_state:
    st.session_state.sidebar_vector = None
if "tracker_period" not in st.session_state:
    st.session_state.tracker_period = "week"
if "dynamo_errors" not in st.session_state:
    st.session_state.dynamo_errors = []
if "media_send" not in st.session_state:
    st.session_state.media_send = False

user_id = st.session_state.user_id

with st.sidebar:
    st.markdown(f'<div class="user-badge">🐈 {user_id[:4]}</div>', unsafe_allow_html=True)
    st.markdown("### 🎯 Detected Emotion")

    emoji = st.session_state.sidebar_emoji
    dom  = st.session_state.sidebar_dominant
    conf = st.session_state.sidebar_confidence
    color = EMOTION_COLORS.get(dom, "#94a3b8")

    st.markdown(f'<div class="big-emoji">{emoji}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="dominant-label" style="color:{color}">{dom}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence-label">{conf*100:.0f}% confidence</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📊 Emotion Bars")

    vec_to_render = st.session_state.sidebar_vector if st.session_state.sidebar_vector else EmotionVector()
    bars_html = render_emotion_bars(vec_to_render)
    st.markdown(f'<div class="result-box">{bars_html}</div>', unsafe_allow_html=True)

    st.divider()
    tip = random.choice(PURRBOT_TIPS)
    st.markdown(f"""
    <div class="purrbot-card">
        <div class="purrbot-card-title">🐾 Purrbot Tip</div>
        <div class="purrbot-card-body">{tip}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚙️ Settings")

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key or groq_key == "your_groq_api_key_here":
        st.warning("GROQ_API_KEY not set")
        key_input = st.text_input("Enter Groq API Key:", type="password")
        if key_input:
            os.environ["GROQ_API_KEY"] = key_input
            st.success("Key set! 😸")
            st.rerun()
    else:
        st.success("Groq API Key set 😸")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.emotion = None
        st.session_state.conversation_id = None
        st.session_state.sidebar_emoji = "😸"
        st.session_state.sidebar_dominant = "waiting"
        st.session_state.sidebar_confidence = 0.0
        st.session_state.sidebar_vector = None
        st.session_state.dynamo_errors = []
        try:
            requests.delete(f"{API_BASE}/emotion/buffer", params={"user_id": user_id}, timeout=3)
        except Exception:
            pass
        st.rerun()

    if st.session_state.dynamo_errors:
        st.warning(f"DynamoDB: {st.session_state.dynamo_errors[-1]}")

    st.markdown(
        '<div style="text-align:center;color:#f0c9a8;font-size:0.65rem;margin-top:0.5rem;">'
        '🐾 EmotiChat v2.0 🐈 🐾</div>',
        unsafe_allow_html=True
    )


st.markdown("""
<div class="main-header">
    <h1>😸 EmotiChat</h1>
    <p>🐾 Your emotion-aware AI companion — senses your face, voice, and words 🐈</p>
</div>
""", unsafe_allow_html=True)


tab_chat, tab_tracker = st.tabs(["💬 Chat", "📊 Emotion Tracker"])


with tab_chat:
    col_cam, col_mic = st.columns(2)

    with col_cam:
        camera_toggle = st.toggle("📸 Camera", value=st.session_state.camera_on, key="cam_toggle")
        if camera_toggle != st.session_state.camera_on:
            st.session_state.camera_on = camera_toggle
            if not camera_toggle:
                try:
                    requests.delete(f"{API_BASE}/emotion/buffer", params={"user_id": user_id}, timeout=3)
                except Exception:
                    pass
            st.rerun()

    with col_mic:
        mic_toggle = st.toggle("🎤 Microphone", value=st.session_state.mic_on, key="mic_toggle")
        if mic_toggle != st.session_state.mic_on:
            st.session_state.mic_on = mic_toggle
            if not mic_toggle:
                try:
                    requests.delete(f"{API_BASE}/emotion/buffer", params={"user_id": user_id}, timeout=3)
                except Exception:
                    pass
            st.rerun()

    if st.session_state.camera_on or st.session_state.mic_on:
        media_html = build_media_html(user_id, st.session_state.camera_on, st.session_state.mic_on)
        components.html(media_html, height=230)

        if st.button("📤 Send Video & Audio Analysis", key="media_send_btn", type="primary", use_container_width=True):
            st.session_state.media_send = True
            st.rerun()

    if st.session_state.media_send:
        st.session_state.media_send = False

        with st.spinner("Purrbot is reading your emotions... 😸"):
            fused, whisper_transcript, combined_text = process_media_send(user_id)

        dom  = fused.fused.dominant
        conf = fused.fused.confidence
        emoji_icon = EMOTION_EMOJI.get(dom, "😸")

        st.session_state.sidebar_emoji      = emoji_icon
        st.session_state.sidebar_dominant   = dom
        st.session_state.sidebar_confidence = conf
        st.session_state.sidebar_vector     = fused.fused
        st.session_state.emotion            = fused

        if whisper_transcript and whisper_transcript.strip():
            display_text = whisper_transcript
        elif st.session_state.camera_on and st.session_state.mic_on:
            display_text = "[📸 Shared video + 🎤 audio]"
        elif st.session_state.camera_on:
            display_text = "[📸 Shared video]"
        else:
            display_text = "[🎤 Shared audio]"

        st.session_state.messages.append({
            "role": "user",
            "content": display_text,
            "emotion_pill": {"emotion": dom, "confidence": conf * 100},
        })

        try:
            history = [
                ChatMessage(role=m["role"], content=m["content"])
                for m in st.session_state.messages[-10:]
            ]
            result = run_async(generate_response(
                message=combined_text,
                fused_emotion=fused,
                conversation_id=st.session_state.conversation_id,
                history=history,
            ))
            reply = result["reply"]
            st.session_state.conversation_id = result["conversation_id"]
        except Exception as e:
            reply = f"Purrbot had a hairball moment: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": reply})

        emotion_dict = {
            "happy": fused.fused.happy, "sad": fused.fused.sad,
            "angry": fused.fused.angry, "fearful": fused.fused.fearful,
            "disgusted": fused.fused.disgusted, "surprised": fused.fused.surprised,
            "neutral": fused.fused.neutral, "contempt": fused.fused.contempt,
        }
        save_ok = save_interaction(
            user_id=user_id, dominant_emotion=dom,
            emotion_vector=emotion_dict, confidence=conf,
            message=display_text, reply=reply,
        )
        if not save_ok:
            st.session_state.dynamo_errors.append("Could not save to DynamoDB. Check AWS credentials.")

        try:
            requests.delete(f"{API_BASE}/emotion/buffer", params={"user_id": user_id}, timeout=3)
        except Exception:
            pass

        st.rerun()

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            if "emotion_pill" in msg:
                pill = msg["emotion_pill"]
                pill_color = EMOTION_COLORS.get(pill["emotion"], "#94a3b8")
                pill_emoji = EMOTION_EMOJI.get(pill["emotion"], "")
                st.markdown(
                    f'<div class="emotion-pill" style="background:{pill_color}20;'
                    f'color:{pill_color};border:1px solid {pill_color}40;">'
                    f'{pill_emoji} {pill["emotion"]} {pill["confidence"]:.0f}%</div>',
                    unsafe_allow_html=True,
                )
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="😸"):
                st.write(msg["content"])

    user_input = st.chat_input("Type a message to Purrbot... 🐾")

    if user_input:
        if st.session_state.camera_on or st.session_state.mic_on:
            fused, whisper_transcript, combined_text = process_media_send(user_id, user_input)
        else:
            text_emotion = run_async(detect_text_emotion(user_input))
            face_vector  = EmotionVector(dominant="neutral", confidence=0.0)
            voice_vector = EmotionVector(dominant="neutral", confidence=0.0)
            fused        = weighted_fusion(face_vector, voice_vector, text_emotion)
            combined_text = user_input

        dom  = fused.fused.dominant
        conf = fused.fused.confidence
        emoji_icon = EMOTION_EMOJI.get(dom, "😸")

        st.session_state.sidebar_emoji      = emoji_icon
        st.session_state.sidebar_dominant   = dom
        st.session_state.sidebar_confidence = conf
        st.session_state.sidebar_vector     = fused.fused
        st.session_state.emotion            = fused

        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "emotion_pill": {"emotion": dom, "confidence": conf * 100},
        })

        try:
            history = [
                ChatMessage(role=m["role"], content=m["content"])
                for m in st.session_state.messages[-10:]
            ]
            result = run_async(generate_response(
                message=combined_text,
                fused_emotion=fused,
                conversation_id=st.session_state.conversation_id,
                history=history,
            ))
            reply = result["reply"]
            st.session_state.conversation_id = result["conversation_id"]
        except Exception as e:
            reply = f"Purrbot had a hairball moment: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": reply})

        emotion_dict = {
            "happy": fused.fused.happy, "sad": fused.fused.sad,
            "angry": fused.fused.angry, "fearful": fused.fused.fearful,
            "disgusted": fused.fused.disgusted, "surprised": fused.fused.surprised,
            "neutral": fused.fused.neutral, "contempt": fused.fused.contempt,
        }
        save_ok = save_interaction(
            user_id=user_id, dominant_emotion=dom,
            emotion_vector=emotion_dict, confidence=conf,
            message=user_input, reply=reply,
        )
        if not save_ok:
            st.session_state.dynamo_errors.append("Could not save to DynamoDB. Check AWS credentials.")

        if st.session_state.camera_on or st.session_state.mic_on:
            try:
                requests.delete(f"{API_BASE}/emotion/buffer", params={"user_id": user_id}, timeout=3)
            except Exception:
                pass

        st.rerun()


with tab_tracker:
    st.markdown("### 📊 Emotion Tracker 🐈")

    period_cols = st.columns(4)
    periods       = ["day", "week", "month", "year"]
    period_labels = ["Day", "Week", "Month", "Year"]
    for i, (p, lbl) in enumerate(zip(periods, period_labels)):
        with period_cols[i]:
            if st.button(
                lbl, key=f"period_{p}", use_container_width=True,
                type="primary" if st.session_state.tracker_period == p else "secondary",
            ):
                st.session_state.tracker_period = p
                st.rerun()

    now    = datetime.now(timezone.utc)
    period = st.session_state.tracker_period

    if period == "day":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=1)
    elif period == "week":
        start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=7)
    elif period == "month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end   = (now.replace(year=now.year+1, month=1, day=1) if now.month == 12
                 else now.replace(month=now.month+1, day=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end   = now.replace(year=now.year+1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

    try:
        history = fetch_emotion_history(user_id, start.isoformat(), end.isoformat())
    except Exception as e:
        history = []
        st.warning(f"Could not load history: {e}")

    if not history:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;">
            <div style="font-size:4rem;">🐾</div>
            <div style="color:#e8a87c;font-size:1rem;margin-top:0.5rem;">No emotion data yet for this period</div>
            <div style="color:#f0c9a8;font-size:0.8rem;margin-top:0.3rem;">
                Start chatting and your emotions will appear here! 😸
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        import plotly.graph_objects as go
        import calendar

        if period == "day":
            morning = []
            afternoon = []
            evening = []
            for item in history:
                ts   = datetime.fromisoformat(item["timestamp"])
                hour = ts.hour
                entry = {"time": ts, "emotion": item.get("dominant_emotion", "neutral")}
                if hour < 12:
                    morning.append(entry)
                elif hour < 18:
                    afternoon.append(entry)
                else:
                    evening.append(entry)

            fig = go.Figure()
            for label, entries in [("Morning", morning), ("Afternoon", afternoon), ("Evening", evening)]:
                for entry in entries:
                    em = entry["emotion"]
                    ec = EMOTION_EMOJI.get(em, "😐")
                    t  = entry["time"]
                    fig.add_trace(go.Scatter(
                        x=[t], y=[label],
                        mode="markers+text",
                        text=[ec],
                        textfont=dict(size=20),
                        textposition="middle center",
                        marker=dict(size=30, color=EMOTION_COLORS.get(em, "#94a3b8"), opacity=0.3),
                        hovertemplate=f"<b>{em}</b><br>{t.strftime('%H:%M')}<extra></extra>",
                        showlegend=False,
                    ))
            fig.update_layout(
                plot_bgcolor="#1a1a18", paper_bgcolor="#1a1a18",
                font=dict(color="#e8a87c"), height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(gridcolor="rgba(196,105,42,0.1)", range=[start, end]),
                yaxis=dict(
                    gridcolor="rgba(196,105,42,0.1)",
                    categoryorder="array",
                    categoryarray=["Evening", "Afternoon", "Morning"]
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

        elif period == "week":
            day_data = {}
            for item in history:
                ts      = datetime.fromisoformat(item["timestamp"])
                day_key = ts.strftime("%Y-%m-%d")
                day_data.setdefault(day_key, []).append(item.get("dominant_emotion", "neutral"))

            week_cols = st.columns(7)
            for i in range(7):
                day     = start + timedelta(days=i)
                day_key = day.strftime("%Y-%m-%d")
                is_today = day.date() == now.date()
                emotions = day_data.get(day_key, [])
                if emotions:
                    most_common = Counter(emotions).most_common(1)[0][0]
                    ec = EMOTION_EMOJI.get(most_common, "😐")
                else:
                    most_common = ""
                    ec = "🐾"
                card_class = "week-card-today" if is_today else "week-card"
                with week_cols[i]:
                    st.markdown(f"""
                    <div class="{card_class}">
                        <div class="week-card-day">{day.strftime("%a")}</div>
                        <div class="week-card-emoji">{ec}</div>
                        <div class="week-card-label">{most_common if most_common else "no data"}</div>
                    </div>
                    """, unsafe_allow_html=True)

        elif period == "month":
            day_data = {}
            for item in history:
                ts = datetime.fromisoformat(item["timestamp"])
                day_data.setdefault(ts.day, []).append(item.get("dominant_emotion", "neutral"))

            cal = calendar.monthcalendar(now.year, now.month)
            st.markdown("""
            <div style="display:grid;grid-template-columns:repeat(7,1fr);gap:2px;margin-bottom:4px;">
                <div style="text-align:center;font-size:0.65rem;color:#c4692a;font-weight:600;">Mon</div>
                <div style="text-align:center;font-size:0.65rem;color:#c4692a;font-weight:600;">Tue</div>
                <div style="text-align:center;font-size:0.65rem;color:#c4692a;font-weight:600;">Wed</div>
                <div style="text-align:center;font-size:0.65rem;color:#c4692a;font-weight:600;">Thu</div>
                <div style="text-align:center;font-size:0.65rem;color:#c4692a;font-weight:600;">Fri</div>
                <div style="text-align:center;font-size:0.65rem;color:#c4692a;font-weight:600;">Sat</div>
                <div style="text-align:center;font-size:0.65rem;color:#c4692a;font-weight:600;">Sun</div>
            </div>
            """, unsafe_allow_html=True)
            for week in cal:
                grid_html = '<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:2px;">'
                for day in week:
                    if day == 0:
                        grid_html += '<div class="month-cell"></div>'
                    else:
                        emotions = day_data.get(day, [])
                        ec = EMOTION_EMOJI.get(Counter(emotions).most_common(1)[0][0], "😐") if emotions else ""
                        grid_html += (
                            f'<div class="month-cell">'
                            f'<div>{day}</div>'
                            f'<div class="month-cell-emoji">{ec}</div>'
                            f'</div>'
                        )
                grid_html += '</div>'
                st.markdown(grid_html, unsafe_allow_html=True)

        elif period == "year":
            month_data = {}
            for item in history:
                ts = datetime.fromisoformat(item["timestamp"])
                month_data.setdefault(ts.month, []).append(item.get("dominant_emotion", "neutral"))

            mnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            months_list = []
            colors_list = []
            counts_list = []
            hover_list  = []
            for m in range(1, 13):
                months_list.append(mnames[m - 1])
                emotions = month_data.get(m, [])
                counts_list.append(len(emotions))
                if emotions:
                    mc = Counter(emotions).most_common(1)[0][0]
                    colors_list.append(EMOTION_COLORS.get(mc, "#94a3b8"))
                    hover_list.append(f"{mc} ({len(emotions)} sessions)")
                else:
                    colors_list.append("rgba(196,105,42,0.2)")
                    hover_list.append("No data")

            fig = go.Figure(data=[go.Bar(
                x=months_list, y=counts_list,
                marker_color=colors_list,
                hovertext=hover_list, hoverinfo="text"
            )])
            fig.update_layout(
                plot_bgcolor="#1a1a18", paper_bgcolor="#1a1a18",
                font=dict(color="#e8a87c"), height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(gridcolor="rgba(196,105,42,0.1)"),
                yaxis=dict(gridcolor="rgba(196,105,42,0.1)", title="Sessions"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        all_emotions   = [item.get("dominant_emotion", "neutral") for item in history]
        total_sessions = len(history)
        if all_emotions:
            top_emotion = Counter(all_emotions).most_common(1)[0][0]
            top_emoji   = EMOTION_EMOJI.get(top_emotion, "😐")
        else:
            top_emotion = "none"
            top_emoji   = "🐾"

        positive_emotions = {"happy", "surprised"}
        positive_count = sum(1 for e in all_emotions if e in positive_emotions)
        positive_pct   = (positive_count / total_sessions * 100) if total_sessions > 0 else 0

        if period == "day":
            prev_start = start - timedelta(days=1)
            prev_end   = start
        elif period == "week":
            prev_start = start - timedelta(days=7)
            prev_end   = start
        elif period == "month":
            prev_end = start
            prev_start = (start.replace(year=start.year-1, month=12) if start.month == 1
                          else start.replace(month=start.month-1))
        else:
            prev_start = start.replace(year=start.year-1)
            prev_end   = start

        try:
            prev_history = fetch_emotion_history(user_id, prev_start.isoformat(), prev_end.isoformat())
        except Exception:
            prev_history = []

        prev_emotions = [item.get("dominant_emotion", "neutral") for item in prev_history]
        prev_positive = sum(1 for e in prev_emotions if e in positive_emotions)
        prev_pct      = (prev_positive / len(prev_history) * 100) if prev_history else 0
        pct_diff      = positive_pct - prev_pct
        trend         = "↑" if pct_diff > 0 else "↓" if pct_diff < 0 else "→"

        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{top_emoji} {top_emotion}</div>
                <div class="stat-label">Top emotion this {period}</div>
            </div>
            """, unsafe_allow_html=True)
        with stat_cols[1]:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{total_sessions}</div>
                <div class="stat-label">Total sessions</div>
            </div>
            """, unsafe_allow_html=True)
        with stat_cols[2]:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{positive_pct:.0f}% {trend}</div>
                <div class="stat-label">Positive vs prev {period}</div>
            </div>
            """, unsafe_allow_html=True)