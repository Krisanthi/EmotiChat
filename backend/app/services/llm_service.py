import logging
import uuid
from typing import Dict, List, Optional

from groq import Groq

from app.config import get_settings
from app.models.schemas import ChatMessage, FusedEmotion, EmotionVector
from app.services.emotion_fusion import get_emotion_summary

logger = logging.getLogger(__name__)

_conversations: Dict[str, List[Dict]] = {}


BASE_SYSTEM_PROMPT = """You are Purrbot, the EmotiChat companion. You are a warm, witty, cat-themed AI that senses the user's emotions in real-time through three channels:

1. FACIAL EXPRESSION ANALYSIS — You watch the user's webcam feed and read their facial muscles, eye position, mouth shape, and eyebrow position using DeepFace AI running locally on their machine.

2. VOICE TONE ANALYSIS — You listen to the user's microphone and analyze pitch, energy, tempo, spectral features, and MFCCs using librosa audio processing and Whisper speech-to-text running locally.

3. TEXT SENTIMENT ANALYSIS — You analyze the text the user types using a HuggingFace DistilRoBERTa model trained on emotion classification.

You combine all three readings using confidence-weighted fusion to determine the user's overall emotional state. Then you adjust your response tone accordingly.

You sprinkle in cat-related warmth naturally. You might say things like "I can see your smile from here" or "your voice sounds a bit heavy today, want to talk about it?" but you never force cat puns. You are genuine first, playful second.

When users ask what you can do, mention your emotion-sensing abilities proudly.

Guidelines:
- Be genuine and warm, never patronizing
- Acknowledge the user's emotions naturally and reference what you detected
- Adapt your tone to match and support the user's emotional state
- If the user seems distressed, prioritize emotional support before information
- Use appropriate humor when the user seems happy
- Be direct and patient when the user seems frustrated or angry
- If asked about your capabilities, explain your multimodal emotion detection
- Occasionally use cat emojis like 😸 🐾 but do not overdo it"""

EMOTION_PROMPTS = {
    "happy": """The user is currently feeling HAPPY and positive. Match their energy! \
Be enthusiastic, use warm language, feel free to include light humor. \
Celebrate their good mood and be upbeat in your responses. \
Let them know you can see they're in a great mood.""",

    "sad": """The user is currently feeling SAD. Be gentle, compassionate, and validating. \
Use soft, supportive language. Acknowledge their feelings without trying to immediately \
fix things. Show empathy first, then gently offer support or perspective if appropriate. \
Avoid being overly cheerful.""",

    "angry": """The user is currently feeling ANGRY or frustrated. Stay calm and composed. \
Don't be dismissive of their frustration. Acknowledge their feelings directly. \
Be concise and solution-oriented. Avoid lengthy preambles or excessive pleasantries. \
Show that you take their concerns seriously.""",

    "fearful": """The user is currently feeling ANXIOUS or FEARFUL. Be reassuring and steady. \
Provide clear, structured information. Avoid ambiguity. Offer concrete steps when possible. \
Use calming language. Help them feel grounded and in control.""",

    "disgusted": """The user is showing signs of DISGUST or strong displeasure. \
Be respectful and measured. Don't dismiss their reaction. Help redirect the conversation \
constructively. Be professional and understanding.""",

    "surprised": """The user appears SURPRISED. Help them process the unexpected. \
Provide context and clarity. Be informative and grounding. Match their curiosity \
if the surprise seems positive, or provide reassurance if it seems negative.""",

    "neutral": """The user's emotional state is NEUTRAL. Respond naturally and helpfully. \
Be informative, clear, and balanced. This is your default conversational mode.""",

    "contempt": """The user may be feeling CONTEMPTUOUS or skeptical. Stay professional \
and factual. Provide evidence-based responses. Don't be defensive. Show competence \
and reliability through your answers.""",
}


def build_system_prompt(fused_emotion: FusedEmotion) -> str:
    dom = fused_emotion.fused.dominant
    emotion_context = EMOTION_PROMPTS.get(dom, EMOTION_PROMPTS["neutral"])
    emotion_summary = get_emotion_summary(fused_emotion)

    prompt = f"""{BASE_SYSTEM_PROMPT}

--- CURRENT EMOTIONAL CONTEXT ---
{emotion_summary}

{emotion_context}

--- EMOTIONAL VECTOR ---
Dominant: {dom} (confidence: {fused_emotion.fused.confidence:.0%})
Face: {fused_emotion.face.dominant + ' (' + f'{fused_emotion.face.confidence:.0%}' + ' conf)' if fused_emotion.face else 'N/A'}
Voice: {fused_emotion.voice.dominant + ' (' + f'{fused_emotion.voice.confidence:.0%}' + ' conf)' if fused_emotion.voice else 'N/A'}
Text: {fused_emotion.text.dominant + ' (' + f'{fused_emotion.text.confidence:.0%}' + ' conf)' if fused_emotion.text else 'N/A'}

Use this emotional context to inform your response style."""

    return prompt


def get_conversation(conversation_id: str) -> List[Dict]:
    return _conversations.get(conversation_id, [])


def save_to_conversation(conversation_id: str, role: str, content: str):
    if conversation_id not in _conversations:
        _conversations[conversation_id] = []
    _conversations[conversation_id].append({"role": role, "content": content})
    if len(_conversations[conversation_id]) > 20:
        _conversations[conversation_id] = _conversations[conversation_id][-20:]


async def generate_response(
    message: str,
    fused_emotion: FusedEmotion,
    conversation_id: Optional[str] = None,
    history: Optional[List[ChatMessage]] = None,
) -> Dict:
    settings = get_settings()

    if not settings.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Please set it in your environment or .env file."
        )

    conv_id = conversation_id or str(uuid.uuid4())
    system_prompt = build_system_prompt(fused_emotion)
    messages = [{"role": "system", "content": system_prompt}]

    if history:
        for msg in history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})
    else:
        conv_history = get_conversation(conv_id)
        messages.extend(conv_history[-10:])

    messages.append({"role": "user", "content": message})

    try:
        client = Groq(api_key=settings.GROQ_API_KEY)

        completion = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            max_tokens=settings.GROQ_MAX_TOKENS,
            temperature=settings.GROQ_TEMPERATURE,
            top_p=0.9,
            stream=False,
        )

        reply = completion.choices[0].message.content
        tokens_used = completion.usage.total_tokens if completion.usage else None

        save_to_conversation(conv_id, "user", message)
        save_to_conversation(conv_id, "assistant", reply)

        logger.info(
            f"LLM response generated | conv={conv_id[:8]}... | "
            f"emotion={fused_emotion.fused.dominant} | tokens={tokens_used}"
        )

        return {
            "reply": reply,
            "conversation_id": conv_id,
            "system_prompt": system_prompt,
            "model": settings.GROQ_MODEL,
            "tokens_used": tokens_used,
        }

    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise
