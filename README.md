

# EmotiChat: Multimodal Emotion-Aware AI Assistant 😸🐾

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
  ![AWS DynamoDB](https://img.shields.io/badge/AWS%20DynamoDB-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
  ![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
  ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

</div>

EmotiChat is a real-time, multimodal AI chatbot that detects user emotions across video, audio, and text streams to provide contextually empathetic responses. Designed with a warm, cat-themed persona ("Purrbot"), the application continuously tracks emotional states over time and visualizes them in an interactive dashboard.

##  Technical Highlights

* **Continuous Multimodal Sampling:** Bypasses standard synchronous UI limitations by utilizing hidden JavaScript to asynchronously sample webcam frames (every 3 seconds) and microphone audio (30-second chunks) in the background.
* **Weighted Emotion Fusion:** Aggregates and averages emotional data across a session, applying a weighted fusion algorithm (Face: 35%, Voice: 30%, Text: 35%) to generate a highly accurate, confident emotional baseline.
* **Local Machine Learning:** Runs heavy inference models locally, including OpenAI Whisper for Speech-to-Text and DeepFace for facial recognition, ensuring user privacy and reducing external API latency.
* **Cloud Architecture:** Leverages AWS DynamoDB to anonymously store and query historical interaction data for long-term emotional tracking.

##  System Architecture

The application is split into a Streamlit frontend and a FastAPI backend, communicating via asynchronous API endpoints to maintain a fluid user experience without page reloads.

1. **Input Streams:**
   * **Visual:** Browser captures frames -> POST to FastAPI -> DeepFace (Local) -> Emotion Vector.
   * **Acoustic & Semantic:** Browser records audio -> POST to FastAPI -> Whisper STT (Local) + Librosa (Acoustic Features) -> Emotion Vector + Transcript.
   * **Text:** User chat input -> DistilRoBERTa (Local) -> Emotion Vector.
2. **Processing Pipeline:** On message send, the backend retrieves the averaged face and voice vectors from the session buffer, fuses them with the text vector, and dynamically adjusts the system prompt for the Groq Llama 3.3 70B LLM.
3. **Storage & Analytics:** The fused emotion, raw message, and AI response are written to AWS DynamoDB using a generated anonymous UUID.

##  Tech Stack

**Machine Learning & AI**
* **Computer Vision:** DeepFace, OpenCV
* **Audio Processing:** Librosa
* **NLP & Speech:** HuggingFace (DistilRoBERTa), OpenAI Whisper
* **LLM Provider:** Groq API (Llama 3.3 70B)

**Backend & Cloud**
* **Framework:** FastAPI, Python 3.11
* **Database:** AWS DynamoDB (boto3)

**Frontend**
* **Framework:** Streamlit
* **Data Visualization:** Plotly (Gantt charts, interactive calendars)
* **Custom Logic:** Vanilla JavaScript for MediaRecorder and continuous DOM sampling

##  Features

* **Interactive Chat UI:** Real-time emotion pills displaying dominant emotions and confidence percentages above user messages.
* **Emotion Tracker Dashboard:** Highly detailed historical views (Day, Week, Month, Year) utilizing Plotly to visualize emotional shifts and session statistics.
* **Frictionless Onboarding:** Zero-click anonymous login utilizing session-based UUIDs for secure, isolated data tracking.

##  Local Setup & Installation

**1. Clone and Setup Environment**
```bash
git clone [https://github.com/yourusername/emotichat.git](https://github.com/yourusername/emotichat.git)
cd emotichat
bash setup.sh
```

**2. Configure Environment Variables**
Copy the `.env.example` file to `.env` and add your required keys:
* `GROQ_API_KEY`
* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`

**3. Run the FastAPI Backend**
The backend must be running to process the continuous media streams.
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**4. Run the Streamlit Frontend**
In a separate terminal instance:
```bash
bash start.sh
```
Navigate to `http://localhost:8502` to begin.
