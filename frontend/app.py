"""
Dear AI -- Mental Health Companion Frontend (Streamlit)

A calming, therapeutic chat interface for the Dear AI mental health companion.
Supports text chat (two-layer responses), voice input/output, conversation
management, and user profile editing.

Run:
    cd frontend
    streamlit run app.py

Requires:
    pip install streamlit httpx websockets audio-recorder-streamlit
"""

import base64
import html as html_mod
import json
import logging
import os
import time
from datetime import datetime

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("DEARAI_API_BASE", "http://localhost:8000")
WS_BASE = os.environ.get("DEARAI_WS_BASE", "ws://localhost:8000")

# Timeout constants (seconds)
API_TIMEOUT_SHORT = 10
API_TIMEOUT_DEFAULT = 15
API_TIMEOUT_LONG = 30
API_TIMEOUT_VOICE = 60
WS_OPEN_TIMEOUT = 5
WS_CLOSE_TIMEOUT = 3
WS_AUTH_TIMEOUT = 5
WS_RECV_TIMEOUT = 10
WS_RECV_TIMEOUT_VOICE = 15
WS_TEXT_DEADLINE = 30
WS_VOICE_DEADLINE = 60

# Validation constants
MIN_PASSWORD_LENGTH = 6
MAX_CONVERSATIONS_LIMIT = 100

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom CSS -- Calming / Therapeutic theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ---- Global ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');

:root {
    --lavender: #C4B7E0;
    --lavender-light: #EDE7F9;
    --sage: #A8C5A0;
    --sage-light: #E3F0E0;
    --warm-bg: #FAF8F5;
    --warm-card: #FFFFFF;
    --text-primary: #3D3D3D;
    --text-secondary: #6B6B6B;
    --text-muted: #9B9B9B;
    --accent: #8B7EC8;
    --accent-hover: #7A6BB7;
    --danger: #D4726A;
    --border: #E8E4E0;
    --shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
    --radius: 16px;
    --radius-sm: 10px;
}

/* Background */
.stApp {
    background-color: var(--warm-bg) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Hide default Streamlit chrome */
#MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {visibility: hidden; height: 0 !important; min-height: 0 !important;}
footer {visibility: hidden; height: 0 !important;}
.stDeployButton {display: none !important;}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background-color: #F5F0EB !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent) !important;
    font-family: 'Lora', serif !important;
}

/* ---- Buttons (primary) ---- */
.stButton > button:not([kind="secondary"]),
button[data-testid="stBaseButton-primary"] {
    background-color: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(139, 126, 200, 0.3) !important;
}

.stButton > button:not([kind="secondary"]):hover,
button[data-testid="stBaseButton-primary"]:hover {
    background-color: var(--accent-hover) !important;
    box-shadow: 0 4px 16px rgba(139, 126, 200, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Secondary / outline buttons (Streamlit type="secondary") */
button[data-testid="stBaseButton-secondary"],
.stButton > button[kind="secondary"] {
    background-color: transparent !important;
    color: var(--accent) !important;
    border: 1.5px solid var(--accent) !important;
    box-shadow: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}

button[data-testid="stBaseButton-secondary"]:hover,
.stButton > button[kind="secondary"]:hover {
    background-color: var(--lavender-light) !important;
    box-shadow: none !important;
    transform: translateY(-1px) !important;
}

/* Danger buttons are styled inline via _danger_button_marker() */

/* ---- Input fields ---- */
.stTextInput input,
.stTextArea textarea,
.stSelectbox [data-baseweb="select"],
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    border-radius: var(--radius-sm) !important;
    border: 1.5px solid var(--border) !important;
    font-family: 'Inter', sans-serif !important;
    background-color: white !important;
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    caret-color: var(--text-primary) !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
    opacity: 1 !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus,
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(139, 126, 200, 0.15) !important;
}

/* Number input */
.stNumberInput input {
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    background-color: white !important;
}

/* Labels */
.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stNumberInput label {
    color: var(--text-secondary) !important;
}

/* ---- Chat messages ---- */
.chat-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 1rem 0;
}

/* Center main chat content area */
section[data-testid="stMain"] .block-container {
    max-width: 850px !important;
}

.message-bubble {
    padding: 1rem 1.25rem;
    border-radius: var(--radius);
    margin-bottom: 0.75rem;
    line-height: 1.6;
    font-size: 0.95rem;
    animation: fadeIn 0.3s ease-in;
    max-width: 85%;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.user-message {
    background: linear-gradient(135deg, var(--lavender), var(--accent));
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 4px;
    box-shadow: 0 2px 10px rgba(139, 126, 200, 0.25);
}

.assistant-message {
    background-color: var(--warm-card);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
    box-shadow: var(--shadow);
}

.layer1-tag {
    display: inline-block;
    background-color: var(--sage-light);
    color: #5A7D52;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    margin-bottom: 0.4rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.layer2-tag {
    display: inline-block;
    background-color: var(--lavender-light);
    color: #6B5EA8;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    margin-bottom: 0.4rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

/* ---- Cards ---- */
.info-card {
    background-color: var(--warm-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}

/* Style st.container(border=True) as info-card equivalent */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--warm-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow) !important;
}

/* ---- Conversation list items ---- */
.conv-item {
    background-color: var(--warm-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: all 0.2s ease;
}

.conv-item:hover {
    border-color: var(--accent);
    box-shadow: 0 2px 8px rgba(139, 126, 200, 0.15);
}

.conv-item-active {
    border-color: var(--accent) !important;
    background-color: var(--lavender-light) !important;
}

/* ---- Divider ---- */
.soft-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* ---- Title ---- */
.app-title {
    font-family: 'Lora', serif;
    color: var(--accent);
    font-weight: 500;
    font-size: 1.8rem;
    margin-bottom: 0.25rem;
}

.app-subtitle {
    color: var(--text-muted);
    font-size: 0.9rem;
    font-weight: 300;
}

/* ---- Typing indicator ---- */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 0.75rem 1rem;
}

.typing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--text-muted);
    animation: typingBounce 1.4s ease-in-out infinite;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typingBounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
}

/* ---- Profile card ---- */
.profile-header {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, var(--lavender-light), var(--sage-light));
    border-radius: var(--radius);
    margin-bottom: 1.5rem;
}

.profile-avatar {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--sage));
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 2rem;
    font-family: 'Lora', serif;
    margin-bottom: 0.75rem;
    box-shadow: 0 4px 16px rgba(139, 126, 200, 0.3);
}

/* ---- Welcome screen ---- */
.welcome-container {
    text-align: center;
    padding: 3rem 1rem;
    max-width: 600px;
    margin: 0 auto;
}

.welcome-container h2 {
    font-family: 'Lora', serif;
    color: var(--accent);
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ---- Voice recording area ---- */
.voice-area {
    background: linear-gradient(135deg, var(--lavender-light), var(--sage-light));
    border-radius: var(--radius);
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}

/* ---- Global text visibility ---- */
.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: var(--text-primary) !important;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: var(--accent) !important;
    font-family: 'Lora', serif !important;
}

/* Ensure alerts / warnings / errors are readable */
.stAlert p {
    color: inherit !important;
}

/* Spinner text */
.stSpinner > div {
    color: var(--text-secondary) !important;
}

/* Selectbox dropdown text */
.stSelectbox [data-baseweb="select"] span {
    color: var(--text-primary) !important;
}
</style>
"""

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Auth
        "access_token": None,
        "refresh_token": None,
        "user_id": None,
        "user_name": None,
        "user_email": None,
        # Navigation
        "page": "login",  # login | register | chat | profile
        # Chat
        "conversations": [],
        "current_conversation_id": None,
        "current_conversation_title": None,
        "messages": [],
        # Two-layer tracking
        "pending_layer1": None,
        "pending_layer2": None,
        "is_thinking": False,
        # Voice
        "last_transcription": None,
        "last_audio_response": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _danger_button_marker():
    """Inject a CSS marker to style the next button as a danger button.

    Uses a unique marker element. In Streamlit's DOM, consecutive widgets
    are siblings inside a vertical block container. We use CSS
    ``[data-testid="stElementContainer"]:has(.danger-btn-marker)
    + [data-testid="stElementContainer"] button`` to reach the next button.

    Falls back to a simpler ``div:has(> .danger-btn-marker) + div button``
    selector for broader compatibility.
    """
    st.markdown(
        '<div class="danger-btn-marker" style="display:none;margin:0;padding:0;height:0;overflow:hidden;"></div>'
        "<style>"
        '[data-testid="stElementContainer"]:has(.danger-btn-marker) + [data-testid="stElementContainer"] button,'
        "div:has(> .stMarkdown .danger-btn-marker) + div button {"
        "  background-color: var(--danger) !important;"
        "  color: white !important;"
        "  border: none !important;"
        "  box-shadow: 0 2px 8px rgba(212, 114, 106, 0.3) !important;"
        "}"
        '[data-testid="stElementContainer"]:has(.danger-btn-marker) + [data-testid="stElementContainer"] button:hover,'
        "div:has(> .stMarkdown .danger-btn-marker) + div button:hover {"
        "  background-color: #C0615A !important;"
        "  box-shadow: 0 4px 12px rgba(212, 114, 106, 0.4) !important;"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# API Client helpers
# ---------------------------------------------------------------------------


def _headers() -> dict:
    """Return authorization headers using the current access token."""
    if st.session_state.access_token:
        return {"Authorization": f"Bearer {st.session_state.access_token}"}
    return {}


def _handle_token_refresh() -> bool:
    """Attempt to refresh the access token. Returns True on success."""
    if not st.session_state.refresh_token:
        return False
    try:
        with httpx.Client(base_url=API_BASE, timeout=API_TIMEOUT_SHORT) as client:
            resp = client.post(
                "/auth/refresh",
                json={"refresh_token": st.session_state.refresh_token},
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.access_token = data["access_token"]
                st.session_state.refresh_token = data["refresh_token"]
                return True
    except Exception as exc:
        logger.warning("Token refresh failed: %s", exc)
    return False


def api_get(path: str, params: dict | None = None) -> httpx.Response | None:
    """Make an authenticated GET request with automatic token refresh."""
    try:
        with httpx.Client(base_url=API_BASE, timeout=API_TIMEOUT_DEFAULT) as client:
            resp = client.get(path, headers=_headers(), params=params)
            if resp.status_code == 401:
                if _handle_token_refresh():
                    resp = client.get(path, headers=_headers(), params=params)
            return resp
    except httpx.ConnectError:
        st.error("Cannot connect to the backend. Is the server running?")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def api_post(path: str, json_data: dict | None = None) -> httpx.Response | None:
    """Make an authenticated POST request with automatic token refresh."""
    try:
        with httpx.Client(base_url=API_BASE, timeout=API_TIMEOUT_LONG) as client:
            resp = client.post(path, headers=_headers(), json=json_data)
            if resp.status_code == 401:
                if _handle_token_refresh():
                    resp = client.post(path, headers=_headers(), json=json_data)
            return resp
    except httpx.ConnectError:
        st.error("Cannot connect to the backend. Is the server running?")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def api_patch(path: str, json_data: dict | None = None) -> httpx.Response | None:
    """Make an authenticated PATCH request with automatic token refresh."""
    try:
        with httpx.Client(base_url=API_BASE, timeout=API_TIMEOUT_DEFAULT) as client:
            resp = client.patch(path, headers=_headers(), json=json_data)
            if resp.status_code == 401:
                if _handle_token_refresh():
                    resp = client.patch(path, headers=_headers(), json=json_data)
            return resp
    except httpx.ConnectError:
        st.error("Cannot connect to the backend. Is the server running?")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def api_delete(path: str) -> httpx.Response | None:
    """Make an authenticated DELETE request with automatic token refresh."""
    try:
        with httpx.Client(base_url=API_BASE, timeout=API_TIMEOUT_DEFAULT) as client:
            resp = client.delete(path, headers=_headers())
            if resp.status_code == 401:
                if _handle_token_refresh():
                    resp = client.delete(path, headers=_headers())
            return resp
    except httpx.ConnectError:
        st.error("Cannot connect to the backend. Is the server running?")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def api_post_voice(path: str, conversation_id: str, audio_bytes: bytes) -> httpx.Response | None:
    """POST voice audio as multipart form data."""
    try:
        with httpx.Client(base_url=API_BASE, timeout=API_TIMEOUT_VOICE) as client:
            resp = client.post(
                path,
                headers=_headers(),
                data={"conversation_id": conversation_id},
                files={"audio": ("recording.wav", audio_bytes, "audio/wav")},
            )
            if resp.status_code == 401:
                if _handle_token_refresh():
                    resp = client.post(
                        path,
                        headers=_headers(),
                        data={"conversation_id": conversation_id},
                        files={"audio": ("recording.wav", audio_bytes, "audio/wav")},
                    )
            return resp
    except httpx.ConnectError:
        st.error("Cannot connect to the backend. Is the server running?")
        return None
    except Exception as e:
        st.error(f"Voice request failed: {e}")
        return None


# ---------------------------------------------------------------------------
# WebSocket chat (threaded approach for Streamlit)
# ---------------------------------------------------------------------------


def ws_send_message(conversation_id: str, content: str) -> dict:
    """Send a message via WebSocket and collect both layer responses.

    Because Streamlit reruns the script on each interaction, we use a
    synchronous wrapper around the ``websockets`` library.  The WebSocket
    connection is opened, authenticated, one message is sent, Layer 1 and
    Layer 2 responses are collected, and then the connection is closed.

    Falls back to the REST ``/chat/text`` endpoint if WebSocket is unavailable.
    """
    result = {"layer1": None, "layer2": None, "error": None}

    try:
        import websockets.sync.client as ws_sync

        token = st.session_state.access_token
        url = f"{WS_BASE}/chat/ws"

        with ws_sync.connect(
            url, open_timeout=WS_OPEN_TIMEOUT, close_timeout=WS_CLOSE_TIMEOUT
        ) as ws:
            # Authenticate via message (avoids exposing token in URL/logs)
            ws.send(json.dumps({"type": "auth", "token": token}))
            auth_msg = json.loads(ws.recv(timeout=WS_AUTH_TIMEOUT))
            if auth_msg.get("type") != "auth_ok":
                raise RuntimeError(f"Auth failed: {auth_msg}")

            # Send user message
            ws.send(
                json.dumps(
                    {
                        "type": "message",
                        "conversation_id": conversation_id,
                        "content": content,
                    }
                )
            )

            # Collect responses (expect layer1, then layer2)
            deadline = time.time() + WS_TEXT_DEADLINE
            while time.time() < deadline:
                try:
                    raw = ws.recv(timeout=WS_RECV_TIMEOUT)
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")

                    if msg_type == "layer1":
                        result["layer1"] = msg.get("content", "")
                    elif msg_type == "layer2":
                        result["layer2"] = msg.get("content", "")
                        break  # Both layers received
                    elif msg_type == "error":
                        result["error"] = msg.get("content", "Unknown error")
                        break
                except TimeoutError:
                    break

        return result

    except ImportError:
        # websockets not installed -- fall back to REST
        return _rest_fallback_chat(conversation_id, content)
    except Exception as exc:
        # WebSocket connection failed -- fall back to REST
        logger.warning("WebSocket text chat failed, falling back to REST: %s", exc)
        return _rest_fallback_chat(conversation_id, content)


def _rest_fallback_chat(conversation_id: str, content: str) -> dict:
    """Fallback: use REST /chat/text when WebSocket is unavailable."""
    result: dict = {"layer1": None, "layer2": None, "error": None}
    resp = api_post(
        "/chat/text",
        json_data={"conversation_id": conversation_id, "content": content},
    )
    if resp and resp.status_code == 200:
        data = resp.json()
        result["layer1"] = data.get("content", "")
    elif resp:
        result["error"] = resp.text
    else:
        result["error"] = "Failed to reach backend"
    return result


# ---------------------------------------------------------------------------
# WebSocket voice chat
# ---------------------------------------------------------------------------


def ws_send_voice(conversation_id: str, audio_bytes: bytes) -> dict:
    """Send voice audio via WebSocket and collect all responses.

    Returns dict with keys: transcription, layer1, layer2, audio (base64), error.
    Falls back to REST /chat/voice on failure.
    """
    result = {
        "transcription": None,
        "language": None,
        "layer1": None,
        "layer2": None,
        "audio": None,
        "error": None,
    }

    try:
        import websockets.sync.client as ws_sync

        token = st.session_state.access_token
        url = f"{WS_BASE}/chat/ws"
        audio_b64 = base64.b64encode(audio_bytes).decode()

        with ws_sync.connect(
            url, open_timeout=WS_OPEN_TIMEOUT, close_timeout=WS_CLOSE_TIMEOUT
        ) as ws:
            # Authenticate via message (avoids exposing token in URL/logs)
            ws.send(json.dumps({"type": "auth", "token": token}))
            auth_msg = json.loads(ws.recv(timeout=WS_AUTH_TIMEOUT))
            if auth_msg.get("type") != "auth_ok":
                raise RuntimeError(f"Auth failed: {auth_msg}")

            ws.send(
                json.dumps(
                    {
                        "type": "voice",
                        "conversation_id": conversation_id,
                        "audio": audio_b64,
                    }
                )
            )

            deadline = time.time() + WS_VOICE_DEADLINE
            while time.time() < deadline:
                try:
                    raw = ws.recv(timeout=WS_RECV_TIMEOUT_VOICE)
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")

                    if msg_type == "transcription":
                        result["transcription"] = msg.get("content", "")
                        result["language"] = msg.get("language", "")
                    elif msg_type == "layer1":
                        result["layer1"] = msg.get("content", "")
                    elif msg_type == "layer2":
                        result["layer2"] = msg.get("content", "")
                    elif msg_type == "audio":
                        result["audio"] = msg.get("audio", "")
                        break  # Audio is the last message
                    elif msg_type == "error":
                        result["error"] = msg.get("content", "Unknown error")
                        break
                except TimeoutError:
                    break

            # If no audio came but we have layer2, we're done
            return result

    except ImportError:
        return _rest_fallback_voice(conversation_id, audio_bytes)
    except Exception as exc:
        logger.warning("WebSocket voice chat failed, falling back to REST: %s", exc)
        return _rest_fallback_voice(conversation_id, audio_bytes)


def _rest_fallback_voice(conversation_id: str, audio_bytes: bytes) -> dict:
    """Fallback: use REST /chat/voice when WebSocket is unavailable."""
    result: dict = {
        "transcription": None,
        "language": None,
        "layer1": None,
        "layer2": None,
        "audio": None,
        "error": None,
    }
    resp = api_post_voice("/chat/voice", conversation_id, audio_bytes)
    if resp and resp.status_code == 200:
        data = resp.json()
        result["layer1"] = data.get("content", "")
        result["transcription"] = data.get("transcription", "")
        result["audio"] = data.get("audio_url", "")
    elif resp:
        result["error"] = resp.text
    else:
        result["error"] = "Failed to reach backend"
    return result


# ---------------------------------------------------------------------------
# Auth functions
# ---------------------------------------------------------------------------


def do_login(email: str, password: str) -> bool:
    """Authenticate and store tokens in session state."""
    resp = api_post("/auth/login", {"email": email, "password": password})
    if resp and resp.status_code == 200:
        data = resp.json()
        st.session_state.access_token = data["access_token"]
        st.session_state.refresh_token = data["refresh_token"]
        # Fetch user profile
        _load_user_profile()
        return True
    return False


def do_register(full_name: str, email: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    resp = api_post(
        "/auth/register",
        {"full_name": full_name, "email": email, "password": password},
    )
    if resp and resp.status_code == 201:
        return True, "Account created successfully. Please log in."
    elif resp:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return False, f"Registration failed: {detail}"
    return False, "Cannot connect to server."


def do_logout():
    """Logout: revoke refresh token and clear session state."""
    if st.session_state.refresh_token:
        api_post("/auth/logout", {"refresh_token": st.session_state.refresh_token})

    for key in [
        "access_token",
        "refresh_token",
        "user_id",
        "user_name",
        "user_email",
        "conversations",
        "current_conversation_id",
        "current_conversation_title",
        "messages",
        "pending_layer1",
        "pending_layer2",
        "is_thinking",
        "last_transcription",
        "last_audio_response",
    ]:
        st.session_state[key] = None if key not in ("conversations", "messages") else []

    st.session_state.page = "login"


def _load_user_profile():
    """Load current user profile into session state."""
    resp = api_get("/users/me")
    if resp and resp.status_code == 200:
        data = resp.json()
        st.session_state.user_id = data["id"]
        st.session_state.user_name = data["full_name"]
        st.session_state.user_email = data["email"]
        st.session_state.page = "chat"


# ---------------------------------------------------------------------------
# Conversation functions
# ---------------------------------------------------------------------------


def load_conversations():
    """Fetch the user's conversation list from the backend."""
    resp = api_get("/conversations", params={"skip": 0, "limit": MAX_CONVERSATIONS_LIMIT})
    if resp and resp.status_code == 200:
        data = resp.json()
        st.session_state.conversations = data.get("conversations", [])


def create_conversation(title: str | None = None, conv_type: str = "friend"):
    """Create a new conversation and set it as current."""
    resp = api_post("/conversations", {"title": title, "type": conv_type})
    if resp and resp.status_code == 201:
        data = resp.json()
        st.session_state.current_conversation_id = data["id"]
        st.session_state.current_conversation_title = data.get("title")
        st.session_state.messages = []
        load_conversations()
        return data
    return None


def load_conversation_messages(conversation_id: str):
    """Load messages for a specific conversation."""
    resp = api_get(f"/conversations/{conversation_id}")
    if resp and resp.status_code == 200:
        data = resp.json()
        st.session_state.current_conversation_id = data["id"]
        st.session_state.current_conversation_title = data.get("title")
        st.session_state.messages = data.get("messages", [])


def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    resp = api_delete(f"/conversations/{conversation_id}")
    if resp and resp.status_code == 204:
        if st.session_state.current_conversation_id == conversation_id:
            st.session_state.current_conversation_id = None
            st.session_state.current_conversation_title = None
            st.session_state.messages = []
        load_conversations()
        return True
    return False


def rename_conversation(conversation_id: str, new_title: str):
    """Rename a conversation."""
    resp = api_patch(f"/conversations/{conversation_id}", {"title": new_title})
    if resp and resp.status_code == 200:
        load_conversations()
        if st.session_state.current_conversation_id == conversation_id:
            st.session_state.current_conversation_title = new_title
        return True
    return False


# ---------------------------------------------------------------------------
# Page: Login
# ---------------------------------------------------------------------------


def render_login_page():
    """Render the login page."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown(
            '<div style="text-align:center;">'
            '<div class="app-title">Dear AI</div>'
            '<div class="app-subtitle">Your compassionate mental health companion</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown("")

        with st.container(border=True):
            st.markdown("### Welcome back")
            st.markdown(
                '<p style="color: var(--text-secondary); font-size: 0.9rem;">'
                "Sign in to continue your journey</p>",
                unsafe_allow_html=True,
            )

            email = st.text_input("Email", placeholder="you@example.com", key="login_email")
            password = st.text_input(
                "Password", type="password", placeholder="Enter your password", key="login_password"
            )

            st.markdown("")

            if st.button("Sign In", use_container_width=True, key="login_btn"):
                if not email or not password:
                    st.warning("Please enter your email and password.")
                else:
                    with st.spinner("Signing in..."):
                        if do_login(email, password):
                            st.rerun()
                        else:
                            st.error("Invalid email or password. Please try again.")

            st.markdown("")
            st.markdown(
                '<hr class="soft-divider">',
                unsafe_allow_html=True,
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(
                    '<p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">'
                    "Don't have an account?</p>",
                    unsafe_allow_html=True,
                )
            with col_b:
                if st.button(
                    "Create Account",
                    use_container_width=True,
                    key="goto_register",
                    type="secondary",
                ):
                    st.session_state.page = "register"
                    st.rerun()


# ---------------------------------------------------------------------------
# Page: Register
# ---------------------------------------------------------------------------


def render_register_page():
    """Render the registration page."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown(
            '<div style="text-align:center;">'
            '<div class="app-title">Dear AI</div>'
            '<div class="app-subtitle">Start your wellness journey</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown("")

        with st.container(border=True):
            st.markdown("### Create your account")

            full_name = st.text_input("Full Name", placeholder="Your name", key="reg_name")
            email = st.text_input("Email", placeholder="you@example.com", key="reg_email")
            password = st.text_input(
                "Password", type="password", placeholder="Create a password", key="reg_password"
            )
            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Confirm your password",
                key="reg_confirm",
            )

            st.markdown("")

            if st.button("Create Account", use_container_width=True, key="register_btn"):
                if not full_name or not email or not password:
                    st.warning("Please fill in all fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(password) < MIN_PASSWORD_LENGTH:
                    st.warning(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
                else:
                    with st.spinner("Creating your account..."):
                        success, msg = do_register(full_name, email, password)
                        if success:
                            st.success(msg)
                            time.sleep(1)
                            st.session_state.page = "login"
                            st.rerun()
                        else:
                            st.error(msg)

            st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(
                    '<p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">'
                    "Already have an account?</p>",
                    unsafe_allow_html=True,
                )
            with col_b:
                if st.button(
                    "Sign In", use_container_width=True, key="goto_login", type="secondary"
                ):
                    st.session_state.page = "login"
                    st.rerun()


# ---------------------------------------------------------------------------
# Page: Chat (main page)
# ---------------------------------------------------------------------------


def render_chat_sidebar():
    """Render the sidebar with conversation list and navigation."""
    with st.sidebar:
        st.markdown(
            '<div class="app-title" style="font-size: 1.4rem;">Dear AI</div>',
            unsafe_allow_html=True,
        )
        safe_name = html_mod.escape(st.session_state.user_name or "User")
        st.markdown(
            f'<p style="color: var(--text-muted); font-size: 0.8rem;">Signed in as {safe_name}</p>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

        # New conversation button
        col_new, col_type = st.columns([2, 1])
        with col_new:
            if st.button("New Chat", use_container_width=True, key="new_chat_btn"):
                conv = create_conversation(conv_type=st.session_state.new_conv_type)
                if conv:
                    st.rerun()
        with col_type:
            conv_type = st.selectbox(
                "Type",
                ["friend", "therapy"],
                index=0,
                key="new_conv_type",
                label_visibility="collapsed",
            )

        st.markdown("")
        st.markdown("**Your Conversations**")

        # Conversation list
        if not st.session_state.conversations:
            load_conversations()

        for conv in st.session_state.conversations:
            conv_id = conv["id"]
            conv_title = conv.get("title") or "Untitled conversation"
            is_active = conv_id == st.session_state.current_conversation_id
            conv_date = ""
            if conv.get("updated_at"):
                try:
                    dt = datetime.fromisoformat(conv["updated_at"])
                    conv_date = dt.strftime("%b %d, %H:%M")
                except Exception:
                    conv_date = ""

            # Each conversation as a row with select and delete
            c1, c2 = st.columns([5, 1])
            with c1:
                display_label = conv_title[:30]
                if conv_date:
                    display_label += f" ({conv_date})"
                if st.button(
                    conv_title[:35] + ("..." if len(conv_title) > 35 else ""),
                    key=f"conv_{conv_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    load_conversation_messages(conv_id)
                    st.rerun()
            with c2:
                _danger_button_marker()
                if st.button("X", key=f"del_{conv_id}"):
                    st.session_state[f"confirm_del_{conv_id}"] = True
                    st.rerun()

            # Deletion confirmation row
            if st.session_state.get(f"confirm_del_{conv_id}"):
                conf1, conf2, conf3 = st.columns([3, 1, 1])
                with conf1:
                    st.markdown(
                        '<p style="color: var(--danger); font-size: 0.8rem; margin-top: 0.4rem;">Delete this conversation?</p>',
                        unsafe_allow_html=True,
                    )
                with conf2:
                    _danger_button_marker()
                    if st.button("Yes", key=f"confirm_yes_{conv_id}"):
                        st.session_state.pop(f"confirm_del_{conv_id}", None)
                        delete_conversation(conv_id)
                        st.rerun()
                with conf3:
                    if st.button("No", key=f"confirm_no_{conv_id}", type="secondary"):
                        st.session_state.pop(f"confirm_del_{conv_id}", None)
                        st.rerun()

        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

        # Navigation
        if st.button("My Profile", use_container_width=True, key="goto_profile", type="secondary"):
            st.session_state.page = "profile"
            st.rerun()

        st.markdown("")

        _danger_button_marker()
        if st.button("Sign Out", use_container_width=True, key="logout_btn"):
            do_logout()
            st.rerun()


def render_message(msg: dict, index: int):
    """Render a single chat message bubble."""
    role = msg.get("role", "user")
    content = html_mod.escape(msg.get("content", ""))
    msg_type = msg.get("type", "text")
    layer = msg.get("layer")  # custom field for two-layer display

    if role == "user":
        st.markdown(
            f'<div style="display:flex; justify-content:flex-end;">'
            f'<div class="message-bubble user-message">{content}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    elif role == "assistant":
        tag_html = ""
        if layer == "layer1":
            tag_html = '<div class="layer1-tag">Quick Response</div><br>'
        elif layer == "layer2":
            tag_html = '<div class="layer2-tag">Detailed Response</div><br>'

        st.markdown(
            f'<div style="display:flex; justify-content:flex-start;">'
            f'<div class="message-bubble assistant-message">'
            f"{tag_html}{content}"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    elif role == "system":
        st.markdown(
            f'<div style="text-align:center; color: var(--text-muted); font-size: 0.8rem; margin: 0.5rem 0;">'
            f"{content}</div>",
            unsafe_allow_html=True,
        )


def render_typing_indicator():
    """Render a typing animation."""
    st.markdown(
        '<div style="display:flex; justify-content:flex-start;">'
        '<div class="message-bubble assistant-message">'
        '<div class="typing-indicator">'
        '<div class="typing-dot"></div>'
        '<div class="typing-dot"></div>'
        '<div class="typing-dot"></div>'
        "</div></div></div>",
        unsafe_allow_html=True,
    )


def render_welcome_screen():
    """Render the welcome screen when no conversation is selected."""
    st.markdown(
        '<div class="welcome-container">'
        "<h2>Welcome to Dear AI</h2>"
        '<p style="color: var(--text-secondary); line-height: 1.8;">'
        "I'm here to listen, support, and walk alongside you on your mental health journey. "
        "Whether you need someone to talk to as a friend, or prefer a more structured "
        "therapeutic conversation, I'm here for you."
        "</p>"
        '<p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 2rem;">'
        "Start a new conversation from the sidebar, or select an existing one."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Quick start cards
    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="info-card">'
            '<p style="font-weight: 600; color: var(--accent);">Friendly Chat</p>'
            '<p style="color: var(--text-secondary); font-size: 0.85rem;">'
            "Talk openly about your feelings, daily life, or anything on your mind. "
            "I'll be a supportive, non-judgmental friend."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="info-card">'
            '<p style="font-weight: 600; color: var(--sage);">Therapy Mode</p>'
            '<p style="color: var(--text-secondary); font-size: 0.85rem;">'
            "Get structured support using evidence-based techniques like CBT, DBT, "
            "and ACT. Guided exercises and coping strategies."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )


def render_chat_page():
    """Render the main chat page."""
    render_chat_sidebar()

    if not st.session_state.current_conversation_id:
        render_welcome_screen()
        return

    # Chat header
    title = st.session_state.current_conversation_title or "Conversation"
    safe_title = html_mod.escape(title)
    st.markdown(
        f'<div style="padding: 0.5rem 0; border-bottom: 1px solid var(--border); margin-bottom: 1rem;">'
        f'<span style="font-family: Lora, serif; font-size: 1.2rem; color: var(--accent);">{safe_title}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Rename conversation
    with st.expander("Conversation settings", expanded=False):
        col_r1, col_r2 = st.columns([3, 1])
        with col_r1:
            new_title = st.text_input(
                "Rename",
                value=title,
                key="rename_input",
                label_visibility="collapsed",
            )
        with col_r2:
            if st.button("Rename", key="rename_btn"):
                if new_title and new_title != title:
                    rename_conversation(st.session_state.current_conversation_id, new_title)
                    st.rerun()

    # Chat messages area
    chat_area = st.container()
    with chat_area:
        messages = st.session_state.messages
        for i, msg in enumerate(messages):
            render_message(msg, i)

        # Show typing indicator if waiting
        if st.session_state.is_thinking:
            render_typing_indicator()

    # Input area with tabs for text and voice
    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    tab_text, tab_voice = st.tabs(["Text", "Voice"])

    with tab_text:
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Message",
                placeholder="Type your message here...",
                key="chat_input",
                label_visibility="collapsed",
            )
        with col_send:
            send_clicked = st.button("Send", key="send_btn", use_container_width=True)

        if send_clicked and user_input:
            _handle_send_text(user_input)

    with tab_voice:
        render_voice_input()


def _handle_send_text(user_input: str):
    """Handle sending a text message."""
    conversation_id = st.session_state.current_conversation_id

    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": user_input, "type": "text"})
    st.session_state.is_thinking = True

    # Send via WebSocket (or REST fallback)
    result = ws_send_message(conversation_id, user_input)

    st.session_state.is_thinking = False

    if result.get("error"):
        st.session_state.messages.append({"role": "system", "content": f"Error: {result['error']}"})
    else:
        if result.get("layer1"):
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["layer1"],
                    "type": "text",
                    "layer": "layer1",
                }
            )
        if result.get("layer2"):
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["layer2"],
                    "type": "text",
                    "layer": "layer2",
                }
            )

    st.rerun()


def render_voice_input():
    """Render voice recording and playback interface."""
    if not st.session_state.current_conversation_id:
        st.info("Select or create a conversation first.")
        return

    st.markdown(
        '<div class="voice-area">'
        '<p style="color: var(--accent); font-weight: 500;">Voice Input</p>'
        '<p style="color: var(--text-secondary); font-size: 0.85rem;">'
        "Record or upload an audio file to send a voice message."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Try to use audio_recorder_streamlit if available
    try:
        from audio_recorder_streamlit import audio_recorder

        st.markdown(
            '<p style="color: var(--text-secondary); font-size: 0.85rem;">Click the microphone to record:</p>',
            unsafe_allow_html=True,
        )
        audio_bytes = audio_recorder(
            text="",
            recording_color="#8B7EC8",
            neutral_color="#6B6B6B",
            icon_size="2x",
            pause_threshold=2.0,
        )

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            if st.button("Send Voice Message", key="send_voice_rec"):
                _handle_send_voice(audio_bytes)
    except ImportError:
        st.markdown(
            '<p style="color: var(--text-muted); font-size: 0.8rem;">'
            "Install <code>audio-recorder-streamlit</code> for microphone recording.</p>",
            unsafe_allow_html=True,
        )

    # File upload fallback
    st.markdown("")
    uploaded_audio = st.file_uploader(
        "Or upload an audio file",
        type=["wav", "mp3", "webm", "ogg", "m4a"],
        key="voice_upload",
    )

    if uploaded_audio:
        audio_bytes = uploaded_audio.read()
        st.audio(audio_bytes)
        if st.button("Send Uploaded Audio", key="send_voice_upload"):
            _handle_send_voice(audio_bytes)

    # Display last transcription
    if st.session_state.last_transcription:
        safe_transcription = html_mod.escape(st.session_state.last_transcription)
        st.markdown(
            f'<div class="info-card">'
            f'<p style="font-weight: 500; color: var(--accent);">Transcription</p>'
            f"<p>{safe_transcription}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Play last audio response
    if st.session_state.last_audio_response:
        st.markdown(
            '<p style="font-weight: 500; color: var(--accent);">Audio Response</p>',
            unsafe_allow_html=True,
        )
        audio_data = base64.b64decode(st.session_state.last_audio_response)
        st.audio(audio_data, format="audio/wav")


def _handle_send_voice(audio_bytes: bytes):
    """Handle sending a voice message."""
    conversation_id = st.session_state.current_conversation_id

    st.session_state.is_thinking = True

    with st.spinner("Processing voice message..."):
        result = ws_send_voice(conversation_id, audio_bytes)

    st.session_state.is_thinking = False

    if result.get("error"):
        st.error(f"Voice processing error: {result['error']}")
        return

    # Store transcription
    if result.get("transcription"):
        st.session_state.last_transcription = result["transcription"]
        st.session_state.messages.append(
            {
                "role": "user",
                "content": result["transcription"],
                "type": "voice",
            }
        )

    # Add assistant responses
    if result.get("layer1"):
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["layer1"],
                "type": "text",
                "layer": "layer1",
            }
        )
    if result.get("layer2"):
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["layer2"],
                "type": "text",
                "layer": "layer2",
            }
        )

    # Store audio response for playback
    if result.get("audio"):
        st.session_state.last_audio_response = result["audio"]

    st.rerun()


# ---------------------------------------------------------------------------
# Page: Profile
# ---------------------------------------------------------------------------


def render_profile_page():
    """Render the user profile page."""
    render_chat_sidebar()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Profile header
        initials = ""
        if st.session_state.user_name:
            parts = st.session_state.user_name.split()
            initials = "".join(p[0].upper() for p in parts[:2])

        safe_profile_name = html_mod.escape(st.session_state.user_name or "User")
        safe_profile_email = html_mod.escape(st.session_state.user_email or "")
        st.markdown(
            f'<div class="profile-header">'
            f'<div class="profile-avatar">{html_mod.escape(initials)}</div>'
            f'<h3 style="margin: 0; color: var(--text-primary);">{safe_profile_name}</h3>'
            f'<p style="color: var(--text-muted); font-size: 0.85rem;">{safe_profile_email}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Load full profile data
        resp = api_get("/users/me")
        if resp and resp.status_code == 200:
            profile = resp.json()
        else:
            st.error("Failed to load profile.")
            if st.button("Back to Chat"):
                st.session_state.page = "chat"
                st.rerun()
            return

        with st.container(border=True):
            st.markdown("### Edit Profile")

            new_name = st.text_input(
                "Full Name",
                value=profile.get("full_name", ""),
                key="profile_name",
            )

            gender_options = ["", "male", "female", "other", "prefer_not_to_say"]
            gender_labels = [
                "Select...",
                "Male",
                "Female",
                "Other",
                "Prefer not to say",
            ]
            current_gender = profile.get("gender") or ""
            gender_index = (
                gender_options.index(current_gender) if current_gender in gender_options else 0
            )

            new_gender = st.selectbox(
                "Gender",
                options=gender_options,
                format_func=lambda x: gender_labels[gender_options.index(x)],
                index=gender_index,
                key="profile_gender",
            )

            new_age = st.number_input(
                "Age",
                min_value=0,
                max_value=150,
                value=profile.get("age") or 0,
                key="profile_age",
            )

            st.markdown("")

            if st.button("Save Changes", use_container_width=True, key="save_profile"):
                update_data = {}
                if new_name and new_name != profile.get("full_name"):
                    update_data["full_name"] = new_name
                if new_gender and new_gender != (profile.get("gender") or ""):
                    update_data["gender"] = new_gender
                if new_age and new_age != (profile.get("age") or 0):
                    update_data["age"] = new_age

                if update_data:
                    resp = api_patch("/users/me", update_data)
                    if resp and resp.status_code == 200:
                        st.success("Profile updated successfully.")
                        _load_user_profile()
                        st.rerun()
                    else:
                        st.error("Failed to update profile.")
                else:
                    st.info("No changes to save.")

        st.markdown("")

        # Account info
        with st.container(border=True):
            st.markdown("### Account Information")

            created_at = profile.get("created_at", "")
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at)
                    created_at = dt.strftime("%B %d, %Y")
                except Exception:
                    pass

            st.markdown(
                f'<p style="color: var(--text-secondary); font-size: 0.9rem;">'
                f"<b>Email:</b> {html_mod.escape(profile.get('email', 'N/A'))}<br>"
                f"<b>Member since:</b> {html_mod.escape(created_at)}<br>"
                f"<b>User ID:</b> <code>{html_mod.escape(str(profile.get('id', 'N/A')))}</code>"
                f"</p>",
                unsafe_allow_html=True,
            )

        st.markdown("")

        if st.button(
            "Back to Chat", use_container_width=True, key="back_to_chat", type="secondary"
        ):
            st.session_state.page = "chat"
            st.rerun()


# ---------------------------------------------------------------------------
# Main app router
# ---------------------------------------------------------------------------


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Dear AI - Mental Health Companion",
        page_icon="\U0001f49c",
        layout="wide",
        initial_sidebar_state="auto",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Check if user is authenticated
    is_authenticated = st.session_state.access_token is not None

    # Auto-redirect to chat if authenticated
    if is_authenticated and st.session_state.page in ("login", "register"):
        st.session_state.page = "chat"

    # Auto-redirect to login if not authenticated
    if not is_authenticated and st.session_state.page not in ("login", "register"):
        st.session_state.page = "login"

    # Route to the correct page
    page = st.session_state.page

    if page == "login":
        render_login_page()
    elif page == "register":
        render_register_page()
    elif page == "chat":
        render_chat_page()
    elif page == "profile":
        render_profile_page()
    else:
        st.session_state.page = "login"
        st.rerun()


if __name__ == "__main__":
    main()
