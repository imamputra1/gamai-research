# src/ai_integration/constants.py
from __future__ import annotations
from typing import Final, List

# =============================================================================
# API ENDPOINTS
# =============================================================================
OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_ENDPOINT: Final[str] = f"{OPENROUTER_BASE_URL}/chat/completions"

# =============================================================================
# MODELS CONFIGURATION
# =============================================================================
# Model default utama (Gratis, Performa Tinggi, Konteks 1M Token)
DEFAULT_LLM_MODEL: Final[str] = "openrouter/owl-alpha"

# Model Tier Atas untuk penalaran kompleks / fallback (Berbayar)
REASONING_MODELS: Final[List[str]] = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-1.5-pro",
]

# =============================================================================
# DIAGNOSTICS / HEALTH CHECK
# =============================================================================
# Model sangat ringan & gratis untuk sekadar cek koneksi (Ping)
CHEAP_PING_MODEL: Final[str] = "google/gemma-2-9b-it:free"
# =============================================================================
# INDONESIAN NLP MODELS
# =============================================================================
# GANTI ke model yang sudah di-Finetuning untuk Sentimen
INDOBERT_MODEL_NAME: str = "w11wo/indonesian-roberta-base-sentiment-classifier"
INDOBERT_BATCH_SIZE: int = 16
INDOBERT_MAX_LENGTH: int = 512

# =============================================================================
# SENTIMENT LABEL MAPPING
# =============================================================================
SENTIMENT_LABEL_MAP: dict[str, str] = {
    "positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative",
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}

# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================
DEFAULT_MAX_TOKENS: int = 4096
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TIMEOUT_SECONDS: int = 60

# =============================================================================
# REASONING PARAMETERS
# =============================================================================
REASONING_TEMPERATURE: float = 0.2
REASONING_MAX_TOKENS: int = 2048

# =============================================================================
# PING PARAMETERS
# =============================================================================
PING_MAX_TOKENS: int = 5
PING_PROMPT: str = "PING"

# =============================================================================
# HTTP STATUS CODES
# =============================================================================
HTTP_OK: int = 200
HTTP_PAYMENT_REQUIRED: int = 402
HTTP_TOO_MANY_REQUESTS: int = 429
HTTP_BAD_GATEWAY: int = 502
HTTP_SERVICE_UNAVAILABLE: int = 503

# =============================================================================
# RETRY CONFIGURATION (Tenacity)
# =============================================================================
RETRY_MAX_ATTEMPTS: int = 3
RETRY_WAIT_MULTIPLIER: int = 2
RETRY_WAIT_MIN: int = 2
RETRY_WAIT_MAX: int = 10

# =============================================================================
# ENVIRONMENT VARIABLE KEYS
# =============================================================================
ENV_OPENROUTER_API_KEY: str = "OPENROUTER_API_KEY"

# =============================================================================
# ERROR MESSAGES
# =============================================================================
ERROR_INSUFFICIENT_BALANCE: str = "Saldo OpenRouter habis (HTTP 402). Periksa billing dashboard."
ERROR_INVALID_CREDENTIALS: str = "Kredensial OpenRouter tidak valid. Periksa OPENROUTER_API_KEY."
ERROR_CONNECTION_FAILED: str = "Tidak dapat terhubung ke OpenRouter. Periksa koneksi internet."

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================
REASONING_SYSTEM_PROMPT: str = (
    "Anda adalah asisten penelitian akademik yang ahli dalam ekonometrika "
    "dan analisis data. Tugas Anda adalah menulis narasi akademik dalam Bahasa "
    "Indonesia yang tajam, objektif, dan sesuai standar tesis S2. Gunakan data "
    "yang diberikan dalam bentuk dictionary. Jangan berhalusinasi atau menambah "
    "data yang tidak ada. Temperature rendah memastikan output deterministik."
)

REASONING_USER_TEMPLATE: str = (
    "Berikut adalah hasil komputasi statistik dari penelitian saya:\n\n"
    "{context_json}\n\n"
    "Silakan tulis narasi akademik yang menjelaskan temuan ini dengan "
    "struktur: (1) Interpretasi hasil, (2) Implikasi teoritis, "
    "(3) Rekomendasi praktis. Gunakan Bahasa Indonesia formal."
)
