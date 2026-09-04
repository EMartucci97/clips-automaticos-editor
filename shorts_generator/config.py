import os

from dotenv import load_dotenv

load_dotenv()

MUAPI_API_KEY = os.getenv("MUAPI_API_KEY", "").strip()
MUAPI_BASE_URL = os.getenv("MUAPI_BASE_URL", "https://api.muapi.ai/api/v1").rstrip("/")

POLL_INTERVAL_SECONDS = float(os.getenv("MUAPI_POLL_INTERVAL", "5"))
POLL_TIMEOUT_SECONDS = float(os.getenv("MUAPI_POLL_TIMEOUT", "600"))

# Local-mode (--mode local) settings — only consulted when running offline.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
ANTHROPIC_WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "medium")
LOCAL_WHISPER_DEVICE = os.getenv("LOCAL_WHISPER_DEVICE", "auto")  # auto / cpu / cuda
# Ruarte content is essentially always Spanish. Without this, faster-whisper
# auto-detects language per run and on long/noisy audio it can mis-detect
# (or drift into) English mid-transcript, producing garbled pseudo-English
# text instead of the real Spanish — which then poisons every downstream LLM
# call (content-type detection, virality scoring, sales-angle matching).
# Empty string restores auto-detect.
LOCAL_WHISPER_LANGUAGE = os.getenv("LOCAL_WHISPER_LANGUAGE", "es").strip() or None
# Vocabulary hint passed to Whisper to bias recognition toward Ruarte's
# financial/market jargon — a generic model otherwise mishears terms like
# "CCL" or "MEP" as near-homophone junk. Empty string disables the hint.
LOCAL_WHISPER_INITIAL_PROMPT = os.getenv(
    "LOCAL_WHISPER_INITIAL_PROMPT",
    "Dólar CCL, dólar MEP, dólar blue, brecha cambiaria, S&P 500, Nasdaq, "
    "tasa de interés, inflación, Merval, bonos, acciones, cripto, Bitcoin, "
    "cartera de inversión, Roberto Ruarte, Ruarte Reports.",
).strip()
LOCAL_OUTPUT_DIR = os.getenv("LOCAL_OUTPUT_DIR", "output")
LOCAL_SUBTITLES_ENABLED = os.getenv("LOCAL_SUBTITLES_ENABLED", "true").strip().lower() == "true"

# One extra LLM pass over the raw transcript, before highlight selection and
# clipping, that catches likely Whisper mis-hearings (mainly acronyms losing
# their first letter — "fore mexicana" instead of "AFORE mexicana"). See
# shorts_generator/local/review.py. Cached per source file; disable if it's
# adding latency you don't want.
TRANSCRIPT_REVIEW_ENABLED = os.getenv("TRANSCRIPT_REVIEW_ENABLED", "true").strip().lower() == "true"

# Speaker identification (--mode local only). Diarizes the audio with
# pyannote-audio and labels each segment ROBERTO/OTHER by matching each
# diarized speaker's voice embedding against a reference built from clean
# solo-Roberto clips — so highlights.py can prioritize when Roberto speaks
# (or responds to a guest) without needing a fixed turn order. Best-effort:
# if pyannote isn't installed, HF_TOKEN is missing, or the reference clips
# aren't present, the pipeline just runs without speaker labels.
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
LOCAL_DIARIZATION_ENABLED = os.getenv("LOCAL_DIARIZATION_ENABLED", "true").strip().lower() == "true"
LOCAL_DIARIZATION_REF_DIR = os.getenv("LOCAL_DIARIZATION_REF_DIR", "fixtures/voice_ref")
# Cosine similarity above this = that diarized speaker cluster is Roberto.
# Measured 0.93 (Roberto) vs 0.37 (a real guest) on the validation fixture —
# wide margin, so this threshold has plenty of headroom either way.
LOCAL_DIARIZATION_SIMILARITY_THRESHOLD = float(os.getenv("LOCAL_DIARIZATION_SIMILARITY_THRESHOLD", "0.6"))

# When enabled, the highlight LLM also scores each candidate against a set of
# proven sales angles (in addition to generic virality) and the two scores
# are blended for the final ranking. See shorts_generator/highlights.py.
SALES_ANGLES_ENABLED = os.getenv("SALES_ANGLES_ENABLED", "true").strip().lower() == "true"
SALES_ANGLE_WEIGHT = float(os.getenv("SALES_ANGLE_WEIGHT", "0.5"))  # 0 = pure virality, 1 = pure sales angle
# If no candidate in a video clears this sales_angle_score, ranking falls back
# to virality-only for that video (with a console notice) instead of blending
# in a near-zero sales angle score for every clip.
SALES_ANGLE_MIN_SCORE = float(os.getenv("SALES_ANGLE_MIN_SCORE", "30"))

# VAD (Voice Activity Detection) settings for faster-whisper
# Default threshold is 0.5; lower = more sensitive, higher = less sensitive
# Default min_speech_duration_ms is 250ms; increase to avoid tiny false positives
# Default min_silence_duration_ms is 2000ms; increase to avoid splitting mid-sentence
# DISABLED by default because VAD is too aggressive on mixed speech/music content
LOCAL_WHISPER_VAD_FILTER = os.getenv("LOCAL_WHISPER_VAD_FILTER", "false").strip().lower() == "true"
_vad_params_env = os.getenv("LOCAL_WHISPER_VAD_PARAMETERS", "")
if _vad_params_env:
    import json
    LOCAL_WHISPER_VAD_PARAMETERS = json.loads(_vad_params_env)
else:
    # Match faster-whisper defaults when VAD is enabled
    LOCAL_WHISPER_VAD_PARAMETERS = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 2000,
        "speech_pad_ms": 400,
    }


def require_api_key() -> str:
    if not MUAPI_API_KEY:
        raise RuntimeError(
            "MUAPI_API_KEY is not set. Add it to your .env file or export it as an env var."
        )
    return MUAPI_API_KEY


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Local mode needs an OpenAI key for highlight ranking. "
            "Add it to your .env or export it, or switch back to --mode api."
        )
    return OPENAI_API_KEY


def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Local mode needs a Gemini key when LLM_PROVIDER=gemini. "
            "Add it to your .env or export it, or switch LLM_PROVIDER back to openai."
        )
    return GEMINI_API_KEY


def require_anthropic_key() -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Local mode needs an Anthropic key when "
            "LLM_PROVIDER=anthropic. Add it to your .env or export it, or switch "
            "LLM_PROVIDER back to openai."
        )
    return ANTHROPIC_API_KEY
