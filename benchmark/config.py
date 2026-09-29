"""Candidate configs + pricing for the transcription/LLM-editing benchmark.

Pure data — no imports from the production pipeline here, so this module is
safe to import from anywhere without side effects.
"""
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
WORK_DIR = BENCHMARK_DIR / "work"
LOCAL_TRANSCRIBE_OUTPUT_DIR = WORK_DIR / "transcribe_cache"
REFERENCE_DIR = BENCHMARK_DIR / "reference"
TRANSCRIPTS_DIR = WORK_DIR / "transcripts"
CAPTIONS_DIR = WORK_DIR / "captions"
CLIP_DETECTION_DIR = WORK_DIR / "clip_detection"
LLM_CACHE_DIR = WORK_DIR / "llm_cache"
COSTS_JSONL = WORK_DIR / "costs.jsonl"

FIXTURES_DIR = BENCHMARK_DIR.parent / "fixtures"

# Test clips: (clip_id, path). Kept short so local iteration on the 3 free
# whisper configs stays fast.
TEST_CLIPS = [
    ("roberto_90s", str(FIXTURES_DIR / "roberto_90s.wav")),
    ("multispeaker_testimonio", str(FIXTURES_DIR / "multispeaker_testimonio_full.wav")),
]

# Reference clip length used to project cost-per-video (real range confirmed
# by the user is ~40-70s; 60s is the reference point).
REFERENCE_CLIP_SECONDS = 60.0
REVENUE_PER_VIDEO_USD = 3.5  # midpoint of the $3-4 the user charges per edited video

TRANSCRIPTION_CONFIGS = {
    "whisper-medium": {
        "kind": "local",
        "whisper_model": "medium",
        "paid": False,
    },
    "whisper-large-v3": {
        "kind": "local",
        "whisper_model": "large-v3",
        "paid": False,
    },
    "whisper-large-v3-turbo": {
        "kind": "local",
        "whisper_model": "large-v3-turbo",
        "paid": False,
    },
    "openai-whisper-1": {
        "kind": "openai-hosted",
        "openai_model": "whisper-1",
        "paid": True,
        # USD per minute of audio, published OpenAI Whisper API rate.
        "price_per_minute_usd": 0.006,
    },
    "openai-gpt-4o-transcribe": {
        "kind": "openai-hosted",
        "openai_model": "gpt-4o-transcribe",
        "paid": True,
        # USD per minute of audio, published OpenAI rate (verify before big
        # decisions -- pricing pages change).
        "price_per_minute_usd": 0.006,
    },
}

LLM_GROUPING_CONFIGS = {
    "mechanical": {
        "kind": "mechanical",
        "paid": False,
    },
    "openai-gpt-4o-mini": {
        "kind": "llm",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "paid": True,
        "price_per_million_input_usd": 0.15,
        "price_per_million_output_usd": 0.60,
    },
    "anthropic-opus-5": {
        "kind": "llm",
        "provider": "anthropic",
        "model": "claude-opus-5",
        "paid": True,
        "price_per_million_input_usd": 5.0,
        "price_per_million_output_usd": 25.0,
    },
}


def estimate_transcription_cost_usd(config_id: str, audio_seconds: float) -> float:
    cfg = TRANSCRIPTION_CONFIGS[config_id]
    if not cfg["paid"]:
        return 0.0
    return (audio_seconds / 60.0) * cfg["price_per_minute_usd"]


def estimate_llm_cost_usd(config_id: str, input_tokens: int, output_tokens: int) -> float:
    cfg = LLM_GROUPING_CONFIGS[config_id]
    if not cfg["paid"]:
        return 0.0
    return (
        (input_tokens / 1_000_000.0) * cfg["price_per_million_input_usd"]
        + (output_tokens / 1_000_000.0) * cfg["price_per_million_output_usd"]
    )
