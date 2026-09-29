"""Step 1: transcribe every test clip with each candidate transcription config.

Usage:
    python -m benchmark.run_transcription --configs whisper-medium,whisper-large-v3
    python -m benchmark.run_transcription --configs openai-whisper-1 --confirm

Local (free) configs run without a confirmation gate. Paid (hosted) configs
print an estimated cost and require --confirm before making any API call.
"""
import argparse
import json
import time

from .config import (
    LOCAL_TRANSCRIBE_OUTPUT_DIR,
    TEST_CLIPS,
    TRANSCRIPTION_CONFIGS,
    TRANSCRIPTS_DIR,
    estimate_transcription_cost_usd,
)
from .cost import audio_duration_seconds
from .media_prep import prepared_clip_path


def _meta_path(config_id: str, clip_id: str):
    return TRANSCRIPTS_DIR / config_id / f"{clip_id}.meta.json"


def _transcript_path(config_id: str, clip_id: str):
    return TRANSCRIPTS_DIR / config_id / f"{clip_id}.json"


def _run_local(config_id: str, cfg: dict, clip_id: str, source_path: str) -> None:
    # transcriber.py reads LOCAL_WHISPER_MODEL and LOCAL_OUTPUT_DIR as
    # module-level constants imported by value (`from ..config import ...`),
    # not live env lookups -- so varying them across configs means
    # reassigning the attributes on the *importing* module, transcriber
    # itself, not shorts_generator.config (confirmed by reading
    # transcriber.py). LOCAL_OUTPUT_DIR in particular MUST be patched: it is
    # where transcribe_local() writes its own {stem}.srt/.words.json cache
    # (keyed only by filename stem, not config), and left unpatched it
    # writes straight into the real production `output/` directory.
    import shorts_generator.local.transcriber as transcriber_mod

    dest_path = prepared_clip_path(config_id, clip_id, source_path)

    original_model = transcriber_mod.LOCAL_WHISPER_MODEL
    original_output_dir = transcriber_mod.LOCAL_OUTPUT_DIR
    transcriber_mod.LOCAL_WHISPER_MODEL = cfg["whisper_model"]
    transcriber_mod.LOCAL_OUTPUT_DIR = str(LOCAL_TRANSCRIBE_OUTPUT_DIR / config_id)
    try:
        start = time.monotonic()
        transcript = transcriber_mod.transcribe_local(str(dest_path))
        elapsed = time.monotonic() - start
    finally:
        transcriber_mod.LOCAL_WHISPER_MODEL = original_model
        transcriber_mod.LOCAL_OUTPUT_DIR = original_output_dir

    _transcript_path(config_id, clip_id).write_text(
        json.dumps(transcript, ensure_ascii=False), encoding="utf-8"
    )
    _meta_path(config_id, clip_id).write_text(
        json.dumps({
            "elapsed_seconds": elapsed,
            "estimated_cost_usd": 0.0,
            "model": cfg["whisper_model"],
            "duration_seconds": transcript.get("duration", 0.0),
        }),
        encoding="utf-8",
    )
    print(f"  [{config_id}] {clip_id}: {elapsed:.1f}s wall, {transcript.get('duration', 0.0):.0f}s audio")


def _openai_hosted_transcript_to_schema(response, audio_seconds: float) -> dict:
    """Converts an OpenAI verbose_json transcription response into the same
    {"duration", "segments":[{"start","end","text","words":[...]}]} shape
    transcribe_local() returns, so downstream scoring/grouping code (which
    already consumes that shape) doesn't need a second code path."""
    segments_out = []
    raw_segments = getattr(response, "segments", None) or []
    raw_words = getattr(response, "words", None) or []

    def words_in(seg_start, seg_end):
        return [
            {"start": float(w.start), "end": float(w.end), "word": (w.word or "").strip()}
            for w in raw_words
            if seg_start <= float(w.start) < seg_end
        ]

    if raw_segments:
        for seg in raw_segments:
            seg_start, seg_end = float(seg.start), float(seg.end)
            segments_out.append({
                "start": seg_start,
                "end": seg_end,
                "text": (seg.text or "").strip(),
                "words": words_in(seg_start, seg_end),
            })
    elif raw_words:
        # No segment-level breakdown returned -- treat the whole clip as one
        # segment so build_ass/_llm_group_words still have something to chew.
        segments_out.append({
            "start": float(raw_words[0].start),
            "end": float(raw_words[-1].end),
            "text": (getattr(response, "text", "") or "").strip(),
            "words": [{"start": float(w.start), "end": float(w.end), "word": (w.word or "").strip()} for w in raw_words],
        })
    else:
        # No word timestamps available at all (seen on some models/response
        # formats) -- record plain text with no words. WER can still run on
        # the text; grouping/caption scoring cannot.
        segments_out.append({
            "start": 0.0,
            "end": audio_seconds,
            "text": (getattr(response, "text", "") or "").strip(),
            "words": [],
        })

    return {"duration": audio_seconds, "segments": segments_out}


def _run_openai_hosted(config_id: str, cfg: dict, clip_id: str, source_path: str, confirmed: bool) -> None:
    from openai import OpenAI
    from shorts_generator.config import require_openai_key

    dest_path = prepared_clip_path(config_id, clip_id, source_path)
    audio_seconds = audio_duration_seconds(str(dest_path))
    est_cost = estimate_transcription_cost_usd(config_id, audio_seconds)

    print(f"  [{config_id}] {clip_id}: est. cost ${est_cost:.4f} for {audio_seconds:.0f}s audio")
    if not confirmed:
        print(f"  [{config_id}] {clip_id}: SKIPPED (pass --confirm to run paid transcription configs)")
        return

    client = OpenAI(api_key=require_openai_key())
    model = cfg["openai_model"]
    start = time.monotonic()
    try:
        with open(dest_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
    except Exception as e:
        # gpt-4o-transcribe (and possibly future hosted models) may reject
        # word/segment timestamp granularities depending on API version --
        # fall back to plain text so the run doesn't hard-fail, but flag it
        # loudly since word timestamps are what the whole pipeline needs.
        print(f"  [{config_id}] {clip_id}: verbose_json/timestamps request failed ({e!r}); retrying as plain text")
        with open(dest_path, "rb") as f:
            response = client.audio.transcriptions.create(model=model, file=f)
    elapsed = time.monotonic() - start

    transcript = _openai_hosted_transcript_to_schema(response, audio_seconds)
    _transcript_path(config_id, clip_id).write_text(
        json.dumps(transcript, ensure_ascii=False), encoding="utf-8"
    )
    _meta_path(config_id, clip_id).write_text(
        json.dumps({
            "elapsed_seconds": elapsed,
            "estimated_cost_usd": est_cost,
            "model": model,
            "duration_seconds": audio_seconds,
        }),
        encoding="utf-8",
    )
    print(f"  [{config_id}] {clip_id}: {elapsed:.1f}s wall, ${est_cost:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True, help="comma-separated config ids from TRANSCRIPTION_CONFIGS")
    parser.add_argument("--confirm", action="store_true", help="required to actually run paid configs")
    args = parser.parse_args()

    config_ids = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in config_ids if c not in TRANSCRIPTION_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown transcription config id(s): {unknown}. Known: {list(TRANSCRIPTION_CONFIGS)}")

    for config_id in config_ids:
        cfg = TRANSCRIPTION_CONFIGS[config_id]
        print(f"=== {config_id} ({cfg['kind']}) ===")
        for clip_id, source_path in TEST_CLIPS:
            if cfg["kind"] == "local":
                _run_local(config_id, cfg, clip_id, source_path)
            elif cfg["kind"] == "openai-hosted":
                _run_openai_hosted(config_id, cfg, clip_id, source_path, args.confirm)
            else:
                raise ValueError(f"Unknown config kind {cfg['kind']!r}")


if __name__ == "__main__":
    main()
