"""Step 2: run cleanup + grouping over each transcript with each candidate
LLM config (or the zero-cost mechanical fallback).

Usage:
    python -m benchmark.run_grouping --transcription whisper-medium --llm mechanical,openai-gpt-4o-mini
    python -m benchmark.run_grouping --transcription whisper-medium --llm anthropic-opus-5 --confirm

Reuses the real production functions (`_llm_cleanup_words`, `_llm_group_words`,
`_group_boundaries` from shorts_generator/local/subtitles.py) so results
reflect the real pipeline. Never touches `output/.llm_cache/` -- redirects
`subtitles._LLM_CACHE_DIR` to a per-llm-config directory under
`benchmark/work/llm_cache/` for the duration of each combo, since the real
cache key does not include provider/model and reusing one shared directory
across configs would silently serve one config's cached answer to another.
"""
import argparse
import json
import time

from .config import (
    CAPTIONS_DIR,
    LLM_CACHE_DIR,
    LLM_GROUPING_CONFIGS,
    TEST_CLIPS,
    TRANSCRIPTS_DIR,
    estimate_llm_cost_usd,
)
from .cost import get_usage, track_llm_usage


def flatten_words(transcript: dict) -> list:
    words = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []) or []:
            words.append({
                "start": float(w["start"]),
                "end": float(w["end"]),
                "word": w["word"],
                "speaker": None,
                "word_start": float(w["start"]),
                "word_end": float(w["end"]),
            })
    return words


def _run_combo(transcription_config: str, llm_config_id: str, clip_id: str, confirmed: bool) -> None:
    import shorts_generator.local.subtitles as subtitles_mod
    from shorts_generator.local.subtitles import DEFAULT_WORDS_PER_GROUP, _group_boundaries

    transcript_path = TRANSCRIPTS_DIR / transcription_config / f"{clip_id}.json"
    if not transcript_path.exists():
        print(f"  SKIP {clip_id}: no transcript at {transcript_path} (run run_transcription first)")
        return
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    words = flatten_words(transcript)
    if not words:
        print(f"  SKIP {clip_id}: transcript has no word-level timestamps")
        return

    combo_id = f"{transcription_config}__{llm_config_id}"
    out_dir = CAPTIONS_DIR / combo_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = LLM_GROUPING_CONFIGS[llm_config_id]
    if cfg["kind"] == "mechanical":
        start = time.monotonic()
        groups = _group_boundaries(words, DEFAULT_WORDS_PER_GROUP)
        elapsed = time.monotonic() - start
        cleaned_words = words
        usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        est_cost = 0.0
    else:
        if cfg["paid"] and not confirmed:
            print(f"  [{llm_config_id}] {clip_id}: est. cost requires --confirm to run (pass --confirm)")
            return
        original_cache_dir = subtitles_mod._LLM_CACHE_DIR
        subtitles_mod._LLM_CACHE_DIR = LLM_CACHE_DIR / llm_config_id
        try:
            with track_llm_usage(cfg["provider"], cfg["model"]):
                start = time.monotonic()
                cleaned_words = subtitles_mod._llm_cleanup_words(words)
                groups = subtitles_mod._llm_group_words(cleaned_words, DEFAULT_WORDS_PER_GROUP)
                elapsed = time.monotonic() - start
                usage = get_usage()
        finally:
            subtitles_mod._LLM_CACHE_DIR = original_cache_dir
        est_cost = estimate_llm_cost_usd(llm_config_id, usage["input_tokens"], usage["output_tokens"])

    (out_dir / f"{clip_id}.json").write_text(
        json.dumps({"words": cleaned_words, "groups": groups}, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / f"{clip_id}.meta.json").write_text(
        json.dumps({
            "elapsed_seconds": elapsed,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "llm_calls": usage["calls"],
            "estimated_cost_usd": est_cost,
        }),
        encoding="utf-8",
    )
    print(
        f"  [{llm_config_id}] {clip_id}: {len(groups)} groups, {elapsed:.1f}s, "
        f"${est_cost:.4f} ({usage['input_tokens']}in/{usage['output_tokens']}out tok)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcription", required=True, help="transcription config id whose transcripts to group")
    parser.add_argument("--llm", required=True, help="comma-separated LLM config ids from LLM_GROUPING_CONFIGS")
    parser.add_argument("--confirm", action="store_true", help="required to actually run paid LLM configs")
    args = parser.parse_args()

    llm_ids = [c.strip() for c in args.llm.split(",") if c.strip()]
    unknown = [c for c in llm_ids if c not in LLM_GROUPING_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown LLM config id(s): {unknown}. Known: {list(LLM_GROUPING_CONFIGS)}")

    for llm_id in llm_ids:
        print(f"=== transcription={args.transcription} llm={llm_id} ===")
        for clip_id, _ in TEST_CLIPS:
            _run_combo(args.transcription, llm_id, clip_id, args.confirm)


if __name__ == "__main__":
    main()
