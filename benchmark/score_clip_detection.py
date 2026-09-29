"""Step 3 (optional): qualitative clip/topic-detection spot-check.

There is no automatic ground truth for "is this a good clip boundary" (no
precision/recall target exists anywhere in the codebase), so this is
explicitly qualitative: for each transcription config, run the same
single-window segment-analysis prompt production uses
(SEGMENT_ANALYSIS_PROMPT / _transcript_lines_for_range / _sanitize_subtopic_clips
from editor_server.py) against the whole test clip, holding the LLM config
fixed (gpt-4o-mini, so only the transcription axis varies), and print the
resulting central_theme + subtopic boundaries side by side per config. A
human (the user) judges by eye whether a worse transcription produces
visibly worse topic boundaries.

Test clips here are short (~90-103s), well under editor_server's windowing
threshold (FULL_VIDEO_WINDOW_SECONDS=900s), so a single analyze_segment-style
call over the whole clip is equivalent to what production would do for a
clip this short -- no need to reimplement the multi-window full-video path.
"""
import argparse
import json

from .config import CAPTIONS_DIR, CLIP_DETECTION_DIR, TEST_CLIPS, TRANSCRIPTS_DIR

SPOT_CHECK_LLM_PROVIDER = "openai"
SPOT_CHECK_LLM_MODEL = "gpt-4o-mini"


def _run_one(config_id: str, clip_id: str, confirmed: bool) -> None:
    import editor_server
    from .cost import get_usage, track_llm_usage

    transcript_path = TRANSCRIPTS_DIR / config_id / f"{clip_id}.json"
    if not transcript_path.exists():
        print(f"  SKIP {clip_id}: no transcript at {transcript_path}")
        return
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    duration = editor_server._transcript_duration(transcript)
    if duration <= 0:
        print(f"  SKIP {clip_id}: transcript has no duration")
        return

    if not confirmed:
        print(f"  [{config_id}] {clip_id}: requires --confirm (makes a real {SPOT_CHECK_LLM_MODEL} call)")
        return

    timed_text = editor_server._transcript_lines_for_range(transcript, 0.0, duration)
    prompt = editor_server.SEGMENT_ANALYSIS_PROMPT.format(text=timed_text, start=0.0, end=duration)

    with track_llm_usage(SPOT_CHECK_LLM_PROVIDER, SPOT_CHECK_LLM_MODEL):
        from .cost import _call_openai_with_usage
        raw = _call_openai_with_usage(prompt, SPOT_CHECK_LLM_MODEL)
        usage = get_usage()

    parsed = editor_server._parse_json_loose(raw)
    result = {
        "central_theme": str(parsed.get("central_theme", "")).strip(),
        "subtopics": editor_server._sanitize_subtopic_clips(parsed.get("subtopics"), 0.0, duration),
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
    }

    out_dir = CLIP_DETECTION_DIR / config_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{clip_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  [{config_id}] {clip_id}: tema='{result['central_theme']}'")
    for st in result["subtopics"]:
        print(f"      [{st['start_time']:.1f}-{st['end_time']:.1f}s] {st['text']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True, help="comma-separated transcription config ids to compare")
    parser.add_argument("--confirm", action="store_true", help="required to actually make LLM calls")
    args = parser.parse_args()

    config_ids = [c.strip() for c in args.configs.split(",") if c.strip()]
    for config_id in config_ids:
        print(f"=== {config_id} ===")
        for clip_id, _ in TEST_CLIPS:
            _run_one(config_id, clip_id, args.confirm)


if __name__ == "__main__":
    main()
