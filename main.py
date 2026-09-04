"""CLI entry point.

Usage:
    python main.py "https://www.youtube.com/watch?v=..." \
        --num-clips 3 --aspect-ratio 9:16
"""
import argparse
import json
import sys

# Windows uses 'charmap' by default, which can't encode Unicode characters
# like →. Reconfigure stdout/stderr to UTF-8 so output works on all platforms.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from shorts_generator import generate_shorts


def _print_single_result(result: dict, header: bool = False) -> None:
    if header:
        print("\n" + "=" * 72)
        print(f"Mode:          {result.get('mode')}")
    else:
        print("\n" + "-" * 72)
    print(f"Source video:  {result['source_video_url']}")

    if result.get("error"):
        print(f"FAILED:        {result['error']}")
        return

    print(f"Highlights:    {len(result['highlights'])} candidates → kept top {len(result['shorts'])}")
    if header:
        print("=" * 72)
    for i, s in enumerate(result["shorts"], 1):
        print(f"\n#{i}  combined={s.get('combined_score')}  virality={s.get('score')}  sales_angle_score={s.get('sales_angle_score')}  {s.get('start_time'):.1f}s → {s.get('end_time'):.1f}s")
        print(f"     title:       {s.get('title')}")
        print(f"     hook:        {s.get('hook_sentence')}")
        if s.get("sales_angle"):
            print(f"     sales angle: {s.get('sales_angle')}")
        if s.get("central_theme"):
            print(f"     theme:       {s.get('central_theme')}")
        if s.get("transcript"):
            print(f"     transcript:  {s.get('transcript')}")
        if s.get("clip_url"):
            print(f"     clip:   {s['clip_url']}")
        else:
            print(f"     clip:   FAILED ({s.get('error')})")


def main() -> int:
    parser = argparse.ArgumentParser(description="AI YouTube Shorts Generator")
    parser.add_argument(
        "url",
        help="YouTube URL, Google Drive file/folder link ('Anyone with the link'), file:// URL, or local file path",
    )
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="api (default, MuAPI) or local (remote URL, file://, or local path + faster-whisper + LLM provider + ffmpeg).",
    )
    parser.add_argument("--num-clips", type=int, default=3, help="How many shorts to render (default: 3)")
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    parser.add_argument("--format", default="720", help="Source download resolution: 360 / 480 / 720 / 1080 (default: 720)")
    parser.add_argument("--language", default=None, help="Force Whisper language code, e.g. 'en' (default: auto-detect)")
    parser.add_argument("--output-json", default=None, help="Write the full result JSON to this path")
    args = parser.parse_args()

    try:
        result = generate_shorts(
            youtube_url=args.url,
            num_clips=args.num_clips,
            aspect_ratio=args.aspect_ratio,
            download_format=args.format,
            language=args.language,
            mode=args.mode,
        )
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1

    if result.get("batch"):
        video_results = result["results"]
        print("\n" + "=" * 72)
        print(f"Mode:          local (batch, {len(video_results)} video(s) from Drive folder)")
        print("=" * 72)
        for video_result in video_results:
            _print_single_result(video_result)
    else:
        _print_single_result(result, header=True)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
