"""End-to-end orchestrator.

Two modes:
  * mode="api"   (default) — MuAPI does download / transcribe / LLM / autocrop.
                              Fast, no local deps, pay-per-call.
  * mode="local"            — yt-dlp + faster-whisper + OpenAI or Gemini + ffmpeg/opencv.
                              Self-hosted, LLM_PROVIDER selects OpenAI or Gemini.
"""
from typing import Dict, List, Optional

from .clipper import crop_highlights
from .downloader import download_youtube
from .highlights import call_muapi_llm, get_highlights
from .transcriber import transcribe


def _run_local(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
) -> Dict:
    from .local.downloader import download_youtube_local
    from .local.drive import (
        download_drive_file,
        download_drive_folder,
        is_drive_folder_url,
        is_drive_url,
    )

    if is_drive_folder_url(youtube_url):
        video_paths = download_drive_folder(youtube_url)
        results = []
        for path in video_paths:
            print(f"[pipeline/local] === processing {path} ===", flush=True)
            try:
                results.append(_process_single_local(path, num_clips, aspect_ratio, language))
            except Exception as e:
                print(f"[pipeline/local] failed for {path}: {e}", flush=True)
                results.append({"mode": "local", "source_video_url": path, "error": str(e)})
        return {"mode": "local", "batch": True, "source": youtube_url, "results": results}

    if is_drive_url(youtube_url):
        source_path = download_drive_file(youtube_url)
    else:
        source_path = download_youtube_local(youtube_url, fmt=download_format)

    return _process_single_local(source_path, num_clips, aspect_ratio, language)


def _process_single_local(
    source_path: str,
    num_clips: int,
    aspect_ratio: str,
    language: Optional[str],
) -> Dict:
    import os

    from .local.clipper import crop_highlights_local
    from .local.llm import call_local_llm
    from .local.review import review_transcript
    from .local.subtitles import burn_subtitles, words_for_window
    from .local.transcriber import transcribe_local
    from .config import (
        LOCAL_DIARIZATION_ENABLED,
        LOCAL_SUBTITLES_ENABLED,
        LOCAL_TRIM_FILLERS_ENABLED,
        LOCAL_TRIM_SILENCE_ENABLED,
    )

    transcript = transcribe_local(source_path, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    transcript = review_transcript(source_path, transcript, llm_fn=call_local_llm)

    if LOCAL_DIARIZATION_ENABLED:
        from .local.diarizer import diarize_and_label
        transcript = diarize_and_label(source_path, transcript)

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_local_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: float(h.get("combined_score", h.get("score", 0))), reverse=True)[:num_clips]
    print(f"[pipeline/local] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights_local(
        source_path, top, aspect_ratio=aspect_ratio,
        transcript_segments=transcript["segments"],
        trim_silence=LOCAL_TRIM_SILENCE_ENABLED,
        trim_fillers=LOCAL_TRIM_FILLERS_ENABLED,
    )

    if LOCAL_SUBTITLES_ENABLED:
        for short in shorts:
            clip_path = short.get("clip_url")
            if not clip_path or not os.path.exists(clip_path):
                continue
            words = short.get("words")
            if words is None:
                words = words_for_window(
                    transcript["segments"], float(short["start_time"]), float(short["end_time"])
                )
            print(f"[subtitles/local] burning {len(words)} words onto {clip_path}", flush=True)
            tmp_path = clip_path + ".subbed.mp4"
            try:
                burn_subtitles(clip_path, words, tmp_path)
                os.replace(tmp_path, clip_path)
            except Exception as e:
                print(f"[subtitles/local] failed for {clip_path}: {e}", flush=True)
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    return {
        "mode": "local",
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def _run_api(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
) -> Dict:
    source_url = download_youtube(youtube_url, fmt=download_format)

    transcript = transcribe(source_url, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_muapi_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: float(h.get("combined_score", h.get("score", 0))), reverse=True)[:num_clips]
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights(source_url, top, aspect_ratio=aspect_ratio)

    return {
        "mode": "api",
        "source_video_url": source_url,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    mode: str = "api",
) -> Dict:
    """Run the full pipeline and return a structured result.

    Args:
        youtube_url: source URL.
        num_clips: how many shorts to render.
        aspect_ratio: e.g. "9:16", "1:1".
        download_format: source resolution ("360" / "480" / "720" / "1080").
        language: ISO-639-1 to force Whisper language detection.
        mode: "api" (default, MuAPI) or "local" (yt-dlp + faster-whisper +
            OpenAI or Gemini + ffmpeg).

    Returns:
        {
          "mode": "api" | "local",
          "source_video_url": str,   # hosted URL (api) or local path (local)
          "transcript": {...},
          "highlights": [...],       # all candidates ranked
          "shorts": [...],           # top `num_clips` with clip_url / local path
        }
    """
    mode = (mode or "api").lower()
    if mode == "local":
        return _run_local(youtube_url, num_clips, aspect_ratio, download_format, language)
    if mode == "api":
        return _run_api(youtube_url, num_clips, aspect_ratio, download_format, language)
    raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.")
