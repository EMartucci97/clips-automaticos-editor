"""Local transcription via mlx-whisper (GPU Apple) o faster-whisper (CPU/CUDA).

Reads a local media file and returns the same shape the highlight generator
expects: {duration, segments[start, end, text]}. Each segment also carries
word-level timestamps (segment["words"] = [{start, end, word}, ...]) used
downstream to burn word-by-word subtitles onto the rendered clips.
"""
import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, Optional

from ..config import (
    LOCAL_OUTPUT_DIR,
    LOCAL_WHISPER_DEVICE,
    LOCAL_WHISPER_INITIAL_PROMPT,
    LOCAL_WHISPER_LANGUAGE,
    LOCAL_WHISPER_MODEL,
)


def _transcript_cache_path(media_path: str) -> Path:
    """Return the .srt cache path for a media file."""
    cache_dir = Path(LOCAL_OUTPUT_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (Path(media_path).stem + ".srt")


def _words_cache_path(media_path: str) -> Path:
    """Return the sidecar .words.json cache path (word-level timestamps)."""
    cache_dir = Path(LOCAL_OUTPUT_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (Path(media_path).stem + ".words.json")


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_srt_timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, millis = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + (millis / 1000.0)


def _write_srt_cache(media_path: str, transcript: Dict) -> Path:
    cache_path = _transcript_cache_path(media_path)
    lines = []
    for idx, segment in enumerate(transcript.get("segments", []), start=1):
        start = _format_srt_timestamp(float(segment["start"]))
        end = _format_srt_timestamp(float(segment["end"]))
        text = str(segment.get("text", "")).strip().replace("\r", "").replace("\n", " ")
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    cache_path.write_text("\n".join(lines), encoding="utf-8")
    return cache_path


def _load_srt_cache(cache_path: Path) -> Dict:
    content = cache_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return {"duration": 0.0, "segments": []}

    segments = []
    for block in re.split(r"\n\s*\n", content):
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[0].split("-->", 1)]
        text = "\n".join(lines[1:]).strip()
        segments.append(
            {
                "start": _parse_srt_timestamp(start_raw),
                "end": _parse_srt_timestamp(end_raw),
                "text": text,
            }
        )

    duration = segments[-1]["end"] if segments else 0.0
    return {"duration": duration, "segments": segments}


def _resolve_device() -> str:
    if LOCAL_WHISPER_DEVICE != "auto":
        return LOCAL_WHISPER_DEVICE
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            # Test that CUDA actually works (catches missing cuBLAS/cuDNN libs)
            torch.zeros(1, device="cuda")
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu"


def _dedupe_word_timestamps(words: list, min_dur: float = 0.02) -> None:
    """faster-whisper's word-level alignment can collapse two or more
    consecutive words onto the exact same (start, end) span -- typically
    zero-duration -- when speech is fast enough (seen e.g. on "bien.
    ¿Vendías" both landing on 181.24:181.24). Anything downstream that
    identifies a word by its own (start, end) -- notably the editor's manual
    transcript_overrides, keyed by `"start:end"`, and the clip-cuts remapper
    (`editing.remap_words_across_cuts`, which drops any word whose span
    collapses to zero duration) -- then either can't tell the words apart or
    silently discards one outright. Enforce strictly increasing,
    non-overlapping timestamps in-place so every word keeps a unique,
    non-zero-duration key.

    Must run over the FULL, flattened word list (all segments concatenated,
    in order) rather than one segment at a time: a zero-duration word can be
    the very FIRST word of its segment (so it has no "previous word" to
    compare against within that segment -- exactly what happened with
    "¿Vendías" opening the segment right after "bien." closed the one
    before it), in which case a per-segment pass silently skips it."""
    for i, cur in enumerate(words):
        if i > 0:
            prev = words[i - 1]
            if cur["start"] < prev["end"]:
                cur["start"] = prev["end"]
        if cur["end"] <= cur["start"]:
            cur["end"] = cur["start"] + min_dur


# Mismo modelo (whisper medium) corriendo en la GPU de Apple vía MLX: ~10x más
# rápido que faster-whisper int8 en CPU (3 min de podcast: 68s vs 775s en un
# M4) con timestamps de palabra casi idénticos (mediana 0.02s de diferencia).
# large-v3-turbo se probó y corre los timestamps hasta 0.5s: descartado.
# LOCAL_WHISPER_ENGINE=faster-whisper vuelve al motor anterior.
_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


# Transcripciones MLX en curso: el reaper de editor_server no suelta el
# modelo mientras sea > 0.
_mlx_lock = threading.Lock()
_mlx_active = 0


def _use_mlx() -> bool:
    engine = os.getenv("LOCAL_WHISPER_ENGINE", "auto").strip().lower()
    if engine == "faster-whisper" or LOCAL_WHISPER_DEVICE == "cuda":
        return False
    if LOCAL_WHISPER_MODEL not in _MLX_REPOS:
        return False
    import platform
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        if engine == "mlx":
            raise RuntimeError("LOCAL_WHISPER_ENGINE=mlx pero mlx-whisper no está instalado")
        return False
    return True


def _transcribe_mlx(media_path: str, language: Optional[str]):
    import mlx_whisper
    repo = _MLX_REPOS[LOCAL_WHISPER_MODEL]
    print(f"[transcribe/local] mlx-whisper model={repo} (GPU)", flush=True)
    # Sin beam search (mlx-whisper no lo implementa): greedy dio el mismo
    # texto que beam_size=5 salvo palabras sueltas dudosas. Sin VAD, igual
    # que el default LOCAL_WHISPER_VAD_FILTER=false de faster-whisper.
    global _mlx_active
    with _mlx_lock:
        _mlx_active += 1
    try:
        result = mlx_whisper.transcribe(
            media_path,
            path_or_hf_repo=repo,
            language=language,
            condition_on_previous_text=False,
            word_timestamps=True,
            initial_prompt=LOCAL_WHISPER_INITIAL_PROMPT or None,
        )
    finally:
        with _mlx_lock:
            _mlx_active -= 1
    segments, words_by_segment = [], []
    for s in result.get("segments", []):
        segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": (s.get("text") or "").strip()})
        words_by_segment.append([
            {"start": float(w["start"]), "end": float(w["end"]), "word": w["word"].strip()}
            for w in s.get("words", [])
        ])
    # Duración real del archivo (el último segmento termina antes si el
    # video cierra en silencio); faster-whisper la daba en info.duration.
    import subprocess
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", media_path],
        capture_output=True, text=True,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        duration = 0.0
    return segments, words_by_segment, duration


def unload_mlx_model() -> bool:
    """Suelta el modelo que mlx-whisper deja cacheado a nivel de clase.
    No hace nada mientras haya una transcripción en curso."""
    try:
        from mlx_whisper.transcribe import ModelHolder
    except ImportError:
        return False
    with _mlx_lock:
        if _mlx_active:
            return False
    if ModelHolder.model is None:
        return False
    ModelHolder.model = None
    ModelHolder.model_path = None
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass
    return True


def _transcribe_faster_whisper(media_path: str, language: Optional[str]):
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[transcribe/local] faster-whisper model={LOCAL_WHISPER_MODEL} device={device}", flush=True)

    from ..config import LOCAL_WHISPER_VAD_FILTER, LOCAL_WHISPER_VAD_PARAMETERS

    model = WhisperModel(LOCAL_WHISPER_MODEL, device=device, compute_type=compute_type)

    transcribe_kwargs = {
        "audio": media_path,
        "language": language,
        "beam_size": 5,
        "condition_on_previous_text": False,
        "word_timestamps": True,
    }
    if LOCAL_WHISPER_INITIAL_PROMPT:
        transcribe_kwargs["initial_prompt"] = LOCAL_WHISPER_INITIAL_PROMPT
    if LOCAL_WHISPER_VAD_FILTER:
        transcribe_kwargs["vad_filter"] = True
        transcribe_kwargs["vad_parameters"] = LOCAL_WHISPER_VAD_PARAMETERS
    else:
        transcribe_kwargs["vad_filter"] = False

    segments_iter, info = model.transcribe(**transcribe_kwargs)

    segments = []
    words_by_segment = []
    for s in segments_iter:
        segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip(),
        })
        words = [
            {"start": float(w.start), "end": float(w.end), "word": w.word.strip()}
            for w in (s.words or [])
        ]
        words_by_segment.append(words)
    return segments, words_by_segment, float(getattr(info, "duration", 0.0))


def transcribe_local(media_path: str, language: Optional[str] = None) -> Dict:
    """Run faster-whisper on a local file path, caching the result as .srt
    (+ a sidecar .words.json with word-level timestamps for subtitles)."""
    language = language or LOCAL_WHISPER_LANGUAGE
    cache_path = _transcript_cache_path(media_path)
    words_path = _words_cache_path(media_path)
    if cache_path.exists():
        source_mtime = os.path.getmtime(media_path)
        cache_mtime = cache_path.stat().st_mtime
        if cache_mtime >= source_mtime:
            print(f"[transcribe/local] reusing cached transcript: {cache_path}", flush=True)
            cached = _load_srt_cache(cache_path)
            # Treat empty cache as invalid (likely from a failed/partial run) — delete and re-transcribe
            if not cached["segments"] or cached["duration"] <= 0.0:
                print(f"[transcribe/local] cache is empty/invalid, deleting: {cache_path}", flush=True)
                cache_path.unlink(missing_ok=True)
                words_path.unlink(missing_ok=True)
            else:
                if words_path.exists():
                    try:
                        words_by_segment = json.loads(words_path.read_text(encoding="utf-8"))
                        for seg, words in zip(cached["segments"], words_by_segment):
                            seg["words"] = words
                    except (json.JSONDecodeError, OSError):
                        pass
                print(
                    f"[transcribe/local] {len(cached['segments'])} cached segments, "
                    f"{cached['duration']:.0f}s of audio",
                    flush=True,
                )
                return cached

    if _use_mlx():
        segments, words_by_segment, duration = _transcribe_mlx(media_path, language)
    else:
        segments, words_by_segment, duration = _transcribe_faster_whisper(media_path, language)

    # Dedupe across the whole transcript at once (not per-segment) so a
    # collision spanning a segment boundary -- like a zero-duration word
    # opening a new segment right after the previous segment's last word --
    # still gets caught; see _dedupe_word_timestamps's docstring.
    _dedupe_word_timestamps([w for words in words_by_segment for w in words])

    duration = duration or (segments[-1]["end"] if segments else 0.0)
    print(f"[transcribe/local] {len(segments)} segments, {duration:.0f}s of audio", flush=True)
    transcript = {"duration": duration, "segments": segments}
    cache_path = _write_srt_cache(media_path, transcript)
    words_path.write_text(json.dumps(words_by_segment), encoding="utf-8")
    print(f"[transcribe/local] wrote cache: {cache_path}", flush=True)
    for seg, words in zip(segments, words_by_segment):
        seg["words"] = words
    return transcript
