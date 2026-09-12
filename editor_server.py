"""Standalone timeline editor: load the full source video, show LLM-detected
highlight zones as draggable markers on a timeline. Phase 1 only — no
silence-removal or format/render functions yet (see plan for later phases).

Usage:
    python3 editor_server.py <source_video_path> [--port 8502] [--num-clips 20]
"""
import argparse
import json
import os
import re
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, abort

from shorts_generator.config import LOCAL_OUTPUT_DIR
from shorts_generator.highlights import _parse_json_loose, get_highlights
from shorts_generator.local import clipper, editing
from shorts_generator.local.diarizer import (
    OTHER_LABEL,
    ROBERTO_LABEL,
    _extract_audio_wav,
    diarize_and_label,
)
from shorts_generator.local.llm import call_local_llm
from shorts_generator.local.subtitles import (
    CAPTION_COLORS,
    CAPTION_POSITIONS,
    CAPTION_SIZES,
    CAPTION_STYLES,
    DEFAULT_CAPTION_COLOR,
    DEFAULT_CAPTION_POSITION,
    DEFAULT_CAPTION_STYLE,
    DEFAULT_FONT_KEY,
    DEFAULT_WORDS_PER_GROUP,
    FONT_CHOICES,
    _llm_cleanup_words,
    _llm_group_words,
    burn_subtitles,
    words_for_window,
)
from shorts_generator.local.transcriber import transcribe_local

EXPORT_ASPECT_RATIO = "9:16"

# Vertical framing modes offered by the editor's "Encuadre" panel:
# "single" = one 9:16 rectangle, "split" = two rectangles stacked 50/50
# (top camera / bottom camera), both cropped out of the same wide source.
CROP_MODES = ("single", "split")
DEFAULT_CROP_MODE = "single"

MUSIC_DIR = Path(__file__).resolve().parent / "assets" / "music"
DEFAULT_MUSIC_VOLUME = 0.18
CTA_OUTRO_DURATION_SECONDS = 6.5
CTA_OUTRO_BLUR_SIGMA = 22.0

app = Flask(__name__)

SOURCE_PATH = None
NUM_CLIPS = 20

_TRANSCRIPT_CACHE = None
_DIARIZED_TRANSCRIPT_CACHE = None
_CLEANED_WORDS_CACHE: dict = {}

SEGMENT_ANALYSIS_PROMPT = """Sos un editor de video que analiza un tramo de transcripción de una entrevista/podcast en español.

Cada línea del texto está marcada con su timestamp absoluto en segundos dentro del video completo, con el formato "[123.4s] texto" (puede tener errores de transcripción, ignorá los que sean obvios). Tu tarea:
1. Identificá el TEMA CENTRAL del tramo completo: una frase corta y concreta.
2. Identificá los SUBTEMAS que se tocan dentro del tramo, en el orden en que aparecen. Si el tramo es corto u homogéneo, puede haber un solo subtema que ocupe todo el rango.
3. Cada subtema tiene que ser un CLIP POTENCIAL: tiene que durar COMO MÍNIMO 40 segundos y alcanzar a contar algo completo e interesante (una anécdota, una reflexión, un dato explicado con su contexto) — no un dato suelto ni una frase aislada. Si varias ideas cortas y relacionadas no llegan solas a los 40 segundos, agrupalas en un mismo subtema en vez de partirlas.
4. Para CADA subtema, indicá start_time y end_time (float, en segundos, timestamps absolutos tomados de las líneas): el tramo exacto donde se desarrolla esa idea, desde que arranca a tocarse hasta que termina — sin cortar una frase a la mitad. Los rangos deben estar dentro de [{start:.1f}, {end:.1f}] y no deben superponerse entre sí.

Devolvé SOLO este JSON, sin texto adicional ni markdown:
{{"central_theme":"string","subtopics":[{{"text":"string","start_time":float,"end_time":float}}]}}

Texto del tramo (con timestamps):
\"\"\"{text}\"\"\""""

# Overlapping fixed-size windows are a purely mechanical device to keep each
# LLM call within a manageable amount of text. They do NOT define clip
# boundaries, and coverage is not required — see FULL_VIDEO_TOPICS_PROMPT.
# The 5-minute overlap between consecutive windows exists so a clip near a
# window edge is still fully contained, with margin, in at least one
# window's interior.
FULL_VIDEO_WINDOW_SECONDS = 900.0
FULL_VIDEO_STRIDE_SECONDS = 600.0

FULL_VIDEO_TOPICS_PROMPT = """Sos un editor de video que busca CLIPS DE ALTA CALIDAD dentro de un tramo de la transcripción de una entrevista/podcast en español. Este tramo va de {start:.1f}s a {end:.1f}s dentro de un video más largo — no tenés el resto del video, solo tenés que ocuparte de este tramo.

Cada línea está marcada con su timestamp absoluto en segundos, formato "[123.4s] texto" (puede tener errores de transcripción, ignorá los obvios).

Tu tarea NO es cubrir el tramo de punta a punta ni dividirlo en partes consecutivas. Tu tarea es ENCONTRAR la mayor cantidad posible de clips genuinamente interesantes: cada clip es un diálogo o desarrollo de UN tema que se sostiene solo, con arranque y cierre de idea — no un dato suelto ni una frase cortada a la mitad.

Reglas:
1. Buscá TODOS los fragmentos interesantes del tramo, no solo el mejor. Maximizá la cantidad de clips encontrados: un tramo de 15 minutos de una buena entrevista normalmente tiene varios clips potenciales, no uno solo.
2. Cada clip tiene que durar COMO MÍNIMO 40 segundos.
3. start_time y end_time salen EXCLUSIVAMENTE de dónde arranca y termina la idea en el texto — nunca cortes una frase ni un pensamiento a la mitad. Si hace falta que el clip dure 90 segundos para cerrar la idea, que dure 90; no fuerces una duración fija.
4. Los clips PUEDEN superponerse entre sí, total o parcialmente — eso es esperado y deseable, no un error. Por ejemplo, un clip corto y filoso sobre un momento puntual y otro más largo que incluye ese mismo momento con más contexto alrededor son AMBOS válidos si cada uno funciona por separado. No fuerces que los clips sean consecutivos ni que no se toquen entre sí.
5. Ignorá sin culpa las partes sin sustancia (chicanas, silencios, cambios de tema sin desarrollo) — no hace falta asignarles ningún clip ni cubrir esos huecos.
6. Si en este tramo no hay nada que valga la pena, devolvé una lista vacía.

Para CADA clip encontrado indicá: title (frase corta y concreta que resuma el tema) y start_time / end_time (float, timestamps absolutos).

Devolvé SOLO este JSON, sin texto adicional ni markdown:
{{"clips":[{{"title":"string","start_time":float,"end_time":float}}]}}

Transcripción del tramo (con timestamps):
\"\"\"{text}\"\"\""""


_highlights_lock = threading.Lock()


def _highlights_cache_path(source_path: str) -> Path:
    stem = Path(source_path).stem
    return Path(LOCAL_OUTPUT_DIR) / f"{stem}.editor_highlights.json"


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON so a crash or a racing writer never leaves torn/partial bytes."""
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _load_or_build_highlights(source_path: str) -> list:
    cache_path = _highlights_cache_path(source_path)
    with _highlights_lock:
        if cache_path.exists():
            print(f"[editor] reusing cached highlights: {cache_path}", flush=True)
            with open(cache_path) as f:
                return json.load(f)

        transcript = transcribe_local(source_path)
        transcript = diarize_and_label(source_path, transcript)
        result = get_highlights(transcript, num_clips=NUM_CLIPS, llm_fn=call_local_llm)
        highlights = result.get("highlights", [])
        for i, h in enumerate(highlights):
            h["id"] = i

        os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
        _atomic_write_json(cache_path, highlights)
        print(f"[editor] wrote {len(highlights)} highlight marker(s) -> {cache_path}", flush=True)
        return highlights


@app.route("/")
def index():
    return send_file(Path(__file__).parent / "static" / "editor.html")


_PREVIEW_LOCK = threading.Lock()


def _preview_video_path(source_path: str) -> Path:
    """Browser-playable proxy for the source video.

    Chrome's <video> element cannot play `video/quicktime` at all
    (canPlayType('video/quicktime') === '') and this project's sources are
    10-bit HEVC .mov files, so serving SOURCE_PATH directly leaves the
    element stuck at readyState 0 forever: loadedmetadata never fires,
    the global `duration` stays 0, and every % computed from it in
    editor.html (zoom window, cut markers, playhead) collapses to 0 —
    cuts are detected and saved correctly but render invisibly at the
    timeline's left edge. Transcoding once to 8-bit H.264/AAC MP4 (cached
    next to the other per-video artifacts) fixes playback without
    touching SOURCE_PATH, which everything else (detection, export)
    keeps reading directly.
    """
    stem = Path(source_path).stem
    preview_path = Path(LOCAL_OUTPUT_DIR) / f"{stem}.preview.mp4"
    if preview_path.exists():
        return preview_path
    with _PREVIEW_LOCK:
        if preview_path.exists():
            return preview_path
        os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
        tmp_path = preview_path.with_name(preview_path.stem + ".tmp.mp4")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", source_path,
            "-c:v", "h264_videotoolbox", "-b:v", "8M",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(tmp_path),
        ]
        print(f"[editor] transcoding browser preview (one-time): {preview_path}", flush=True)
        subprocess.run(cmd, check=True)
        os.replace(tmp_path, preview_path)
        print(f"[editor] preview ready: {preview_path}", flush=True)
        return preview_path


@app.route("/video")
def video():
    return send_file(_preview_video_path(SOURCE_PATH), conditional=True)


def _waveform_cache_path(source_path: str) -> Path:
    stem = Path(source_path).stem
    return Path(LOCAL_OUTPUT_DIR) / f"{stem}.waveform.json"


def _build_waveform(source_path: str, samples_per_sec: float = 20.0) -> list:
    """Downsampled peak amplitudes (0-1) for drawing a waveform on the
    timeline. Reuses the mono 16kHz wav diarize_and_label already extracts
    and caches, so this is cheap after the first highlight analysis.

    Resolution is fixed per second of audio (not a fixed total point count)
    so that zoomed-in views of the timeline still show real waveform detail
    instead of a handful of blocky bars stretched across the whole file."""
    cache_path = _waveform_cache_path(source_path)
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    wav_path = _extract_audio_wav(source_path)
    with wave.open(wav_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

    audio_duration = n_frames / float(framerate) if framerate else 0.0
    num_points = max(1, int(audio_duration * samples_per_sec))

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if samples.size == 0:
        peaks = [0.0] * num_points
    else:
        chunk_size = max(1, len(samples) // num_points)
        trimmed = samples[: chunk_size * num_points]
        peak_vals = np.abs(trimmed.reshape(-1, chunk_size)).max(axis=1)
        peaks = (peak_vals / float(np.iinfo(dtype).max)).clip(0, 1).round(4).tolist()
        while len(peaks) < num_points:
            peaks.append(peaks[-1] if peaks else 0.0)

    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(peaks, f)
    return peaks


@app.route("/api/waveform", methods=["GET"])
def api_waveform():
    return jsonify(_build_waveform(SOURCE_PATH))


def _get_transcript(source_path: str) -> dict:
    """Full local transcript (with word-level timestamps), loaded once per
    server run. transcribe_local caches to disk itself, so this is cheap
    even the first time (highlight generation already ran it)."""
    global _TRANSCRIPT_CACHE
    if _TRANSCRIPT_CACHE is None:
        _TRANSCRIPT_CACHE = transcribe_local(source_path)
    return _TRANSCRIPT_CACHE


def _get_diarized_transcript(source_path: str) -> dict:
    """Transcript with each segment tagged by speaker (see diarizer.py),
    loaded once per server run. diarize_and_label's own diarization pass is
    disk-cached (.diar.json), so this is cheap even on first call here."""
    global _DIARIZED_TRANSCRIPT_CACHE
    if _DIARIZED_TRANSCRIPT_CACHE is None:
        _DIARIZED_TRANSCRIPT_CACHE = diarize_and_label(source_path, _get_transcript(source_path))
        _apply_transcript_overrides(_DIARIZED_TRANSCRIPT_CACHE, _load_overrides(source_path))
    return _DIARIZED_TRANSCRIPT_CACHE


def _overrides_cache_path(source_path: str) -> Path:
    stem = Path(source_path).stem
    return Path(LOCAL_OUTPUT_DIR) / f"{stem}.transcript_overrides.json"


def _load_overrides(source_path: str) -> dict:
    cache_path = _overrides_cache_path(source_path)
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {"words": {}}


def _save_overrides(source_path: str, overrides: dict) -> None:
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    dest = _overrides_cache_path(source_path)
    # Write-then-rename so a concurrent request reading the file never sees a
    # half-written/truncated JSON body (os.replace is atomic on POSIX and Windows).
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(overrides, f, indent=2, ensure_ascii=False)
    os.replace(tmp, dest)


def _word_key(start: float, end: float) -> str:
    return f"{round(float(start), 3)}:{round(float(end), 3)}"


def _apply_transcript_overrides(transcript: dict, overrides: dict) -> None:
    """Patch manual corrections (text and/or speaker) onto the matching
    words in-place, keyed by each word's own original absolute timestamps
    so overrides survive regardless of segment/word indexing.

    Whisper's word-level alignment occasionally collapses two different
    consecutive words onto the exact same (start, end) span (a real
    transcription artifact on fast speech -- new transcriptions are now
    de-duped in transcriber.py, but already-cached transcripts and any other
    future edge case can still produce one). If that happens, a single
    override keyed by that shared timestamp must never be allowed to apply
    to more than one word -- that's what silently duplicated "bien" onto
    "¿Vendías" in production. Consume each override key at most once, in
    transcript order, so a collision degrades to "one word not patched"
    instead of "wrong word overwritten"."""
    words_overrides = overrides.get("words") or {}
    if not words_overrides:
        return
    consumed = set()
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []) or []:
            key = _word_key(w["start"], w["end"])
            if key in consumed:
                continue
            entry = words_overrides.get(key)
            if not entry:
                continue
            consumed.add(key)
            if "text" in entry:
                w["word"] = entry["text"]
            if "speaker" in entry:
                w["speaker_override"] = entry["speaker"]


def _transcript_text_for_range(transcript: dict, start: float, end: float) -> str:
    """Word-accurate transcript text for [start, end), so it tracks a
    highlight's exact boundaries (not just whichever whole segments it
    overlaps)."""
    words_out = []
    for seg in transcript.get("segments", []):
        if seg["end"] < start or seg["start"] > end:
            continue
        words = seg.get("words") or []
        if words:
            for w in words:
                mid = (w["start"] + w["end"]) / 2
                if start <= mid <= end:
                    words_out.append(w["word"])
        elif seg["start"] >= start and seg["end"] <= end:
            words_out.append(seg["text"].strip())
    return " ".join(w for w in words_out if w).strip()


def _transcript_lines_for_range(transcript: dict, start: float, end: float) -> str:
    """Per-segment transcript lines tagged with absolute timestamps, e.g.
    '[123.4s] texto...', so the LLM can point back at real time ranges for
    each subtopic instead of just returning prose."""
    lines = []
    for seg in transcript.get("segments", []):
        if seg["end"] < start or seg["start"] > end:
            continue
        text = seg["text"].strip()
        if text:
            lines.append(f"[{seg['start']:.1f}s] {text}")
    return "\n".join(lines)


MIN_SUBTOPIC_SECONDS = 40.0


def _sanitize_subtopic_clips(raw_subtopics: object, start: float, end: float) -> list:
    """Normalize LLM subtopic output into clip dicts clamped to the parent
    highlight's own [start, end] range; drop anything degenerate; merge any
    subtopic shorter than MIN_SUBTOPIC_SECONDS into its neighbor so short,
    isolated snippets don't end up as standalone clip candidates."""
    if not isinstance(raw_subtopics, list):
        return []

    cleaned = []
    for item in raw_subtopics:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        try:
            s = float(item.get("start_time"))
            e = float(item.get("end_time"))
        except (TypeError, ValueError):
            continue
        s = max(start, min(s, end))
        e = max(start, min(e, end))
        if e - s < 1.0:
            continue
        cleaned.append({"text": text, "start_time": s, "end_time": e})

    cleaned.sort(key=lambda c: c["start_time"])

    merged = []
    for c in cleaned:
        if merged and merged[-1]["end_time"] - merged[-1]["start_time"] < MIN_SUBTOPIC_SECONDS:
            prev = merged[-1]
            prev["end_time"] = c["end_time"]
            prev["text"] = f"{prev['text']} {c['text']}"
        else:
            merged.append(dict(c))

    while len(merged) >= 2 and merged[-1]["end_time"] - merged[-1]["start_time"] < MIN_SUBTOPIC_SECONDS:
        last = merged.pop()
        merged[-1]["end_time"] = last["end_time"]
        merged[-1]["text"] = f"{merged[-1]['text']} {last['text']}"

    return merged


def _transcript_duration(transcript: dict) -> float:
    duration = transcript.get("duration")
    if duration:
        return float(duration)
    segments = transcript.get("segments", [])
    return float(segments[-1]["end"]) if segments else 0.0


def _clip_iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two (start, end) time ranges."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _sanitize_clip_candidates(raw_clips: object, duration: float) -> list:
    """Normalize LLM clip-candidate output: clamp ranges to [0, duration],
    drop anything under the minimum clip length, and collapse near-identical
    duplicates that show up when the same clip gets detected in more than
    one overlapping window. Distinct clips are deliberately left overlapping
    or nested — overlap is a feature of this model, not noise to remove."""
    if not isinstance(raw_clips, list):
        return []

    cleaned = []
    for item in raw_clips:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        try:
            s = float(item.get("start_time"))
            e = float(item.get("end_time"))
        except (TypeError, ValueError):
            continue
        s = max(0.0, min(s, duration))
        e = max(0.0, min(e, duration))
        # Small tolerance below the 40s floor for LLM rounding — this is a
        # sanity check, not the definition of a valid clip.
        if e - s < MIN_SUBTOPIC_SECONDS - 5.0:
            continue
        cleaned.append({"title": title, "start_time": s, "end_time": e})

    # Longer candidates first so a duplicate pair keeps its more complete
    # version when only one survives the merge below.
    cleaned.sort(key=lambda c: (c["start_time"], -(c["end_time"] - c["start_time"])))

    deduped = []
    for c in cleaned:
        c_range = (c["start_time"], c["end_time"])
        if any(_clip_iou(c_range, (d["start_time"], d["end_time"])) >= 0.7 for d in deduped):
            continue
        deduped.append(c)

    deduped.sort(key=lambda c: c["start_time"])
    return deduped


@app.route("/api/transcript_range", methods=["GET"])
def api_transcript_range():
    try:
        start = float(request.args.get("start", 0))
        end = float(request.args.get("end", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid start/end"}), 400

    transcript = _get_transcript(SOURCE_PATH)
    text = _transcript_text_for_range(transcript, start, end)
    return jsonify({"text": text})


def _get_cleaned_words(source_words: list, start: float, end: float) -> list:
    """Cleaned version (Whisper mis-hear fixes, spoken numbers -> digits) of
    a words_for_window window, via the same _llm_cleanup_words pass
    build_ass runs before burning the real export -- cached in-memory by
    (start, end) so /api/captions_range and /api/caption_groups, which are
    always requested for the same window back-to-back, share one LLM call
    instead of duplicating it."""
    key = (start, end)
    if key not in _CLEANED_WORDS_CACHE:
        _CLEANED_WORDS_CACHE[key] = _llm_cleanup_words(source_words)
    return _CLEANED_WORDS_CACHE[key]


@app.route("/api/captions_range", methods=["GET"])
def api_captions_range():
    try:
        start = float(request.args.get("start", 0))
        end = float(request.args.get("end", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid start/end"}), 400

    transcript = _get_diarized_transcript(SOURCE_PATH)
    words = words_for_window(transcript.get("segments", []), start, end)
    cleaned_words = _get_cleaned_words(words, start, end)
    return jsonify({"words": words, "cleaned_words": cleaned_words})


@app.route("/api/caption_groups", methods=["GET"])
def api_caption_groups():
    """Real caption-box boundaries for the editor's live preview, computed
    the same way the final render decides them (subtitles.build_ass: first
    _llm_cleanup_words, then _llm_group_words on the cleaned tokens), not the
    JS mechanical fallback groupCaptionBoundaries() -- that fallback only
    breaks on punctuation, and this project's transcripts carry none, so it
    always degraded to a fixed-size hard cap and never actually showed
    phrase-level grouping. Indices are into the RAW word list
    /api/captions_range returns for this start/end (not the cleaned one),
    so the editor's caption-block drag/timing system -- which only knows how
    to address raw, one-object-per-original-word entries -- can keep using
    these boundaries to slice clipEditor.captionWords unchanged."""
    try:
        start = float(request.args.get("start", 0))
        end = float(request.args.get("end", 0))
        words_per_group = int(request.args.get("words", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid start/end/words"}), 400

    transcript = _get_diarized_transcript(SOURCE_PATH)
    words = words_for_window(transcript.get("segments", []), start, end)
    if words_per_group <= 1 or not words:
        groups = [[i, i + 1] for i in range(len(words))]
    else:
        groups = [list(g) for g in _llm_group_words(words, words_per_group)]
    return jsonify({"groups": groups})


@app.route("/api/word_override", methods=["POST"])
def api_word_override():
    body = request.get_json(force=True) or {}
    try:
        word_start = float(body.get("word_start"))
        word_end = float(body.get("word_end"))
    except (TypeError, ValueError):
        return jsonify({"error": "word_start/word_end required"}), 400

    text = body.get("text")
    speaker = body.get("speaker")
    if speaker is not None and speaker not in (ROBERTO_LABEL, OTHER_LABEL):
        return jsonify({"error": "invalid speaker"}), 400
    if text is None and speaker is None:
        return jsonify({"error": "text or speaker required"}), 400

    overrides = _load_overrides(SOURCE_PATH)
    entry = overrides.setdefault("words", {}).setdefault(_word_key(word_start, word_end), {})
    if text is not None:
        entry["text"] = str(text)
    if speaker is not None:
        entry["speaker"] = speaker
    _save_overrides(SOURCE_PATH, overrides)

    transcript = _get_diarized_transcript(SOURCE_PATH)
    _apply_transcript_overrides(transcript, overrides)
    return jsonify({"ok": True})


@app.route("/api/word_overrides_bulk", methods=["POST"])
def api_word_overrides_bulk():
    """Same as /api/word_override but applies a whole batch in one
    read-modify-write, so a client-side bulk edit (e.g. strip punctuation or
    toggle case across a clip's words) can't race itself across concurrent
    requests and corrupt/lose overrides."""
    body = request.get_json(force=True) or {}
    items = body.get("words")
    if not isinstance(items, list) or not items:
        return jsonify({"error": "words required"}), 400

    overrides = _load_overrides(SOURCE_PATH)
    for item in items:
        try:
            word_start = float(item.get("word_start"))
            word_end = float(item.get("word_end"))
        except (TypeError, ValueError):
            continue
        text = item.get("text")
        speaker = item.get("speaker")
        if speaker is not None and speaker not in (ROBERTO_LABEL, OTHER_LABEL):
            continue
        if text is None and speaker is None:
            continue
        entry = overrides.setdefault("words", {}).setdefault(_word_key(word_start, word_end), {})
        if text is not None:
            entry["text"] = str(text)
        if speaker is not None:
            entry["speaker"] = speaker
    _save_overrides(SOURCE_PATH, overrides)

    transcript = _get_diarized_transcript(SOURCE_PATH)
    _apply_transcript_overrides(transcript, overrides)
    return jsonify({"ok": True})


@app.route("/api/analyze_segment", methods=["POST"])
def api_analyze_segment():
    body = request.get_json(force=True) or {}
    text = str(body.get("text") or "").strip()
    if not text:
        return jsonify({"central_theme": "", "subtopics": []})

    try:
        start = float(body.get("start"))
        end = float(body.get("end"))
    except (TypeError, ValueError):
        return jsonify({"error": "start/end required"}), 400
    if end <= start:
        return jsonify({"error": "end must be after start"}), 400

    transcript = _get_transcript(SOURCE_PATH)
    timed_text = _transcript_lines_for_range(transcript, start, end) or text

    prompt = SEGMENT_ANALYSIS_PROMPT.format(text=timed_text, start=start, end=end)
    try:
        raw = call_local_llm(prompt)
        parsed = _parse_json_loose(raw)
    except Exception as e:
        print(f"[editor] analyze_segment failed: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "central_theme": str(parsed.get("central_theme", "")).strip(),
        "subtopics": _sanitize_subtopic_clips(parsed.get("subtopics"), start, end),
    })


def _overlapping_windows(duration: float, window_seconds: float, stride_seconds: float) -> list:
    """Build fixed-size, overlapping [start, end) windows covering
    [0, duration]. Purely a mechanical device to bound how much text goes
    into a single LLM call — it does not define clip boundaries and does
    not need to produce non-overlapping or gap-free coverage on its own."""
    if duration <= window_seconds:
        return [(0.0, duration)]
    windows = []
    start = 0.0
    while True:
        end = min(start + window_seconds, duration)
        windows.append((start, end))
        if end >= duration:
            break
        start += stride_seconds
    return windows


@app.route("/api/analyze_full_video", methods=["POST"])
def api_analyze_full_video():
    transcript = _get_transcript(SOURCE_PATH)
    duration = _transcript_duration(transcript)
    if duration <= 0:
        return jsonify({"error": "no transcript available"}), 400

    # One LLM call per overlapping window instead of a single call over the
    # whole transcript, purely so each call sees a manageable amount of
    # text — this is not a content decision. Consecutive windows overlap by
    # FULL_VIDEO_WINDOW_SECONDS - FULL_VIDEO_STRIDE_SECONDS so any single
    # clip is fully contained, with margin, inside at least one window;
    # duplicate detections from the overlap get merged in
    # _sanitize_clip_candidates. Clips are intentionally allowed to overlap
    # each other — see FULL_VIDEO_TOPICS_PROMPT.
    raw_clips = []
    windows = _overlapping_windows(duration, FULL_VIDEO_WINDOW_SECONDS, FULL_VIDEO_STRIDE_SECONDS)
    for win_start, win_end in windows:
        timed_text = _transcript_lines_for_range(transcript, win_start, win_end)
        if timed_text.strip():
            prompt = FULL_VIDEO_TOPICS_PROMPT.format(text=timed_text, start=win_start, end=win_end)
            try:
                raw = call_local_llm(prompt)
                parsed = _parse_json_loose(raw)
                raw_clips.extend(parsed.get("clips") or [])
            except Exception as e:
                print(f"[editor] analyze_full_video window [{win_start:.0f},{win_end:.0f}] failed: {e}", flush=True)

    if not raw_clips:
        return jsonify({"error": "no clips detected"}), 500

    clips = _sanitize_clip_candidates(raw_clips, duration)

    cache_path = _highlights_cache_path(SOURCE_PATH)
    with _highlights_lock:
        highlights = []
        if cache_path.exists():
            with open(cache_path) as f:
                highlights = json.load(f)

        # Drop any temas from a previous run of this endpoint before appending
        # the fresh batch, so re-running doesn't pile up stale duplicates.
        highlights = [h for h in highlights if h.get("kind") != "tema"]
        next_id = max((h["id"] for h in highlights), default=-1) + 1

        for clip in clips:
            highlights.append({
                "id": next_id,
                "kind": "tema",
                "title": clip["title"],
                "central_theme": clip["title"],
                "start_time": clip["start_time"],
                "end_time": clip["end_time"],
                "sales_angle": "",
                "score": None,
                "sales_angle_score": None,
                "hook_sentence": "",
                # Each clip is already a complete, self-contained candidate (not
                # a container to be split further), so it points to itself here
                # — this lets the frontend reuse it directly instead of firing
                # an on-demand /api/analyze_segment call when it's opened.
                "subtopics": [{
                    "text": clip["title"],
                    "start_time": clip["start_time"],
                    "end_time": clip["end_time"],
                }],
            })
            next_id += 1

        os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
        _atomic_write_json(cache_path, highlights)
        print(f"[editor] wrote {len(clips)} full-video clip(s) -> {cache_path}", flush=True)

    return jsonify(highlights)


SILENCE_MIN_GAP_SECONDS = 0.3
SILENCE_PAD_SECONDS = 0.08
SILENCE_NOISE_DB = "-30dB"

_SILENCE_RE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_RE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def _audio_silences(start: float, end: float) -> list[dict]:
    """Ground-truth silence intervals from the actual audio (ffmpeg
    silencedetect), independent of Whisper's word timestamps.

    Word-gap detection alone under-detects: Whisper's word boundaries are
    often not tight around a pause — the silence gets absorbed into the
    start/end of an adjacent word instead of showing up as a gap between
    words (sometimes a whole pause ends up *inside* a single word's
    timestamp span, e.g. a short filler word stretched to cover it) — so a
    real pause well over SILENCE_MIN_GAP_SECONDS can show as a sub-threshold
    or nonexistent word gap. Runs against the cached mono wav already
    produced for diarization rather than the huge source .mov."""
    wav_path = _extract_audio_wav(SOURCE_PATH)
    duration = end - start
    # -ss MUST come before -i: this seeks the input directly (fast, and
    # resets the filter's timestamps to start at 0 at the seek point). If
    # -ss is placed after -i instead, ffmpeg decodes the *entire* file from
    # t=0 to get an accurate seek, and silencedetect logs true source-
    # absolute timestamps rather than timestamps relative to `start` — a
    # previous version of this code assumed the latter and added `start` to
    # already-absolute values, which shifted every real silence out of the
    # window (dropped by clamping) while coincidentally shifting unrelated
    # silences from near the start of the file into the window as bogus cuts.
    cmd = [
        "ffmpeg", "-nostats", "-ss", str(start), "-i", wav_path,
        "-t", str(duration),
        "-af", f"silencedetect=noise={SILENCE_NOISE_DB}:d={SILENCE_MIN_GAP_SECONDS}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    intervals = []
    pending_start = None
    for line in stderr.splitlines():
        m_start = _SILENCE_RE_START.search(line)
        if m_start:
            pending_start = float(m_start.group(1))
            continue
        m_end = _SILENCE_RE_END.search(line)
        if m_end and pending_start is not None:
            s = start + max(0.0, pending_start)
            e = start + float(m_end.group(1))
            intervals.append((max(start, s), min(end, e)))
            pending_start = None
    return intervals


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    intervals = sorted(intervals)
    merged: list[list[float]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


@app.route("/api/detect_silences", methods=["GET"])
def api_detect_silences():
    """Silence detection for the clip editor's 'quitar silencios' tool.

    Unions two independent signals so neither one's blind spot becomes a
    missed cut: Whisper word-timestamp gaps (cheap, usually fine) and
    ffmpeg's actual audio silencedetect (ground truth, catches the pauses
    Whisper's word boundaries swallow — see _audio_silences). Padded
    slightly on each side so a cut doesn't clip the tail/head of speech."""
    try:
        start = float(request.args.get("start", 0))
        end = float(request.args.get("end", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid start/end"}), 400
    if end <= start:
        return jsonify({"error": "end must be after start"}), 400

    transcript = _get_transcript(SOURCE_PATH)
    words = []
    for seg in transcript.get("segments", []):
        if seg["end"] < start or seg["start"] > end:
            continue
        for w in (seg.get("words") or []):
            if w["end"] >= start and w["start"] <= end:
                words.append(w)
    words.sort(key=lambda w: w["start"])

    raw_intervals = []
    cursor = start
    for w in words:
        w_start = max(start, w["start"])
        if w_start - cursor >= SILENCE_MIN_GAP_SECONDS:
            raw_intervals.append((cursor, w_start))
        cursor = max(cursor, min(w["end"], end))
    if end - cursor >= SILENCE_MIN_GAP_SECONDS:
        raw_intervals.append((cursor, end))

    try:
        raw_intervals.extend(_audio_silences(start, end))
    except Exception as exc:
        print(f"[editor] audio silencedetect failed, using word gaps only: {exc}", flush=True)

    gaps = [
        {"start_time": round(s + SILENCE_PAD_SECONDS, 2), "end_time": round(e - SILENCE_PAD_SECONDS, 2)}
        for s, e in _merge_intervals(raw_intervals)
    ]
    gaps = [g for g in gaps if g["end_time"] - g["start_time"] > 0.05]
    return jsonify({"gaps": gaps})


@app.route("/api/detect_fillers", methods=["GET"])
def api_detect_fillers():
    """Filler-word ('muletilla') detection for the clip editor's 'quitar
    muletillas' tool. Reuses editing.find_filler_cuts() -- the same
    conservative LLM pass (FILLER_PROMPT) the full batch pipeline runs under
    LOCAL_TRIM_FILLERS_ENABLED -- so the editor's interactive cuts match
    what a from-scratch render would flag. Returns gaps in the same
    absolute-time {start_time, end_time} shape as /api/detect_silences, so
    the client can merge them into the same state.cuts list and everything
    downstream (export, live-preview cut-skipping, caption remap) works
    unchanged."""
    try:
        start = float(request.args.get("start", 0))
        end = float(request.args.get("end", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid start/end"}), 400
    if end <= start:
        return jsonify({"error": "end must be after start"}), 400

    transcript = _get_transcript(SOURCE_PATH)
    words = []
    for seg in transcript.get("segments", []):
        if seg["end"] < start or seg["start"] > end:
            continue
        for w in (seg.get("words") or []):
            if w["end"] >= start and w["start"] <= end:
                words.append(w)
    words.sort(key=lambda w: w["start"])

    cuts = editing.find_filler_cuts(words)
    gaps = [
        {"start_time": round(s, 2), "end_time": round(e, 2)}
        for s, e in cuts
        if e - s > 0.02
    ]
    return jsonify({"gaps": gaps})


@app.route("/api/highlights", methods=["GET"])
def api_highlights():
    highlights = _load_or_build_highlights(SOURCE_PATH)
    return jsonify(highlights)


@app.route("/api/music_tracks", methods=["GET"])
def api_music_tracks():
    if not MUSIC_DIR.is_dir():
        return jsonify([])
    tracks = sorted(p.name for p in MUSIC_DIR.iterdir() if p.suffix.lower() in (".mp3", ".wav", ".m4a"))
    return jsonify(tracks)


@app.route("/api/music_tracks/<path:filename>", methods=["GET"])
def api_music_track_file(filename):
    # Serves the raw track for in-browser preview playback (mixed client-side
    # with the <video>, since the real mix only happens server-side on export).
    path = (MUSIC_DIR / filename).resolve()
    if MUSIC_DIR.resolve() not in path.parents or not path.is_file():
        abort(404)
    return send_file(path)


# A preset is a named bundle of the look-and-feel settings a clip is exported
# with. Fields mirror the ones api_patch_highlight accepts; applying a preset in
# the editor replays them through that same PATCH, so validation lives there.
PRESET_FIELDS = (
    "caption_style",
    "font",
    "caption_color",
    "caption_size",
    "caption_position",
    "caption_words",
    "captions_enabled",
    "speed",
    "music_track",
    "music_volume",
)
_presets_lock = threading.Lock()


def _presets_path() -> Path:
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    return Path(LOCAL_OUTPUT_DIR) / "clip_presets.json"


def _load_presets() -> dict:
    path = _presets_path()
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


@app.route("/api/presets", methods=["GET"])
def api_presets():
    presets = _load_presets()
    return jsonify([{"name": name, "settings": presets[name]} for name in sorted(presets)])


@app.route("/api/presets/<path:name>", methods=["PUT"])
def api_put_preset(name):
    clean_name = str(name).strip()
    if not clean_name or len(clean_name) > 60:
        return jsonify({"error": "invalid preset name"}), 400
    body = request.get_json(force=True) or {}
    settings = {k: body[k] for k in PRESET_FIELDS if k in body}
    if not settings:
        return jsonify({"error": "no preset fields in body"}), 400

    with _presets_lock:
        presets = _load_presets()
        presets[clean_name] = settings
        _atomic_write_json(_presets_path(), presets)
    return jsonify({"name": clean_name, "settings": settings})


@app.route("/api/presets/<path:name>", methods=["DELETE"])
def api_delete_preset(name):
    clean_name = str(name).strip()
    with _presets_lock:
        presets = _load_presets()
        if clean_name not in presets:
            return jsonify({"error": f"preset '{clean_name}' not found"}), 404
        presets.pop(clean_name)
        _atomic_write_json(_presets_path(), presets)
    return jsonify({"deleted": clean_name})


@app.route("/api/highlights/<int:highlight_id>", methods=["PATCH"])
def api_patch_highlight(highlight_id):
    cache_path = _highlights_cache_path(SOURCE_PATH)
    body = request.get_json(force=True) or {}

    with _highlights_lock:
        if not cache_path.exists():
            return jsonify({"error": "no highlights cached yet"}), 404

        with open(cache_path) as f:
            highlights = json.load(f)

        match = next((h for h in highlights if h.get("id") == highlight_id), None)
        if match is None:
            return jsonify({"error": f"highlight {highlight_id} not found"}), 404
        if "start_time" in body:
            match["start_time"] = float(body["start_time"])
        if "end_time" in body:
            match["end_time"] = float(body["end_time"])
        if "transcript_override" in body:
            match["transcript_override"] = str(body["transcript_override"])
        if "color" in body:
            if body["color"]:
                match["color"] = str(body["color"])
            else:
                match.pop("color", None)
        if "captions_enabled" in body:
            match["captions_enabled"] = bool(body["captions_enabled"])
        if "speed" in body:
            match["speed"] = float(body["speed"])
        if "font" in body:
            font_key = str(body["font"])
            if font_key not in FONT_CHOICES:
                return jsonify({"error": f"unknown font '{font_key}'"}), 400
            match["font"] = font_key
        if "caption_color" in body:
            color_key = str(body["caption_color"])
            if color_key not in CAPTION_COLORS:
                return jsonify({"error": f"unknown caption_color '{color_key}'"}), 400
            match["caption_color"] = color_key
        if "caption_size" in body:
            size_key = str(body["caption_size"])
            if size_key not in CAPTION_SIZES:
                return jsonify({"error": f"unknown caption_size '{size_key}'"}), 400
            match["caption_size"] = size_key
        if "caption_position" in body:
            position_key = str(body["caption_position"])
            if position_key not in CAPTION_POSITIONS:
                return jsonify({"error": f"unknown caption_position '{position_key}'"}), 400
            match["caption_position"] = position_key
        if "caption_words" in body:
            words_count = int(body["caption_words"])
            if words_count not in (1, 2, 3):
                return jsonify({"error": f"unknown caption_words '{words_count}'"}), 400
            match["caption_words"] = words_count
        if "caption_style" in body:
            style_key = str(body["caption_style"])
            if style_key not in CAPTION_STYLES:
                return jsonify({"error": f"unknown caption_style '{style_key}'"}), 400
            match["caption_style"] = style_key
        if "cuts" in body:
            match["cuts"] = [
                {"start_time": float(c["start_time"]), "end_time": float(c["end_time"])}
                for c in body["cuts"]
            ]
        if "crop_enabled" in body:
            match["crop_enabled"] = bool(body["crop_enabled"])
        if "crop_keyframes" in body:
            match["crop_keyframes"] = [
                {
                    "t": float(k["t"]),
                    "cx": float(k["cx"]),
                    "cy": float(k["cy"]),
                    "zoom": float(k["zoom"]),
                }
                for k in body["crop_keyframes"]
            ]
        if "crop_mode" in body:
            crop_mode = str(body["crop_mode"])
            if crop_mode not in CROP_MODES:
                return jsonify({"error": f"unknown crop_mode '{crop_mode}'"}), 400
            match["crop_mode"] = crop_mode
        if "split_keyframes" in body:
            match["split_keyframes"] = [
                {
                    "t": float(k["t"]),
                    "top": _pane_spec(k.get("top")),
                    "bottom": _pane_spec(k.get("bottom")),
                }
                for k in body["split_keyframes"]
            ]
        if "music_track" in body:
            track = body["music_track"]
            if track:
                if not (MUSIC_DIR / str(track)).is_file():
                    return jsonify({"error": f"unknown music_track '{track}'"}), 400
                match["music_track"] = str(track)
            else:
                match.pop("music_track", None)
        if "music_volume" in body:
            match["music_volume"] = max(0.0, min(1.0, float(body["music_volume"])))
        if "caption_time_overrides" in body:
            match["caption_time_overrides"] = [
                {
                    "word_start": float(o["word_start"]),
                    "word_end": float(o["word_end"]),
                    "start": float(o["start"]),
                    "end": float(o["end"]),
                }
                for o in body["caption_time_overrides"]
            ]

        _atomic_write_json(cache_path, highlights)

    return jsonify(match)


def _pane_spec(raw) -> dict:
    """One split-screen pane's framing, clamped to the ranges the editor's
    crop box can actually produce: cx/cy are 0-1 fractions of the source
    frame (box centre) and zoom is the 1-4 UI multiplier."""
    raw = raw or {}
    return {
        "cx": max(0.0, min(1.0, float(raw.get("cx", 0.5)))),
        "cy": max(0.0, min(1.0, float(raw.get("cy", 0.5)))),
        "zoom": max(1.0, min(4.0, float(raw.get("zoom", 1.0)))),
    }


def _manual_crop_keyframes(h: dict):
    """(mode, keyframes) for this highlight's manual framing, or (mode, [])
    when it has none and the automatic face-tracking reframe should run
    instead."""
    mode = h.get("crop_mode", DEFAULT_CROP_MODE)
    if mode not in CROP_MODES:
        mode = DEFAULT_CROP_MODE
    key = "split_keyframes" if mode == "split" else "crop_keyframes"
    return mode, list(h.get(key) or [])


def _export_output_path(source_path: str, highlight_id: int) -> Path:
    stem = Path(source_path).stem
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    return Path(LOCAL_OUTPUT_DIR) / f"{stem}.clip{highlight_id}.mp4"


@app.route("/api/highlights/<int:highlight_id>/export", methods=["POST"])
def api_export_highlight(highlight_id):
    """Real render of one highlight, applying every setting the editor
    persists for it (cuts, captions incl. the LLM number/time cleanup that
    burn_subtitles runs by default, speed) -- mirrors dashboard.py's
    working 'Regenerar clip' flow (lines ~590-650) so settings configured
    here actually reach a real output video instead of only living in the
    highlights JSON cache."""
    cache_path = _highlights_cache_path(SOURCE_PATH)
    with _highlights_lock:
        if not cache_path.exists():
            return jsonify({"error": "no highlights cached yet"}), 404
        with open(cache_path) as f:
            highlights = json.load(f)
        h = next((x for x in highlights if x.get("id") == highlight_id), None)
        if h is None:
            return jsonify({"error": f"highlight {highlight_id} not found"}), 404
        h = dict(h)  # snapshot -- render below can take a while

    start_time = float(h["start_time"])
    end_time = float(h["end_time"])
    duration = end_time - start_time
    if duration <= 0:
        return jsonify({"error": "invalid start_time/end_time"}), 400

    out_path = str(_export_output_path(SOURCE_PATH, highlight_id))

    cut_segments = [
        (float(c["start_time"]) - start_time, float(c["end_time"]) - start_time)
        for c in h.get("cuts", [])
    ]
    cut_segments = [(s, e) for s, e in cut_segments if e > s] or None

    crop_enabled = bool(h.get("crop_enabled", False))
    crop_mode, manual_keyframes = _manual_crop_keyframes(h)

    try:
        if crop_enabled and manual_keyframes:
            # The editor's own rectangles win: what the vertical preview shows
            # is what gets rendered. Automatic active-speaker tracking below is
            # only the fallback for a clip whose crop was switched on but never
            # framed by hand.
            clipper.crop_clip_manual(
                SOURCE_PATH, start_time, end_time, EXPORT_ASPECT_RATIO, out_path,
                keyframes=manual_keyframes,
                mode=crop_mode,
                cut_segments=cut_segments,
            )
        else:
            clipper.crop_clip_local(
                SOURCE_PATH, start_time, end_time, EXPORT_ASPECT_RATIO, out_path,
                cut_segments=cut_segments,
                crop_enabled=crop_enabled,
            )

        if h.get("captions_enabled", True):
            transcript = _get_diarized_transcript(SOURCE_PATH)
            words = words_for_window(
                transcript.get("segments", []), start_time, end_time,
                time_overrides=h.get("caption_time_overrides"),
            )
            if cut_segments:
                keep = editing.keep_segments_from_cuts(cut_segments, duration)
                words = editing.remap_words_across_cuts(words, keep)

            font_key = h.get("font", DEFAULT_FONT_KEY)
            caption_size = h.get("caption_size", "m")
            tmp_subbed = out_path + ".subbed.mp4"
            burn_subtitles(
                out_path, words, tmp_subbed,
                font_name=FONT_CHOICES.get(font_key, FONT_CHOICES[DEFAULT_FONT_KEY]),
                size=CAPTION_SIZES.get(caption_size, CAPTION_SIZES["m"]),
                position=h.get("caption_position", DEFAULT_CAPTION_POSITION),
                style=h.get("caption_style", DEFAULT_CAPTION_STYLE),
                color=h.get("caption_color", DEFAULT_CAPTION_COLOR),
                words_per_group=h.get("caption_words", DEFAULT_WORDS_PER_GROUP),
            )
            os.replace(tmp_subbed, out_path)

        speed = float(h.get("speed", 1.0))
        if speed != 1.0:
            tmp_sped = out_path + ".sped.mp4"
            clipper.apply_speed_local(out_path, speed, tmp_sped)
            os.replace(tmp_sped, out_path)

        source_duration = clipper._ffprobe_duration(SOURCE_PATH)
        cta_duration = min(CTA_OUTRO_DURATION_SECONDS, max(source_duration - end_time, 0))
        if cta_duration >= 0.5:
            tmp_carded = out_path + ".carded.mp4"
            clipper.append_blurred_cta_outro(
                out_path, SOURCE_PATH, end_time, tmp_carded,
                duration=cta_duration, blur_sigma=CTA_OUTRO_BLUR_SIGMA,
            )
            os.replace(tmp_carded, out_path)

        music_track = h.get("music_track")
        if music_track:
            music_path = MUSIC_DIR / str(music_track)
            if not music_path.is_file():
                return jsonify({"error": f"music track not found: {music_track}"}), 400
            volume = float(h.get("music_volume", DEFAULT_MUSIC_VOLUME))
            tmp_mixed = out_path + ".mixed.mp4"
            clipper.mix_background_music(out_path, str(music_path), volume, tmp_mixed)
            os.replace(tmp_mixed, out_path)
    except Exception as exc:
        print(f"[editor] export failed for highlight {highlight_id}: {exc!r}", flush=True)
        return jsonify({"error": str(exc)}), 500

    return jsonify({"ok": True, "path": out_path, "url": f"/api/highlights/{highlight_id}/export/file"})


@app.route("/api/highlights/<int:highlight_id>/export/file", methods=["GET"])
def api_export_highlight_file(highlight_id):
    out_path = _export_output_path(SOURCE_PATH, highlight_id)
    if not out_path.exists():
        return jsonify({"error": "not exported yet"}), 404
    return send_file(out_path, conditional=True)


def main():
    global SOURCE_PATH, NUM_CLIPS
    parser = argparse.ArgumentParser(description="Timeline editor for one local source video")
    parser.add_argument("source", help="Path to the local source video")
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--num-clips", type=int, default=20,
                         help="How many candidate highlight zones to request from the LLM")
    args = parser.parse_args()

    SOURCE_PATH = os.path.abspath(args.source)
    if not os.path.exists(SOURCE_PATH):
        raise SystemExit(f"source video not found: {SOURCE_PATH}")
    NUM_CLIPS = args.num_clips

    print(f"[editor] source: {SOURCE_PATH}", flush=True)
    print(f"[editor] http://localhost:{args.port}", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
