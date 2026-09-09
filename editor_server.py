"""Standalone timeline editor: load the full source video, show LLM-detected
highlight zones as draggable markers on a timeline. Phase 1 only — no
silence-removal or format/render functions yet (see plan for later phases).

Usage:
    python3 editor_server.py <source_video_path> [--port 8502] [--num-clips 20]
"""
import argparse
import json
import os
import threading
import wave
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file

from shorts_generator.config import LOCAL_OUTPUT_DIR
from shorts_generator.highlights import _parse_json_loose, get_highlights
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
    FONT_CHOICES,
    words_for_window,
)
from shorts_generator.local.transcriber import transcribe_local

app = Flask(__name__)

SOURCE_PATH = None
NUM_CLIPS = 20

_TRANSCRIPT_CACHE = None
_DIARIZED_TRANSCRIPT_CACHE = None

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


@app.route("/video")
def video():
    return send_file(SOURCE_PATH, conditional=True)


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
    so overrides survive regardless of segment/word indexing."""
    words_overrides = overrides.get("words") or {}
    if not words_overrides:
        return
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []) or []:
            entry = words_overrides.get(_word_key(w["start"], w["end"]))
            if not entry:
                continue
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


@app.route("/api/captions_range", methods=["GET"])
def api_captions_range():
    try:
        start = float(request.args.get("start", 0))
        end = float(request.args.get("end", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid start/end"}), 400

    transcript = _get_diarized_transcript(SOURCE_PATH)
    words = words_for_window(transcript.get("segments", []), start, end)
    return jsonify({"words": words})


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


@app.route("/api/detect_silences", methods=["GET"])
def api_detect_silences():
    """Word-level pause detection for the clip editor's 'quitar silencios'
    tool. Reuses the same Whisper word timestamps already produced for
    transcription/highlights instead of a separate audio-amplitude silence
    pass — a gap between two spoken words within [start, end] longer than
    SILENCE_MIN_GAP_SECONDS is a candidate cut, padded slightly on each side
    so the cut doesn't clip the tail/head of adjacent speech."""
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

    gaps = []
    cursor = start
    for w in words:
        w_start = max(start, w["start"])
        if w_start - cursor >= SILENCE_MIN_GAP_SECONDS:
            gaps.append({
                "start_time": round(cursor + SILENCE_PAD_SECONDS, 2),
                "end_time": round(w_start - SILENCE_PAD_SECONDS, 2),
            })
        cursor = max(cursor, min(w["end"], end))
    if end - cursor >= SILENCE_MIN_GAP_SECONDS:
        gaps.append({
            "start_time": round(cursor + SILENCE_PAD_SECONDS, 2),
            "end_time": round(end - SILENCE_PAD_SECONDS, 2),
        })

    gaps = [g for g in gaps if g["end_time"] - g["start_time"] > 0.05]
    return jsonify({"gaps": gaps})


@app.route("/api/highlights", methods=["GET"])
def api_highlights():
    highlights = _load_or_build_highlights(SOURCE_PATH)
    return jsonify(highlights)


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
