"""Standalone timeline editor: load the full source video, show LLM-detected
highlight zones as draggable markers on a timeline. Phase 1 only — no
silence-removal or format/render functions yet (see plan for later phases).

Usage:
    python3 editor_server.py <source_video_path> [--port 8502] [--num-clips 20]
"""
import argparse
import hashlib
import io
import json
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # ops de pyannote sin kernel MPS van a CPU
import re
import unicodedata
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path

import math
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory, abort

from shorts_generator.config import LOCAL_OUTPUT_DIR
from shorts_generator.highlights import _parse_json_loose, get_highlights
from shorts_generator.local import clipper, editing
from shorts_generator.local.encoding import FINAL_VIDEO_ARGS, intermediate_encodes
from shorts_generator.local.diarizer import (
    OTHER_LABEL,
    ROBERTO_LABEL,
    _extract_audio_wav,
    diarize_and_label,
)
from shorts_generator.local.llm import call_local_llm
from shorts_generator.local.subtitles import (
    FONTS_DIR,
    CAPTION_COLORS,
    CAPTION_POSITIONS,
    CAPTION_SIZES,
    CAPTION_SIZE_RANGE,
    CAPTION_STYLES,
    DEFAULT_CAPTION_COLOR,
    DEFAULT_CAPTION_POSITION,
    DEFAULT_CAPTION_STYLE,
    DEFAULT_FONT_KEY,
    DEFAULT_WORDS_PER_GROUP,
    FONT_CHOICES,
    _group_boundaries,
    _layout_metrics,
    _pil_font,
    _split_groups_to_fit,
    apply_group_keys,
    arrange_caption_group_keys,
    burn_subtitles,
    correct_clip_words,
    resolve_caption_color,
    resolve_caption_pos,
    resolve_caption_size,
    words_for_window,
)
from shorts_generator.local import transcriber as local_transcriber
from shorts_generator.local.transcriber import transcribe_local

EXPORT_ASPECT_RATIO = "9:16"

# Vertical framing modes offered by the editor's "Encuadre" panel:
# "single" = one 9:16 rectangle, "split" = two rectangles stacked 50/50
# (top camera / bottom camera), both cropped out of the same wide source.
CROP_MODES = ("single", "split")
DEFAULT_CROP_MODE = "single"

MUSIC_DIR = Path(__file__).resolve().parent / "assets" / "music"
DEFAULT_MUSIC_VOLUME = 0.18
CTA_OUTRO_DURATION_SECONDS = 4.5
CTA_OUTRO_BLUR_SIGMA = 22.0

# Text templates ("Plantillas") overlaid on the first seconds of a vertical
# clip. The rendering lives in clipper.TEXT template helpers; here we only
# persist/validate what the editor's panel lets the user choose.
TEXT_TEMPLATE_IDS = clipper.TEMPLATE_IDS
DEFAULT_TEMPLATE_DURATION = 4.0
TEMPLATE_DURATION_RANGE = (1.0, 15.0)

# Free-form overlay layers ("Capas"): PNGs the user drops onto the frame and
# then moves/resizes on the preview, same coordinate contract as the template
# card (x/y = top-left corner, w = width, all fractions of the output frame).
OVERLAY_DIR = Path(__file__).resolve().parent / "output" / "overlay_assets"
OVERLAY_MAX_BYTES = 12 * 1024 * 1024
OVERLAY_EXTENSIONS = {".png", ".webp", ".gif", ".jpg", ".jpeg"}
OVERLAY_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
OVERLAY_VIDEO_MAX_BYTES = 300 * 1024 * 1024
MAX_OVERLAYS_PER_CLIP = 12

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
    # Proxy liviano (lado corto 720p, 2.5 Mbps): el <video> del editor decodifica
    # y bufferea ~4x menos que con el 1080p a 8 Mbps de antes. El export sigue
    # leyendo SOURCE_PATH, y el front toma el tamaño real del frame de
    # /api/source_info (no del videoWidth del proxy). El sufijo nuevo hace que
    # los .preview.mp4 viejos queden huérfanos en vez de reusarse.
    preview_path = Path(LOCAL_OUTPUT_DIR) / f"{stem}.preview720.mp4"
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
            "-vf", "scale=w='if(gt(iw,ih),-2,min(720,iw))':h='if(gt(iw,ih),min(720,ih),-2)'",
            "-c:v", "h264_videotoolbox", "-b:v", "2500k",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(tmp_path),
        ]
        print(f"[editor] transcoding browser preview (one-time): {preview_path}", flush=True)
        subprocess.run(cmd, check=True)
        os.replace(tmp_path, preview_path)
        print(f"[editor] preview ready: {preview_path}", flush=True)
        return preview_path


_SOURCE_INFO = None


@app.route("/api/source_info")
def api_source_info():
    """Tamaño real del frame de la fuente (el proxy de /video es más chico)."""
    global _SOURCE_INFO
    if _SOURCE_INFO is None:
        w, h, fps = clipper._ffprobe_video_info(SOURCE_PATH)
        _SOURCE_INFO = {"width": w, "height": h, "fps": fps}
    return jsonify(_SOURCE_INFO)


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
    # Atómico: el precalentado y un GET /api/waveform pueden calcularla a la
    # vez, y un lector no debe ver el JSON a medio escribir.
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with open(tmp_path, "w") as f:
        json.dump(peaks, f)  # compacto: miles de puntos, sin indent
    os.replace(tmp_path, cache_path)
    return peaks


# Precalentado al arrancar: antes el primer GET de /video, /api/highlights y
# /api/waveform hacía TODO el trabajo pesado (transcode del proxy,
# transcripción + diarización + LLM) dentro del request, con la página
# colgada. Ahora arranca apenas levanta el server, en dos hilos (el proxy va
# por VideoToolbox y no compite con whisper/pyannote en CPU). Los endpoints
# siguen llamando a las mismas funciones: si llegan antes de que termine,
# esperan el lock y reusan el resultado, sin duplicar trabajo.
_WARMUP_STATE = {"preview": "pendiente", "analysis": "pendiente", "error": None}


def _warmup_preview() -> None:
    try:
        _WARMUP_STATE["preview"] = "generando proxy de video…"
        _preview_video_path(SOURCE_PATH)
        _WARMUP_STATE["preview"] = "listo"
    except Exception as exc:
        _WARMUP_STATE["preview"] = "error"
        _WARMUP_STATE["error"] = f"proxy: {exc!r}"
        print(f"[editor] warmup preview failed: {exc!r}", flush=True)


def _warmup_analysis() -> None:
    try:
        if not _highlights_cache_path(SOURCE_PATH).exists():
            _WARMUP_STATE["analysis"] = "transcribiendo, identificando voces y buscando highlights…"
        _load_or_build_highlights(SOURCE_PATH)
        _WARMUP_STATE["analysis"] = "armando forma de onda…"
        _build_waveform(SOURCE_PATH)
        _WARMUP_STATE["analysis"] = "cargando transcripción…"
        _get_diarized_transcript(SOURCE_PATH)
        _WARMUP_STATE["analysis"] = "listo"
    except Exception as exc:
        _WARMUP_STATE["analysis"] = "error"
        _WARMUP_STATE["error"] = f"análisis: {exc!r}"
        print(f"[editor] warmup analysis failed: {exc!r}", flush=True)


@app.route("/api/warmup", methods=["GET"])
def api_warmup():
    return jsonify(_WARMUP_STATE)


@app.route("/api/thumb", methods=["GET"])
def api_thumb():
    """Miniatura JPEG (alto 96px) del frame en `t` segundos, para el carril de
    video de la timeline. Sale del proxy 720p (no de la fuente 10-bit) y se
    cachea en disco por t cuantizado a 0.1s, así el front pide poquitos frames
    por zoom y los repetidos no vuelven a lanzar ffmpeg."""
    try:
        t = max(0.0, round(float(request.args.get("t", "0")), 1))
    except ValueError:
        abort(400)
    proxy = _preview_video_path(SOURCE_PATH)
    thumb_dir = Path(LOCAL_OUTPUT_DIR) / "thumbs" / Path(SOURCE_PATH).stem
    thumb_path = thumb_dir / f"{int(t * 10)}.jpg"
    if not thumb_path.exists():
        os.makedirs(thumb_dir, exist_ok=True)
        tmp_path = thumb_path.with_name(f"{thumb_path.stem}.tmp-{threading.get_ident()}.jpg")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{t:.1f}", "-i", str(proxy),
            "-frames:v", "1", "-vf", "scale=-2:96", "-q:v", "5",
            str(tmp_path),
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0 or not tmp_path.exists():
            abort(500)
        os.replace(tmp_path, thumb_path)
    resp = send_file(thumb_path, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


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
                w["overridden"] = True
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


def _clamp01(value, default: float = 0.0) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return default


def _sanitize_overlays(raw: object) -> list:
    """Normalises the editor's overlay layer list. Geometry is stored as
    fractions of the output frame (x/y = top-left corner, w = width; the height
    follows the PNG's own aspect), which is the same contract clipper's
    burn_overlay_items consumes -- so what the panel drags is literally what
    gets composited.

    `file` is forced back to a bare name inside OVERLAY_DIR: these values come
    from a PATCH body and end up in a filesystem path at export time."""
    if not isinstance(raw, list):
        return []
    items = []
    for entry in raw[:MAX_OVERLAYS_PER_CLIP]:
        if not isinstance(entry, dict):
            continue
        name = os.path.basename(str(entry.get("file", "")))
        if not name or not (OVERLAY_DIR / name).is_file():
            continue
        start = max(float(entry.get("start", 0.0) or 0.0), 0.0)
        end = entry.get("end")
        try:
            end = None if end in (None, "") else max(float(end), start + 0.1)
        except (TypeError, ValueError):
            end = None
        items.append({
            "id": str(entry.get("id") or uuid.uuid4().hex[:8]),
            "kind": "video" if name.lower().endswith(".mp4") and entry.get("kind") == "video" else "png",
            "file": name,
            "label": str(entry.get("label") or name)[:80],
            # x/y are deliberately allowed a little outside 0-1 so a layer can
            # be bled off the edge of the frame on purpose.
            "x": min(max(float(entry.get("x", 0.1) or 0.0), -1.0), 2.0),
            "y": min(max(float(entry.get("y", 0.1) or 0.0), -1.0), 2.0),
            "w": min(max(float(entry.get("w", 0.3) or 0.05), 0.01), 3.0),
            "start": start,
            "end": end,
            "opacity": _clamp01(entry.get("opacity", 1.0), 1.0),
            "anim_in": (str(entry.get("anim_in") or "none").lower()
                        if str(entry.get("anim_in") or "none").lower() in clipper.OVERLAY_ANIMS
                        else "none"),
            "anim_dur": min(max(float(entry.get("anim_dur") or clipper.OVERLAY_ANIM_DEFAULT_SECONDS), 0.1), 2.0),
            # Capas de video: desde que segundo del archivo arranca (recorte de
            # cabeza), duracion total del archivo, volumen propio, mute y ojo.
            "src_in": max(float(entry.get("src_in", 0.0) or 0.0), 0.0),
            "src_dur": max(float(entry.get("src_dur", 0.0) or 0.0), 0.0),
            # Velocidad de la capa de video (segundos de archivo por segundo de clip).
            "speed": min(max(float(entry.get("speed", 1.0) or 1.0), 0.25), 4.0),
            "volume": min(max(float(entry.get("volume", 1.0) if entry.get("volume") is not None else 1.0), 0.0), 2.0),
            "muted": bool(entry.get("muted", False)),
            "hidden": bool(entry.get("hidden", False)),
            # Canal del timeline: piezas de una capa cortada comparten canal.
            "track": str(entry.get("track") or "")[:40] or None,
            # PNG: esquinas (fraccion del ancho de la capa) y borde hacia adentro
            # (fraccion del ancho del cuadro, como el borde del encuadre).
            "radius": min(max(float(entry.get("radius", 0.0) or 0.0), 0.0), 0.5),
            "border_w": min(max(float(entry.get("border_w", 0.0) or 0.0), 0.0), 0.05),
            "border_color": (str(entry.get("border_color"))
                             if re.fullmatch(r"#[0-9a-fA-F]{6}", str(entry.get("border_color") or ""))
                             else "#ffffff"),
            "crop": _overlay_crop(entry.get("crop")),
            # Capa de texto: el PNG en `file` ya es el render; esto es para
            # re-editarlo desde el panel.
            "text": _text_spec(entry.get("text")),
        })
    return items


def _overlay_crop(raw):
    """Encuadre de una capa: {top,bottom,left,right} en fracciones del medio
    original (0-0.45 c/u, cada eje deja al menos 10%). None = sin recorte."""
    if not isinstance(raw, dict):
        return None
    try:
        c = {k: min(max(float(raw.get(k, 0.0) or 0.0), 0.0), 0.45)
             for k in ("top", "bottom", "left", "right")}
    except (TypeError, ValueError):
        return None
    for a, b in (("top", "bottom"), ("left", "right")):
        if c[a] + c[b] > 0.9:
            c[b] = 0.9 - c[a]
    return c if any(v > 0.0005 for v in c.values()) else None


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
    """Words for a words_for_window window, matching build_ass/burn_subtitles'
    use_llm=False default (benchmark showed the LLM cleanup pass -- Whisper
    mis-hear fixes, spoken numbers -> digits -- isn't worth its cost: the
    mechanical path scored the same without it, see benchmark/work/report.md).
    No LLM call, no cache needed -- kept as a function so callers/signature
    don't have to change if a cleanup pass is ever reintroduced."""
    return source_words


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


def _valid_group_keys(h, words_per_group):
    """Keys saved by "Acomodar subtitulos", only while they still describe the
    grouping the user asked for: they were generated for one specific
    `caption_words`, so changing it falls back to the automatic grouping."""
    keys = h.get("caption_group_keys")
    if keys and h.get("caption_group_words") == words_per_group:
        return keys
    return None


def _stored_group_keys(hid, words_per_group):
    try:
        hid = int(hid)
    except (TypeError, ValueError):
        return None
    cache_path = _highlights_cache_path(SOURCE_PATH)
    with _highlights_lock:
        if not cache_path.exists():
            return None
        with open(cache_path) as f:
            highlights = json.load(f)
    h = next((x for x in highlights if x.get("id") == hid), None)
    return _valid_group_keys(h, words_per_group) if h else None


@app.route("/api/caption_groups", methods=["GET"])
def api_caption_groups():
    """Real caption-box boundaries for the editor's live preview, computed
    the same way the final render decides them (subtitles.build_ass with its
    use_llm=False default: _group_boundaries on the raw tokens, then
    _split_groups_to_fit so no group is wider than one line). _group_boundaries
    is NOT the JS mechanical fallback groupCaptionBoundaries() -- that one only
    breaks on punctuation, and this project's transcripts carry none, so it
    always degraded to a fixed-size hard cap and never actually showed
    phrase-level grouping. _group_boundaries is the Python mechanical grouper
    benchmarked in benchmark/work/report.md, which tied Opus 5 on every
    quality metric at $0/video, so there's no LLM call here any more.
    Indices are into the RAW word list
    /api/captions_range returns for this start/end (not the cleaned one),
    so the editor's caption-block drag/timing system -- which only knows how
    to address raw, one-object-per-original-word entries -- can keep using
    these boundaries to slice clipEditor.captionWords unchanged.

    The one-line re-split needs the same font/size/position/style/resolution
    build_ass would use, so the caller (loadCaptionGroups in the editor)
    passes the highlight's caption look plus the frame it's previewing --
    without these the preview can't know how wide a line actually is and
    would show groups build_ass would have split, exactly the bug this
    endpoint exists to avoid."""
    try:
        start = float(request.args.get("start", 0))
        end = float(request.args.get("end", 0))
        words_per_group = int(request.args.get("words", 1))
        frame_w = max(int(request.args.get("frame_w", 1080)), 16)
        frame_h = max(int(request.args.get("frame_h", 1920)), 16)
    except (TypeError, ValueError):
        return jsonify({"error": "invalid start/end/words/frame_w/frame_h"}), 400

    transcript = _get_diarized_transcript(SOURCE_PATH)
    words = words_for_window(transcript.get("segments", []), start, end)
    if words_per_group <= 1 or not words:
        groups = [[i, i + 1] for i in range(len(words))]
    else:
        stored_keys = _stored_group_keys(request.args.get("hid"), words_per_group)
        if stored_keys:
            groups = list(apply_group_keys(words, stored_keys))
        else:
            groups = list(_group_boundaries(words, words_per_group))
        font_key = request.args.get("font", DEFAULT_FONT_KEY)
        font_name = FONT_CHOICES.get(font_key, FONT_CHOICES[DEFAULT_FONT_KEY])
        size = resolve_caption_size(request.args.get("size"))
        position = request.args.get("position", DEFAULT_CAPTION_POSITION)
        style = request.args.get("style", DEFAULT_CAPTION_STYLE)
        pos = resolve_caption_pos(request.args.get("pos_x"), request.args.get("pos_y"))
        layout = _layout_metrics(size, position, style, frame_h, pos=pos)
        pil_font = _pil_font(font_name, layout["size"])
        max_text_width = frame_w - 2 * layout["margin_lr"] - 2 * layout["pad_x"]
        groups = _split_groups_to_fit(groups, words, pil_font, max_text_width, True)
        groups = [list(g) for g in groups]
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


# Job en background (antes: request síncrono de varios minutos, con las
# ventanas una tras otra). POST lo arranca, GET devuelve el progreso; el
# frontend recarga /api/highlights al terminar.
_FULL_ANALYSIS = {"running": False, "done": 0, "total": 0, "error": None, "count": None}
_FULL_ANALYSIS_LOCK = threading.Lock()
FULL_VIDEO_PARALLEL_WINDOWS = 3  # Gemini free tolera esto sin 429; el retry/fallback de llm.py sigue igual


@app.route("/api/analyze_full_video", methods=["GET"])
def api_analyze_full_video_status():
    return jsonify(_FULL_ANALYSIS)


@app.route("/api/analyze_full_video", methods=["POST"])
def api_analyze_full_video():
    transcript = _get_transcript(SOURCE_PATH)
    duration = _transcript_duration(transcript)
    if duration <= 0:
        return jsonify({"error": "no transcript available"}), 400
    with _FULL_ANALYSIS_LOCK:
        if not _FULL_ANALYSIS["running"]:
            _FULL_ANALYSIS.update(running=True, done=0, total=0, error=None, count=None)
            threading.Thread(
                target=_run_full_analysis_job, args=(transcript, duration),
                daemon=True, name="full-analysis",
            ).start()
    return jsonify(_FULL_ANALYSIS), 202


def _run_full_analysis_job(transcript: dict, duration: float) -> None:
    try:
        _FULL_ANALYSIS["count"] = _full_analysis(transcript, duration)
    except Exception as exc:
        print(f"[editor] analyze_full_video failed: {exc!r}", flush=True)
        _FULL_ANALYSIS["error"] = str(exc)
    finally:
        _FULL_ANALYSIS["running"] = False


def _full_analysis(transcript: dict, duration: float) -> int:

    # One LLM call per overlapping window instead of a single call over the
    # whole transcript, purely so each call sees a manageable amount of
    # text — this is not a content decision. Consecutive windows overlap by
    # FULL_VIDEO_WINDOW_SECONDS - FULL_VIDEO_STRIDE_SECONDS so any single
    # clip is fully contained, with margin, inside at least one window;
    # duplicate detections from the overlap get merged in
    # _sanitize_clip_candidates. Clips are intentionally allowed to overlap
    # each other — see FULL_VIDEO_TOPICS_PROMPT.
    windows = _overlapping_windows(duration, FULL_VIDEO_WINDOW_SECONDS, FULL_VIDEO_STRIDE_SECONDS)
    _FULL_ANALYSIS["total"] = len(windows)

    def analyze_window(win):
        win_start, win_end = win
        try:
            timed_text = _transcript_lines_for_range(transcript, win_start, win_end)
            if not timed_text.strip():
                return []
            prompt = FULL_VIDEO_TOPICS_PROMPT.format(text=timed_text, start=win_start, end=win_end)
            try:
                raw = call_local_llm(prompt)
                parsed = _parse_json_loose(raw)
                return parsed.get("clips") or []
            except Exception as e:
                print(f"[editor] analyze_full_video window [{win_start:.0f},{win_end:.0f}] failed: {e}", flush=True)
                return []
        finally:
            _FULL_ANALYSIS["done"] += 1

    # map() conserva el orden de las ventanas, así el merge de duplicados
    # de _sanitize_clip_candidates ve lo mismo que con el loop secuencial.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=FULL_VIDEO_PARALLEL_WINDOWS) as pool:
        raw_clips = [c for clips in pool.map(analyze_window, windows) for c in clips]

    if not raw_clips:
        raise RuntimeError("no clips detected")

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

    return len(clips)


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


@app.route("/api/whole_video_clip", methods=["POST"])
def api_whole_video_clip():
    """Crea (o reutiliza) un clip que cubre el video entero, para videos
    cortos que no hace falta dividir en temas. Idempotente: si ya existe
    uno, lo devuelve."""
    try:
        body = request.get_json(force=True, silent=True) or {}
        duration = float(body.get("duration") or 0)
        if duration <= 0:
            return jsonify({"error": "duration required"}), 400
        cache_path = _highlights_cache_path(SOURCE_PATH)
        with _highlights_lock:
            highlights = []
            if cache_path.exists():
                with open(cache_path) as f:
                    highlights = json.load(f)
            existing = next((h for h in highlights if h.get("kind") == "entero"), None)
            if existing:
                return jsonify(existing)
            title = os.path.splitext(os.path.basename(SOURCE_PATH))[0]
            clip = {
                "id": max((h["id"] for h in highlights), default=-1) + 1,
                "kind": "entero",
                "title": title,
                "central_theme": title,
                "start_time": 0.0,
                "end_time": duration,
                "sales_angle": "",
                "score": None,
                "sales_angle_score": None,
                "hook_sentence": "",
                "subtopics": [{"text": title, "start_time": 0.0, "end_time": duration}],
            }
            highlights.append(clip)
            os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
            _atomic_write_json(cache_path, highlights)
        return jsonify(clip)
    except Exception as e:
        print(f"[editor] whole_video_clip failed: {e!r}", flush=True)
        return jsonify({"error": str(e)}), 500


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


_music_peaks_cache: dict = {}


@app.route("/api/music_peaks/<path:filename>", methods=["GET"])
def api_music_peaks(filename):
    """Peaks (0-1, 20/s) of a music track, to draw its waveform inside the
    music blocks of the timeline. Cached in memory per file+mtime."""
    path = (MUSIC_DIR / filename).resolve()
    if MUSIC_DIR.resolve() not in path.parents or not path.is_file():
        abort(404)
    key = (str(path), path.stat().st_mtime)
    if key not in _music_peaks_cache:
        sps = 20
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "8000",
             "-f", "s16le", "-"],
            capture_output=True, check=True,
        ).stdout
        samples = np.abs(np.frombuffer(raw, dtype=np.int16).astype(np.float32))
        chunk = 8000 // sps
        n = len(samples) // chunk
        peaks = (samples[: n * chunk].reshape(-1, chunk).max(axis=1) / 32767.0) if n else np.zeros(0)
        top = float(peaks.max()) if peaks.size else 0.0
        if top > 0:
            peaks = peaks / top
        _music_peaks_cache[key] = {
            "sps": sps,
            "duration": len(samples) / 8000.0,
            "peaks": [round(float(x), 3) for x in peaks.clip(0, 1)],
        }
    return jsonify(_music_peaks_cache[key])


# A preset is a named bundle of the look-and-feel settings a clip is exported
# with. Fields mirror the ones api_patch_highlight accepts; applying a preset in
# the editor replays them through that same PATCH, so validation lives there.
PRESET_FIELDS = (
    "caption_style",
    "font",
    "caption_color",
    "caption_size",
    "caption_position",
    "caption_pos_x",
    "caption_pos_y",
    "caption_words",
    "captions_enabled",
    "speed",
    "music_track",
    "music_volume",
    "cta_id",
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


# "Capas predeterminadas": un set de capas guardado para insertarlo de un click
# en otro clip. Un JSON por plantilla en output/overlay_presets/ ({name,
# clip_len, overlays}); los archivos de las capas ya viven en OVERLAY_DIR,
# compartido entre videos. clip_len = duracion del video final donde se armo,
# para que el front pueda anclar las capas al final del clip nuevo.
OVERLAY_PRESET_DIR = Path(__file__).resolve().parent / "output" / "overlay_presets"


def _overlay_preset_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()).strip("_")
    return slug[:60]


@app.route("/api/overlay_presets", methods=["GET"])
def api_overlay_presets():
    out = []
    if OVERLAY_PRESET_DIR.is_dir():
        for p in sorted(OVERLAY_PRESET_DIR.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({
                "slug": p.stem,
                "name": data.get("name") or p.stem,
                "clip_len": data.get("clip_len"),
                "overlays": _sanitize_overlays(data.get("overlays")),
            })
    return jsonify(out)


@app.route("/api/overlay_presets", methods=["POST"])
def api_save_overlay_preset():
    body = request.get_json(force=True) or {}
    name = str(body.get("name") or "").strip()
    slug = _overlay_preset_slug(name)
    if not name or not slug:
        return jsonify({"error": "invalid name"}), 400
    overlays = _sanitize_overlays(body.get("overlays"))
    if not overlays:
        return jsonify({"error": "no overlays"}), 400
    clip_len = body.get("clip_len")
    data = {"name": name, "clip_len": float(clip_len) if clip_len else None, "overlays": overlays}
    OVERLAY_PRESET_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(OVERLAY_PRESET_DIR / f"{slug}.json", data)
    return jsonify({"slug": slug, **data})


@app.route("/api/overlay_presets/<slug>", methods=["DELETE"])
def api_delete_overlay_preset(slug):
    p = OVERLAY_PRESET_DIR / f"{os.path.basename(slug)}.json"
    if not p.is_file():
        return jsonify({"error": "not found"}), 404
    p.unlink()
    return jsonify({"deleted": slug})


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
        if "audio_enhance" in body:
            match["audio_enhance"] = bool(body["audio_enhance"])
        if "border" in body:
            match["border"] = _border_spec(body["border"]) or {}
        if "speed" in body:
            match["speed"] = float(body["speed"])
        if "font" in body:
            font_key = str(body["font"])
            if font_key not in FONT_CHOICES:
                return jsonify({"error": f"unknown font '{font_key}'"}), 400
            match["font"] = font_key
        if "caption_color" in body:
            # Either a named key or a "#RRGGBB" from the editor's colour picker.
            color_key = str(body["caption_color"])
            if color_key not in CAPTION_COLORS and resolve_caption_color(color_key) is None:
                return jsonify({"error": f"unknown caption_color '{color_key}'"}), 400
            match["caption_color"] = color_key
        if "caption_size" in body:
            # Either a legacy s/m/l/xl key or a raw number of script units at
            # the 1920-tall canvas (the editor's numeric stepper). Stored as
            # given; resolve_caption_size does the interpreting at render time.
            raw_size = body["caption_size"]
            if isinstance(raw_size, str) and raw_size in CAPTION_SIZES:
                match["caption_size"] = raw_size
            else:
                try:
                    size_number = float(raw_size)
                except (TypeError, ValueError):
                    return jsonify({"error": f"unknown caption_size '{raw_size}'"}), 400
                lo, hi = CAPTION_SIZE_RANGE
                if not (lo <= size_number <= hi):
                    return jsonify({"error": f"caption_size out of range ({lo}-{hi})"}), 400
                match["caption_size"] = int(round(size_number))
        if "caption_position" in body:
            position_key = str(body["caption_position"])
            if position_key not in CAPTION_POSITIONS:
                return jsonify({"error": f"unknown caption_position '{position_key}'"}), 400
            match["caption_position"] = position_key
        # Free caption position, dragged over the editor's visualizer: 0-1
        # fractions of the output frame, or null to go back to the
        # bottom/middle/top preset in caption_position. Both axes always travel
        # together (the editor PATCHes them as a pair), but each is validated on
        # its own so a partial body can't wedge one axis at a stale value.
        for axis in ("caption_pos_x", "caption_pos_y"):
            if axis not in body:
                continue
            raw_axis = body[axis]
            if raw_axis is None:
                match[axis] = None
                continue
            try:
                axis_value = float(raw_axis)
            except (TypeError, ValueError):
                return jsonify({"error": f"invalid {axis} '{raw_axis}'"}), 400
            if not (0.0 <= axis_value <= 1.0):
                return jsonify({"error": f"{axis} must be between 0 and 1"}), 400
            match[axis] = axis_value
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
        if "trim" in body:
            match["trim"] = _trim_spec(body["trim"]) or {}
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
        if "music_segments" in body:
            raw = body["music_segments"]
            if raw is None:
                match.pop("music_segments", None)
            else:
                if not isinstance(raw, list):
                    return jsonify({"error": "music_segments must be a list"}), 400
                segs = []
                for sg in raw[:200]:
                    st = max(0.0, float(sg.get("start", 0) or 0))
                    en = sg.get("end")
                    en = None if en is None else float(en)
                    if en is not None and en - st < 0.05:
                        continue
                    seg = {
                        "id": str(sg.get("id") or uuid.uuid4().hex[:8]),
                        "start": round(st, 3),
                        "end": None if en is None else round(en, 3),
                        "src_in": round(max(0.0, float(sg.get("src_in", 0) or 0)), 3),
                    }
                    # Volumen propio del tramo (0-1); sin el campo = volumen general.
                    if sg.get("volume") is not None:
                        seg["volume"] = round(max(0.0, min(1.0, float(sg["volume"]))), 3)
                    segs.append(seg)
                segs.sort(key=lambda x: x["start"])
                match["music_segments"] = segs
        if "music_volume" in body:
            match["music_volume"] = max(0.0, min(1.0, float(body["music_volume"])))
        if "voice_volume" in body:
            match["voice_volume"] = round(max(0.0, min(2.0, float(body["voice_volume"]))), 3)
        if "cta_id" in body:
            cta_id = body["cta_id"]
            if cta_id:
                if cta_id != CTA_NONE_ID and not any(c["id"] == cta_id for c in _all_ctas()):
                    return jsonify({"error": f"unknown cta_id '{cta_id}'"}), 400
                match["cta_id"] = str(cta_id)
            else:
                match.pop("cta_id", None)
        # Text template ("Plantilla") overlaid on the clip's first seconds.
        # template=None/"" means no card; the kicker/headline stay stored so
        # switching the template off and on again doesn't lose the text the
        # user (or the hook generator) wrote.
        if "template" in body:
            template_id = body["template"]
            if not template_id:
                match.pop("template", None)
            else:
                template_id = str(template_id)
                if template_id not in TEXT_TEMPLATE_IDS:
                    return jsonify({"error": f"unknown template '{template_id}'"}), 400
                match["template"] = template_id
        for text_field in ("template_kicker", "template_headline"):
            if text_field in body:
                match[text_field] = str(body[text_field] or "").strip()
        if "template_duration" in body:
            try:
                template_duration = float(body["template_duration"])
            except (TypeError, ValueError):
                return jsonify({"error": "invalid template_duration"}), 400
            lo, hi = TEMPLATE_DURATION_RANGE
            if not (lo <= template_duration <= hi):
                return jsonify({"error": f"template_duration out of range ({lo}-{hi})"}), 400
            match["template_duration"] = template_duration
        if "template_logo" in body:
            match["template_logo"] = bool(body["template_logo"])
        # Where the card sits on the frame and how big it is. Same fractions
        # the preview drags with, so moving the block in the editor and burning
        # it at export read the identical numbers.
        for geo_field, lo, hi in (
            ("template_x", -1.0, 2.0),
            ("template_y", -1.0, 2.0),
            ("template_scale", 0.15, 3.0),
            ("template_width_frac", 0.15, 1.0),
            ("template_headline_size", 0.02, 0.14),
        ):
            if geo_field in body:
                try:
                    value = float(body[geo_field])
                except (TypeError, ValueError):
                    return jsonify({"error": f"invalid {geo_field}"}), 400
                match[geo_field] = min(max(value, lo), hi)
        if "template_headline_color" in body:
            color = str(body["template_headline_color"] or "").strip()
            if color and not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                return jsonify({"error": "invalid template_headline_color"}), 400
            match["template_headline_color"] = color
        if "template_headline_color_runs" in body:
            headline_len = len(match.get("template_headline", "") or "")
            runs = []
            for run in body["template_headline_color_runs"] or []:
                if not isinstance(run, dict):
                    return jsonify({"error": "invalid template_headline_color_runs"}), 400
                try:
                    start = int(run["start"])
                    end = int(run["end"])
                except (KeyError, TypeError, ValueError):
                    return jsonify({"error": "invalid template_headline_color_runs"}), 400
                start = max(0, min(start, headline_len))
                end = max(0, min(end, headline_len))
                if end <= start:
                    continue
                run_color = str(run.get("color") or "").strip()
                if not re.fullmatch(r"#[0-9a-fA-F]{6}", run_color):
                    return jsonify({"error": "invalid template_headline_color_runs"}), 400
                runs.append({"start": start, "end": end, "color": run_color})
            match["template_headline_color_runs"] = runs
        if "template_bg_color" in body:
            bg_color = str(body["template_bg_color"] or "").strip()
            if bg_color and not re.fullmatch(r"#[0-9a-fA-F]{6}", bg_color):
                return jsonify({"error": "invalid template_bg_color"}), 400
            match["template_bg_color"] = bg_color
        if "template_bg_opacity" in body:
            try:
                bg_opacity = float(body["template_bg_opacity"])
            except (TypeError, ValueError):
                return jsonify({"error": "invalid template_bg_opacity"}), 400
            match["template_bg_opacity"] = max(0.0, min(1.0, bg_opacity))
        if "overlays" in body:
            match["overlays"] = _sanitize_overlays(body["overlays"])
        if "caption_splits" in body:
            try:
                match["caption_splits"] = sorted({round(float(k), 3) for k in body["caption_splits"] or []})
            except (TypeError, ValueError):
                return jsonify({"error": "invalid caption_splits"}), 400
        if "caption_group_keys" in body and not body["caption_group_keys"]:
            match.pop("caption_group_keys", None)
            match.pop("caption_group_words", None)
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


# Deshacer/rehacer del editor: vuelve campos a un valor que el propio server ya
# había guardado antes (el front lo tomó de un PATCH o del highlight cargado),
# así que se escribe tal cual, sin pasar por la validación campo a campo del
# PATCH. `unset` = campos que no existían antes del cambio.
_RESTORE_PROTECTED = {"id"}


@app.route("/api/highlights/<int:highlight_id>/restore", methods=["POST"])
def api_restore_highlight(highlight_id):
    cache_path = _highlights_cache_path(SOURCE_PATH)
    body = request.get_json(force=True) or {}
    fields = body.get("fields") or {}
    unset = body.get("unset") or []
    if not isinstance(fields, dict) or not isinstance(unset, list):
        return jsonify({"error": "bad restore body"}), 400

    with _highlights_lock:
        if not cache_path.exists():
            return jsonify({"error": "no highlights cached yet"}), 404
        with open(cache_path) as f:
            highlights = json.load(f)
        match = next((h for h in highlights if h.get("id") == highlight_id), None)
        if match is None:
            return jsonify({"error": f"highlight {highlight_id} not found"}), 404
        for k, v in fields.items():
            if k not in _RESTORE_PROTECTED:
                match[k] = v
        for k in unset:
            if k not in _RESTORE_PROTECTED:
                match.pop(k, None)
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


def _trim_spec(raw):
    """Clamped edge-trim {top,bottom,left,right} (fractions of the source
    frame) or None when the clip has no trim."""
    if not isinstance(raw, dict):
        return None
    out = {}
    for k in ("top", "bottom", "left", "right"):
        try:
            out[k] = max(0.0, min(0.45, float(raw.get(k) or 0.0)))
        except (TypeError, ValueError):
            out[k] = 0.0
    return out if any(v > 0 for v in out.values()) else None


def _border_spec(raw):
    """Frame border {width, color} (width = fraction of the frame width, max
    5%) or None when the clip has no border."""
    if not isinstance(raw, dict):
        return None
    try:
        width = max(0.0, min(0.05, float(raw.get("width") or 0.0)))
    except (TypeError, ValueError):
        return None
    color = str(raw.get("color") or "#ffffff")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        color = "#ffffff"
    return {"width": width, "color": color} if width > 0 else None


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


@app.route("/api/template_preview.png")
def api_template_preview():
    """The template card rendered exactly as the export will burn it, as a
    transparent PNG the editor lays over its 9:16 preview. Rendering it
    server-side instead of re-implementing the layout in CSS is what keeps
    "preview = export" true here: font metrics, wrapping and auto-shrinking
    all come from the same PIL code that renders the real frame."""
    template_id = request.args.get("template", "")
    if template_id not in TEXT_TEMPLATE_IDS:
        abort(404)
    # The card is rendered for the real output frame, not always 9:16: type
    # sizes are relative to the frame height, so the same headline is a
    # different shape on a vertical clip and on a wide one. The editor sends
    # the frame it is actually previewing.
    try:
        frame_w = max(int(request.args.get("frame_w", 1080)), 16)
        frame_h = max(int(request.args.get("frame_h", 1920)), 16)
    except (TypeError, ValueError):
        frame_w, frame_h = 1080, 1920
    # Rendered to a per-request temp file and served from memory: the editor
    # re-fetches this on every keystroke, so a single shared path on disk would
    # let one request serve another's half-written PNG.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        png_path = tmp.name
    def _float_arg(name):
        raw = request.args.get(name)
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _runs_arg(name):
        raw = request.args.get(name)
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except ValueError:
            return None
        return parsed if isinstance(parsed, list) else None

    try:
        rendered = clipper.render_template_block_png(
            png_path, frame_w, frame_h,
            template=template_id,
            kicker=request.args.get("kicker", ""),
            headline=request.args.get("headline", ""),
            logo_path=None if request.args.get("logo", "1") != "0" else "",
            width_frac=_float_arg("width"),
            headline_color=request.args.get("color") or None,
            headline_size_frac=_float_arg("font_size"),
            headline_color_runs=_runs_arg("color_runs"),
            bg_color=request.args.get("bg_color") or None,
            bg_opacity=_float_arg("bg_opacity"),
        )
        if rendered is None:
            abort(404)
        _, block_w, block_h = rendered
        with open(png_path, "rb") as f:
            payload = f.read()
    finally:
        if os.path.exists(png_path):
            os.remove(png_path)
    response = send_file(io.BytesIO(payload), mimetype="image/png")
    response.headers["Cache-Control"] = "no-store"
    # The editor sizes the draggable box from the block's own aspect; these
    # save it a second round-trip to measure the decoded image.
    response.headers["X-Block-W"] = str(block_w)
    response.headers["X-Block-H"] = str(block_h)
    response.headers["Access-Control-Expose-Headers"] = "X-Block-W, X-Block-H"
    return response


def _upload_video_overlay(file, ext: str):
    """Capa de video: se re-codifica una vez a H.264 mp4 con su audio (max 1280 de
    ancho) para que el <video> del preview la reproduzca siempre (iPhone .mov es
    HEVC) y el export no cargue con el original pesado."""
    payload = file.read(OVERLAY_VIDEO_MAX_BYTES + 1)
    if len(payload) > OVERLAY_VIDEO_MAX_BYTES:
        return jsonify({"error": "video too large (max 300 MB)"}), 400
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(payload).hexdigest()[:16] + ".mp4"
    path = OVERLAY_DIR / name
    if not path.exists():
        src = OVERLAY_DIR / f"{name}.src{ext}"
        tmp = OVERLAY_DIR / f"{name}.tmp.mp4"
        src.write_bytes(payload)
        try:
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                "-map", "0:v:0", "-map", "0:a:0?",
                "-vf", "scale='min(1280,iw)':-2",
                "-c:v", "h264_videotoolbox", "-b:v", "4000k", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
                "-movflags", "+faststart", str(tmp),
            ], check=True)
            os.replace(tmp, path)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            return jsonify({"error": f"not a readable video: {e}"}), 400
        finally:
            src.unlink(missing_ok=True)
    width, height, _fps = clipper._ffprobe_video_info(str(path))
    return jsonify({
        "file": name,
        "kind": "video",
        "label": os.path.basename(file.filename)[:80],
        "width": width,
        "height": height,
        "duration": clipper._ffprobe_duration(str(path)),
        "has_audio": clipper._has_audio_stream(str(path)),
    })


@app.route("/api/overlays", methods=["POST"])
def api_upload_overlay():
    """Takes one image from the editor's "Agregar PNG" button and parks it in
    output/overlay_assets under a content-addressed name, so re-uploading the
    same logo twice doesn't pile up copies. Returns the geometry the editor
    needs to place it at a sane starting size."""
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "no file"}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext in OVERLAY_VIDEO_EXTENSIONS:
        return _upload_video_overlay(file, ext)
    if ext not in OVERLAY_EXTENSIONS:
        return jsonify({"error": f"unsupported extension '{ext}'"}), 400
    payload = file.read(OVERLAY_MAX_BYTES + 1)
    if len(payload) > OVERLAY_MAX_BYTES:
        return jsonify({"error": "file too large (max 12 MB)"}), 400

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(payload).hexdigest()[:16] + ext
    path = OVERLAY_DIR / name
    if not path.exists():
        path.write_bytes(payload)

    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as img:
            width, height = img.size
    except Exception as e:
        path.unlink(missing_ok=True)
        return jsonify({"error": f"not a readable image: {e}"}), 400

    return jsonify({
        "file": name,
        "label": os.path.basename(file.filename)[:80],
        "width": width,
        "height": height,
    })


def _text_spec(raw):
    """Parametros de una capa de texto (medidas en px de un cuadro de 1080 de
    ancho). None si la capa no es de texto."""
    if not isinstance(raw, dict):
        return None
    def num(key, default, lo, hi):
        try:
            return min(max(float(raw.get(key, default)), lo), hi)
        except (TypeError, ValueError):
            return default
    def hexc(key, default):
        v = str(raw.get(key) or "")
        return v if re.fullmatch(r"#[0-9a-fA-F]{6}", v) else default
    return {
        "text": str(raw.get("text") or "")[:600],
        "size": num("size", 64, 16, 240),
        "weight": int(num("weight", 900, 100, 900)),
        "color": hexc("color", "#ffffff"),
        "bg": hexc("bg", "#000000"),
        "bg_op": num("bg_op", 0.6, 0.0, 1.0),
        "align": raw.get("align") if raw.get("align") in ("left", "center", "right") else "center",
        "pad": num("pad", 24, 0, 120),
        "max_w": num("max_w", 0.8, 0.2, 1.0),
        "min_h": num("min_h", 0, 0, 1920),
        "spans": _text_spans(raw.get("spans"), len(str(raw.get("text") or "")[:600])),
    }


def _text_spans(raw, n):
    """Tramos pintados de una capa de texto: [{s,e,size?,weight?,color?}] con
    s/e indices de caracter (e exclusivo). Solo quedan los que tienen algun
    estilo y largo > 0."""
    out = []
    for sp in (raw if isinstance(raw, list) else [])[:200]:
        if not isinstance(sp, dict):
            continue
        try:
            s, e = max(0, min(int(sp.get("s", 0)), n)), max(0, min(int(sp.get("e", 0)), n))
        except (TypeError, ValueError):
            continue
        if e <= s:
            continue
        item = {"s": s, "e": e}
        try:
            if sp.get("size") is not None:
                item["size"] = min(max(float(sp["size"]), 16), 240)
            if sp.get("weight") is not None:
                item["weight"] = int(min(max(float(sp["weight"]), 100), 900))
        except (TypeError, ValueError):
            pass
        if re.fullmatch(r"#[0-9a-fA-F]{6}", str(sp.get("color") or "")):
            item["color"] = sp["color"]
        if len(item) > 2:
            out.append(item)
    return out


@app.route("/api/text_layer", methods=["POST"])
def api_text_layer():
    """Renderiza el PNG de una capa de texto (mismo archivo para preview y
    export). Nombre por hash de los parametros: el mismo texto no se duplica."""
    spec = _text_spec(request.get_json(silent=True) or {})
    if spec is None or not spec["text"].strip():
        return jsonify({"error": "texto vacio"}), 400
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    name = "txt_" + hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16] + ".png"
    path = OVERLAY_DIR / name
    try:
        w, h = clipper.render_text_box_png(str(path), spec)
    except Exception as e:
        return jsonify({"error": f"render: {e}"}), 500
    return jsonify({"file": name, "width": w, "height": h, "text": spec})


def _overlay_media_info(path: Path) -> dict:
    """Miniaturas (una tira JPEG) y picos de audio de una capa de video, para
    dibujar su canal en la timeline igual que el principal: imagen arriba,
    onda abajo. Se cachea al lado del asset (mismo nombre por hash)."""
    info_path = path.with_name(path.name + ".media.json")
    if info_path.exists():
        with open(info_path) as f:
            return json.load(f)
    dur = 0.0
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True).stdout.strip()
        dur = float(out or 0)
    except ValueError:
        pass
    n = max(1, min(60, int(math.ceil(dur / 0.5)) if dur else 1))
    every = (dur / n) if dur else 1.0
    sprite = path.with_name(path.name + ".strip.jpg")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
        "-vf", f"fps={1.0 / every:.6f},scale=-2:48,tile={n}x1",
        "-frames:v", "1", "-q:v", "5", str(sprite)])
    has_audio = clipper._has_audio_stream(str(path))
    peaks = []
    if has_audio:
        raw = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
             "-ar", "8000", "-f", "s16le", "-"], capture_output=True).stdout
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        n_pts = max(1, int((samples.size / 8000.0) * 20))
        if samples.size:
            chunk = max(1, samples.size // n_pts)
            vals = np.abs(samples[: chunk * n_pts].reshape(-1, chunk)).max(axis=1)
            peaks = (vals / 32767.0).clip(0, 1).round(4).tolist()
    info = {"duration": dur, "thumbs_n": n, "thumbs_every": every,
            "has_thumbs": sprite.exists(), "has_audio": has_audio,
            "peaks": peaks, "peaks_per_sec": 20}
    with open(info_path, "w") as f:
        json.dump(info, f)
    return info


@app.route("/api/overlay_media/<path:filename>", methods=["GET"])
def api_overlay_media(filename):
    name = os.path.basename(filename)
    path = OVERLAY_DIR / name
    if not path.is_file():
        abort(404)
    try:
        return jsonify(_overlay_media_info(path))
    except Exception as exc:
        return jsonify({"error": repr(exc)}), 500


@app.route("/api/overlay_strip/<path:filename>", methods=["GET"])
def api_overlay_strip(filename):
    name = os.path.basename(filename)
    path = OVERLAY_DIR / name
    if not path.is_file():
        abort(404)
    _overlay_media_info(path)
    sprite = path.with_name(path.name + ".strip.jpg")
    if not sprite.exists():
        abort(404)
    return send_file(sprite, mimetype="image/jpeg")


@app.route("/api/overlays/<path:filename>", methods=["GET"])
def api_overlay_file(filename):
    name = os.path.basename(filename)
    if not (OVERLAY_DIR / name).is_file():
        abort(404)
    return send_from_directory(OVERLAY_DIR, name)


# --- Biblioteca de CTAs (sección "CTA" del panel) ---------------------------
# Cada clip elige qué outro lleva (h["cta_id"]). Sin cta_id = el CTA original
# del podcast Constantini, así los clips viejos exportan igual que antes.
#   type "card":  video real desenfocado + portada (opcional) + texto, como el
#                 original, con line1/highlight/line2 propios.
#   type "video": un video ya armado que se pega tal cual después del clip.
CTA_DIR = Path(__file__).resolve().parent / "output" / "cta_assets"
CTA_NONE_ID = "none"
CTA_DEFAULT_ID = "constantini"
CTA_BUILTINS = [{
    "id": CTA_DEFAULT_ID,
    "name": "Podcast Constantini",
    "type": "card",
    **clipper.DEFAULT_CTA_TEXT,
    "image": None,  # None = assets/cta_portada.png
    "builtin": True,
}]
CTA_IMAGE_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg"}
CTA_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
CTA_VIDEO_MAX_BYTES = 500 * 1024 * 1024
_cta_lock = threading.Lock()


def _cta_library_path() -> Path:
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    return Path(LOCAL_OUTPUT_DIR) / "cta_library.json"


def _load_custom_ctas() -> list:
    path = _cta_library_path()
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save_custom_ctas(ctas: list) -> None:
    path = _cta_library_path()
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(ctas, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _all_ctas() -> list:
    return CTA_BUILTINS + _load_custom_ctas()


def _find_cta(cta_id):
    """cta_id del highlight -> entrada de la biblioteca, o None si es "Sin
    CTA". Un id que ya no existe (CTA borrado) cae al default en vez de
    romper el export."""
    if cta_id == CTA_NONE_ID:
        return None
    ctas = _all_ctas()
    for c in ctas:
        if c["id"] == (cta_id or CTA_DEFAULT_ID):
            return c
    return ctas[0]


def _cta_image_path(cta: dict):
    """Ruta de la portada para clipper: None = portada Constantini, "" = sin
    imagen."""
    if cta.get("builtin"):
        return None
    return str(CTA_DIR / cta["image"]) if cta.get("image") else ""


@app.route("/api/ctas", methods=["GET"])
def api_list_ctas():
    return jsonify({"ctas": _all_ctas(), "default": CTA_DEFAULT_ID, "none": CTA_NONE_ID})


@app.route("/api/ctas", methods=["POST"])
def api_create_cta():
    """Alta de un CTA. multipart: type (card|video), name, y para card
    line1/highlight/line2 + file (portada opcional); para video, file."""
    try:
        cta_type = request.form.get("type", "card")
        name = (request.form.get("name") or "").strip()[:60]
        if cta_type not in ("card", "video"):
            return jsonify({"error": f"unknown type '{cta_type}'"}), 400
        if not name:
            return jsonify({"error": "falta el nombre"}), 400
        file = request.files.get("file")
        CTA_DIR.mkdir(parents=True, exist_ok=True)
        cta_id = "cta_" + hashlib.sha1(f"{name}{time.time()}".encode()).hexdigest()[:10]
        entry = {"id": cta_id, "name": name, "type": cta_type}

        if cta_type == "card":
            entry["line1"] = (request.form.get("line1") or "").strip()[:80]
            entry["highlight"] = (request.form.get("highlight") or "").strip()[:40]
            entry["line2"] = (request.form.get("line2") or "").strip()[:120]
            entry["image"] = None
            try:
                entry["duration"] = round(min(max(float(request.form.get("duration") or CTA_OUTRO_DURATION_SECONDS), 1.0), 15.0), 1)
            except ValueError:
                entry["duration"] = CTA_OUTRO_DURATION_SECONDS
            if file is not None and file.filename:
                ext = os.path.splitext(file.filename)[1].lower()
                if ext not in CTA_IMAGE_EXTENSIONS:
                    return jsonify({"error": f"imagen no soportada '{ext}'"}), 400
                payload = file.read(OVERLAY_MAX_BYTES + 1)
                if len(payload) > OVERLAY_MAX_BYTES:
                    return jsonify({"error": "imagen muy pesada (máx 12 MB)"}), 400
                img_name = hashlib.sha1(payload).hexdigest()[:16] + ext
                (CTA_DIR / img_name).write_bytes(payload)
                entry["image"] = img_name
            if not (entry["line1"] or entry["highlight"] or entry["line2"] or entry["image"]):
                return jsonify({"error": "el CTA necesita texto o imagen"}), 400
        else:
            if file is None or not file.filename:
                return jsonify({"error": "falta el video"}), 400
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in CTA_VIDEO_EXTENSIONS:
                return jsonify({"error": f"video no soportado '{ext}'"}), 400
            raw = CTA_DIR / f"{cta_id}.upload{ext}"
            file.save(raw)
            if raw.stat().st_size > CTA_VIDEO_MAX_BYTES:
                raw.unlink(missing_ok=True)
                return jsonify({"error": "video muy pesado (máx 500 MB)"}), 400
            # Normalizado a mp4 h264/aac: el navegador lo reproduce en el
            # preview (un .mov/.webm cualquiera no siempre) y el export lo
            # concatena sin sorpresas de códec.
            video_name = f"{cta_id}.mp4"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                     "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                     "-movflags", "+faststart", str(CTA_DIR / video_name)],
                    check=True,
                )
            except subprocess.CalledProcessError:
                (CTA_DIR / video_name).unlink(missing_ok=True)
                return jsonify({"error": "no se pudo leer el video"}), 400
            finally:
                raw.unlink(missing_ok=True)
            entry["video"] = video_name
            entry["duration"] = round(clipper._ffprobe_duration(str(CTA_DIR / video_name)), 2)

        with _cta_lock:
            ctas = _load_custom_ctas()
            ctas.append(entry)
            _save_custom_ctas(ctas)
        return jsonify(entry)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ctas/<cta_id>", methods=["DELETE"])
def api_delete_cta(cta_id):
    """Borra un CTA propio (los builtin no). Los archivos quedan en
    cta_assets: son chicos y un clip ya exportado no los necesita."""
    with _cta_lock:
        ctas = _load_custom_ctas()
        kept = [c for c in ctas if c["id"] != cta_id]
        if len(kept) == len(ctas):
            return jsonify({"error": "CTA inexistente o no borrable"}), 404
        _save_custom_ctas(kept)
    return jsonify({"ok": True})


@app.route("/api/ctas/<cta_id>/overlay.png", methods=["GET"])
def api_cta_overlay(cta_id):
    """PNG portada+texto del CTA a tamaño de cuadro, el mismo que compone el
    export -- el preview lo pone encima del video desenfocado."""
    cta = _find_cta(cta_id)
    if cta is None or cta["type"] != "card":
        abort(404)
    frame_w = max(64, min(int(request.args.get("frame_w", 1080)), 2160))
    frame_h = max(64, min(int(request.args.get("frame_h", 1920)), 3840))
    cache_dir = CTA_DIR / "_preview"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(json.dumps(
        [cta.get("line1"), cta.get("highlight"), cta.get("line2"), cta.get("image"),
         frame_w, frame_h], ensure_ascii=False).encode()).hexdigest()[:16]
    out = cache_dir / f"{key}.png"
    if not out.exists():
        clipper._render_cta_overlay_png(
            str(out), frame_w, frame_h,
            text={k: cta.get(k, "") for k in ("line1", "highlight", "line2")},
            portada_path=_cta_image_path(cta),
        )
    return send_from_directory(cache_dir, out.name)


@app.route("/api/ctas/<cta_id>/video", methods=["GET"])
def api_cta_video(cta_id):
    cta = _find_cta(cta_id)
    if cta is None or cta["type"] != "video" or not (CTA_DIR / cta["video"]).is_file():
        abort(404)
    return send_from_directory(CTA_DIR, cta["video"], conditional=True)


@app.route("/api/fonts/<path:filename>", methods=["GET"])
def api_font_file(filename):
    name = os.path.basename(filename)
    if not (FONTS_DIR / name).is_file():
        abort(404)
    return send_from_directory(FONTS_DIR, name)


TEMPLATE_HOOK_PROMPT = """Sos un editor de contenido que arma placas de texto para clips verticales de un podcast en español rioplatense.

Te paso la transcripción de un clip. Escribí una placa que funcione como gancho para que alguien que scrollea se detenga a mirarlo.

Reglas:
- "kicker": 2 a 4 palabras, el tema o la categoría del clip (va en una cajita naranja arriba, en mayúsculas). Ej: "DOLAR", "PLAZO FIJO", "ERROR COMUN".
- "headline": el gancho. Una sola frase de 6 a 14 palabras, concreta y específica de lo que se dice en ESTE clip. Nada de clickbait genérico ni de promesas que el clip no cumple. Sin signos de exclamación. Sin comillas.
- Ambos en español rioplatense, sin emojis.

Respondé SOLO con JSON:
{{"kicker": "...", "headline": "..."}}

Transcripción del clip:
{text}
"""


@app.route("/api/highlights/<int:highlight_id>/template_hook", methods=["POST"])
def api_template_hook(highlight_id):
    """Generates the template's kicker + headline from what is actually said
    in this clip. Returns them for the editor to drop into the (editable)
    inputs -- it deliberately does NOT persist anything, so a generated hook
    the user doesn't like never overwrites one they typed by hand: the panel
    PATCHes it like any manual edit once they keep it."""
    cache_path = _highlights_cache_path(SOURCE_PATH)
    with _highlights_lock:
        if not cache_path.exists():
            return jsonify({"error": "no highlights cached yet"}), 404
        with open(cache_path) as f:
            highlights = json.load(f)
        h = next((x for x in highlights if x.get("id") == highlight_id), None)
        if h is None:
            return jsonify({"error": f"highlight {highlight_id} not found"}), 404
        h = dict(h)

    start = float(h["start_time"])
    end = float(h["end_time"])
    transcript = _get_transcript(SOURCE_PATH)
    text = _transcript_lines_for_range(transcript, start, end)
    if not text.strip():
        text = str(h.get("transcript_override") or h.get("text") or "").strip()
    if not text:
        return jsonify({"error": "no transcript text for this clip"}), 400

    try:
        parsed = _parse_json_loose(call_local_llm(TEMPLATE_HOOK_PROMPT.format(text=text)))
    except Exception as e:
        print(f"[editor] template_hook failed: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "kicker": str(parsed.get("kicker", "")).strip(),
        "headline": str(parsed.get("headline", "")).strip(),
    })


_retranscribe_lock = threading.Lock()
_RETRANSCRIBE_PAD = 0.4  # silencio entre tramos para que whisper no pegue palabras de tomas distintas


def _kept_ranges(h) -> list:
    """[start, end] del clip menos sus `cuts`, en tiempo absoluto de la fuente."""
    start, end = float(h["start_time"]), float(h["end_time"])
    cuts = sorted(
        (max(start, float(c["start_time"])), min(end, float(c["end_time"])))
        for c in (h.get("cuts") or [])
    )
    kept, t = [], start
    for a, b in cuts:
        if b <= a:
            continue
        if a > t:
            kept.append((t, a))
        t = max(t, b)
    if end > t:
        kept.append((t, end))
    return [(a, b) for a, b in kept if b - a >= 0.05]


@app.route("/api/highlights/<int:highlight_id>/retranscribe", methods=["POST"])
def api_retranscribe(highlight_id):
    """"Retranscribir selección": vuelve a pasar whisper SOLO por lo que queda
    del clip después de los cortes, y reemplaza las palabras de la transcripción
    en [start_time, end_time] por las nuevas (mapeadas a tiempo de la fuente).

    Por qué: en videos con varias tomas de la misma frase, whisper sobre el
    video entero transcribe la frase una sola vez y alinea sus palabras a la
    toma equivocada (la que después se corta). La transcripción muestra todas
    las palabras, pero el visualizador no, porque caen dentro de los cortes.

    Las palabras dentro de los cortes se descartan: si después se deshace un
    corte, ese tramo queda sin subtítulos hasta volver a retranscribir."""
    global _TRANSCRIPT_CACHE, _DIARIZED_TRANSCRIPT_CACHE
    cache_path = _highlights_cache_path(SOURCE_PATH)
    with _highlights_lock:
        if not cache_path.exists():
            return jsonify({"error": "no highlights cached yet"}), 404
        with open(cache_path) as f:
            h = next((dict(x) for x in json.load(f) if x.get("id") == highlight_id), None)
    if h is None:
        return jsonify({"error": f"highlight {highlight_id} not found"}), 404
    S, E = float(h["start_time"]), float(h["end_time"])
    kept = _kept_ranges(h)
    if not kept:
        return jsonify({"error": "el clip no tiene tramos sin cortar"}), 400

    if not _retranscribe_lock.acquire(blocking=False):
        return jsonify({"error": "ya hay una retranscripción en curso"}), 409
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "kept.wav")
            # Read from the cached 16k mono diarization wav (not the multi-GB
            # source .mov) and input-seek to the first kept span, so trims are
            # relative to `seek` (see _audio_silences for the -ss gotcha).
            src_wav = _extract_audio_wav(SOURCE_PATH)
            seek = max(0.0, min(a for a, _ in kept) - 1.0)
            parts, labels = [], []
            for i, (a, b) in enumerate(kept):
                parts.append(
                    f"[0:a]atrim=start={a - seek:.4f}:end={b - seek:.4f},asetpts=PTS-STARTPTS,"
                    f"aresample=16000,apad=pad_dur={_RETRANSCRIBE_PAD}[a{i}]"
                )
                labels.append(f"[a{i}]")
            graph = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(kept)}:v=0:a=1[out]"
            res = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{seek:.4f}", "-i", src_wav,
                 "-filter_complex", graph, "-map", "[out]", "-ac", "1", "-ar", "16000", wav],
                capture_output=True, text=True,
            )
            if res.returncode != 0:
                raise RuntimeError(f"ffmpeg: {res.stderr[-400:]}")
            segs, words_by_seg, _ = local_transcriber._transcribe_dropping_dictation(
                wav, local_transcriber.LOCAL_WHISPER_LANGUAGE)

        # Offsets de cada tramo en el wav concatenado.
        offsets, o = [], 0.0
        for a, b in kept:
            offsets.append(o)
            o += (b - a) + _RETRANSCRIBE_PAD

        def seg_of(t):
            return max(0, min(len(kept) - 1, sum(1 for x in offsets if x <= t) - 1))

        new_words = []
        for words in words_by_seg:
            for w in words:
                text = (w.get("word") or "").strip()
                if not text:
                    continue
                ts, te = float(w["start"]), float(w["end"])
                i = seg_of(ts)
                L = kept[i][1] - kept[i][0]
                # Arranca en el silencio de relleno: whisper adelanta el inicio
                # de la palabra que abre el tramo siguiente.
                if ts - offsets[i] >= L - 0.02 and i + 1 < len(kept):
                    i += 1
                    ts = offsets[i]
                    L = kept[i][1] - kept[i][0]
                a, b = kept[i]
                ws = a + min(max(ts - offsets[i], 0.0), L)
                we = a + min(max(te - offsets[i], 0.0), L)
                if we - ws < 0.08:  # cruza un corte o quedó aplastada contra el borde
                    we = min(ws + 0.15, b)
                    ws = min(ws, max(a, we - 0.08))
                ws, we = min(max(round(ws, 3), a), b), min(max(round(we, 3), a), b)
                new_words.append({"start": ws, "end": we, "word": text})
        new_words.sort(key=lambda w: w["start"])
        local_transcriber._dedupe_word_timestamps(new_words)
        if not new_words:
            raise RuntimeError("whisper no devolvió palabras")

        # Segmentos nuevos: uno por tramo conservado (así la diarización y el
        # .srt quedan prolijos).
        new_segments = []
        for a, b in kept:
            ws_ = [w for w in new_words if a <= w["start"] < b + 1e-6]
            if ws_:
                new_segments.append({
                    "start": ws_[0]["start"], "end": ws_[-1]["end"],
                    "text": " ".join(w["word"] for w in ws_), "words": ws_,
                })

        transcript = transcribe_local(SOURCE_PATH)  # fresh copy from disk cache
        out_segments = []
        for seg in transcript["segments"]:
            words = seg.get("words") or []
            before = [w for w in words if (w["start"] + w["end"]) / 2 < S]
            after = [w for w in words if (w["start"] + w["end"]) / 2 > E]
            if not words:
                if seg["end"] <= S or seg["start"] >= E:
                    out_segments.append(seg)
                continue
            for piece in (before, after):
                if piece:
                    out_segments.append({
                        "start": piece[0]["start"], "end": piece[-1]["end"],
                        "text": " ".join(w["word"] for w in piece) if len(piece) != len(words) else seg["text"],
                        "words": piece,
                    })
        out_segments.extend(new_segments)
        out_segments.sort(key=lambda s: s["start"])
        old_count = sum(1 for seg in transcript["segments"] for w in (seg.get("words") or [])
                        if S <= (w["start"] + w["end"]) / 2 <= E)

        new_transcript = {"duration": transcript["duration"],
                          "segments": [{k: s[k] for k in ("start", "end", "text")} for s in out_segments]}
        local_transcriber._write_srt_cache(SOURCE_PATH, new_transcript)
        local_transcriber._words_cache_path(SOURCE_PATH).write_text(
            json.dumps([s.get("words") or [] for s in out_segments]), encoding="utf-8")

        # Correcciones manuales viejas del rango ya no apuntan a ninguna palabra.
        overrides = _load_overrides(SOURCE_PATH)
        kept_ov = {}
        for k, v in (overrides.get("words") or {}).items():
            try:
                a, b = (float(x) for x in k.split(":"))
            except ValueError:
                kept_ov[k] = v
                continue
            if not (S <= (a + b) / 2 <= E):
                kept_ov[k] = v
        overrides["words"] = kept_ov
        _save_overrides(SOURCE_PATH, overrides)

        _TRANSCRIPT_CACHE = None
        _DIARIZED_TRANSCRIPT_CACHE = None
    except Exception as e:
        print(f"[editor] retranscribe failed: {e}", flush=True)
        return jsonify({"error": str(e)}), 500
    finally:
        _retranscribe_lock.release()

    # Agrupamiento / tiempos manuales viejos apuntaban a las palabras viejas.
    with _highlights_lock:
        with open(cache_path) as f:
            highlights = json.load(f)
        for x in highlights:
            if x.get("id") == highlight_id:
                for k in ("caption_group_keys", "caption_group_words", "caption_time_overrides"):
                    x.pop(k, None)
                x["caption_splits"] = []
        _atomic_write_json(cache_path, highlights)
    print(f"[editor] retranscribe clip {highlight_id}: {old_count} -> {len(new_words)} words "
          f"over {len(kept)} kept ranges", flush=True)
    return jsonify({"ok": True, "old_words": old_count, "new_words": len(new_words),
                    "kept_ranges": len(kept)})


@app.route("/api/highlights/<int:highlight_id>/arrange_captions", methods=["POST"])
def api_arrange_captions(highlight_id):
    """"Acomodar subtitulos": regroups the clip's captions with the LLM and
    persists the result as `caption_group_keys` (word_start of each box's first
    word) tied to the `caption_words` it was generated for. Replaces any manual
    `caption_splits`, which were cuts on top of the old grouping."""
    cache_path = _highlights_cache_path(SOURCE_PATH)
    with _highlights_lock:
        if not cache_path.exists():
            return jsonify({"error": "no highlights cached yet"}), 404
        with open(cache_path) as f:
            highlights = json.load(f)
        h = next((x for x in highlights if x.get("id") == highlight_id), None)
        if h is None:
            return jsonify({"error": f"highlight {highlight_id} not found"}), 404
        h = dict(h)

    try:
        words_per_group = int(h.get("caption_words", DEFAULT_WORDS_PER_GROUP))
        transcript = _get_diarized_transcript(SOURCE_PATH)
        words = words_for_window(
            transcript.get("segments", []), float(h["start_time"]), float(h["end_time"]),
            time_overrides=h.get("caption_time_overrides"),
        )
        if words_per_group <= 1 or not words:
            return jsonify({"error": "con 1 palabra a la vez no hay nada que acomodar"}), 400
        # 1) Corregir palabras mal escritas / mal oídas (1:1, se guardan como
        # correcciones de palabra, igual que un doble clic manual). 2) Agrupar
        # sobre el texto ya corregido: una corrección puede cambiar puntuación.
        corrections = correct_clip_words(words, context=str(h.get("title") or ""))
        if corrections:
            overrides = _load_overrides(SOURCE_PATH)
            for c in corrections:
                overrides.setdefault("words", {})[_word_key(c["word_start"], c["word_end"])] = {
                    **overrides.get("words", {}).get(_word_key(c["word_start"], c["word_end"]), {}),
                    "text": c["to"],
                    "auto_from": c["from"],
                }
            _save_overrides(SOURCE_PATH, overrides)
            _apply_transcript_overrides(transcript, overrides)
            words = words_for_window(
                transcript.get("segments", []), float(h["start_time"]), float(h["end_time"]),
                time_overrides=h.get("caption_time_overrides"),
            )
        keys = arrange_caption_group_keys(words, words_per_group)
    except Exception as e:
        print(f"[editor] arrange_captions failed: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

    with _highlights_lock:
        with open(cache_path) as f:
            highlights = json.load(f)
        match = next((x for x in highlights if x.get("id") == highlight_id), None)
        if match is None:
            return jsonify({"error": f"highlight {highlight_id} not found"}), 404
        match["caption_group_keys"] = keys
        match["caption_group_words"] = words_per_group
        match["caption_splits"] = []
        _atomic_write_json(cache_path, highlights)
    return jsonify({
        "caption_group_keys": keys,
        "caption_group_words": words_per_group,
        "caption_splits": [],
        "corrections": [{"from": c["from"].strip(), "to": c["to"].strip()}
                        for c in corrections if not c.get("tail")],
    })


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

    final_path = str(_export_output_path(SOURCE_PATH, highlight_id))
    with _export_jobs_lock:
        job = _export_jobs.get(highlight_id)
        if job and job["state"] == "running":
            # Doble click / segunda pestaña: no arrancar otro render que pise
            # el mismo archivo (era el 'moov atom not found' del log).
            return jsonify(_export_job_public(job)), 409
        job = {
            "state": "running", "stage": "Preparando…", "step": 0,
            "total": _export_total_steps(h), "error": None,
            "started": time.time(), "finished": None,
            "url": f"/api/highlights/{highlight_id}/export/file",
        }
        _export_jobs[highlight_id] = job
    threading.Thread(
        target=_run_export_job, args=(highlight_id, h, final_path, job), daemon=True,
    ).start()
    return jsonify(_export_job_public(job)), 202


@app.route("/api/highlights/<int:highlight_id>/export/status", methods=["GET"])
def api_export_status(highlight_id):
    with _export_jobs_lock:
        job = _export_jobs.get(highlight_id)
        if job is None:
            return jsonify({"state": "idle"})
        return jsonify(_export_job_public(job))


# Export en segundo plano: antes el POST duraba todo el render (minutos) y el
# navegador podía cortar la conexión ("Failed to fetch") aunque el server
# siguiera trabajando. Ahora el POST devuelve al toque y el frontend consulta
# /export/status, que dice en qué etapa está.
_export_jobs: dict = {}
_export_jobs_lock = threading.Lock()
MAX_CONCURRENT_EXPORTS = 2
_EXPORT_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_EXPORTS)


def _export_job_public(job: dict) -> dict:
    out = dict(job)
    end = job["finished"] or time.time()
    out["elapsed"] = round(end - job["started"], 1)
    return out


def _export_total_steps(h: dict) -> int:
    # recorte + capas + outro, más los opcionales: audio, subs, velocidad, música
    return (3 + bool(h.get("audio_enhance")) + bool(h.get("captions_enabled", True))
            + (float(h.get("speed", 1.0)) != 1.0) + bool(h.get("music_track")))


def _run_export_job(highlight_id: int, h: dict, final_path: str, job: dict) -> None:
    def stage(label: str) -> None:
        with _export_jobs_lock:
            job["step"] += 1
            job["stage"] = label
        print(f"[editor] export {highlight_id}: {job['step']}/{job['total']} {label}", flush=True)

    # Se renderiza sobre un archivo de trabajo y recién al final se reemplaza
    # el clip exportado: el anterior sigue reproducible mientras tanto y un
    # render que falla a la mitad no deja un mp4 roto en su lugar.
    out_path = final_path + ".work.mp4"
    try:
        with _export_jobs_lock:
            job["stage"] = "En cola…"
        # Cada export ya usa varios cores (libx264/DeepFilter): más de 2 a la
        # vez no termina antes, solo reparte la CPU y dispara la RAM.
        with _EXPORT_SLOTS:
            with _export_jobs_lock:
                job["stage"] = "Preparando…"
            _render_export(highlight_id, h, out_path, stage)
        os.replace(out_path, final_path)
        with _export_jobs_lock:
            job.update(state="done", stage="Listo", finished=time.time())
    except Exception as exc:
        print(f"[editor] export failed for highlight {highlight_id}: {exc!r}", flush=True)
        with _export_jobs_lock:
            job.update(state="error", error=str(exc), finished=time.time())
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def _render_export(highlight_id: int, h: dict, out_path: str, stage) -> None:
    # Todas las pasadas que re-codifican antes del outro son intermedias
    # (h264_videotoolbox, ~2x más rápido); solo la concatenación del outro
    # --la última que toca el video-- usa libx264. Ver encoding.py.
    with intermediate_encodes():
        _render_export_passes(highlight_id, h, out_path, stage)


def _render_export_passes(highlight_id: int, h: dict, out_path: str, stage) -> None:
    final_encoded = False
    start_time = float(h["start_time"])
    end_time = float(h["end_time"])
    duration = end_time - start_time

    cut_segments = [
        (float(c["start_time"]) - start_time, float(c["end_time"]) - start_time)
        for c in h.get("cuts", [])
    ]
    cut_segments = [(s, e) for s, e in cut_segments if e > s] or None

    crop_enabled = bool(h.get("crop_enabled", False))
    crop_mode, manual_keyframes = _manual_crop_keyframes(h)
    trim = _trim_spec(h.get("trim"))

    stage("Recortando y armando el clip…")
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
            trim=trim,
        )
    else:
        clipper.crop_clip_local(
            SOURCE_PATH, start_time, end_time, EXPORT_ASPECT_RATIO, out_path,
            cut_segments=cut_segments,
            crop_enabled=crop_enabled,
            trim=trim,
        )

    if h.get("audio_enhance"):
        stage("Mejorando audio (DeepFilterNet)…")
        # Antes de subtítulos/velocidad/música: limpia SOLO la voz del
        # clip ya cortado (DeepFilterNet3), no el video fuente entero.
        # Reusa el WAV que ya generó el preview (misma base + cortes): si el
        # usuario lo escuchó en el editor, acá no se vuelve a correr la red.
        tmp_enhanced = out_path + ".enhanced.mp4"
        enhanced_wav = clipper.enhance_audio_preview(SOURCE_PATH, start_time, end_time, cut_segments)
        clipper.mux_enhanced_audio(out_path, enhanced_wav, tmp_enhanced)
        os.replace(tmp_enhanced, out_path)

    # Volumen de la voz (audio principal del clip): antes de outro/capas/música
    # para que solo afecte a la voz y no al audio del CTA ni a la música.
    voice_volume = h.get("voice_volume")
    voice_volume = 1.0 if voice_volume is None else float(voice_volume)
    if abs(voice_volume - 1.0) > 1e-3:
        tmp_voice = out_path + ".voice.mp4"
        clipper.apply_voice_gain(out_path, voice_volume, tmp_voice)
        os.replace(tmp_voice, out_path)

    speed = float(h.get("speed") or 1.0)
    speed_applied = False
    if h.get("captions_enabled", True):
        stage("Quemando subtítulos…")
        transcript = _get_diarized_transcript(SOURCE_PATH)
        words = words_for_window(
            transcript.get("segments", []), start_time, end_time,
            time_overrides=h.get("caption_time_overrides"),
        )
        if cut_segments:
            keep = editing.keep_segments_from_cuts(cut_segments, duration)
            words = editing.remap_words_across_cuts(words, keep)

        font_key = h.get("font", DEFAULT_FONT_KEY)
        tmp_subbed = out_path + ".subbed.mp4"
        if speed != 1.0:
            stage(f"Quemando subtítulos y velocidad {speed}x…")
        burn_subtitles(
            out_path, words, tmp_subbed,
            font_name=FONT_CHOICES.get(font_key, FONT_CHOICES[DEFAULT_FONT_KEY]),
            size=resolve_caption_size(h.get("caption_size")),
            position=h.get("caption_position", DEFAULT_CAPTION_POSITION),
            style=h.get("caption_style", DEFAULT_CAPTION_STYLE),
            color=h.get("caption_color", DEFAULT_CAPTION_COLOR),
            words_per_group=h.get("caption_words", DEFAULT_WORDS_PER_GROUP),
            pos=resolve_caption_pos(h.get("caption_pos_x"), h.get("caption_pos_y")),
            caption_splits=h.get("caption_splits"),
            caption_group_keys=_valid_group_keys(h, h.get("caption_words", DEFAULT_WORDS_PER_GROUP)),
            speed=speed,  # misma pasada que los subtítulos (ass → setpts)
        )
        os.replace(tmp_subbed, out_path)
        speed_applied = True

    if speed != 1.0 and not speed_applied:
        stage(f"Aplicando velocidad {speed}x…")
        tmp_sped = out_path + ".sped.mp4"
        clipper.apply_speed_local(out_path, speed, tmp_sped)
        os.replace(tmp_sped, out_path)

    # Overlay layers (the text template plus any PNG the user dropped on
    # the frame) on top of everything the clip already shows. They go after
    # the captions and after the speed change so their time windows are
    # real seconds of the final video (compositing before apply_speed_local
    # would stretch or shrink them with the footage), and before the CTA
    # outro so they stay inside the clip instead of leaking into the
    # appended outro. All of them are composited in one encode.
    stage("Capas y plantilla…")
    overlay_items = []
    scratch_pngs = []
    frame_w, frame_h, _fps = clipper._ffprobe_video_info(out_path)

    template_id = h.get("template")
    if template_id in TEXT_TEMPLATE_IDS:
        block_png = out_path + ".tmplblock.png"
        width_frac = float(h.get("template_width_frac") or clipper.TEMPLATE_BLOCK_W)
        block = clipper.render_template_block_png(
            block_png, frame_w, frame_h,
            template=template_id,
            kicker=h.get("template_kicker", ""),
            headline=h.get("template_headline", ""),
            logo_path=None if h.get("template_logo", True) else "",
            width_frac=width_frac,
            headline_color=h.get("template_headline_color") or None,
            headline_size_frac=h.get("template_headline_size"),
            headline_color_runs=h.get("template_headline_color_runs"),
            bg_color=h.get("template_bg_color") or None,
            bg_opacity=h.get("template_bg_opacity"),
        )
        if block:
            scratch_pngs.append(block_png)
            overlay_items.append({
                "png": block_png,
                "x": float(h.get("template_x", clipper.TEMPLATE_DEFAULT_X)),
                "y": float(h.get("template_y", clipper.TEMPLATE_DEFAULT_Y)),
                "w": width_frac * float(
                    h.get("template_scale", clipper.TEMPLATE_DEFAULT_SCALE)),
                "start": 0.0,
                "end": float(h.get("template_duration", DEFAULT_TEMPLATE_DURATION)),
                # Entrada de zocalo de noticiero: entra de costado y llega
                # desenfocada. El preview del editor replica estos mismos
                # numeros (ver ov-news-in en static/editor.html).
                "slide_in": clipper.TEMPLATE_SLIDE_SECONDS,
                "slide_from": clipper.TEMPLATE_SLIDE_FROM,
                "blur_in": clipper.TEMPLATE_BLUR_SIGMA_AT_1080,
            })

    for layer in _sanitize_overlays(h.get("overlays")):
        if layer["hidden"]:
            continue
        png_path = str(OVERLAY_DIR / layer["file"])
        if layer["kind"] == "png" and (layer["radius"] > 0 or layer["border_w"] > 0 or layer["crop"]):
            styled = f"{out_path}.ov{len(scratch_pngs)}.png"
            if clipper.style_overlay_png(png_path, styled, max(int(round(frame_w * layer["w"])), 2),
                                         layer["radius"], frame_w * layer["border_w"],
                                         layer["border_color"], crop=layer["crop"]):
                scratch_pngs.append(styled)
                png_path = styled
        overlay_items.append({
            "png": png_path,
            "video": layer["kind"] == "video",
            # Solo las usa el video: el PNG ya viene con esquinas/borde horneados.
            "radius": layer["radius"] if layer["kind"] == "video" else 0.0,
            "border_w": layer["border_w"] if layer["kind"] == "video" else 0.0,
            "border_color": layer["border_color"],
            "crop": layer["crop"] if layer["kind"] == "video" else None,
            "src_in": layer["src_in"],
            "speed": layer["speed"],
            "volume": 0.0 if layer["muted"] else layer["volume"],
            "x": layer["x"],
            "y": layer["y"],
            "w": layer["w"],
            "start": layer["start"],
            "end": layer["end"],
            "opacity": layer["opacity"],
            # A logo or a sticker the user placed by hand should just be
            # there, not swim in; the card is the thing that announces
            # itself, so only it keeps the default fades.
            "fade_in": 0.2 if layer["start"] > 0 else 0.0,
            "fade_out": 0.2 if layer["end"] is not None else 0.0,
            "anim": layer["anim_in"],
            "anim_dur": layer["anim_dur"],
        })

    # El marco va al final de la lista, o sea arriba de todo, y solo sobre el
    # clip (antes del outro CTA). Sigue el limite del encuadre: en split cada
    # mitad lleva su propio marco.
    border = _border_spec(h.get("border"))
    if border:
        border_png = out_path + ".border.png"
        panes = 2 if (crop_enabled and manual_keyframes and crop_mode == "split") else 1
        if clipper.render_border_png(border_png, frame_w, frame_h, border["width"], border["color"], panes):
            scratch_pngs.append(border_png)
            overlay_items.append({
                "png": border_png, "x": 0.0, "y": 0.0, "w": 1.0,
                "start": 0.0, "end": None, "fade_in": 0.0, "fade_out": 0.0,
            })

    # Capas (o musica) que terminan despues del clip: estirar el video (ultimo
    # cuadro congelado + silencio) hasta la ultima, asi el clip termina recien
    # cuando termina la pieza mas larga. El outro CTA va despues de la cola.
    try:
        explicit_ends = [float(it["end"]) for it in overlay_items if it.get("end") is not None]
        if h.get("music_track"):
            explicit_ends += [float(sg["end"]) for sg in (h.get("music_segments") or [])
                              if sg.get("end") is not None]
        tail = (max(explicit_ends) - clipper._ffprobe_duration(out_path)) if explicit_ends else 0.0
        if tail > 0.05:
            tmp_tail = out_path + ".tail.mp4"
            clipper.extend_frozen_tail(out_path, tmp_tail, tail)
            os.replace(tmp_tail, out_path)
    except Exception:
        for png in scratch_pngs:
            if os.path.exists(png):
                os.remove(png)
        raise

    if overlay_items:
        tmp_layers = out_path + ".layers.mp4"
        try:
            written = clipper.burn_overlay_items(out_path, tmp_layers, overlay_items)
            if written:
                os.replace(tmp_layers, out_path)
        finally:
            for png in scratch_pngs:
                if os.path.exists(png):
                    os.remove(png)

    cta = _find_cta(h.get("cta_id"))
    if cta is not None and cta["type"] == "video":
        stage("Outro con CTA…")
        tmp_carded = out_path + ".carded.mp4"
        with intermediate_encodes(False):
            clipper.append_video_outro(out_path, str(CTA_DIR / cta["video"]), tmp_carded)
        final_encoded = True
        os.replace(tmp_carded, out_path)

    card_cta = cta is not None and cta["type"] == "card"
    if card_cta:
        stage("Outro con CTA…")
        source_duration = clipper._ffprobe_duration(SOURCE_PATH)
        cta_seconds = float(cta.get("duration") or CTA_OUTRO_DURATION_SECONDS)
        cta_duration = min(cta_seconds, max(source_duration - end_time, 0))
    else:
        cta_duration = 0.0
    cta_start = end_time
    if card_cta and cta_duration < 0.5:
        # The clip runs to (almost) the end of the source, so there is no
        # following footage to blur under the CTA. Reuse the clip's own
        # last seconds instead: they play once normally and once more
        # blurred behind the CTA. Capped at the clip's length so a very
        # short clip never reaches before its own start.
        cta_duration = min(cta_seconds, duration)
        cta_start = end_time - cta_duration
    if cta_duration >= 0.5:
        tmp_carded = out_path + ".carded.mp4"
        # The outro has to keep the clip's own framing: with a manual crop
        # (and above all in 2-camera "split" mode) the raw source frame is
        # a 16:9 wide shot, so letting append_blurred_cta_outro center-crop
        # it to 9:16 lands on whatever sits between the two cameras -- a
        # wall, the desk -- instead of the stacked panes the viewer was
        # just watching. Reframing the outro window through the same
        # keyframes keeps the last framing on screen while it blurs out
        # (_manual_keyframe_cuts clamps every keyframe before the window to
        # 0.0 and lets the latest one win, which is exactly the framing in
        # effect at the clip's end).
        outro_src = None
        try:
            if crop_enabled and manual_keyframes:
                outro_src = out_path + ".ctasrc.mp4"
                clipper.crop_clip_manual(
                    SOURCE_PATH, cta_start, cta_start + cta_duration,
                    EXPORT_ASPECT_RATIO, outro_src,
                    keyframes=manual_keyframes,
                    mode=crop_mode,
                    trim=trim,
                )
            with intermediate_encodes(False):
                clipper.append_blurred_cta_outro(
                    out_path, SOURCE_PATH, cta_start, tmp_carded,
                    duration=cta_duration, blur_sigma=CTA_OUTRO_BLUR_SIGMA,
                    outro_clip_path=outro_src,
                    cta_text={k: cta.get(k, "") for k in ("line1", "highlight", "line2")},
                    portada_path=_cta_image_path(cta),
                    trim=trim,
                )
            final_encoded = True
        finally:
            if outro_src and os.path.exists(outro_src):
                os.remove(outro_src)
        os.replace(tmp_carded, out_path)

    if not final_encoded:
        # Sin outro (caso raro): lo último que se codificó fue intermedio
        # (videotoolbox, archivo ~4x más pesado) -- pasada final a libx264.
        tmp_final = out_path + ".final.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", out_path,
             *FINAL_VIDEO_ARGS, "-c:a", "copy", tmp_final],
            check=True,
        )
        os.replace(tmp_final, out_path)

    music_track = h.get("music_track")
    if music_track:
        stage("Mezclando música…")
        music_path = MUSIC_DIR / str(music_track)
        if not music_path.is_file():
            raise RuntimeError(f"music track not found: {music_track}")
        volume = float(h.get("music_volume", DEFAULT_MUSIC_VOLUME))
        tmp_mixed = out_path + ".mixed.mp4"
        clipper.mix_background_music(out_path, str(music_path), volume, tmp_mixed,
                                     segments=h.get("music_segments"))
        os.replace(tmp_mixed, out_path)


@app.route("/api/highlights/<int:highlight_id>/export/file", methods=["GET"])
def api_export_highlight_file(highlight_id):
    out_path = _export_output_path(SOURCE_PATH, highlight_id)
    if not out_path.exists():
        return jsonify({"error": "not exported yet"}), 404
    return send_file(out_path, conditional=True)


@app.route("/api/highlights/<int:highlight_id>/audio_enhanced", methods=["GET"])
def api_audio_enhanced_preview(highlight_id):
    """Voz del clip pasada por DeepFilterNet3, para el `<audio>` secundario
    del preview (mismo patrón que la música de fondo) -- no toca el video
    fuente ni requiere haber exportado el clip todavía."""
    cache_path = _highlights_cache_path(SOURCE_PATH)
    with _highlights_lock:
        if not cache_path.exists():
            return jsonify({"error": "no highlights cached yet"}), 404
        with open(cache_path) as f:
            highlights = json.load(f)
        h = next((x for x in highlights if x.get("id") == highlight_id), None)
        if h is None:
            return jsonify({"error": f"highlight {highlight_id} not found"}), 404
        h = dict(h)

    start_time = float(h["start_time"])
    end_time = float(h["end_time"])
    duration = end_time - start_time
    if duration <= 0:
        return jsonify({"error": "invalid start_time/end_time"}), 400
    cut_segments = [
        (float(c["start_time"]) - start_time, float(c["end_time"]) - start_time)
        for c in h.get("cuts", [])
    ]
    cut_segments = [(s, e) for s, e in cut_segments if e > s] or None

    try:
        wav_path = clipper.enhance_audio_preview(SOURCE_PATH, start_time, end_time, cut_segments)
    except Exception as exc:
        print(f"[editor] audio preview enhance failed for highlight {highlight_id}: {exc!r}", flush=True)
        return jsonify({"error": str(exc)}), 500
    return send_file(wav_path, mimetype="audio/wav", conditional=True)


MODEL_IDLE_UNLOAD_S = 600  # 10 min sin uso → se liberan DeepFilterNet y pyannote


def _idle_model_reaper() -> None:
    """Libera los modelos pesados que quedaron cargados si nadie los usa
    hace MODEL_IDLE_UNLOAD_S. Con varios editores abiertos a la vez esto es
    lo que evita que cada uno retenga ~1-2GB para siempre."""
    from shorts_generator.local import diarizer, transcriber
    while True:
        time.sleep(60)
        try:
            if clipper._deepfilter_worker.close_if_idle(MODEL_IDLE_UNLOAD_S):
                print("[idle] worker de DeepFilterNet liberado", flush=True)
            if diarizer.unload_if_idle(MODEL_IDLE_UNLOAD_S):
                print("[idle] pipeline de pyannote liberado", flush=True)
            # Whisper corre una sola vez por video (después todo sale del
            # cache .srt): el modelo se suelta en el primer tick tras usarlo.
            # unload_mlx_model no hace nada mientras haya una en curso.
            if transcriber.unload_mlx_model():
                print("[idle] modelo de whisper (MLX) liberado", flush=True)
        except Exception as e:
            print(f"[idle] reaper: {e}", flush=True)


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
    threading.Thread(target=_warmup_preview, daemon=True, name="warmup-preview").start()
    threading.Thread(target=_warmup_analysis, daemon=True, name="warmup-analysis").start()
    threading.Thread(target=_idle_model_reaper, daemon=True, name="idle-reaper").start()
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
