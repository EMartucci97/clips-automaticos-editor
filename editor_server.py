"""Standalone timeline editor: load the full source video, show LLM-detected
highlight zones as draggable markers on a timeline. Phase 1 only — no
silence-removal or format/render functions yet (see plan for later phases).

Usage:
    python3 editor_server.py <source_video_path> [--port 8502] [--num-clips 20]
"""
import argparse
import json
import os
import wave
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file

from shorts_generator.config import LOCAL_OUTPUT_DIR
from shorts_generator.highlights import _parse_json_loose, get_highlights
from shorts_generator.local.diarizer import _extract_audio_wav, diarize_and_label
from shorts_generator.local.llm import call_local_llm
from shorts_generator.local.transcriber import transcribe_local

app = Flask(__name__)

SOURCE_PATH = None
NUM_CLIPS = 20

_TRANSCRIPT_CACHE = None

SEGMENT_ANALYSIS_PROMPT = """Sos un editor de video que analiza un tramo de transcripción de una entrevista/podcast en español.

Te paso el texto transcripto de un tramo puntual (puede tener errores de transcripción, ignorá los que sean obvios). Tu tarea:
1. Identificá el TEMA CENTRAL del tramo: una frase corta y concreta.
2. Identificá los SUBTEMAS que se tocan dentro de ese tramo, en el orden en que aparecen. Si el tramo es corto o homogéneo, puede haber un solo subtema (igual al tema central).

Devolvé SOLO este JSON, sin texto adicional ni markdown:
{{"central_theme":"string","subtopics":["string", "string"]}}

Texto del tramo:
\"\"\"{text}\"\"\""""


def _highlights_cache_path(source_path: str) -> Path:
    stem = Path(source_path).stem
    return Path(LOCAL_OUTPUT_DIR) / f"{stem}.editor_highlights.json"


def _load_or_build_highlights(source_path: str) -> list:
    cache_path = _highlights_cache_path(source_path)
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
    with open(cache_path, "w") as f:
        json.dump(highlights, f, indent=2, ensure_ascii=False)
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


@app.route("/api/analyze_segment", methods=["POST"])
def api_analyze_segment():
    body = request.get_json(force=True) or {}
    text = str(body.get("text") or "").strip()
    if not text:
        return jsonify({"central_theme": "", "subtopics": []})

    prompt = SEGMENT_ANALYSIS_PROMPT.format(text=text)
    try:
        raw = call_local_llm(prompt)
        parsed = _parse_json_loose(raw)
    except Exception as e:
        print(f"[editor] analyze_segment failed: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "central_theme": str(parsed.get("central_theme", "")).strip(),
        "subtopics": [str(s).strip() for s in (parsed.get("subtopics") or []) if str(s).strip()],
    })


@app.route("/api/highlights", methods=["GET"])
def api_highlights():
    highlights = _load_or_build_highlights(SOURCE_PATH)
    return jsonify(highlights)


@app.route("/api/highlights/<int:highlight_id>", methods=["PATCH"])
def api_patch_highlight(highlight_id):
    cache_path = _highlights_cache_path(SOURCE_PATH)
    if not cache_path.exists():
        return jsonify({"error": "no highlights cached yet"}), 404

    with open(cache_path) as f:
        highlights = json.load(f)

    match = next((h for h in highlights if h.get("id") == highlight_id), None)
    if match is None:
        return jsonify({"error": f"highlight {highlight_id} not found"}), 404

    body = request.get_json(force=True) or {}
    if "start_time" in body:
        match["start_time"] = float(body["start_time"])
    if "end_time" in body:
        match["end_time"] = float(body["end_time"])
    if "transcript_override" in body:
        match["transcript_override"] = str(body["transcript_override"])

    with open(cache_path, "w") as f:
        json.dump(highlights, f, indent=2, ensure_ascii=False)

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
