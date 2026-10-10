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


def _transcribe_engine(media_path: str, language: Optional[str]):
    if _use_mlx():
        return _transcribe_mlx(media_path, language)
    return _transcribe_faster_whisper(media_path, language)


# Tomas dictadas: alguien fuera de cámara le dice la frase al orador en voz
# baja y él la repite fuerte. Whisper sobre todo el audio transcribe la frase
# una sola vez y estira sus palabras sobre la toma baja. Regla: un tramo de voz
# >= DICTATION_DB más bajo que la voz principal, cuyo texto se repite en los
# segundos siguientes, no se subtitula (gana la repetición fuerte). Si el tramo
# bajo NO se repite (ej. el orador baja la voz, otro hablante), se conserva.
# LOCAL_WHISPER_DICTATION_DB=0 lo desactiva.
_DICTATION_DB = float(os.getenv("LOCAL_WHISPER_DICTATION_DB", "8"))
_DICTATION_LOOKAHEAD = 25.0   # s después del tramo bajo donde buscar la repetición
_DICTATION_MATCH = 0.6        # fracción de palabras del tramo bajo que reaparecen en orden
_DICTATION_MIN_WORDS = 3      # frases más cortas ("dale", "bueno") nunca se descartan
_SR = 16000


def _load_wav16k(media_path: str):
    import subprocess
    import numpy as np
    res = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", media_path, "-vn",
         "-ac", "1", "-ar", str(_SR), "-f", "s16le", "-"],
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg: {res.stderr.decode(errors='ignore')[-300:]}")
    return np.frombuffer(res.stdout, dtype=np.int16)


def _write_wav16k(path: str, samples) -> None:
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SR)
        w.writeframes(samples.tobytes())


def _speech_regions(samples, below_db: float) -> list:
    """Tramos de voz (separados por >= 0,25 s de silencio) como (inicio, fin,
    bajo). `bajo` = su nivel (p80 de frames de 50 ms) está >= below_db por
    debajo de la voz principal (nivel que cubre el 60% del tiempo hablado,
    ponderado por duración)."""
    import numpy as np
    hop = int(_SR * 0.05)
    n = len(samples) // hop
    if n == 0:
        return []
    x = samples[: n * hop].astype(np.float32) / 32768.0
    db = 20 * np.log10(np.sqrt((x.reshape(n, hop) ** 2).mean(1)) + 1e-6)
    voiced = db > -45
    regs, i = [], 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j, gap, k = i, 0, i
        while k < n:
            if voiced[k]:
                j, gap = k, 0
            else:
                gap += 1
                if gap >= 5:
                    break
            k += 1
        if (j - i + 1) >= 8:  # >= 0,4 s
            regs.append((i, j + 1))
        i = k + 1
    if len(regs) < 3:
        return []
    levels = np.array([np.percentile(db[a:b][voiced[a:b]], 80) for a, b in regs])
    durs = np.array([b - a for a, b in regs], dtype=float)
    order = np.argsort(levels)
    cum = np.cumsum(durs[order]) / durs.sum()
    ref = levels[order][min(len(order) - 1, int(np.searchsorted(cum, 0.6)))]
    return [(a * 0.05, b * 0.05, bool(lv < ref - below_db)) for (a, b), lv in zip(regs, levels)]


def _norm_tokens(words) -> list:
    """Palabras normalizadas para comparar tomas: sin tildes ni puntuación y
    sin la 's' final (whisper alterna "rusa"/"rusas", "la"/"las")."""
    import unicodedata
    out = []
    for w in words:
        t = unicodedata.normalize("NFD", (w.get("word") or "").lower())
        t = re.sub(r"[^a-z0-9ñ]", "", "".join(c for c in t if unicodedata.category(c) != "Mn"))
        if len(t) > 2 and t.endswith("s"):
            t = t[:-1]
        if t:
            out.append(t)
    return out


def _lcs_len(a: list, b: list) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


_PIECE_MIN = 1.5  # s: frases más cortas se juntan con la siguiente (whisper alucina con audios muy cortos)


def _looks_hallucinated(words: list) -> bool:
    """Basura típica de whisper en audios cortos: caracteres no latinos o la
    misma palabra repetida en loop."""
    text = " ".join(w["word"] for w in words)
    if re.search(r"[^\x00-\u024f\s¿¡«»“”…—–'’]", text):
        return True
    tok = _norm_tokens(words)
    return len(tok) >= 8 and len(set(tok)) / len(tok) < 0.4


def _transcribe_dropping_dictation(media_path: str, language: Optional[str]):
    """Como _transcribe_engine, pero aplicando la regla de tomas dictadas.

    Si hay tramos en voz baja, whisper NO se pasa sobre el audio entero: cuando
    ve dos tomas de la misma frase seguidas (la dictada y la fuerte, o dos tomas
    fuertes) las colapsa en una y pierde/estira palabras (probado: silenciar o
    atenuar el tramo bajo, o cortar solo alrededor de él, no alcanza). Se
    transcribe cada frase (tramo entre silencios) por separado y cada tramo
    bajo cuyo texto reaparece en el audio fuerte de los siguientes
    _DICTATION_LOOKAHEAD s se descarta."""
    if _DICTATION_DB <= 0:
        return _transcribe_engine(media_path, language)
    import tempfile
    samples = _load_wav16k(media_path)
    regions = _speech_regions(samples, _DICTATION_DB)
    if not any(q for _, _, q in regions):
        return _transcribe_engine(media_path, language)
    duration = len(samples) / _SR
    n_quiet = sum(1 for _, _, q in regions if q)
    print(f"[transcribe/local] {n_quiet} tramos en voz baja: transcribo frase por frase y chequeo repeticiones", flush=True)

    # Pedazos: un tramo bajo solo; tramos fuertes seguidos se juntan hasta
    # _PIECE_MIN. Bordes en la mitad de cada silencio, así no se corta ninguna palabra.
    groups = []
    for a, b, q in regions:
        if groups and not q and not groups[-1][2] and groups[-1][1] - groups[-1][0] < _PIECE_MIN:
            groups[-1] = (groups[-1][0], b, False)
        else:
            groups.append((a, b, q))
    pieces = []
    for i, (a, b, q) in enumerate(groups):
        pa = 0.0 if i == 0 else (groups[i - 1][1] + a) / 2
        pb = duration if i == len(groups) - 1 else (b + groups[i + 1][0]) / 2
        pieces.append((pa, pb, a, b, q))

    out = []  # (start, end, quiet, segs, words_by_seg) en tiempo absoluto
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "piece.wav")
        for pa, pb, a, b, quiet in pieces:
            _write_wav16k(path, samples[int(pa * _SR):int(pb * _SR)])
            # Sin pasarle la frase anterior como prompt: probado, whisper
            # evita repetir lo que está en el prompt y se saltea la toma fuerte.
            segs, wbs, _ = _transcribe_engine(path, language)
            p_segs, p_words = [], []
            for seg, ws in zip(segs, wbs):
                ws = [
                    {"start": round(min(pa + w["start"], pb), 3),
                     "end": round(min(pa + w["end"], pb), 3),
                     "word": w["word"].strip()}
                    for w in ws if (w.get("word") or "").strip()
                ]
                if ws and _looks_hallucinated(ws):
                    print(f"[transcribe/local] descarto alucinación {pa:.2f}-{pb:.2f}: {' '.join(w['word'] for w in ws)[:80]!r}", flush=True)
                    continue
                if ws:
                    p_segs.append({"start": ws[0]["start"], "end": ws[-1]["end"],
                                   "text": " ".join(w["word"] for w in ws)})
                    p_words.append(ws)
            out.append((a, b, quiet, p_segs, p_words))

    segments, words_by_segment = [], []
    for a, b, quiet, p_segs, p_words in out:
        if quiet and p_words:
            q_tok = _norm_tokens([w for ws in p_words for w in ws])
            later = _norm_tokens([
                w for a2, _, q2, _, pw in out if not q2 and a2 >= b - 0.01
                for ws in pw for w in ws if w["start"] <= b + _DICTATION_LOOKAHEAD
            ])
            text = " ".join(s["text"] for s in p_segs)
            if len(q_tok) >= _DICTATION_MIN_WORDS and _lcs_len(q_tok, later) >= _DICTATION_MATCH * len(q_tok):
                print(f"[transcribe/local] dictado descartado {a:.2f}-{b:.2f}: {text!r}", flush=True)
                continue
            print(f"[transcribe/local] voz baja conservada {a:.2f}-{b:.2f}: {text!r}", flush=True)
        segments.extend(p_segs)
        words_by_segment.extend(p_words)
    return segments, words_by_segment, duration


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

    segments, words_by_segment, duration = _transcribe_dropping_dictation(media_path, language)

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
