"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models).
"""
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..config import LOCAL_OUTPUT_DIR
from .encoding import probe_display_video, video_encode_args
from .editing import (
    _snap_forward,
    find_cut_segments,
    keep_segments_from_cuts,
    remap_time,
    remap_words_across_cuts,
)


def _slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "clip"


def _clip_slug(highlight: Dict) -> str:
    """Prefer the detected sales angle, fall back to the clip's topic/title."""
    angle = highlight.get("sales_angle")
    if angle and not highlight.get("sales_angle_fallback"):
        return _slugify(angle)
    return _slugify(highlight.get("title") or "clip")


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


MAX_TRIM_PER_AXIS = 0.45


def _trim_filter(source_path: str, trim: Optional[Dict]) -> Optional[str]:
    """ffmpeg `crop=w:h:x:y` that drops the editor's edge trim (fractions 0-1
    of the source frame per side: top/bottom/left/right), or None when there is
    nothing to trim. Applied while cutting the subclip so every later stage
    (manual framing, face tracking, split) sees the trimmed frame as the whole
    frame and its normalized coordinates stay relative to it."""
    if not trim:
        return None
    t, b, l, r = (
        max(0.0, min(MAX_TRIM_PER_AXIS, float(trim.get(k) or 0.0)))
        for k in ("top", "bottom", "left", "right")
    )
    if t + b + l + r <= 1e-6:
        return None
    src_w, src_h = _ffprobe_resolution(source_path)
    x = int(round(src_w * l))
    y = int(round(src_h * t))
    w = int(round(src_w * (1.0 - l - r)))
    h = int(round(src_h * (1.0 - t - b)))
    w = max(2, w - (w % 2))
    h = max(2, h - (h % 2))
    x = max(0, min(src_w - w, x))
    y = max(0, min(src_h - h, y))
    return f"crop={w}:{h}:{x}:{y}"


def _cut_subclip(source_path: str, start: float, end: float, out_path: str,
                 intermediate: bool = False, trim: Optional[Dict] = None) -> str:
    """ffmpeg -ss start -t dur → re-encoded mp4 with audio.

    -ss MUST come before -i (input seek): placed after -i, ffmpeg decodes the
    whole source from t=0 up to `start` — on a 55-min HEVC podcast a clip at
    minute 40 took ~166s instead of ~9s. Input seek is still frame-accurate
    here because we re-encode (accurate_seek drops the pre-roll frames)."""
    trim_vf = _trim_filter(source_path, trim)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", source_path,
        "-t", f"{end - start:.3f}",
        *(["-vf", trim_vf] if trim_vf else []),
        *video_encode_args(intermediate),
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _cut_subclip_multi(
    source_path: str,
    start: float,
    end: float,
    cut_segments: List[Tuple[float, float]],
    out_path: str,
    intermediate: bool = False,
    trim: Optional[Dict] = None,
) -> str:
    """Like `_cut_subclip`, but also removes each (start, end) clip-relative
    span in `cut_segments` (silence gaps / filler words -- see editing.py) in
    one ffmpeg filter_complex trim+concat pass, instead of a plain -ss/-to
    cut. Falls back to `_cut_subclip` when there's nothing left to cut after
    merging, so unaffected callers see identical behaviour to before."""
    duration = end - start
    keep = keep_segments_from_cuts(cut_segments, duration)
    if not keep or (len(keep) == 1 and keep[0][0] <= 1e-6 and keep[0][1] >= duration - 1e-6):
        return _cut_subclip(source_path, start, end, out_path, intermediate, trim=trim)

    trim_vf = _trim_filter(source_path, trim)
    trim_chain = f",{trim_vf}" if trim_vf else ""
    # Input-seek a little before `start` (see _cut_subclip) so ffmpeg doesn't
    # decode the source from t=0; after an input seek the stream timestamps
    # restart at 0 at `seek`, so the trims are relative to it.
    seek = max(0.0, start - 2.0)
    filter_parts = []
    concat_inputs = []
    for i, (s, e) in enumerate(keep):
        abs_s, abs_e = start - seek + s, start - seek + e
        filter_parts.append(
            f"[0:v]trim=start={abs_s:.3f}:end={abs_e:.3f},setpts=PTS-STARTPTS{trim_chain}[v{i}];"
            f"[0:a]atrim=start={abs_s:.3f}:end={abs_e:.3f},asetpts=PTS-STARTPTS[a{i}];"
        )
        concat_inputs.append(f"[v{i}][a{i}]")
    filter_complex = (
        "".join(filter_parts)
        + "".join(concat_inputs)
        + f"concat=n={len(keep)}:v=1:a=1[outv][outa]"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{seek:.3f}",
        "-t", f"{end - seek + 0.5:.3f}",  # input option: stop reading past `end`
        "-i", source_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        *video_encode_args(intermediate),
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def apply_speed_local(in_path: str, speed: float, out_path: str) -> str:
    """Re-encodes in_path at `speed`x using ffmpeg's time-stretch filters
    (setpts for video, atempo for audio) instead of changing the sample
    rate, so voice pitch is preserved — no "chipmunk" effect. `atempo`
    supports 0.5-100.0 in one pass, comfortably covering the 1.0-1.3x range
    the editor exposes, so no filter chaining is needed."""
    if speed == 1.0:
        if os.path.abspath(in_path) != os.path.abspath(out_path):
            os.replace(in_path, out_path)
        return out_path
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-filter_complex", f"[0:v]setpts=PTS/{speed}[v];[0:a]atempo={speed}[a]",
        "-map", "[v]", "-map", "[a]",
        *video_encode_args(),
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


# "Mejorar audio": DeepFilterNet3, elegido a mano por el usuario contra un
# ffmpeg afftdn (comparación en vivo, clip "Nordelta") -- regenera la voz y
# saca ruido de ambiente (roce de micrófono, etc.) mucho mejor que un filtro
# clásico, corre local (CPU) y no consume tokens. Instalado en un venv propio
# ~/.deepfilter-venv, aislado del venv del proyecto (mismo patrón que
# ~/.graphify-venv) porque trae su propio torch/torchaudio.
DEEPFILTER_PYTHON = os.getenv(
    "DEEPFILTER_PYTHON", os.path.expanduser("~/.deepfilter-venv/bin/python")
)
_DEEPFILTER_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "deepfilter_worker.py")

# Cache por contenido (no por highlight id/mtime): si el clip no cambió, un
# re-export o una segunda apertura del preview reutiliza el wav ya procesado
# en vez de correr la red de nuevo (0.57s por 25s de audio en M4, pero suma
# si el usuario reabre el editor seguido).
_AUDIO_ENHANCE_CACHE_DIR = Path(LOCAL_OUTPUT_DIR) / ".audio_enhance_cache"


def _audio_enhance_cache_key(in_path: str) -> str:
    digest = hashlib.sha256()
    with open(in_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_deepfilter() -> None:
    if not os.path.exists(DEEPFILTER_PYTHON):
        raise RuntimeError(
            f"DeepFilterNet no está instalado en {DEEPFILTER_PYTHON}. Instalar con:\n"
            "  python3 -m venv ~/.deepfilter-venv && "
            "~/.deepfilter-venv/bin/pip install deepfilternet torch torchaudio soundfile"
        )


class _DeepFilterWorker:
    """Un único proceso de DeepFilterNet3 con el modelo cargado UNA vez,
    reusado por todos los clips (en vez de invocar el CLI y recargar el
    modelo -- ~2s -- en cada request). Se arranca perezoso, en el primer
    enhance real, no al importar este módulo.

    Lock porque la inferencia es CPU y no hay ninguna ganancia en paralelizar
    dos requests contra el mismo proceso -- serializarlas es gratis y evita
    pisarse los stdin/stdout. Si el proceso murió (crash, kill manual), se
    reintenta una vez con un proceso nuevo antes de darse por vencido."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._last_used = 0.0

    def close_if_idle(self, max_idle_s: float) -> bool:
        """Mata el proceso (y libera su RAM) si nadie lo usó en max_idle_s.
        El próximo enhance lo vuelve a levantar (~2s de carga). No bloquea:
        si hay un enhance en curso, no hace nada."""
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if self._proc is None or self._proc.poll() is not None:
                return False
            if time.time() - self._last_used < max_idle_s:
                return False
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None
            return True
        finally:
            self._lock.release()

    def _spawn(self) -> None:
        proc = subprocess.Popen(
            [DEEPFILTER_PYTHON, _DEEPFILTER_WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Salteamos líneas que no sean JSON válido -- alguna dependencia de
        # deep-filter (torch/torchaudio) podría imprimir un warning suelto a
        # stdout durante la carga del modelo; no vale la pena que tire abajo
        # todo el worker por eso.
        status = None
        while True:
            ready_line = proc.stdout.readline()
            if not ready_line:
                raise RuntimeError(
                    "El worker de DeepFilterNet se cerró al arrancar (revisar instalación en "
                    f"{DEEPFILTER_PYTHON})"
                )
            try:
                status = json.loads(ready_line)
            except json.JSONDecodeError:
                continue
            break
        if not status.get("ready"):
            raise RuntimeError(f"El worker de DeepFilterNet no pudo cargar el modelo: {status.get('error')}")
        self._proc = proc

    def _request(self, in_path: str, out_path: str) -> str:
        assert self._proc is not None
        req = json.dumps({"in": in_path, "out": out_path}) + "\n"
        self._proc.stdin.write(req)
        self._proc.stdin.flush()
        # Mismo motivo que en _spawn: ignorar cualquier línea que no sea JSON
        # válido en vez de reventar con ella.
        while True:
            line = self._proc.stdout.readline()
            if not line:
                return ""
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            return line

    def enhance(self, in_path: str, out_path: str) -> None:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._spawn()
            try:
                line = self._request(in_path, out_path)
            except (BrokenPipeError, OSError):
                line = ""
            if not line:
                self._spawn()  # murió a mitad de un request -- un solo reintento
                line = self._request(in_path, out_path)
                if not line:
                    raise RuntimeError("El worker de DeepFilterNet se cayó dos veces seguidas")
            self._last_used = time.time()
            result = json.loads(line)
            if not result.get("ok"):
                raise RuntimeError(f"DeepFilterNet falló: {result.get('error')}")


_deepfilter_worker = _DeepFilterWorker()


def _enhanced_wav_cached(raw_wav: str) -> Path:
    """Runs DeepFilterNet3 on `raw_wav` (mono, any sr) and returns the cached
    enhanced wav's path, reusing a previous run if this exact audio (by
    content hash) was already processed."""
    _require_deepfilter()
    cache_key = _audio_enhance_cache_key(raw_wav)
    _AUDIO_ENHANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_wav = _AUDIO_ENHANCE_CACHE_DIR / f"{cache_key}.wav"
    if not cached_wav.exists():
        # El sufijo va ANTES de .wav: save_audio del worker elige el formato
        # por extensión, y "xxx.wav.tmp-123" no es una que reconozca.
        # pid + thread: el editor corre threaded=True, dos requests en el mismo
        # proceso compartían el mismo tmp y se pisaban.
        tmp_cached = str(cached_wav.with_suffix(f".tmp-{os.getpid()}-{threading.get_ident()}.wav"))
        _deepfilter_worker.enhance(raw_wav, tmp_cached)
        os.replace(tmp_cached, cached_wav)
    return cached_wav


def enhance_audio_local(in_path: str, out_path: str) -> str:
    """Corre DeepFilterNet3 sobre el audio de `in_path` y lo vuelve a mezclar
    con el video original (copiado tal cual, sin recodificar) en `out_path`.
    Pensado para un clip YA cortado (25s-2min), no para el video fuente
    entero -- el usuario lo pidió así explícitamente porque procesar todo el
    video sería mucho más lento y no hace falta: solo el clip exportado se
    escucha."""
    _require_deepfilter()
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_wav = os.path.join(tmp_dir, "raw.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", in_path,
             "-vn", "-ac", "1", "-ar", "48000", raw_wav],
            check=True,
        )
        cached_wav = _enhanced_wav_cached(raw_wav)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", in_path, "-i", str(cached_wav),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            out_path,
        ]
        subprocess.run(cmd, check=True)
    return out_path


# Margen/redondeo de la base mejorada: la red corre UNA vez sobre el rango del
# clip SIN cortes, estirado a múltiplos de _ENHANCE_BASE_STEP con margen. Así
# mover un corte, o correr inicio/fin unos segundos, reusa la misma base y
# solo re-aplica atrim/concat sobre el WAV ya limpio (~0.1s en vez de ~30s).
_ENHANCE_BASE_MARGIN = 2.0
_ENHANCE_BASE_STEP = 5.0


def _enhanced_base_range(source_path: str, start: float, end: float) -> Tuple[float, float, Path]:
    base_start = max(0.0, math.floor((start - _ENHANCE_BASE_MARGIN) / _ENHANCE_BASE_STEP) * _ENHANCE_BASE_STEP)
    base_end = math.ceil((end + _ENHANCE_BASE_MARGIN) / _ENHANCE_BASE_STEP) * _ENHANCE_BASE_STEP
    st = os.stat(source_path)
    key_src = json.dumps([os.path.abspath(source_path), st.st_size, int(st.st_mtime), base_start, base_end])
    key = hashlib.sha256(key_src.encode()).hexdigest()[:32]
    return base_start, base_end, _AUDIO_ENHANCE_CACHE_DIR / f"base_{key}.wav"


def _enhanced_base_wav(source_path: str, start: float, end: float) -> Tuple[float, Path]:
    """(base_start, wav) con la voz mejorada de [base_start, base_end] del source."""
    _require_deepfilter()
    base_start, base_end, base_wav = _enhanced_base_range(source_path, start, end)
    _AUDIO_ENHANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if base_wav.exists():
        return base_start, base_wav
    with _preview_key_lock(base_wav.name):
        if base_wav.exists():
            return base_start, base_wav
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_wav = os.path.join(tmp_dir, "raw.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-ss", f"{base_start:.3f}", "-t", f"{base_end - base_start:.3f}", "-i", source_path,
                 "-vn", "-ac", "1", "-ar", "48000", raw_wav],
                check=True,
            )
            tmp_base = base_wav.with_suffix(f".tmp-{os.getpid()}-{threading.get_ident()}.wav")
            _deepfilter_worker.enhance(raw_wav, str(tmp_base))
            os.replace(tmp_base, base_wav)
    return base_start, base_wav


def enhance_audio_preview(
    source_path: str, start: float, end: float,
    cut_segments: Optional[List[Tuple[float, float]]],
) -> str:
    """WAV de voz mejorada (DeepFilterNet3) del rango/cortes del clip, para el
    `<audio>` secundario del preview y para el export (mismo archivo, así el
    export no vuelve a correr la red). Los cortes se aplican con el mismo
    atrim/concat que usa el corte de video, así queda sincronizado."""
    duration = end - start
    keep = keep_segments_from_cuts(cut_segments, duration) if cut_segments else [(0.0, duration)]
    base_start, base_wav = _enhanced_base_wav(source_path, start, end)
    key_src = json.dumps([base_wav.name, round(start, 3), [[round(s, 3), round(e, 3)] for s, e in keep]])
    key = hashlib.sha256(key_src.encode()).hexdigest()[:32]
    preview_wav = _AUDIO_ENHANCE_CACHE_DIR / f"preview_{key}.wav"
    if preview_wav.exists():
        return str(preview_wav)
    with _preview_key_lock(key):
        if preview_wav.exists():
            return str(preview_wav)
        offset = start - base_start
        filter_parts, concat_inputs = [], []
        for i, (s, e) in enumerate(keep):
            filter_parts.append(
                f"[0:a]atrim=start={offset + s:.3f}:end={offset + e:.3f},asetpts=PTS-STARTPTS[a{i}];"
            )
            concat_inputs.append(f"[a{i}]")
        filter_complex = (
            "".join(filter_parts) + "".join(concat_inputs)
            + f"concat=n={len(keep)}:v=0:a=1[outa]"
        )
        tmp_preview = preview_wav.with_suffix(f".tmp-{os.getpid()}-{threading.get_ident()}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(base_wav),
             "-filter_complex", filter_complex, "-map", "[outa]",
             "-ac", "1", "-ar", "48000", str(tmp_preview)],
            check=True,
        )
        os.replace(tmp_preview, preview_wav)
    return str(preview_wav)


def mux_enhanced_audio(in_path: str, enhanced_wav: str, out_path: str) -> str:
    """Reemplaza el audio de `in_path` por `enhanced_wav` (video copiado)."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", in_path, "-i", enhanced_wav,
         "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
         "-shortest", out_path],
        check=True,
    )
    return out_path


def apply_voice_gain(in_path: str, gain: float, out_path: str) -> str:
    """Sube/baja el audio principal del clip (la voz) por `gain` (1.0 = igual).
    alimiter evita que un gain > 1 sature los picos. Video copiado."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", in_path,
         "-map", "0:v", "-map", "0:a?",
         "-af", f"volume={gain:.3f},alimiter=limit=0.97:level=disabled",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", out_path],
        check=True,
    )
    return out_path


_preview_locks: Dict[str, threading.Lock] = {}
_preview_locks_guard = threading.Lock()


def _preview_key_lock(key: str) -> threading.Lock:
    with _preview_locks_guard:
        return _preview_locks.setdefault(key, threading.Lock())


def _music_segments_filter(segments: list, duration: float, volume: float,
                           fade_seconds: float) -> Optional[str]:
    """filter_complex chunk that builds [music] from the looped input 1 using
    the editor's music segments ({start, end|None, src_in}, in output
    seconds). Each segment plays the track from src_in, delayed to start.
    Returns None when no segment falls inside the output."""
    segs = []
    for sg in segments:
        st = max(0.0, float(sg.get("start", 0) or 0))
        en = sg.get("end")
        en = duration if en is None else min(float(en), duration)
        if en - st < 0.05:
            continue
        # Volumen propio del tramo; sin el campo, el general de la música.
        vol = volume if sg.get("volume") is None else max(0.0, min(1.0, float(sg["volume"])))
        segs.append((st, en, max(0.0, float(sg.get("src_in", 0) or 0)), vol))
    if not segs:
        return None
    parts = [f"[1:a]asplit={len(segs)}" + "".join(f"[ms{i}]" for i in range(len(segs)))
             if len(segs) > 1 else "[1:a]anull[ms0]"]
    labels = []
    for i, (st, en, src_in, vol) in enumerate(segs):
        length = en - st
        # Fade-in largo solo cuando arranca la canción desde cero en el 0
        # (comportamiento histórico); los cortes intermedios llevan fades
        # cortos para que no hagan click.
        fin = fade_seconds if (st < 0.01 and src_in < 0.01) else min(0.03, length / 4)
        fout = min(0.03, length / 4) if en < duration - 0.01 else 0
        chain = (f"[ms{i}]atrim={src_in}:{src_in + length},asetpts=PTS-STARTPTS,"
                 f"afade=t=in:st=0:d={fin}")
        if fout:
            chain += f",afade=t=out:st={length - fout}:d={fout}"
        chain += f",volume={vol}"
        ms = int(round(st * 1000))
        if ms:
            chain += f",adelay={ms}:all=1"
        parts.append(chain + f"[mc{i}]")
        labels.append(f"[mc{i}]")
    if len(labels) > 1:
        parts.append("".join(labels) + f"amix=inputs={len(labels)}:duration=longest:"
                     f"dropout_transition=0:normalize=0[music]")
    else:
        parts.append(f"{labels[0]}anull[music]")
    return ";".join(parts)


def mix_background_music(in_path: str, music_path: str, volume: float, out_path: str,
                          fade_seconds: float = 1.0, segments: Optional[list] = None) -> str:
    """Loops `music_path` under in_path's existing audio via ffmpeg amix,
    trimmed to in_path's duration with a short fade-in so the loop seam
    doesn't click. No fade-out is applied at the end -- the music holds its
    volume through the final frame instead of dipping down, per user
    preference. Video stream is copied untouched; only audio is re-encoded.
    `volume` is linear (0.0-1.0)."""
    duration = _ffprobe_duration(in_path)
    # normalize=0 keeps the clip's own voice track at its original level --
    # amix's default normalize=1 would divide every input's volume by the
    # input count, quietly ducking the dialogue just from adding a track.
    music_chain = None
    if segments is not None:
        music_chain = _music_segments_filter(segments, duration, volume, fade_seconds)
        if music_chain is None:
            # Todos los tramos borrados: el clip sale sin música.
            shutil.copyfile(in_path, out_path)
            return out_path
    if music_chain is None:
        music_chain = (f"[1:a]atrim=0:{duration},afade=t=in:st=0:d={fade_seconds},"
                       f"volume={volume}[music]")
    filter_complex = (
        f"{music_chain};"
        f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-stream_loop", "-1", "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _ffprobe_video_info(path: str) -> Tuple[int, int, float]:
    """(width, height, fps) of a media file's first video stream, ya rotado
    como lo decodifica ffmpeg (ver probe_display_video)."""
    w, h, fps_str = probe_display_video(path)
    fps = 30.0
    if fps_str and "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) else 30.0
    return w, h, fps


# Safe area for the CTA outro, as fractions of the frame. Traced from the
# framing the user drew over a real 9:16 export: the social apps paint their
# own chrome (caption + handle at the bottom, action column on the right,
# progress bar) over these strips, so nothing legible may live there.
# The area is an L, not a rectangle -- below CTA_SAFE_NOTCH_TOP the right
# edge steps in to CTA_SAFE_NOTCH_RIGHT, which is where the like/comment/
# share column starts.
CTA_SAFE_SIDE = 0.056        # ~60 px on a 1080-wide frame
CTA_SAFE_TOP = 0.137         # ~263 px on a 1920-tall frame
CTA_SAFE_BOTTOM = 0.208      # ~400 px on a 1920-tall frame
CTA_SAFE_NOTCH_TOP = 0.604   # ~1159 px: top of the action-button column
CTA_SAFE_NOTCH_RIGHT = 0.791  # ~854 px: left edge of that column
# Fraction of the safe width the text is allowed to use. Filling the safe box
# edge to edge technically respects it but still reads as "pegado al borde",
# so leave a gutter inside it.
CTA_TEXT_WIDTH = 0.86

# Horizontal (16:9) export: en Reels el cuadro va letterboxeado en el medio de
# la pantalla, asi que caption/handle quedan abajo, FUERA del video. Lo unico
# que pisa el cuadro es la columna like/comentar, desde ~55% del alto hacia
# abajo y ~89% del ancho hacia la derecha (medido sobre captura real de IG).
# Con las fracciones 9:16 el bloque quedaba en 708x525 sobre 1920x1080: chico.
CTA_SAFE_LANDSCAPE = {
    "side": 0.03, "top": 0.05, "bottom": 0.05,
    "notch_top": 0.50, "notch_right": 0.86,
}
# El mismo texto se ve chico en horizontal (el video ocupa ~30% del alto del
# telefono), asi que la fuente -- y con ella la portada -- crece un 50%.
CTA_LANDSCAPE_SCALE = 1.5


def _cta_safe_fractions(canvas_w: int, canvas_h: int) -> dict:
    if canvas_w > canvas_h:
        return CTA_SAFE_LANDSCAPE
    return {
        "side": CTA_SAFE_SIDE, "top": CTA_SAFE_TOP, "bottom": CTA_SAFE_BOTTOM,
        "notch_top": CTA_SAFE_NOTCH_TOP, "notch_right": CTA_SAFE_NOTCH_RIGHT,
    }


def _cta_safe_box(canvas_w: int, canvas_h: int, block_w: int, block_h: int):
    """Picks the biggest place the CTA block can sit in, given that the block
    is centered horizontally on the frame and the safe area is L-shaped.
    Only two rectangles inside that L can be maximal for a centered block:
    the wide-but-short one above the action column, and the narrow-but-tall
    one to its left. Returns (left, top, avail_w, avail_h) for the winner --
    the caller scales the block down to fit."""
    f = _cta_safe_fractions(canvas_w, canvas_h)
    side = int(canvas_w * f["side"])
    top = int(canvas_h * f["top"])
    bottom = canvas_h - int(canvas_h * f["bottom"])
    notch_top = int(canvas_h * f["notch_top"])
    notch_right = int(canvas_w * f["notch_right"])
    center = canvas_w / 2

    wide = (canvas_w - 2 * side, max(notch_top - top, 1), top)
    # Centered on the frame, so the usable half-width is the smaller side.
    narrow = (int(2 * min(center - side, notch_right - center)),
              max(bottom - top, 1), top)

    best = None
    for avail_w, avail_h, box_top in (wide, narrow):
        fit = min(avail_w / max(block_w, 1), avail_h / max(block_h, 1), 1.0)
        if best is None or fit > best[0]:
            best = (fit, avail_w, avail_h, box_top)
    _, avail_w, avail_h, box_top = best
    return int(center - avail_w / 2), box_top, avail_w, avail_h


# Texto del CTA original (campaña podcast Constantini). Los CTAs de la
# biblioteca del editor pasan el suyo con las mismas claves.
DEFAULT_CTA_TEXT = {
    "line1": "Comenta",
    "highlight": "EDUARDO",
    "line2": "y te mando la entrevista completa",
}
DEFAULT_CTA_PORTADA = Path(__file__).resolve().parents[2] / "assets" / "cta_portada.png"


def _render_cta_text_lines(width: int, height: int, scale: int, font_path: str,
                           max_text_w: int = 0, text: Optional[dict] = None):
    """Renders the two CTA text lines ("Comenta 'EDUARDO'" / "y te mando la
    entrevista completa" by default, or `text`'s line1/highlight/line2) onto
    tightly-cropped RGBA canvases; an empty line comes back as None. Font size 1
    starts proportional to height, then shrinks if needed so line 1 fits
    within `max_text_w` (defaults to the safe-area width) -- on a narrow
    vertical crop (e.g. 606x1080 from a 16:9 source) a height-only size
    otherwise overflows and clips the text at both edges.
    Returns (line1_canvas, line2_canvas, size1)."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    canvas_w = width * scale
    if not max_text_w:
        max_text_w = int(canvas_w * (1 - 2 * CTA_SAFE_SIDE) * CTA_TEXT_WIDTH)

    # Was 0.075/0.032 of the height: that filled the frame edge to edge and
    # ran straight into the social UI strips.
    boost = CTA_LANDSCAPE_SCALE if width > height else 1.0
    size1 = int(height * scale * 0.055 * boost)
    size2 = int(height * scale * 0.026 * boost)

    WHITE, GOLD, BLACK = (255, 255, 255, 255), (245, 185, 33, 255), (0, 0, 0, 255)
    text = text or DEFAULT_CTA_TEXT
    part_a = (text.get("line1") or "").strip()
    highlight = (text.get("highlight") or "").strip()
    part_b = f"\"{highlight}\"" if highlight else ""
    if part_a and part_b:
        part_a += " "
    line2 = (text.get("line2") or "").strip()

    probe = ImageDraw.Draw(PILImage.new("RGBA", (1, 1)))

    def _line1_width(size):
        font = ImageFont.truetype(font_path, size)
        stroke = max(int(size * 0.07), 2)
        return probe.textlength(part_a, font=font) + probe.textlength(part_b, font=font) + 4 * stroke

    def _line2_width(size):
        font = ImageFont.truetype(font_path, size)
        stroke = max(int(size * 0.09), 2)
        return probe.textlength(line2, font=font) + 2 * stroke

    w1 = _line1_width(size1)
    if w1 > max_text_w:
        size1 = max(int(size1 * max_text_w / w1), 8)
    w2 = _line2_width(size2) if line2 else 0
    if w2 > max_text_w:
        size2 = max(int(size2 * max_text_w / w2), 8)

    font1 = ImageFont.truetype(font_path, size1)
    font2 = ImageFont.truetype(font_path, size2)
    stroke1 = max(int(size1 * 0.07), 2)
    stroke2 = max(int(size2 * 0.09), 2)

    scratch_h = size1 * 2
    line1_canvas = PILImage.new("RGBA", (canvas_w, scratch_h), (0, 0, 0, 0))
    d1 = ImageDraw.Draw(line1_canvas)
    origin_y = scratch_h // 2
    w_a = d1.textlength(part_a, font=font1) + 2 * stroke1
    w_b = d1.textlength(part_b, font=font1) + 2 * stroke1
    x0 = (line1_canvas.width - (w_a + w_b)) / 2
    d1.text((x0, origin_y), part_a, font=font1, fill=WHITE, stroke_width=stroke1,
             stroke_fill=BLACK, anchor="lm")
    if part_b:
        d1.text((x0 + w_a, origin_y), part_b, font=font1, fill=GOLD, stroke_width=stroke1,
                 stroke_fill=BLACK, anchor="lm")
    bbox1 = line1_canvas.getbbox()
    line1_canvas = line1_canvas.crop(bbox1) if bbox1 else None

    line2_canvas = None
    if line2:
        line2_canvas = PILImage.new("RGBA", (canvas_w, scratch_h), (0, 0, 0, 0))
        d2 = ImageDraw.Draw(line2_canvas)
        d2.text((line2_canvas.width / 2, scratch_h // 2), line2, font=font2, fill=WHITE,
                 stroke_width=stroke2, stroke_fill=BLACK, anchor="mm")
        bbox2 = line2_canvas.getbbox()
        line2_canvas = line2_canvas.crop(bbox2) if bbox2 else None

    return line1_canvas, line2_canvas, size1


def _stack_cta_pieces(pieces):
    """Apila verticalmente [(canvas|None, gap_antes)] centrados; ignora los
    None. Devuelve el bloque RGBA o None si no queda nada."""
    from PIL import Image as PILImage

    pieces = [(c, g) for c, g in pieces if c is not None]
    if not pieces:
        return None
    block_w = max(c.width for c, _ in pieces)
    block_h = sum(c.height for c, _ in pieces) + sum(g for _, g in pieces[1:])
    block = PILImage.new("RGBA", (block_w, block_h), (0, 0, 0, 0))
    y = 0
    for i, (c, g) in enumerate(pieces):
        if i:
            y += g
        block.alpha_composite(c, ((block_w - c.width) // 2, y))
        y += c.height
    return block


def _render_cta_text_png(out_path: str, width: int, height: int) -> str:
    """Renders the "Comenta 'EDUARDO'" CTA as a clean, anti-aliased
    transparent PNG using the bundled Montserrat Black font -- replaces the
    old background-removed raster text, which had a rough, speckled cutout
    edge around the letters. Rendered at 3x and downsampled for smoother
    edges, sized/centered to overlay directly onto a `width`x`height` frame."""
    from PIL import Image as PILImage

    from .subtitles import FONTS_DIR

    scale = 3
    font_path = str(FONTS_DIR / "Montserrat-Black.ttf")
    canvas_w, canvas_h = width * scale, height * scale
    safe_w = int(canvas_w * (1 - 2 * CTA_SAFE_SIDE))
    text_w = int(safe_w * CTA_TEXT_WIDTH)
    line1_canvas, line2_canvas, size1 = _render_cta_text_lines(
        width, height, scale, font_path, max_text_w=text_w)

    gap = int(size1 * 0.28)
    block = _stack_cta_pieces([(line1_canvas, 0), (line2_canvas, gap)])
    canvas = PILImage.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    if block is not None:
        _place_cta_block(canvas, block)

    canvas = canvas.resize((width, height), PILImage.LANCZOS)
    canvas.save(out_path)
    return out_path


def _place_cta_block(canvas, block):
    """Shrinks the composed CTA block to the safe box and pastes it there,
    centered. Belt and braces on top of the per-line width clamp: the block's
    height depends on font metrics and on the portada's aspect, so this is the
    one place that can actually guarantee nothing pokes into the social UI."""
    from PIL import Image as PILImage

    left, top, avail_w, avail_h = _cta_safe_box(
        canvas.width, canvas.height, block.width, block.height)
    fit = min(avail_w / block.width, avail_h / block.height, 1.0)
    if fit < 1.0:
        block = block.resize((max(int(block.width * fit), 1),
                              max(int(block.height * fit), 1)), PILImage.LANCZOS)
    # Centrado vertical sobre el CUADRO, no sobre la caja segura: la caja es una
    # L que termina arriba de la columna de botones sociales, asi que centrar en
    # ella dejaba el bloque alto. Se apunta al centro del frame y se recorta
    # dentro de la caja, con lo cual el bloque baja todo lo que la zona segura
    # permite (en 1080x1920 queda pegado al borde del notch).
    y = (canvas.height - block.height) // 2
    y = max(top, min(y, top + avail_h - block.height))
    canvas.alpha_composite(block, (left + (avail_w - block.width) // 2, y))
    return canvas


def _render_cta_overlay_png(out_path: str, width: int, height: int,
                            text: Optional[dict] = None,
                            portada_path: Optional[str] = None) -> str:
    """Like `_render_cta_text_png`, but also stacks the show's "portada"
    graphic (photo collage + "LA OTRA CARA DEL BILLONARIO" headline, cropped
    from the original end-card image) above the CTA text, so the outro shows
    both -- not just the bare text -- on top of the blurred running video.
    The portada is scaled to fit width/height budgets so this still works on
    both landscape and portrait frames.

    `text` (line1/highlight/line2) and `portada_path` let the editor's CTA
    library swap both; portada_path=None keeps the Constantini portada and
    "" means no image at all (text only)."""
    from PIL import Image as PILImage

    from .subtitles import FONTS_DIR

    scale = 3
    font_path = str(FONTS_DIR / "Montserrat-Black.ttf")
    canvas_w, canvas_h = width * scale, height * scale
    safe_w = int(canvas_w * (1 - 2 * CTA_SAFE_SIDE))
    text_w = int(safe_w * CTA_TEXT_WIDTH)
    line1_canvas, line2_canvas, size1 = _render_cta_text_lines(
        width, height, scale, font_path, max_text_w=text_w, text=text)

    text_gap = int(size1 * 0.28)
    text_block = _stack_cta_pieces([(line1_canvas, 0), (line2_canvas, text_gap)])
    text_block_w = text_block.width if text_block is not None else 0

    if portada_path is None:
        portada_path = str(DEFAULT_CTA_PORTADA)
    portada_resized = None
    if portada_path:
        portada = PILImage.open(portada_path).convert("RGBA")
        # Portada a bit narrower than the text so the stack reads as a pyramid;
        # the whole block is fitted to the safe box afterwards anyway.
        portada_w = min(int(safe_w * 0.72), text_block_w or int(safe_w * 0.72))
        portada_h = max(int(portada_w * portada.height / portada.width), 1)
        portada_resized = portada.resize((portada_w, portada_h), PILImage.LANCZOS)

    block = _stack_cta_pieces([(portada_resized, 0), (text_block, int(size1 * 0.5))])

    canvas = PILImage.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    if block is not None:
        _place_cta_block(canvas, block)

    canvas = canvas.resize((width, height), PILImage.LANCZOS)
    canvas.save(out_path)
    return out_path


def append_blurred_cta_outro(
    in_path: str,
    source_path: str,
    cta_start_time: float,
    out_path: str,
    duration: float = 3.5,
    blur_sigma: float = 22.0,
    blur_ramp_seconds: float = 0.25,
    overlay_delay_seconds: float = 0.25,
    overlay_fade_seconds: float = 0.25,
    outro_clip_path: Optional[str] = None,
    cta_text: Optional[dict] = None,
    portada_path: Optional[str] = None,
    trim: Optional[Dict] = None,
) -> str:
    """Appends a CTA outro after in_path's video+audio, made of the *actual
    following footage* from `source_path` (starting at `cta_start_time`,
    normally the highlight's own end_time) instead of a static end-card --
    so the background keeps visibly running in the same scene/set, with an
    incremental blur (gblur's own sigma is ramped up directly over time via
    `sendcmd`, not a sharp/blurred opacity crossfade -- crossfading two fixed
    extremes stays visually indistinguishable from sharp for over half the
    duration since blending in a bit of a blurred frame barely changes the
    pixels, so the blur only became noticeable in the last stretch instead of
    building up steadily) and the clean rendered "Comenta EDUARDO" text on
    top. The blur reaches full strength quickly (`blur_ramp_seconds`, then
    holds -- sendcmd's last command sticks) and the portada/text overlay only
    fades in afterwards (`overlay_delay_seconds` + `overlay_fade_seconds`),
    so the background is already visibly blurry before the CTA shows up
    instead of both happening at once. The outro gets no original dialogue
    audio (silent track instead -- background music, if any, is mixed in
    afterwards over the extended duration) since it isn't part of the actual
    conversation. No transition/fade is applied between in_path and the
    outro: because the outro starts exactly where in_path's highlight ends,
    the cut is already visually continuous.

    `outro_clip_path` lets the caller hand in that following footage *already
    reframed* the same way the clip itself was (the editor passes a manually
    cropped cut, above all in 2-camera "split" mode). Without it the raw
    source frame is used, and center-cropping a 16:9 wide shot down to 9:16
    lands on whatever sits between the two cameras instead of on the stacked
    panes the viewer was just watching."""
    w, h, fps = _ffprobe_video_info(in_path)

    own_raw_outro = outro_clip_path is None
    raw_outro = outro_clip_path or (out_path + ".rawoutro.mp4")
    cta_png = out_path + ".ctatext.png"
    try:
        if own_raw_outro:
            trim_vf = _trim_filter(source_path, trim)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", f"{cta_start_time:.3f}", "-i", source_path,
                    "-t", f"{duration}",
                    *(["-vf", trim_vf] if trim_vf else []),
                    "-an",
                    *video_encode_args(intermediate=True),
                    raw_outro,
                ],
                check=True,
            )
        _render_cta_overlay_png(cta_png, w, h, text=cta_text, portada_path=portada_path)

        # Ramp sigma up over just blur_ramp_seconds (denser steps since the
        # window is short), then leave it alone -- sendcmd's last command
        # keeps holding, so sigma stays at blur_sigma for the rest of the
        # outro without needing further commands.
        ramp_seconds = min(blur_ramp_seconds, duration)
        steps_count = max(int(ramp_seconds * 6), 8)
        blur_commands = ";".join(
            f"{ramp_seconds * i / (steps_count - 1):.3f} gblur@ctablur sigma "
            f"{blur_sigma * i / (steps_count - 1):.3f}"
            for i in range(steps_count)
        )

        filter_complex = (
            f"[1:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},setsar=1,fps={fps}[oraw];"
            f"[oraw]sendcmd=c='{blur_commands}',"
            f"gblur@ctablur=sigma=0:steps=3[oblurred];"
            f"[2:v]format=rgba,fade=t=in:st={overlay_delay_seconds:.3f}:"
            f"d={overlay_fade_seconds:.3f}:alpha=1[ctatxt];"
            f"[oblurred][ctatxt]overlay=(W-w)/2:(H-h)/2:format=auto[ov];"
            f"[ov]format=yuv420p[v1];"
            f"[0:v][0:a][v1][3:a]concat=n=2:v=1:a=1[outv][outa]"
        )
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", in_path,
            "-i", raw_outro,
            "-loop", "1", "-t", f"{duration}", "-i", cta_png,
            "-f", "lavfi", "-t", f"{duration}", "-i", "anullsrc=r=44100:cl=stereo",
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            *video_encode_args(),
            "-c:a", "aac", "-b:a", "128k",
            out_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        tmps = [cta_png] + ([raw_outro] if own_raw_outro else [])
        for tmp in tmps:
            if os.path.exists(tmp):
                os.remove(tmp)
    return out_path


def _has_audio_stream(path: str) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip()
    return bool(out)


def append_video_outro(in_path: str, outro_video: str, out_path: str) -> str:
    """Pega un video ya armado (CTA de la biblioteca del editor) después del
    clip. El outro se escala entero dentro del cuadro del clip (barras negras
    si la proporción no coincide, nunca recorta texto del CTA) y se lleva al
    mismo fps; si no trae audio va silencio, así la música de fondo que se
    mezcla después sigue sonando por encima."""
    w, h, fps = _ffprobe_video_info(in_path)
    outro_audio = _has_audio_stream(outro_video)
    outro_dur = _ffprobe_duration(outro_video)
    audio_src = "[1:a]" if outro_audio else "[2:a]"
    filter_complex = (
        f"[1:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={fps},"
        f"format=yuv420p[v1];"
        f"{audio_src}aresample=44100,aformat=channel_layouts=stereo[a1];"
        f"[0:a]aresample=44100,aformat=channel_layouts=stereo[a0];"
        f"[0:v][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-i", outro_video,
        "-f", "lavfi", "-t", f"{outro_dur:.3f}", "-i", "anullsrc=r=44100:cl=stereo",
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        *video_encode_args(),
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


# --- Text templates (overlay cards on top of the first seconds of a clip) ---
#
# These are the "plantillas de texto" the editor offers for vertical clips: a
# static card composited over the *already finished* frame (crop + burned
# subtitles + speed all applied), visible only for the first few seconds and
# then faded out. Unlike the CTA outro they do not extend the video, they just
# sit on top of it -- which is why they run last in the render chain but before
# the outro is appended.
#
# Layout is top-anchored on purpose: the whole point of the feature is to add a
# headline *above* the talking heads, where nothing of the conversation is
# happening, and where the burned captions (bottom third) can't collide with
# it.
TEMPLATE_IDS = ("noticia",)

# Same reasoning as the CTA safe area, but only the top strip matters here:
# the block never reaches down to the action column or the caption strip.
TEMPLATE_SAFE_SIDE = 0.06
TEMPLATE_SAFE_TOP = 0.085

TEMPLATE_KICKER_BG = (240, 106, 24, 255)   # the news-card orange
TEMPLATE_TEXT = (255, 255, 255, 255)
TEMPLATE_MAX_HEADLINE_LINES = 4

# The card is a movable, resizable block now: x/y are the fractions of the
# frame where its top-left corner sits and scale multiplies its base width
# (TEMPLATE_BLOCK_W of the frame). The defaults reproduce the original fixed
# placement in the top-left of the safe area.
TEMPLATE_BLOCK_W = 1.0 - 2 * TEMPLATE_SAFE_SIDE
TEMPLATE_DEFAULT_X = TEMPLATE_SAFE_SIDE
TEMPLATE_DEFAULT_Y = TEMPLATE_SAFE_TOP
TEMPLATE_DEFAULT_SCALE = 1.0

# Entrada de la placa, al estilo de un zocalo de noticiero: entra deslizandose
# desde el costado y llega desenfocada, resolviendose sobre el final del
# recorrido. Son segundos y un sigma de desenfoque medido sobre un frame de
# 1080 px de ancho (se escala con el ancho real, si no un clip horizontal de
# 1920 recibiria la mitad de desenfoque relativo).
TEMPLATE_SLIDE_SECONDS = 0.55
TEMPLATE_SLIDE_FROM = "left"
TEMPLATE_BLUR_SIGMA_AT_1080 = 14.0

# Entradas sobrias para las capas PNG (nombres tomados de animate.css, MIT, pero con
# recorridos cortos). Curva easeOutCubic. Espejo en static/editor.html (OVERLAY_ANIM_*).
OVERLAY_ANIMS = ("fade", "rise", "drop", "slide_left", "slide_right", "zoom")
OVERLAY_ANIM_TRAVEL_Y = 0.05    # rise/drop: fraccion del alto del frame
OVERLAY_ANIM_TRAVEL_X = 0.06    # slide_*: fraccion del ancho del frame
OVERLAY_ANIM_ZOOM_FROM = 0.9    # zoom: escala inicial
OVERLAY_ANIM_DEFAULT_SECONDS = 0.4


def _template_logo_path() -> Optional[str]:
    """The user's transparent PNG logo for the top-left slot, if they dropped
    one into assets/. Missing logo is not an error -- the card just renders
    without it."""
    from .subtitles import FONTS_DIR

    for name in ("overlay_logo.png", "logo.png"):
        candidate = FONTS_DIR.parent / name
        if candidate.is_file():
            return str(candidate)
    return None


def _wrap_to_width_tracked(draw, text: str, font, max_w: float) -> list:
    """Like _wrap_to_width, but each output line also carries its [start, end)
    character range in the original `text` (paragraphs and the '\\n' that
    separates them both counted) -- so per-range colors (headline_color_runs,
    offsets into the stored headline) can be sliced onto the exact wrapped
    lines. The rendered line is text[start:end] itself, not a re-joined copy,
    so it keeps whatever original inter-word whitespace was there."""
    lines = []
    pos = 0
    for paragraph in text.split("\n"):
        p_start = pos
        pos += len(paragraph) + 1  # +1 for the '\n' this split ate
        words = paragraph.split()
        if not words:
            lines.append(("", p_start, p_start))
            continue
        search_from = 0
        current = ""
        line_start = line_end = None
        for word in words:
            w_off = p_start + paragraph.find(word, search_from)
            search_from = (w_off - p_start) + len(word)
            probe = f"{current} {word}".strip()
            if not current or draw.textlength(probe, font=font) <= max_w:
                current = probe
                if line_start is None:
                    line_start = w_off
                line_end = w_off + len(word)
            else:
                lines.append((text[line_start:line_end], line_start, line_end))
                current = word
                line_start = w_off
                line_end = w_off + len(word)
        lines.append((text[line_start:line_end], line_start, line_end))
    return lines


def _wrap_to_width(draw, text: str, font, max_w: float) -> list:
    """Greedy word wrap against a real font metric. A single word longer than
    max_w still gets its own line (never dropped) -- the caller shrinks the
    font size until the whole block fits. Manual line breaks (from the
    in-place editor's Enter key) are kept as forced breaks -- each paragraph
    is wrapped on its own instead of being flattened into one run of words."""
    return [line for line, _start, _end in _wrap_to_width_tracked(draw, text, font, max_w)]


def _draw_soft_shadow(layer, blur_radius: int):
    """Blurs an RGBA layer that only contains the shadow silhouette. Cheaper
    and softer than a stroke outline, and closer to how these news cards look
    over a photo -- a stroke reads as a cartoon border at this font size."""
    from PIL import ImageFilter

    return layer.filter(ImageFilter.GaussianBlur(blur_radius))


def _crop_fracs(crop: Optional[dict]) -> Tuple[float, float, float, float]:
    """(left, top, keep_w, keep_h) en fracciones; keep_* >= 0.1."""
    c = crop or {}
    g = lambda k: min(max(float(c.get(k) or 0.0), 0.0), 0.45)
    left, right, top, bottom = g("left"), g("right"), g("top"), g("bottom")
    return left, top, max(1 - left - right, 0.1), max(1 - top - bottom, 0.1)


def _crop_box(w: int, h: int, crop: Optional[dict]) -> Tuple[int, int, int, int]:
    left, top, kw, kh = _crop_fracs(crop)
    x0, y0 = int(round(w * left)), int(round(h * top))
    return x0, y0, max(x0 + int(round(w * kw)), x0 + 1), max(y0 + int(round(h * kh)), y0 + 1)


def _hex_rgb(color: str) -> Tuple[int, int, int]:
    try:
        return tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, AttributeError):
        return (255, 255, 255)


def _rounded_mask_ring(px_w: int, px_h: int, radius_frac: float, border_px: float):
    """(mascara, anillo) en modo L de px_w x px_h: rectangulo redondeado (radio =
    radius_frac * ancho, tope medio lado corto) y borde de border_px hacia adentro
    que sigue las esquinas; anillo None si no hay borde. None si no hay nada que
    aplicar. Supersampling x4 para bordes suaves."""
    from PIL import Image as PILImage, ImageDraw

    ss = 4
    W, H = px_w * ss, px_h * ss
    rad = min(max(radius_frac, 0.0) * px_w, min(px_w, px_h) / 2) * ss
    bpx = int(round(max(border_px, 0.0) * ss))
    if rad <= 0 and bpx <= 0:
        return None
    mask = PILImage.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, H - 1], radius=rad, fill=255)
    mask = mask.resize((px_w, px_h), PILImage.LANCZOS)
    ring = None
    if bpx > 0:
        ring = PILImage.new("L", (W, H), 0)
        ImageDraw.Draw(ring).rounded_rectangle([0, 0, W - 1, H - 1], radius=rad, outline=255, width=bpx)
        ring = ring.resize((px_w, px_h), PILImage.LANCZOS)
    return mask, ring


TEXT_BOX_SCALE = 2   # se renderiza al doble de 1080 para que no se vea blando al agrandar


def render_text_box_png(out_path: str, spec: dict) -> Tuple[int, int]:
    """Capa de texto del editor: un cuadro con fondo (color + opacidad) y el
    texto en Montserrat. Medidas de `spec` en px de un cuadro de 1080 de ancho;
    el PNG sale a TEXT_BOX_SCALE x. `spec["spans"]` = [{s,e,size?,weight?,color?}]
    pinta tramos del texto (indices de caracter) encima del estilo base. Respeta
    los saltos de linea y corta por palabras al llegar a max_w; cada linea toma
    el alto de su letra mas grande y todas las letras comparten linea de base.
    Devuelve (ancho, alto) en px de 1080 (lo que usa el editor para el tamano)."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    from .subtitles import FONTS_DIR
    k = TEXT_BOX_SCALE
    pad = int(round(float(spec.get("pad", 24)) * k))
    max_w = int(round(float(spec.get("max_w", 0.8)) * 1080 * k))
    fonts = {}

    def font_for(size, weight):
        key = (size, weight)
        if key not in fonts:
            f = ImageFont.truetype(str(FONTS_DIR / "_source" / "Montserrat-Variable.ttf"), size)
            try:
                f.set_variation_by_axes([weight])
            except Exception:
                f = ImageFont.truetype(str(FONTS_DIR / "Montserrat-Black.ttf"), size)
            fonts[key] = f
        return fonts[key]

    text = str(spec.get("text") or " ")
    base = (int(round(float(spec.get("size", 64)) * k)), int(spec.get("weight", 900)),
            str(spec.get("color") or "#ffffff"))
    styles = [base] * len(text)
    for sp in spec.get("spans") or []:
        st = (int(round(float(sp["size"]) * k)) if sp.get("size") else None,
              int(sp["weight"]) if sp.get("weight") else None, sp.get("color"))
        for i in range(max(0, int(sp.get("s", 0))), min(len(text), int(sp.get("e", 0)))):
            cur = styles[i]
            styles[i] = tuple(n if n is not None else c for n, c in zip(st, cur))

    def groups(chars):
        # [(texto, estilo)] juntando caracteres consecutivos con el mismo estilo
        out = []
        for ch, st in chars:
            if out and out[-1][1] == st:
                out[-1][0].append(ch)
            else:
                out.append(([ch], st))
        return [("".join(t), st) for t, st in out]

    def width(chars):
        return sum(font_for(st[0], st[1]).getlength(t) for t, st in groups(chars))

    inner = max(max_w - 2 * pad, base[0])
    lines = []   # cada linea: [(ch, estilo)]
    pos = 0
    for para in text.split("\n"):
        chars = list(zip(para, styles[pos:pos + len(para)]))
        pos += len(para) + 1
        words, w = [], []
        for c in chars:
            if c[0] == " ":
                words.append(w)
                w = [c]      # el espacio viaja con la palabra siguiente
            else:
                w.append(c)
        words.append(w)
        cur = []
        for word in words:
            if cur and width(cur + word) > inner:
                lines.append(cur)
                cur = word[1:] if word and word[0][0] == " " else word
            else:
                cur = cur + word
        lines.append(cur if cur else [("", base)])

    laid = []
    for ln in lines:
        gs = [g for g in groups(ln) if g[0]] or [("", base)]
        fs = [font_for(st[0], st[1]) for _, st in gs]
        asc = max(f.getmetrics()[0] for f in fs)
        desc = max(f.getmetrics()[1] for f in fs)
        line_h = int(round(max(st[0] for _, st in gs) * 1.2))
        laid.append((gs, fs, asc, desc, line_h, sum(f.getlength(t) for (t, _), f in zip(gs, fs))))
    # El cuadro mide max_w de ancho (y min_h de alto si se pidio): la letra
    # cambia el texto adentro, no el cuadro. Solo crece si el texto no entra
    # (una palabra mas larga que el ancho o mas lineas que el alto).
    text_w = max(1, int(math.ceil(max(l[5] for l in laid))))
    content_h = sum(l[4] for l in laid)
    W = max(max_w, text_w + 2 * pad)
    H = max(int(round(float(spec.get("min_h", 0) or 0) * k)), content_h + 2 * pad)
    bg = _hex_rgb(str(spec.get("bg") or "#000000"))
    bg_a = int(round(min(max(float(spec.get("bg_op", 0.6)), 0.0), 1.0) * 255))
    img = PILImage.new("RGBA", (W, H), bg + (bg_a,))
    draw = ImageDraw.Draw(img)
    align = spec.get("align") or "center"
    y = (H - content_h) / 2
    for gs, fs, asc, desc, line_h, lw in laid:
        # Linea de base comun, centrada en la caja de line_h por ascent+descent
        # de las fuentes (no del texto): mismas alturas con o sin acentos.
        baseline = y + (line_h - (asc + desc)) / 2 + asc
        x = pad if align == "left" else (W - pad - lw if align == "right" else (W - lw) / 2)
        for (t, st), f in zip(gs, fs):
            draw.text((x, baseline), t, font=f, fill=_hex_rgb(st[2]) + (255,), anchor="ls")
            x += f.getlength(t)
        y += line_h
    img.save(out_path)
    return int(round(W / k)), int(round(H / k))


def style_overlay_png(
    src: str,
    out_path: str,
    px_w: int,
    radius_frac: float,
    border_px: float,
    color: str,
    crop: Optional[dict] = None,
) -> bool:
    """Copia de una capa PNG recortada por `crop` ({top,bottom,left,right},
    fracciones del original) y escalada a `px_w`, con esquinas redondeadas (radio =
    radius_frac * ancho, tope medio lado corto) y un borde de `border_px` hacia
    adentro que sigue las esquinas. Igual que el preview del editor. Supersampling
    x4 para bordes suaves. False si no hay nada que aplicar o falla."""
    from PIL import Image as PILImage, ImageChops

    try:
        with PILImage.open(src) as im:
            im = im.convert("RGBA")
            if crop:
                box = _crop_box(im.width, im.height, crop)
                im = im.crop(box)
            px_h = max(int(round(px_w * im.height / max(im.width, 1))), 2)
            im = im.resize((px_w, px_h), PILImage.LANCZOS)
    except Exception:
        return False
    parts = _rounded_mask_ring(px_w, px_h, radius_frac, border_px)
    if parts is None and not crop:
        return False
    mask, ring = parts or (None, None)
    if mask is not None:
        im.putalpha(ImageChops.multiply(im.getchannel("A"), mask))
    if ring is not None:
        solid = PILImage.new("RGBA", (px_w, px_h), _hex_rgb(color) + (255,))
        solid.putalpha(ring)
        im = PILImage.alpha_composite(im, solid)
    im.save(out_path)
    return True


def render_border_png(
    out_path: str,
    frame_w: int,
    frame_h: int,
    width_frac: float,
    color: str,
    panes: int = 1,
) -> bool:
    """Full-frame transparent PNG with a solid ring of `color` along the edge
    of each pane (1 = whole frame, 2 = the two stacked halves of the split
    mode). The ring's thickness is `width_frac` of the frame width, drawn
    inward, so the editor preview (same fraction of its own width) matches.
    Returns False when there is nothing to draw."""
    from PIL import Image as PILImage, ImageDraw

    px = int(round(frame_w * max(0.0, float(width_frac))))
    if px < 1:
        return False
    try:
        rgb = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, AttributeError):
        rgb = (255, 255, 255)
    img = PILImage.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pane_h = frame_h // panes
    for i in range(panes):
        top = i * pane_h
        bottom = frame_h - 1 if i == panes - 1 else top + pane_h - 1
        draw.rectangle([0, top, frame_w - 1, bottom], outline=rgb + (255,), width=px)
    img.save(out_path)
    return True


def render_template_block_png(
    out_path: str,
    frame_w: int,
    frame_h: int,
    template: str = "noticia",
    kicker: str = "",
    headline: str = "",
    logo_path: Optional[str] = None,
    width_frac: Optional[float] = None,
    headline_color: Optional[str] = None,
    headline_size_frac: Optional[float] = None,
    headline_color_runs: Optional[list] = None,
    bg_color: Optional[str] = None,
    bg_opacity: Optional[float] = None,
) -> Optional[Tuple[str, int, int]]:
    """Renders the card as a *tight* transparent PNG -- just the block, with no
    frame-sized padding around it -- at its base size for a frame_w x frame_h
    output, i.e. a block TEMPLATE_BLOCK_W of the frame wide. Returns
    (path, width, height) in output pixels, or None when there is nothing to
    draw.

    Tight is what makes the card draggable: the editor positions and scales
    these exact bytes with CSS and the export pastes the same bytes at the same
    fractions, so the panel cannot drift from the render. Type sizes stay
    relative to the frame *height* (as they were when the card was drawn
    straight onto the frame), which is why the block has to be rendered for the
    real output aspect -- a 9:16 card and a 16:9 card are not the same shape.

    The "noticia" template stacks, from the top down: the logo (optional), an
    orange kicker box with white uppercase text (optional), and the headline in
    white with a soft drop shadow. Drawn at 3x and downsampled, like the CTA
    text, so the edges stay clean after the overlay is re-encoded."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    from .subtitles import FONTS_DIR

    kicker = (kicker or "").strip()
    headline = (headline or "").strip()
    if not kicker and not headline:
        return None
    if template not in TEMPLATE_IDS:
        template = TEMPLATE_IDS[0]

    width_frac = TEMPLATE_BLOCK_W if not width_frac else max(0.15, min(1.0, width_frac))

    def _hex_to_rgba(hex_color, default):
        s = (hex_color or "").strip().lstrip("#")
        if len(s) != 6:
            return default
        try:
            r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        except ValueError:
            return default
        return (r, g, b, 255)

    headline_fill = _hex_to_rgba(headline_color, TEMPLATE_TEXT)
    headline_len = len(headline)
    color_runs = []
    for r in headline_color_runs or []:
        if not isinstance(r, dict):
            continue
        try:
            start = max(0, min(int(r["start"]), headline_len))
            end = max(0, min(int(r["end"]), headline_len))
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        color_runs.append({"start": start, "end": end, "color": r.get("color")})

    scale = 3
    block_w = max(int(frame_w * width_frac), 8)
    canvas_w = block_w * scale
    # Reference height for every type size: the frame, not the block, so the
    # card keeps the proportions it had when it was drawn onto the frame.
    ref_h = max(int(frame_h * scale), 8)
    canvas_h = ref_h
    max_w = canvas_w

    headline_font_path = str(FONTS_DIR / "Montserrat-Black.ttf")
    kicker_font_path = str(FONTS_DIR / "Montserrat-ExtraBold.ttf")

    canvas = PILImage.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    y = 0
    bleed = 0  # extra rows the blurred shadow needs below the last baseline

    if logo_path is None:
        logo_path = _template_logo_path()
    if logo_path and os.path.exists(logo_path):
        logo = PILImage.open(logo_path).convert("RGBA")
        # Fit inside a box rather than forcing a width: a wide wordmark and a
        # square badge both have to land at a sane size without being cropped.
        box_w, box_h = int(canvas_w * 0.386), int(ref_h * 0.055)
        fit = min(box_w / logo.width, box_h / logo.height, 1.0)
        logo = logo.resize((max(int(logo.width * fit), 1),
                            max(int(logo.height * fit), 1)), PILImage.LANCZOS)
        canvas.alpha_composite(logo, (0, y))
        y += logo.height + int(ref_h * 0.022)

    if kicker:
        size = int(ref_h * 0.025)
        font = ImageFont.truetype(kicker_font_path, size)
        text = kicker.upper()
        while size > 8 and draw.textlength(text, font=font) > max_w * 0.9:
            size = int(size * 0.94)
            font = ImageFont.truetype(kicker_font_path, size)
        pad_x, pad_y = int(size * 0.62), int(size * 0.42)
        text_w = draw.textlength(text, font=font)
        box_w = int(text_w) + 2 * pad_x
        box_h = size + 2 * pad_y
        draw.rectangle([0, y, box_w, y + box_h], fill=TEMPLATE_KICKER_BG)
        draw.text((pad_x, y + box_h / 2), text, font=font,
                  fill=TEMPLATE_TEXT, anchor="lm")
        y += box_h + int(ref_h * 0.024)

    if headline:
        base_frac = 0.058 if not headline_size_frac else max(0.02, min(0.14, headline_size_frac))
        size = int(ref_h * base_frac)
        font = ImageFont.truetype(headline_font_path, size)
        lines = _wrap_to_width(draw, headline, font, max_w)
        # Shrink until the headline fits the line budget *and* the longest word
        # actually fits the block width (a long word can't be wrapped away).
        while size > 12 and (
            len(lines) > TEMPLATE_MAX_HEADLINE_LINES
            or max(draw.textlength(line, font=font) for line in lines) > max_w
        ):
            size = int(size * 0.94)
            font = ImageFont.truetype(headline_font_path, size)
            lines = _wrap_to_width(draw, headline, font, max_w)

        tracked_lines = _wrap_to_width_tracked(draw, headline, font, max_w)

        line_h = int(size * 1.16)
        block_h = line_h * len(tracked_lines)
        blur = max(int(size * 0.09), 3)
        offset = max(int(size * 0.06), 2)
        shadow = PILImage.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        for i, (line, _ls, _le) in enumerate(tracked_lines):
            shadow_draw.text((offset, y + i * line_h + offset), line,
                             font=font, fill=(0, 0, 0, 200))
        canvas.alpha_composite(_draw_soft_shadow(shadow, blur))

        def _segments_for_line(line_start, line_end):
            # Cut the line at every color_run boundary that falls inside it, so
            # each resulting sub-range is either fully covered by one run (its
            # color) or fully uncovered (None -> headline_fill).
            cuts = {line_start, line_end}
            for r in color_runs:
                if r["start"] < line_end and r["end"] > line_start:
                    cuts.add(max(r["start"], line_start))
                    cuts.add(min(r["end"], line_end))
            points = sorted(cuts)
            segments = []
            for a, b in zip(points, points[1:]):
                color = next(
                    (r["color"] for r in color_runs if r["start"] <= a and r["end"] >= b),
                    None,
                )
                segments.append((a, b, color))
            return segments

        for i, (line, line_start, line_end) in enumerate(tracked_lines):
            if not color_runs:
                draw.text((0, y + i * line_h), line, font=font, fill=headline_fill)
                continue
            x = 0.0
            for a, b, color in _segments_for_line(line_start, line_end):
                seg_text = headline[a:b]
                if not seg_text:
                    continue
                fill = _hex_to_rgba(color, headline_fill) if color else headline_fill
                draw.text((x, y + i * line_h), seg_text, font=font, fill=fill)
                x += draw.textlength(seg_text, font=font)
        y += block_h
        bleed = blur * 2 + offset

    content_h = max(min(y + bleed, canvas_h), scale)
    block = canvas.crop((0, 0, canvas_w, content_h))

    if bg_color:
        bg_rgba = _hex_to_rgba(bg_color, None)
        if bg_rgba:
            opacity = 0.55 if bg_opacity is None else max(0.0, min(1.0, bg_opacity))
            fill = (bg_rgba[0], bg_rgba[1], bg_rgba[2], int(255 * opacity))
            base = PILImage.new("RGBA", block.size, fill)
            base.alpha_composite(block)
            block = base

    out_h = max(int(round(content_h / scale)), 1)
    block = block.resize((block_w, out_h), PILImage.LANCZOS)
    block.save(out_path)
    return out_path, block_w, out_h


def extend_frozen_tail(in_path: str, out_path: str, seconds: float) -> str:
    """Estira in_path `seconds` al final: el ultimo cuadro queda congelado
    (tpad clone) y el audio sigue en silencio (apad). Es la "cola" de las capas
    que el editor deja pasar del final del clip, para que burn_overlay_items
    tenga donde dibujarlas (recorta todo a la duracion de su entrada)."""
    seconds = max(float(seconds), 0.0)
    has_audio = _has_audio_stream(in_path)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-vf", f"tpad=stop_mode=clone:stop_duration={seconds:.3f}",
    ]
    if has_audio:
        cmd += ["-af", f"apad=pad_dur={seconds:.3f}", "-c:a", "aac", "-b:a", "128k"]
    cmd += [*video_encode_args(intermediate=True), "-movflags", "+faststart", out_path]
    subprocess.run(cmd, check=True)
    return out_path


def burn_overlay_items(
    in_path: str,
    out_path: str,
    items: list,
) -> Optional[str]:
    """Composites a stack of still PNGs over in_path in a single encode and
    writes out_path. Returns None (and writes nothing) when there is nothing to
    composite, so callers can treat it as a no-op step.

    Each item is a dict with the geometry expressed as fractions of the output
    frame -- the same numbers the editor stores and draws with -- plus its own
    time window:

        png      path to an RGBA still
        x, y     top-left corner, 0-1 of the frame
        w        width, 0-1 of the frame (height follows the PNG's aspect)
        start    seconds into the clip when it appears (default 0)
        end      seconds when it disappears (default: end of clip)
        fade_in / fade_out   seconds (default 0.3 / 0.5)
        opacity  0-1 (default 1)
        slide_in seconds of the news-ticker entrance (default 0 = no slide)
        slide_from  'left' (default) or 'right'
        blur_in  blur sigma at the start of that entrance, measured on a
                 1080-wide frame (default 0 = no blur); ignored without slide_in
        anim, anim_dur  soft entrance from OVERLAY_ANIMS (fade/rise/drop/
                 slide_left/slide_right/zoom) lasting anim_dur seconds; the
                 alpha ramp is stretched to match. Ignored with slide_in.

    Visibility is gated purely by the alpha fades: `fade=t=in:st=S` leaves the
    alpha plane at zero before S and `fade=t=out` leaves it at zero after, so
    every item can be looped for the whole clip and still only show inside its
    own window. Audio is stream-copied -- this step only ever touches pixels."""
    items = [it for it in (items or []) if it.get("png") and os.path.exists(it["png"])]
    if not items:
        return None
    # A "video" item (capa de video) reuses the same keys, but `png` points at an
    # H.264 mp4 that plays from its own `start` (no loop; when it ends the
    # layer just disappears). Its audio is never used.

    frame_w, frame_h, _fps = _ffprobe_video_info(in_path)
    clip_duration = _ffprobe_duration(in_path)
    if not frame_w or not frame_h or clip_duration <= 0:
        return None

    # Items whose window is empty are dropped up front: the loop below numbers
    # ffmpeg inputs by position, so skipping one mid-loop would shift the rest.
    def _visible(it):
        st = max(float(it.get("start", 0.0) or 0.0), 0.0)
        en = it.get("end")
        en = clip_duration if en is None else min(float(en), clip_duration)
        return en > st
    items = [it for it in items if _visible(it)]
    if not items:
        return None

    inputs, chains, overlays = [], [], []
    scratch = []   # mascaras/anillos temporales de capas de video
    audio_tags = []   # capas de video con audio propio, ya retrasadas a su start
    main_has_audio = _has_audio_stream(in_path)
    prev = "0:v"
    for idx, it in enumerate(items, start=1):
        start = max(float(it.get("start", 0.0) or 0.0), 0.0)
        end = it.get("end")
        end = clip_duration if end is None else min(float(end), clip_duration)
        if end <= start:
            continue
        fade_in = max(float(it.get("fade_in", 0.3) or 0.0), 0.0)
        fade_out = max(float(it.get("fade_out", 0.5) or 0.0), 0.0)
        # Una capa que llega al final del video no se desvanece: el fade dejaria
        # ver el video de abajo en los ultimos cuadros (frame "colado" al final).
        if end >= clip_duration - 0.05:
            fade_out = 0.0
        span = end - start
        fade_in = min(fade_in, span / 2)
        fade_out = min(fade_out, span / 2)
        opacity = min(max(float(it.get("opacity", 1.0) or 0.0), 0.0), 1.0)
        px_w = max(int(round(frame_w * float(it.get("w", 1.0)))), 2)
        pos_x = int(round(frame_w * float(it.get("x", 0.0))))
        pos_y = int(round(frame_h * float(it.get("y", 0.0))))

        # Optional news-ticker entrance: the item travels in from off-frame with
        # an ease-out while a blurred copy of itself dissolves on top of it, so
        # it reads as arriving out of focus and settling. Items without
        # slide_in emit exactly the same graph as before.
        slide_in = min(max(float(it.get("slide_in", 0.0) or 0.0), 0.0), span)
        blur_sigma = max(float(it.get("blur_in", 0.0) or 0.0), 0.0) if slide_in > 0 else 0.0

        anim = str(it.get("anim") or "none").lower()
        if slide_in > 0 or anim not in OVERLAY_ANIMS:
            anim = None
        anim_dur = min(max(float(it.get("anim_dur") or OVERLAY_ANIM_DEFAULT_SECONDS), 0.1), span)

        is_video = bool(it.get("video"))
        # Velocidad de la capa de video: `speed` s de archivo por s de clip.
        speed = min(max(float(it.get("speed", 1.0) or 1.0), 0.25), 4.0) if is_video else 1.0
        pts_f = (f"setpts=(PTS-STARTPTS)/{speed:.4f}+{start:.3f}/TB" if abs(speed - 1.0) > 1e-3
                 else f"setpts=PTS-STARTPTS+{start:.3f}/TB")
        if is_video:
            # Si el archivo se termina antes que la ventana de la capa, se congela
            # su ultimo cuadro: con eof_action=pass se colaria el video de abajo.
            pts_f += f",tpad=stop_mode=clone:stop_duration={span + 0.5:.3f}"
        ii = inputs.count("-i") + 1   # indice ffmpeg de la entrada de este item
        extra_inputs = []
        head = f"[{ii}:v]"
        base = ["format=rgba", f"scale={px_w}:-1:flags=lanczos"]
        crop_f = None
        if is_video and it.get("crop"):
            # Encuadre de la capa: se recorta el original antes de escalar, asi
            # px_w es el ancho de lo que queda (igual que en el editor).
            cl, ct, ckw, ckh = _crop_fracs(it["crop"])
            crop_f = (f"crop=trunc(iw*{ckw:.5f}/2)*2:trunc(ih*{ckh:.5f}/2)*2"
                      f":trunc(iw*{cl:.5f}):trunc(ih*{ct:.5f})")
        if is_video:
            # Shift the layer's own timeline so its first frame lands at `start`.
            base.insert(0, pts_f)
            if crop_f:
                base.insert(1, crop_f)
            # Esquinas/borde de la capa de video: mascara (alphamerge) y anillo
            # fijos de px_w x px_h, mismas cuentas que style_overlay_png.
            vw, vh, _f = _ffprobe_video_info(it["png"])
            if crop_f:
                _l, _t, ckw, ckh = _crop_fracs(it["crop"])
                vw, vh = int(vw * ckw) // 2 * 2, int(vh * ckh) // 2 * 2
            px_h = max(int(round(px_w * vh / max(vw, 1))), 2)
            parts = _rounded_mask_ring(px_w, px_h, float(it.get("radius") or 0.0),
                                       frame_w * float(it.get("border_w") or 0.0))
            if parts is not None:
                from PIL import Image as PILImage
                mask, ring = parts
                mask_path = f"{out_path}.vm{idx}.png"
                mask.save(mask_path)
                scratch.append(mask_path)
                extra_inputs += ["-loop", "1", "-t", f"{clip_duration:.3f}", "-i", mask_path]
                chains.append(f"[{ii}:v]{pts_f},"
                              f"{crop_f + ',' if crop_f else ''}format=rgba,"
                              f"scale={px_w}:{px_h}:flags=lanczos[vs{idx}]")
                chains.append(f"[{ii + 1}:v]format=gray[mk{idx}]")
                chains.append(f"[vs{idx}][mk{idx}]alphamerge[vm{idx}]")
                head = f"[vm{idx}]"
                if ring is not None:
                    ring_path = f"{out_path}.vr{idx}.png"
                    solid = PILImage.new("RGBA", (px_w, px_h), _hex_rgb(str(it.get("border_color") or "#ffffff")) + (255,))
                    solid.putalpha(ring)
                    solid.save(ring_path)
                    scratch.append(ring_path)
                    extra_inputs += ["-loop", "1", "-t", f"{clip_duration:.3f}", "-i", ring_path]
                    chains.append(f"[vm{idx}][{ii + 2}:v]overlay=0:0:format=auto[vr{idx}]")
                    head = f"[vr{idx}]"
                base = ["format=rgba"]
        if opacity < 1.0:
            base.append(f"colorchannelmixer=aa={opacity:.3f}")

        # A zero-length fade still has to be emitted whenever the item starts
        # late or ends early: it is the fade that zeroes the alpha outside the
        # window, not the input duration.
        # While sliding, the alpha ramp has to last at least as long as the
        # travel, or the card would be fully opaque halfway across the frame.
        appear = max(fade_in, slide_in, anim_dur if anim else 0.0)
        fades = []
        if start > 0 or appear > 0:
            fades.append(f"fade=t=in:st={start:.3f}:d={max(appear, 0.001):.3f}:alpha=1")
        if end < clip_duration - 0.05 or fade_out > 0:
            fades.append(f"fade=t=out:st={max(end - fade_out, start):.3f}"
                         f":d={max(fade_out, 0.001):.3f}:alpha=1")

        if blur_sigma > 0:
            # gblur's sigma is in pixels, so it has to scale with the real frame
            # width or the same card would look twice as sharp on a 1920-wide
            # horizontal clip as on the 1080-wide vertical one.
            sigma = blur_sigma * frame_w / 1080.0
            chains.append(f"{head}{','.join(base)},split=2[sh{idx}][bl{idx}]")
            chains.append(f"[sh{idx}]{','.join(fades or ['null'])}[ov{idx}]")
            chains.append(
                f"[bl{idx}]gblur=sigma={sigma:.2f}"
                f",fade=t=in:st={start:.3f}:d={min(0.15, slide_in / 3):.3f}:alpha=1"
                f",fade=t=out:st={start:.3f}:d={slide_in:.3f}:alpha=1[ovb{idx}]"
            )
            layers = [f"ovb{idx}", f"ov{idx}"]
        else:
            tail = []
            if anim == "zoom":
                # Ultimo en la cadena: solo el overlay ve el frame cambiar de tamano.
                z = (f"({OVERLAY_ANIM_ZOOM_FROM}+{1 - OVERLAY_ANIM_ZOOM_FROM:.3f}"
                     f"*(1-pow(1-max(0,min(1,(t-{start:.3f})/{anim_dur:.3f})),3)))")
                tail.append(f"scale=w='iw*{z}':h='ih*{z}':eval=frame:flags=bilinear")
            chains.append(f"{head}{','.join(base + fades + tail)}[ov{idx}]")
            layers = [f"ov{idx}"]

        if slide_in > 0:
            # Single quotes keep ffmpeg from reading the commas/colons inside the
            # expression as filtergraph separators.
            off = "(W+40)" if (it.get("slide_from") or "left").lower() == "right" else "(0-w-40)"
            prog = f"max(0,min(1,(t-{start:.3f})/{slide_in:.3f}))"
            ease = f"(1-pow(1-{prog},3))"
            x_expr = f"'{off}+({pos_x}-{off})*{ease}'"
        else:
            x_expr = str(pos_x)
        y_expr = str(pos_y)

        if anim:
            prog = f"max(0,min(1,(t-{start:.3f})/{anim_dur:.3f}))"
            rest = f"(1-(1-pow(1-{prog},3)))"   # 1 al empezar, 0 al llegar
            if anim in ("rise", "drop"):
                dy = int(round(frame_h * OVERLAY_ANIM_TRAVEL_Y)) * (1 if anim == "rise" else -1)
                y_expr = f"'{pos_y}+{dy}*{rest}'"
            elif anim in ("slide_left", "slide_right"):
                dx = int(round(frame_w * OVERLAY_ANIM_TRAVEL_X)) * (-1 if anim == "slide_left" else 1)
                x_expr = f"'{pos_x}+{dx}*{rest}'"
            elif anim == "zoom":
                if is_video:
                    vw, vh, _f = _ffprobe_video_info(it["png"])
                    if crop_f:
                        _l, _t, ckw, ckh = _crop_fracs(it["crop"])
                        vw, vh = vw * ckw, vh * ckh
                    nat_h = px_w * vh / max(vw, 1)
                else:
                    from PIL import Image as PILImage
                    with PILImage.open(it["png"]) as im:
                        nat_h = px_w * im.height / max(im.width, 1)
                x_expr = f"'{pos_x + px_w / 2:.1f}-w/2'"
                y_expr = f"'{pos_y + nat_h / 2:.1f}-h/2'"

        if is_video:
            src_in = max(float(it.get("src_in", 0.0) or 0.0), 0.0)
            volume = min(max(float(it.get("volume", 0.0) if it.get("volume") is not None else 0.0), 0.0), 2.0)
            layer_audio = volume > 0.001 and _has_audio_stream(it["png"])
            trim_args = ["-ss", f"{src_in:.3f}", "-t", f"{span * speed:.3f}"]
            if layer_audio:
                inputs += [*trim_args, "-i", it["png"]]
                delay_ms = int(round(start * 1000))
                # atempo acepta 0.5-2 por filtro: se encadena para 0.25x / 4x.
                tempo, left = [], speed
                while left > 2.0 + 1e-6:
                    tempo.append("atempo=2.0"); left /= 2.0
                while left < 0.5 - 1e-6:
                    tempo.append("atempo=0.5"); left /= 0.5
                if abs(left - 1.0) > 1e-3:
                    tempo.append(f"atempo={left:.4f}")
                chains.append(
                    f"[{ii}:a]aresample=48000,aformat=channel_layouts=stereo,"
                    f"{''.join(t + ',' for t in tempo)}"
                    f"volume={volume:.3f},adelay={delay_ms}|{delay_ms}[la{idx}]"
                )
                audio_tags.append(f"[la{idx}]")
            else:
                inputs += ["-an", *trim_args, "-i", it["png"]]
        else:
            inputs += ["-loop", "1", "-t", f"{clip_duration:.3f}", "-i", it["png"]]
        inputs += extra_inputs
        for lay in layers:
            tag = f"bg{len(overlays) + 1}"
            overlays.append(
                f"[{prev}][{lay}]overlay=x={x_expr}:y={y_expr}"
                f":eof_action=pass:format=auto[{tag}]"
            )
            prev = tag

    if not chains:
        for pth in scratch:
            try:
                os.remove(pth)
            except OSError:
                pass
        return None

    tail = [f"[{prev}]format=yuv420p[v]"]
    if audio_tags:
        # El audio de las capas de video se suma al del clip; sin normalize para
        # que el volumen elegido sea el que suena.
        srcs = (["[0:a]"] if main_has_audio else []) + audio_tags
        if len(srcs) == 1:
            tail.append(f"{srcs[0]}anull[aout]")
        else:
            tail.append(f"{''.join(srcs)}amix=inputs={len(srcs)}:duration=first"
                        f":dropout_transition=0:normalize=0[aout]")
        audio_args = ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    else:
        audio_args = ["-map", "0:a?", "-c:a", "copy"]
    filter_complex = ";".join(chains + overlays + tail)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        *video_encode_args(),
        *audio_args,
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        for pth in scratch:
            try:
                os.remove(pth)
            except OSError:
                pass
    return out_path


def _active_speaker_centers(in_path: str, shot_overrides: Optional[Dict[int, int]] = None):
    """Try the mouth-motion + audio active-speaker tracker. Returns None
    (falling back to naive largest-face tracking) if mediapipe isn't
    installed or anything about the analysis fails."""
    try:
        from .active_speaker import compute_active_speaker_centers
        return compute_active_speaker_centers(in_path, shot_overrides=shot_overrides)
    except Exception as e:
        print(f"[clip/local] active-speaker detection unavailable, falling back to largest-face: {e}", flush=True)
        return None


def _fit_crop_size(src_w: int, src_h: int, aspect_ratio: str) -> Tuple[int, int]:
    """Largest crop that fits inside a src_w x src_h frame at aspect_ratio,
    rounded down to even pixel dimensions (ffmpeg/H.264 requirement)."""
    target_ratio = _ratio(aspect_ratio)
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))
    return crop_w, crop_h


def _track_and_crop(
    in_path: str,
    out_w: int,
    out_h: int,
    shot_overrides: Optional[Dict[int, Union[int, List[int]]]] = None,
) -> str:
    """Per-frame face/active-speaker-tracked crop+resize of `in_path` down to
    exactly `out_w`x`out_h`, tracking whichever face is actively speaking
    (mouth motion correlated with audio) when possible, falling back to
    naive largest-face tracking otherwise.

    Returns the path to a silent (no audio) mp4 written next to `in_path`
    (`<in_path>.track.silent.mp4`) — caller is responsible for muxing audio
    back in and cleaning up the temp file."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_ratio = out_w / out_h

    speaker_centers = _active_speaker_centers(in_path, shot_overrides=shot_overrides)
    face_cascade = None
    if speaker_centers is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    silent_path = in_path + ".track.silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (out_w, out_h))

    last_center: Optional[Tuple[float, float]] = None
    last_crop_h: Optional[float] = None
    pos_smoothing = 0.15  # how aggressively to chase a new face position
    zoom_smoothing = 0.15  # how aggressively to chase a new zoom level
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        target = None
        target_face_h = None
        is_cut = False
        if speaker_centers is not None:
            if frame_idx < len(speaker_centers) and speaker_centers[frame_idx] is not None:
                t = speaker_centers[frame_idx]
                target = (t.cx, t.cy)
                target_face_h = t.face_h
                is_cut = t.is_cut
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                # Pick the largest face — usually the speaker.
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                target = (x + w / 2, y + h / 2)

        # A face bbox height lets us frame tightly on one pane of a
        # multi-camera grid instead of always taking the full frame height
        # (which, for a 2x2 grid, would straddle two unrelated panes).
        if target_face_h and target_face_h > 1:
            target_crop_h = min(src_h, max(out_h * 0.5, target_face_h * 3.8))
        else:
            target_crop_h = float(src_h)
        if target_crop_h * target_ratio > src_w:
            target_crop_h = src_w / target_ratio

        if target is not None:
            cx, cy = target
            if last_center is None or is_cut:
                last_center = (cx, cy)
                last_crop_h = target_crop_h
            else:
                lx, ly = last_center
                last_center = (
                    lx + (cx - lx) * pos_smoothing,
                    ly + (cy - ly) * pos_smoothing,
                )
                last_crop_h = last_crop_h + (target_crop_h - last_crop_h) * zoom_smoothing
        if last_center is None:
            last_center = (src_w / 2, src_h / 2)
            last_crop_h = float(src_h)

        cx, cy = last_center
        cur_crop_h = max(2, int(last_crop_h) - (int(last_crop_h) % 2))
        cur_crop_w = max(2, int(cur_crop_h * target_ratio))
        cur_crop_w = min(cur_crop_w, src_w)
        cur_crop_h = min(cur_crop_h, src_h)
        x0 = max(0, min(src_w - cur_crop_w, int(cx) - cur_crop_w // 2))
        y0 = max(0, min(src_h - cur_crop_h, int(cy) - cur_crop_h // 2))
        cropped = frame[y0:y0 + cur_crop_h, x0:x0 + cur_crop_w]
        if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
            cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        writer.write(cropped)
        frame_idx += 1

    cap.release()
    writer.release()
    return silent_path


def _reframe_vertical(
    in_path: str,
    out_path: str,
    aspect_ratio: str,
    shot_overrides: Optional[Dict[int, Union[int, List[int]]]] = None,
) -> str:
    """Crop the cut clip to the target aspect ratio, tracking whichever face
    is actively speaking (mouth motion correlated with audio) when possible,
    falling back to naive largest-face tracking otherwise."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Output resolution is fixed for the whole clip (largest crop that fits
    # the source at the target ratio). The per-frame crop *window* can be
    # smaller/positioned anywhere within the frame — e.g. to isolate one
    # pane of a multi-camera grid layout — and gets resized up to this.
    out_w, out_h = _fit_crop_size(src_w, src_h, aspect_ratio)

    silent_path = _track_and_crop(in_path, out_w, out_h, shot_overrides=shot_overrides)
    try:
        # Mux audio from the cut clip back onto the silent reframed video.
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", silent_path,
            "-i", in_path,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest",
            out_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(silent_path):
            os.remove(silent_path)
    return out_path


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    shot_overrides: Optional[Dict[int, Union[int, List[int]]]] = None,
    cut_segments: Optional[List[Tuple[float, float]]] = None,
    crop_enabled: bool = True,
    trim: Optional[Dict] = None,
) -> str:
    """Cut (+ optionally reframe) one highlight, returning the local mp4 path.

    `shot_overrides`: optional {shot_index: track_id} map to force which
    face is treated as the speaker for specific camera shots (see
    `active_speaker.list_shots_for_review` to discover shot indices/track
    ids for a given source_path/start_time/end_time before regenerating).

    `cut_segments`: optional list of (start, end) spans, clip-relative to
    start_time, to remove from inside the highlight (silence gaps / filler
    words -- see editing.py). When given, use
    `editing.remap_words_across_cuts()` on the caption word list too, so
    burned timestamps stay in sync with the shortened output.

    `crop_enabled`: when False, skip the vertical reframe and keep the
    source's original framing/aspect ratio (just the cuts applied) --
    matches the editor's "Recorte vertical" toggle being off."""
    if not crop_enabled:
        if cut_segments:
            _cut_subclip_multi(source_path, start_time, end_time, cut_segments, out_path, trim=trim)
        else:
            _cut_subclip(source_path, start_time, end_time, out_path, trim=trim)
        return out_path

    cut_path = out_path + ".cut.mp4"
    try:
        if cut_segments:
            _cut_subclip_multi(source_path, start_time, end_time, cut_segments, cut_path, intermediate=True, trim=trim)
        else:
            _cut_subclip(source_path, start_time, end_time, cut_path, intermediate=True, trim=trim)
        _reframe_vertical(cut_path, out_path, aspect_ratio, shot_overrides=shot_overrides)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def _ffprobe_resolution(path: str) -> Tuple[int, int]:
    """(width, height) of a video's first video stream, ya rotado."""
    w, h, _ = probe_display_video(path)
    return w, h


def _manual_crop_rect(
    src_w: int, src_h: int, cx: float, cy: float, zoom: float, pane_aspect: float
) -> Tuple[int, int, int, int]:
    """Pixel crop rect (w, h, x, y) for one normalized editor selection.

    Mirrors static/editor.html's crop-box maths exactly so the rectangle the
    user drags is the rectangle ffmpeg cuts: `cx`/`cy` are 0-1 fractions of
    the source frame (the box *centre*), `zoom` is the UI multiplier where
    1 = the box spans the full frame height, and the width is always derived
    from `pane_aspect` (9:16 for a single camera, 9:8 for one half of a
    50/50 stack). The box is clamped inside the frame instead of letting the
    centre drift off-screen -- same clamp the overlay applies -- and both
    dimensions are forced even for H.264."""
    zoom = max(1.0, float(zoom))
    crop_h = src_h / zoom
    crop_w = crop_h * pane_aspect
    if crop_w > src_w:
        crop_w = float(src_w)
        crop_h = crop_w / pane_aspect
    w = max(2, int(crop_w) - (int(crop_w) % 2))
    h = max(2, int(crop_h) - (int(crop_h) % 2))
    x = int(round(float(cx) * src_w - w / 2.0))
    y = int(round(float(cy) * src_h - h / 2.0))
    x = max(0, min(src_w - w, x))
    y = max(0, min(src_h - h, y))
    return w, h, x, y


def _manual_keyframe_cuts(
    keyframes: List[Dict],
    start_time: float,
    duration: float,
    cut_segments: Optional[List[Tuple[float, float]]],
) -> List[Tuple[float, Dict]]:
    """Editor keyframes → [(output_start_time, keyframe)] on the *cut* clip's
    own timeline, sorted and de-duplicated.

    Keyframe `t` is an absolute source timestamp (the editor keeps every
    timeline layer on one axis -- see
    feedback_timeline_single_source_of_truth.md), so it has to be shifted to
    clip-relative time and then squeezed through the same cut map the video
    itself goes through, or a keyframe placed after a removed span would fire
    late. A keyframe landing *inside* a removed span is snapped forward to
    where the video resumes instead of being dropped: the editor seeds a
    clip's first keyframe at its start_time, which "quitar silencios" very
    often removes, and dropping it threw the user's framing away for the
    whole clip. The first entry is always forced to 0.0 so the clip opens on
    the earliest keyframe rather than on an unframed gap."""
    kfs = sorted(keyframes, key=lambda k: float(k.get("t", 0.0)))
    keep = keep_segments_from_cuts(cut_segments, duration) if cut_segments else None

    points: List[Tuple[float, Dict]] = []
    for kf in kfs:
        rel = float(kf.get("t", 0.0)) - start_time
        if keep is not None:
            snapped = _snap_forward(rel, keep)
            if snapped is None:
                continue  # past every kept span: this framing never plays
            mapped = remap_time(snapped, keep)
            if mapped is None:
                continue
            rel = mapped
        rel = max(0.0, min(rel, duration))
        points.append((rel, kf))

    if not points:
        return []
    points[0] = (0.0, points[0][1])
    # Collapse keyframes that collapsed onto the same output timestamp (two
    # keyframes straddling a removed span): the later one wins, matching the
    # editor's "this framing applies from here until the next" semantics.
    deduped: List[Tuple[float, Dict]] = []
    for t, kf in points:
        if deduped and abs(deduped[-1][0] - t) < 1e-3:
            deduped[-1] = (t, kf)
        else:
            deduped.append((t, kf))
    return deduped


def _pane_filters(
    src_w: int, src_h: int, spec: Dict, out_w: int, out_h: int, pane_aspect: float
) -> str:
    """crop+scale filter string for one pane from one {cx, cy, zoom} spec."""
    w, h, x, y = _manual_crop_rect(
        src_w, src_h,
        float(spec.get("cx", 0.5)), float(spec.get("cy", 0.5)), float(spec.get("zoom", 1.0)),
        pane_aspect,
    )
    return f"crop={w}:{h}:{x}:{y},scale={out_w}:{out_h},setsar=1"


def crop_clip_manual(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    keyframes: List[Dict],
    mode: str = "single",
    cut_segments: Optional[List[Tuple[float, float]]] = None,
    trim: Optional[Dict] = None,
) -> str:
    """Cut one highlight and reframe it from the editor's *manual* crop
    rectangles, so the exported clip matches the vertical preview exactly.

    This is the manual counterpart to `crop_clip_local`, which reframes by
    automatic active-speaker face tracking and therefore ignores whatever the
    user dragged. Callers pick between them on whether the highlight actually
    has keyframes.

    `mode="single"` renders one 9:16 crop; `mode="split"` renders two crops of
    the *same* source frame stacked 50/50 (top camera / bottom camera), each
    pane being `out_w` x `out_h/2`.

    `keyframes` are the editor's points on the absolute source time axis:
    `{t, cx, cy, zoom}` for single mode, `{t, top: {...}, bottom: {...}}` for
    split. Framing is a hard cut at each keyframe (no panning), which is
    rendered as one trim+crop segment per interval concatenated back
    together -- ffmpeg's `crop` filter only re-evaluates x/y per frame, not
    w/h, so a time-varying *zoom* cannot be expressed as a single crop
    expression."""
    duration = end_time - start_time
    cut_path = out_path + ".cut.mp4"
    try:
        if cut_segments:
            _cut_subclip_multi(source_path, start_time, end_time, cut_segments, cut_path, intermediate=True, trim=trim)
        else:
            _cut_subclip(source_path, start_time, end_time, cut_path, intermediate=True, trim=trim)

        src_w, src_h = _ffprobe_resolution(cut_path)
        out_w, out_h = _fit_crop_size(src_w, src_h, aspect_ratio)

        if mode == "split":
            pane_h = out_h // 2
            pane_h = max(2, pane_h - (pane_h % 2))
            # The stack, not _fit_crop_size, defines the real output height:
            # an odd half would otherwise leave the two panes short of out_h.
            pane_aspect = out_w / float(pane_h)
        else:
            pane_h = out_h
            pane_aspect = out_w / float(out_h)

        points = _manual_keyframe_cuts(keyframes, start_time, duration, cut_segments)
        if not points:
            centre = {"cx": 0.5, "cy": 0.5, "zoom": 1.0}
            points = [(0.0, {"top": centre, "bottom": centre} if mode == "split" else centre)]

        def specs_for(kf: Dict) -> List[Dict]:
            if mode == "split":
                return [kf.get("top") or {}, kf.get("bottom") or {}]
            return [kf]

        # Single framing for the whole clip: no trim/concat needed, so keep
        # the cheap one-pass -filter_complex form.
        cut_duration = _ffprobe_duration(cut_path)
        boundaries = [t for t, _ in points] + [cut_duration]

        chain_parts: List[str] = []
        segment_labels: List[str] = []
        for i, (_, kf) in enumerate(points):
            seg_start, seg_end = boundaries[i], boundaries[i + 1]
            if seg_end - seg_start <= 1e-3 and len(points) > 1:
                continue
            src_label = "0:v"
            if len(points) > 1:
                chain_parts.append(
                    f"[0:v]trim=start={seg_start:.3f}:end={seg_end:.3f},"
                    f"setpts=PTS-STARTPTS[seg{i}];"
                )
                src_label = f"seg{i}"
            panes = specs_for(kf)
            if mode == "split":
                chain_parts.append(f"[{src_label}]split=2[{src_label}t][{src_label}b];")
                chain_parts.append(
                    f"[{src_label}t]{_pane_filters(src_w, src_h, panes[0], out_w, pane_h, pane_aspect)}[top{i}];"
                )
                chain_parts.append(
                    f"[{src_label}b]{_pane_filters(src_w, src_h, panes[1], out_w, pane_h, pane_aspect)}[bot{i}];"
                )
                chain_parts.append(f"[top{i}][bot{i}]vstack=inputs=2[v{i}];")
            else:
                chain_parts.append(
                    f"[{src_label}]{_pane_filters(src_w, src_h, panes[0], out_w, pane_h, pane_aspect)}[v{i}];"
                )
            segment_labels.append(f"[v{i}]")

        filter_complex = "".join(chain_parts)
        if len(segment_labels) > 1:
            filter_complex += "".join(segment_labels) + f"concat=n={len(segment_labels)}:v=1:a=0[vout]"
        else:
            # One segment: rename its label rather than concat=n=1, which
            # ffmpeg rejects.
            filter_complex += f"{segment_labels[0]}null[vout]"

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", cut_path,
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "0:a:0?",
            *video_encode_args(),
            "-c:a", "aac", "-b:a", "128k",
            out_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def _ffprobe_duration(path: str) -> float:
    """Duration in seconds of a media file, via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _build_external_pane(
    source: str,
    start: float,
    end: float,
    needed_duration: float,
    out_w: int,
    pane_h: int,
    fps: float,
    tmp_prefix: str,
) -> str:
    """Cut `source[start:end]`, loop (if shorter) or trim (if longer) it to
    exactly `needed_duration` seconds, and simple-center-crop/scale it (no
    face tracking) to `out_w`x`pane_h` at `fps`. Returns the path to the
    resulting silent (no audio) mp4."""
    cut_path = tmp_prefix + ".cut.mp4"
    pane_path = tmp_prefix + ".pane.mp4"
    try:
        _cut_subclip(source, start, end, cut_path, intermediate=True)
        src_duration = _ffprobe_duration(cut_path)
        vf = (
            f"fps={fps},"
            f"scale=w={out_w}:h={pane_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{pane_h}"
        )
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        if src_duration < needed_duration:
            # Source is shorter than the pane needs — loop it indefinitely,
            # then -t below trims the loop to the exact needed duration.
            cmd += ["-stream_loop", "-1"]
        cmd += [
            "-i", cut_path,
            "-t", f"{needed_duration:.3f}",
            "-vf", vf,
            "-an",
            *video_encode_args(intermediate=True),
            pane_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return pane_path


def crop_clip_split(
    main_source: str,
    main_start: float,
    main_end: float,
    aspect_ratio: str,
    out_path: str,
    top: Optional[Dict] = None,
    bottom: Optional[Dict] = None,
    shot_overrides: Optional[Dict[int, Union[int, List[int]]]] = None,
) -> str:
    """Stack two half-height panes into one `aspect_ratio` canvas.

    `top`/`bottom`, when provided, are dicts {"source": path, "start": float,
    "end": float} describing an external clip/segment to use for that pane
    instead of the main clip. When None, that pane uses the normal
    face/active-speaker-tracked crop of `main_source` between
    `main_start`/`main_end`.

    Only `main_source`'s audio (main_start-main_end) ends up in the output;
    any external pane video is silent/muted in the composite.
    """
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    needed_duration = main_end - main_start
    main_cut_path = out_path + ".main.cut.mp4"
    combined_silent_path = out_path + ".combined.silent.mp4"
    temp_files = [main_cut_path, combined_silent_path]

    try:
        _cut_subclip(main_source, main_start, main_end, main_cut_path, intermediate=True)

        cap = cv2.VideoCapture(main_cut_path)
        if not cap.isOpened():
            raise RuntimeError(f"could not open {main_cut_path}")
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        out_w, out_h = _fit_crop_size(src_w, src_h, aspect_ratio)
        pane_h = out_h // 2
        pane_h = max(2, pane_h - (pane_h % 2))

        pane_paths = []
        for slot_name, pane_spec in (("top", top), ("bottom", bottom)):
            if pane_spec is None:
                raw_silent = _track_and_crop(main_cut_path, out_w, pane_h, shot_overrides=shot_overrides)
                pane_path = out_path + f".{slot_name}.pane.mp4"
                os.replace(raw_silent, pane_path)
            else:
                pane_path = _build_external_pane(
                    pane_spec["source"], float(pane_spec["start"]), float(pane_spec["end"]),
                    needed_duration, out_w, pane_h, fps,
                    out_path + f".{slot_name}.ext",
                )
            pane_paths.append(pane_path)
            temp_files.append(pane_path)

        top_path, bottom_path = pane_paths

        # Stack the two silent panes into one silent out_w x (pane_h*2) video.
        stack_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", top_path,
            "-i", bottom_path,
            "-filter_complex", "[0:v][1:v]vstack=inputs=2[v]",
            "-map", "[v]",
            *video_encode_args(),
            combined_silent_path,
        ]
        subprocess.run(stack_cmd, check=True)

        # Mux the main clip's audio back onto the combined silent video.
        mux_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", combined_silent_path,
            "-i", main_cut_path,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest",
            out_path,
        ]
        subprocess.run(mux_cmd, check=True)
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    transcript_segments: Optional[List[Dict]] = None,
    trim_silence: bool = True,
    trim_fillers: bool = True,
) -> List[Dict]:
    """Crop each highlight into its own clip.

    `transcript_segments`: optional full transcript (`transcript["segments"]`
    from transcribe_local). When given, dead-air silences and/or LLM-flagged
    filler words are cut out of each clip in the same ffmpeg pass (see
    editing.py), and each result dict gets a `"words"` list already remapped
    onto the shortened clip's timeline -- pass that straight to
    `burn_subtitles()` instead of recomputing `words_for_window()`."""
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    used_names: set = set()
    for i, h in enumerate(highlights, 1):
        slug = _clip_slug(h)
        filename = f"{i:02d}_{slug}.mp4"
        n = 2
        while filename in used_names:
            filename = f"{i:02d}_{slug}-{n}.mp4"
            n += 1
        used_names.add(filename)
        out_path = os.path.join(out_dir, filename)
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        start_time = float(h["start_time"])
        end_time = float(h["end_time"])
        cut_segments = None
        remapped_words = None
        if transcript_segments is not None and (trim_silence or trim_fillers):
            from .subtitles import words_for_window
            words = words_for_window(transcript_segments, start_time, end_time)
            cut_segments = find_cut_segments(
                words, end_time - start_time,
                remove_silence=trim_silence, remove_fillers=trim_fillers,
            )
            if cut_segments:
                print(f"[clip/local] {i}: trimming {len(cut_segments)} silence/filler span(s)", flush=True)
                keep = keep_segments_from_cuts(cut_segments, end_time - start_time)
                remapped_words = remap_words_across_cuts(words, keep)
            else:
                remapped_words = words
        try:
            crop_clip_local(
                source_path,
                start_time,
                end_time,
                aspect_ratio,
                out_path,
                cut_segments=cut_segments,
            )
            result = {**h, "clip_url": out_path}
            if remapped_words is not None:
                result["words"] = remapped_words
            results.append(result)
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
