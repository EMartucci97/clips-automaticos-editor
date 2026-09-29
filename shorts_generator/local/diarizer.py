"""Speaker identification for --mode local: labels each transcript segment
ROBERTO or OTHER so highlights.py can prioritize when Roberto speaks (or
responds to a guest), without assuming a fixed turn order.

Approach: pyannote-audio diarizes the clip into anonymous speaker clusters
and returns one voice embedding per cluster. Each cluster is matched by
cosine similarity against a reference embedding built once from clean
solo-Roberto clips (see LOCAL_DIARIZATION_REF_DIR).

Everything here is best-effort — diarize_and_label() catches its own errors
so a missing pyannote install, missing HF_TOKEN, or a diarization failure
just runs the pipeline without speaker labels instead of breaking it.
"""
import glob
import json
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # ops de pyannote sin kernel MPS van a CPU
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import (
    HF_TOKEN,
    LOCAL_DIARIZATION_REF_DIR,
    LOCAL_DIARIZATION_SIMILARITY_THRESHOLD,
    LOCAL_OUTPUT_DIR,
)

DIARIZATION_PIPELINE_NAME = "pyannote/speaker-diarization-3.1"
ROBERTO_LABEL = "ROBERTO"
OTHER_LABEL = "OTHER"

_pipeline = None
_roberto_ref_embedding: Optional[np.ndarray] = None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from pyannote.audio import Pipeline
        _pipeline = Pipeline.from_pretrained(DIARIZATION_PIPELINE_NAME, token=HF_TOKEN or None)
        # GPU de Apple (MPS): ~5x más rápido que CPU con turnos idénticos
        # (medido sobre 3 min del podcast). _run_pipeline cae a CPU si falla.
        if torch.backends.mps.is_available():
            _pipeline.to(torch.device("mps"))
    return _pipeline


_pipeline_last_used = 0.0
_pipeline_lock = threading.Lock()


def unload_if_idle(max_idle_s: float) -> bool:
    """Suelta el pipeline de pyannote (~1GB entre RAM y memoria MPS) si no se
    usó en max_idle_s. Se recarga solo en la próxima diarización (~5s)."""
    global _pipeline
    if _pipeline is None or not _pipeline_lock.acquire(blocking=False):
        return False
    try:
        if _pipeline is None or time.time() - _pipeline_last_used < max_idle_s:
            return False
        _pipeline = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        return True
    finally:
        _pipeline_lock.release()


def _run_pipeline(audio_path: str, **kwargs):
    with _pipeline_lock:
        global _pipeline_last_used
        try:
            return _run_pipeline_unlocked(audio_path, **kwargs)
        finally:
            _pipeline_last_used = time.time()


def _run_pipeline_unlocked(audio_path: str, **kwargs):
    pipeline = _get_pipeline()
    try:
        return pipeline(audio_path, **kwargs)
    except RuntimeError as e:
        if "mps" not in str(e).lower():
            raise
        import torch
        print(f"[diarize/local] MPS falló ({e}); reintentando en CPU", flush=True)
        pipeline.to(torch.device("cpu"))
        return pipeline(audio_path, **kwargs)


def _roberto_reference_path() -> Path:
    return Path(LOCAL_DIARIZATION_REF_DIR) / ".roberto_embedding.npy"


def _get_roberto_reference() -> np.ndarray:
    """Enrollment: average voice embedding across the reference clips.
    Cached to disk (keyed by the ref clip set) since it's the same every run."""
    global _roberto_ref_embedding
    if _roberto_ref_embedding is not None:
        return _roberto_ref_embedding

    ref_clips = sorted(glob.glob(os.path.join(LOCAL_DIARIZATION_REF_DIR, "*.wav")))
    if not ref_clips:
        raise RuntimeError(f"no reference clips found in {LOCAL_DIARIZATION_REF_DIR}")

    cache_path = _roberto_reference_path()
    newest_ref_mtime = max(os.path.getmtime(p) for p in ref_clips)
    if cache_path.exists() and cache_path.stat().st_mtime >= newest_ref_mtime:
        _roberto_ref_embedding = np.load(cache_path)
        return _roberto_ref_embedding

    print(f"[diarize/local] enrolling Roberto's voice from {len(ref_clips)} reference clip(s)", flush=True)
    embeddings = []
    for clip in ref_clips:
        out = _run_pipeline(clip, num_speakers=1)
        embeddings.append(out.speaker_embeddings[0])
    _roberto_ref_embedding = np.mean(embeddings, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, _roberto_ref_embedding)
    return _roberto_ref_embedding


def _diarization_cache_path(media_path: str) -> Path:
    cache_dir = Path(LOCAL_OUTPUT_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (Path(media_path).stem + ".diar.json")


def _extract_audio_wav(media_path: str) -> str:
    """pyannote loads audio via torchaudio, which is unreliable on some mp4/
    mkv containers straight off the downloader — extract to mono 16kHz wav
    first (ffmpeg, same tool clipper.py already depends on). Cached like the
    transcript sidecars in transcriber.py."""
    cache_dir = Path(LOCAL_OUTPUT_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav_path = cache_dir / (Path(media_path).stem + ".diar_audio.wav")
    if wav_path.exists() and wav_path.stat().st_mtime >= os.path.getmtime(media_path):
        return str(wav_path)

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", media_path, "-ac", "1", "-ar", "16000", str(wav_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")
    return str(wav_path)


def diarize_local(media_path: str) -> List[Dict]:
    """Diarize media_path and return [{start, end, speaker}, ...] with
    speaker in (ROBERTO_LABEL, OTHER_LABEL). Caches to a .diar.json sidecar
    the same way transcribe_local caches .srt."""
    cache_path = _diarization_cache_path(media_path)
    if cache_path.exists() and cache_path.stat().st_mtime >= os.path.getmtime(media_path):
        print(f"[diarize/local] reusing cached diarization: {cache_path}", flush=True)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    roberto_ref = _get_roberto_reference()
    audio_wav = _extract_audio_wav(media_path)

    print(f"[diarize/local] diarizing {media_path}", flush=True)
    out = _run_pipeline(audio_wav)
    labels = out.speaker_diarization.labels()
    embeddings = out.speaker_embeddings

    roberto_cluster = None
    best_sim = float("-inf")
    for label, emb in zip(labels, embeddings):
        sim = _cosine_sim(roberto_ref, emb)
        print(f"[diarize/local] {label}: similarity to Roberto = {sim:.3f}", flush=True)
        if sim > best_sim:
            best_sim = sim
            roberto_cluster = label if sim >= LOCAL_DIARIZATION_SIMILARITY_THRESHOLD else None

    turns = []
    for turn, _, speaker in out.speaker_diarization.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": ROBERTO_LABEL if speaker == roberto_cluster else OTHER_LABEL,
        })

    cache_path.write_text(json.dumps(turns), encoding="utf-8")
    return turns


def label_transcript_speakers(transcript: Dict, turns: List[Dict]) -> Dict:
    """Tag each transcript segment with the speaker whose diarization turn
    overlaps it the most. Segments with no overlapping turn are left
    unlabeled (speaker=None) rather than guessed."""
    for seg in transcript.get("segments", []):
        seg_start, seg_end = seg["start"], seg["end"]
        best_speaker = None
        best_overlap = 0.0
        for turn in turns:
            overlap = min(seg_end, turn["end"]) - max(seg_start, turn["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        seg["speaker"] = best_speaker
    return transcript


def diarize_and_label(media_path: str, transcript: Dict) -> Dict:
    """Best-effort entry point: diarize + label, or leave the transcript
    untouched (no 'speaker' keys) if anything about it fails."""
    try:
        turns = diarize_local(media_path)
        transcript = label_transcript_speakers(transcript, turns)
        labeled = sum(1 for s in transcript["segments"] if s.get("speaker") == ROBERTO_LABEL)
        print(f"[diarize/local] {labeled}/{len(transcript['segments'])} segments labeled ROBERTO", flush=True)
    except Exception as e:
        print(f"[diarize/local] speaker identification unavailable, continuing without it: {e}", flush=True)
    return transcript
