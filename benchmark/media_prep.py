"""Copies each test clip into a per-config subdirectory before transcribing.

transcriber.py's cache is keyed only by filename stem + mtime (see
transcribe_local in shorts_generator/local/transcriber.py), not by model or
any other param. Reusing one shared copy of a clip across configs would mean
the second config to run just gets served the first config's cached .srt.
Giving every config its own copy of the source file sidesteps that entirely
-- each copy has its own stem-derived cache path under its own directory.
"""
import shutil
from pathlib import Path
from typing import Tuple

from .config import TRANSCRIPTS_DIR


def prepared_clip_path(config_id: str, clip_id: str, source_path: str) -> Path:
    """Copy source_path into benchmark/work/transcripts/{config_id}/ (if not
    already there) and return the copy's path. Idempotent: skips the copy if
    the destination already exists and is not older than the source."""
    ext = Path(source_path).suffix
    dest_dir = TRANSCRIPTS_DIR / config_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{clip_id}{ext}"
    if not dest_path.exists() or dest_path.stat().st_mtime < Path(source_path).stat().st_mtime:
        shutil.copy2(source_path, dest_path)
    return dest_path


def prepare_all(config_id: str, clips: list) -> list:
    """clips: list of (clip_id, source_path). Returns list of (clip_id, dest_path)."""
    return [(clip_id, str(prepared_clip_path(config_id, clip_id, src))) for clip_id, src in clips]
