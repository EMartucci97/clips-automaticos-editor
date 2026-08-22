"""Google Drive input support (single file or whole folder).

Uses `gdown`, which downloads the same way a signed-out browser would — no
OAuth, no API key. That means the file or folder must be shared as
"Anyone with the link" (Viewer is enough).
"""
import os
from pathlib import Path
from typing import List, Optional

from ..config import LOCAL_OUTPUT_DIR

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def is_drive_url(source: str) -> bool:
    return "drive.google.com" in source


def is_drive_folder_url(source: str) -> bool:
    return is_drive_url(source) and "/folders/" in source


def _import_gdown():
    try:
        import gdown  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "gdown is required to download from Google Drive. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e
    return gdown


def download_drive_file(url: str, out_dir: Optional[str] = None) -> str:
    """Download a single Drive file share link, return its local path."""
    gdown = _import_gdown()
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    print(f"[download/drive] fetching file: {url}", flush=True)
    path = gdown.download(url=url, output=os.path.join(out_dir, ""), quiet=False)
    if not path or not os.path.exists(path):
        raise RuntimeError(
            f"Could not download from Google Drive: {url}. "
            "Make sure it's shared as 'Anyone with the link'."
        )
    print(f"[download/drive] ready: {path}", flush=True)
    return path


def download_drive_folder(url: str, out_dir: Optional[str] = None) -> List[str]:
    """Download every video file in a public Drive folder.

    Returns local paths of the video files, sorted by name.
    """
    gdown = _import_gdown()
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    dest = os.path.join(out_dir, "drive_batch")
    os.makedirs(dest, exist_ok=True)
    print(f"[download/drive] fetching folder: {url}", flush=True)
    gdown.download_folder(url=url, output=dest, quiet=False, use_cookies=False)

    video_paths = sorted(
        (str(p.resolve()) for p in Path(dest).rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS),
        key=str.lower,
    )
    if not video_paths:
        raise RuntimeError(
            f"No video files found in Drive folder: {url}. "
            "Make sure it's shared as 'Anyone with the link' and contains video files."
        )
    print(f"[download/drive] {len(video_paths)} video(s) ready in {dest}", flush=True)
    return video_paths
