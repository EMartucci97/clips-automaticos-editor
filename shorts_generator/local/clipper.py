"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models).
"""
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

from ..config import LOCAL_OUTPUT_DIR


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


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end → re-encoded mp4 with audio."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", source_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
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


def _reframe_vertical(
    in_path: str,
    out_path: str,
    aspect_ratio: str,
    shot_overrides: Optional[Dict[int, int]] = None,
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

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    speaker_centers = _active_speaker_centers(in_path, shot_overrides=shot_overrides)
    face_cascade = None
    if speaker_centers is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Output resolution is fixed for the whole clip (largest crop that fits
    # the source at the target ratio). The per-frame crop *window* can be
    # smaller/positioned anywhere within the frame — e.g. to isolate one
    # pane of a multi-camera grid layout — and gets resized up to this.
    out_w, out_h = crop_w, crop_h

    silent_path = out_path + ".silent.mp4"
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
    os.remove(silent_path)
    return out_path


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    shot_overrides: Optional[Dict[int, int]] = None,
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path.

    `shot_overrides`: optional {shot_index: track_id} map to force which
    face is treated as the speaker for specific camera shots (see
    `active_speaker.list_shots_for_review` to discover shot indices/track
    ids for a given source_path/start_time/end_time before regenerating)."""
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(cut_path, out_path, aspect_ratio, shot_overrides=shot_overrides)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
) -> List[Dict]:
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
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
