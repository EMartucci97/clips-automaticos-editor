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
from typing import Dict, List, Optional, Tuple, Union

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
        _cut_subclip(source, start, end, cut_path)
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
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
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
        _cut_subclip(main_source, main_start, main_end, main_cut_path)

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
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
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
