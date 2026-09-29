"""Two-fixed-camera vertical reframe.

Simpler alternative to active_speaker.py's continuous per-frame face
tracking. This source is shot as one static wide two-shot for the whole
interview (both people always in frame, angled toward each other rather
than the camera) — not a multi-cam edit that cuts between solo shots. So
instead of chasing a face frame by frame, this detects each person's fixed
position once (sampling several frames across the whole clip) and treats
"camera A" / "camera B" as two fixed crop sectors of that one wide shot,
hard-switched between following the existing diarization timeline
(diarizer.py) — already far more reliable than the mouth-motion+audio
heuristic since it looks at the whole source's audio, not per-frame guesses.

Face angle here (~30-45 degrees, looking at each other, not the lens) is
too steep for mediapipe FaceMesh's short-range detector, which reliably
finds nothing on this footage — so camera positions are detected with
mediapipe's face_detection solution (full-range model), which handles the
angle and distance. Because that model doesn't expose mouth landmarks,
figuring out which detected face is Roberto uses a plain pixel-difference
motion proxy (variance of frame-to-frame differences in the lower part of
each face box during Roberto's own diarization spans) rather than a mouth
aspect ratio.

Falls back to the regular active-speaker tracker when it can't find two
consistent faces or diarization isn't set up.
"""
import os
import subprocess
from typing import Dict, List, Tuple

import numpy as np

from .clipper import _cut_subclip, _fit_crop_size, _reframe_vertical
from .diarizer import ROBERTO_LABEL, diarize_local

_LOOSE_ZOOM_FACTOR = 8.0        # crop_h = face_h * factor: a wide "todo un plano" shot, not tight on the face
_MIN_HOLD_SECONDS = 0.35        # merge diarization turns shorter than this into the previous camera (avoid flicker on "sí", "claro", laughs)
_DETECT_MAX_SAMPLES = 24        # frames sampled across the whole clip to locate the two fixed faces
_DETECT_MIN_VALID_FRAMES = 3    # need at least this many frames with 2 confident faces to trust the average
_MIN_FACE_HEIGHT_FRAC = 0.05    # ignore tiny detections (e.g. a face in a background photo)
_MOTION_SAMPLE_PAIRS = 8        # frame pairs sampled per side when scoring mouth-region motion


class Camera:
    __slots__ = ("cx", "cy", "face_h")

    def __init__(self, cx: float, cy: float, face_h: float):
        self.cx, self.cy, self.face_h = cx, cy, face_h


def _probe_dims_and_fps(video_path: str) -> Tuple[int, int, float]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    duration = frame_count / fps if fps else 0.0
    return img_w, img_h, fps, duration


def _detect_faces(face_detector, frame, img_w: int, img_h: int) -> List[Tuple[float, float, float]]:
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detector.process(rgb)
    dets = result.detections or []
    faces = []
    for d in dets:
        box = d.location_data.relative_bounding_box
        h_px = box.height * img_h
        if h_px < img_h * _MIN_FACE_HEIGHT_FRAC:
            continue
        cx = (box.xmin + box.width / 2) * img_w
        cy = (box.ymin + box.height / 2) * img_h
        faces.append((cx, cy, h_px))
    return faces


def _detect_two_cameras(video_path: str, img_w: int, img_h: int, fps: float, duration: float) -> Tuple[Camera, Camera]:
    """Sample frames across the whole clip and average the leftmost/
    rightmost consistently-detected face into two fixed cameras."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)

    left_samples, right_samples = [], []
    try:
        n = max(1, _DETECT_MAX_SAMPLES)
        for i in range(n):
            t = duration * (i + 0.5) / n
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ret, frame = cap.read()
            if not ret:
                continue
            faces = _detect_faces(face_detector, frame, img_w, img_h)
            if len(faces) < 2:
                continue
            faces.sort(key=lambda f: f[2], reverse=True)
            two = sorted(faces[:2], key=lambda f: f[0])
            left_samples.append(two[0])
            right_samples.append(two[1])
    finally:
        cap.release()
        face_detector.close()

    if len(left_samples) < _DETECT_MIN_VALID_FRAMES:
        raise RuntimeError("two_camera: could not find two consistent faces in this clip")

    def _avg(samples):
        return Camera(
            cx=float(np.mean([s[0] for s in samples])),
            cy=float(np.mean([s[1] for s in samples])),
            face_h=float(np.mean([s[2] for s in samples])),
        )

    return _avg(left_samples), _avg(right_samples)


def _motion_score(video_path: str, cam: Camera, fps: float, spans_s: List[Tuple[float, float]]) -> float:
    """Proxy for mouth movement: mean frame-to-frame pixel difference in the
    lower half of `cam`'s face box (roughly the mouth/chin area) across a
    handful of close frame pairs sampled from `spans_s`."""
    import cv2

    if not spans_s:
        return 0.0

    half = cam.face_h * 0.6
    x0 = int(max(0, cam.cx - half))
    x1 = int(cam.cx + half)
    y0 = int(cam.cy)
    y1 = int(cam.cy + half)

    cap = cv2.VideoCapture(video_path)
    total, count = 0.0, 0
    try:
        spans_sorted = sorted(spans_s, key=lambda s: s[1] - s[0], reverse=True)[:_MOTION_SAMPLE_PAIRS]
        for s, e in spans_sorted:
            mid_t = (s + e) / 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid_t * fps))
            ret1, f1 = cap.read()
            ret2, f2 = cap.read()
            if not (ret1 and ret2):
                continue
            crop1 = cv2.cvtColor(f1[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            crop2 = cv2.cvtColor(f2[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            if crop1.size == 0 or crop2.size == 0 or crop1.shape != crop2.shape:
                continue
            diff = cv2.absdiff(crop1, crop2)
            total += float(np.mean(diff))
            count += 1
    finally:
        cap.release()

    return total / count if count else 0.0


def _camera_for_segment(video_path: str, img_w: int, img_h: int, fps: float, span_start: float, span_end: float,
                         midline_x: float, phys_side: str, fallback_cam: Camera, max_samples: int = 3) -> Camera:
    """Re-detect the face on `phys_side` (of `midline_x`) within one
    schedule segment, rather than trusting one global average for the whole
    clip — this static shot can still drift/reframe partway through a long
    highlight, so each segment gets its own fresh read, falling back to the
    clip-wide average if nothing is found."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
    samples = []
    try:
        duration = max(0.01, span_end - span_start)
        n = max(1, min(max_samples, int(duration) + 1))
        for i in range(n):
            t = span_start + duration * (i + 0.5) / n
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ret, frame = cap.read()
            if not ret:
                continue
            faces = _detect_faces(face_detector, frame, img_w, img_h)
            candidates = [f for f in faces if (f[0] < midline_x) == (phys_side == "left")]
            if not candidates:
                continue
            cx, cy, h = max(candidates, key=lambda f: f[2])
            samples.append((cx, cy, h))
    finally:
        cap.release()
        face_detector.close()

    if not samples:
        return fallback_cam

    return Camera(
        cx=float(np.mean([s[0] for s in samples])),
        cy=float(np.mean([s[1] for s in samples])),
        face_h=float(np.mean([s[2] for s in samples])),
    )


def _assign_roberto_side(video_path: str, left: Camera, right: Camera, fps: float,
                          turns: List[Dict], clip_start: float, clip_end: float) -> str:
    roberto_spans = []
    for t in turns:
        if t["speaker"] != ROBERTO_LABEL or t["end"] <= clip_start or t["start"] >= clip_end:
            continue
        s = max(clip_start, t["start"]) - clip_start
        e = min(clip_end, t["end"]) - clip_start
        if e - s >= 0.5:
            roberto_spans.append((s, e))

    if not roberto_spans:
        raise RuntimeError("no usable ROBERTO spans to correlate camera side")

    left_score = _motion_score(video_path, left, fps, roberto_spans)
    right_score = _motion_score(video_path, right, fps, roberto_spans)
    if left_score == 0.0 and right_score == 0.0:
        raise RuntimeError("motion correlation found no signal on either side")
    return "left" if left_score >= right_score else "right"


def _camera_schedule(turns: List[Dict], clip_start: float, clip_end: float) -> List[Tuple[float, float, str]]:
    """Collapse diarization turns overlapping [clip_start, clip_end] into a
    list of (rel_start, rel_end, side) spans, times relative to the clip,
    side in {"roberto", "other"}. Gaps between turns and stretches shorter
    than _MIN_HOLD_SECONDS keep whichever camera was already active, to
    avoid flicker on short interjections ('claro', 'sí', laughs)."""
    clip_duration = clip_end - clip_start

    relevant = []
    for t in turns:
        if t["end"] <= clip_start or t["start"] >= clip_end:
            continue
        s = max(clip_start, t["start"]) - clip_start
        e = min(clip_end, t["end"]) - clip_start
        side = "roberto" if t["speaker"] == ROBERTO_LABEL else "other"
        relevant.append([s, e, side])
    relevant.sort(key=lambda span: span[0])

    if not relevant:
        return [(0.0, clip_duration, "roberto")]

    schedule: List[List] = []
    if relevant[0][0] > 0.0:
        schedule.append([0.0, relevant[0][0], relevant[0][2]])
    prev_end = 0.0
    for s, e, side in relevant:
        if s > prev_end:
            fill_side = schedule[-1][2] if schedule else side
            schedule.append([prev_end, s, fill_side])
        schedule.append([s, e, side])
        prev_end = e
    if prev_end < clip_duration:
        schedule.append([prev_end, clip_duration, schedule[-1][2]])

    merged = [schedule[0]]
    for span in schedule[1:]:
        if span[1] - span[0] < _MIN_HOLD_SECONDS:
            merged[-1][1] = span[1]
        else:
            merged.append(span)

    return [(s, e, side) for s, e, side in merged]


def _render_two_camera(in_path: str, out_w: int, out_h: int,
                        segments: List[Tuple[float, float, Camera]], fps: float,
                        fixed_crop_h: float) -> str:
    import cv2

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_ratio = out_w / out_h

    # Same crop size for every segment — only cx/cy (position) changes on a
    # camera switch, never the zoom level, so an A/B switch never looks like
    # a zoom.
    crop_h = min(src_h, fixed_crop_h)
    if crop_h * target_ratio > src_w:
        crop_h = src_w / target_ratio
    crop_windows = [(e, cam.cx, cam.cy, crop_h) for s, e, cam in segments]

    silent_path = in_path + ".twocam.silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (out_w, out_h))

    frame_idx = 0
    seg_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        while seg_idx + 1 < len(crop_windows) and t >= crop_windows[seg_idx][0]:
            seg_idx += 1
        _, cx, cy, crop_h = crop_windows[seg_idx]

        cur_crop_h = max(2, int(crop_h) - (int(crop_h) % 2))
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


def crop_clip_two_camera(source_path: str, start_time: float, end_time: float,
                          aspect_ratio: str, out_path: str) -> str:
    """Cut + reframe one highlight using two fixed camera sectors (Roberto /
    guest) hard-switched by diarization, plus a subtle cosmetic zoom
    oscillation for dynamism. Falls back to the regular active-speaker
    tracker (clipper._reframe_vertical) if it can't set up two cameras —
    e.g. diarization unavailable, or fewer than two consistent faces found."""
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path, intermediate=True)
        try:
            turns = diarize_local(source_path)
            img_w, img_h, fps, duration = _probe_dims_and_fps(cut_path)
            left_cam, right_cam = _detect_two_cameras(cut_path, img_w, img_h, fps, duration)
            midline_x = (left_cam.cx + right_cam.cx) / 2
            roberto_side = _assign_roberto_side(cut_path, left_cam, right_cam, fps, turns, start_time, end_time)
            other_side = "right" if roberto_side == "left" else "left"
            phys_by_role = {"roberto": roberto_side, "other": other_side}
            cam_by_phys = {"left": left_cam, "right": right_cam}
            schedule = _camera_schedule(turns, start_time, end_time)
            # Re-detect each segment's face position instead of trusting one
            # global average for the whole clip — this static shot can still
            # drift/reframe partway through a long highlight.
            segments = [
                (s, e, _camera_for_segment(cut_path, img_w, img_h, fps, s, e, midline_x,
                                            phys_by_role[role], cam_by_phys[phys_by_role[role]]))
                for s, e, role in schedule
            ]
        except Exception as e:
            print(f"[two_camera] falling back to active-speaker tracking: {e}", flush=True)
            _reframe_vertical(cut_path, out_path, aspect_ratio)
            return out_path

        out_w, out_h = _fit_crop_size(img_w, img_h, aspect_ratio)
        # One crop size for the whole clip — derived from the wider of the two
        # detected faces so neither camera looks more zoomed-in than the other.
        fixed_crop_h = max(left_cam.face_h, right_cam.face_h) * _LOOSE_ZOOM_FACTOR
        fixed_crop_h = max(out_h * 0.5, fixed_crop_h)
        silent_path = _render_two_camera(cut_path, out_w, out_h, segments, fps, fixed_crop_h)
        try:
            mux_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", silent_path,
                "-i", cut_path,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "128k",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest",
                out_path,
            ]
            subprocess.run(mux_cmd, check=True)
        finally:
            if os.path.exists(silent_path):
                os.remove(silent_path)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path
