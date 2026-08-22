"""Active-speaker detection for vertical reframing.

For multi-speaker footage (interviews, podcasts), picks which detected face
to keep centered in the vertical crop by correlating each face's mouth
motion with the audio's voice-activity envelope, instead of just tracking
whichever face happens to be biggest in frame.

Also detects hard scene cuts (camera switches, e.g. solo shot <-> 2-shot
<-> 4-way grid), since these sources are edited with real camera cuts, not
a single static angle — carrying track identity or crop-position smoothing
across a cut produces a crop that chases the wrong region for a while.

The heuristic (mouth motion + audio) isn't perfect on heavily-edited
multi-camera footage. `list_shots_for_review()` / `render_shot_image()`
support a manual-override review step: the clip is split into shots (by
cut), one representative frame per shot is rendered with each detected
face's track id boxed, and a human can say "shot 2 -> track 5" instead of
trusting the heuristic for that shot. `compute_active_speaker_centers()`
accepts those overrides and uses them verbatim, skipping the heuristic
entirely for overridden shots.
"""
import subprocess
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

# mediapipe FaceMesh landmark indices for the inner lip (mouth open/close).
_MOUTH_TOP = 13
_MOUTH_BOTTOM = 14
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291


class SpeakerTarget(NamedTuple):
    cx: float
    cy: float
    face_h: float  # bounding-box height in px — lets the crop zoom to the face
    is_cut: bool  # True if this frame is a hard scene cut vs. the previous one


def _mouth_aspect_ratio(landmarks, img_w: int, img_h: int) -> float:
    def pt(i):
        lm = landmarks[i]
        return np.array([lm.x * img_w, lm.y * img_h])
    vertical = np.linalg.norm(pt(_MOUTH_TOP) - pt(_MOUTH_BOTTOM))
    horizontal = np.linalg.norm(pt(_MOUTH_LEFT) - pt(_MOUTH_RIGHT))
    return float(vertical / horizontal) if horizontal > 1e-6 else 0.0


def _face_bbox(landmarks, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Returns (centroid_x, centroid_y, bbox_width, bbox_height) in px."""
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return cx, cy, w, h


def _read_audio_rms(video_path: str, frame_count: int, fps: float) -> np.ndarray:
    """Decode audio to mono PCM and return one RMS value per video frame."""
    sr = 16000
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    rms = np.zeros(frame_count, dtype=np.float32)
    if pcm.size == 0 or fps <= 0:
        return rms
    samples_per_frame = sr / fps
    for i in range(frame_count):
        start = int(i * samples_per_frame)
        end = int((i + 1) * samples_per_frame)
        window = pcm[start:end]
        if window.size:
            rms[i] = float(np.sqrt(np.mean(window ** 2)))
    return rms


class _Track:
    __slots__ = ("centroid", "mar_history")

    def __init__(self, centroid: Tuple[float, float]):
        self.centroid = centroid
        self.mar_history: List[float] = []


def _detect_cuts(gray_frames: List[np.ndarray]) -> List[bool]:
    """Flag frames whose content differs sharply from the previous frame —
    a real camera switch (solo shot -> 2-shot -> 4-way grid, etc.), not
    just motion within a continuous shot."""
    n = len(gray_frames)
    cuts = [False] * n
    for i in range(1, n):
        diff = np.mean(np.abs(gray_frames[i].astype(np.int16) - gray_frames[i - 1].astype(np.int16)))
        if diff > 28.0:  # empirical threshold on 0-255 mean abs diff, downsampled frame
            cuts[i] = True
    return cuts


def _shots_from_cuts(cuts: List[bool]) -> List[Tuple[int, int]]:
    """Group frame indices into (start, end) shot ranges, split at cuts."""
    n = len(cuts)
    if n == 0:
        return []
    shots = []
    start = 0
    for i in range(1, n):
        if cuts[i]:
            shots.append((start, i - 1))
            start = i
    shots.append((start, n - 1))
    return shots


class _Analysis(NamedTuple):
    n: int
    fps: float
    img_w: int
    img_h: int
    cuts: List[bool]
    shots: List[Tuple[int, int]]
    frame_to_shot: List[int]
    all_track_ids: List[int]
    centroid_by_track: Dict[int, List[Optional[Tuple[float, float]]]]
    width_by_track: Dict[int, List[Optional[float]]]
    height_by_track: Dict[int, List[Optional[float]]]
    mar_by_track: Dict[int, List[Optional[float]]]
    rms: np.ndarray
    voice_threshold: float


def _detect_faces(face_mesh, rgb: np.ndarray, img_w: int, img_h: int) -> List[Dict]:
    result = face_mesh.process(rgb)
    faces = []
    for lm in (result.multi_face_landmarks or []):
        cx, cy, fw, fh = _face_bbox(lm.landmark, img_w, img_h)
        mar = _mouth_aspect_ratio(lm.landmark, img_w, img_h)
        faces.append({"centroid": (cx, cy), "face_w": fw, "face_h": fh, "mar": mar})
    return faces


def _detect_faces_via_quadrants(quad_mesh, frame: np.ndarray, img_w: int, img_h: int) -> List[Dict]:
    """Fallback for 2x2 grid composite shots (all 4 speakers in one frame,
    each in their own panel): mediapipe's face detector reliably finds 0
    faces when run on the full composited frame (confirmed empirically —
    not a resolution issue, upscaling the whole frame doesn't help either),
    but finds each face fine when each panel is cropped out and processed
    in isolation. Only called when whole-frame detection finds nothing."""
    import cv2

    half_w, half_h = img_w // 2, img_h // 2
    if half_w < 20 or half_h < 20:
        return []
    faces = []
    for x0, y0 in [(0, 0), (half_w, 0), (0, half_h), (half_w, half_h)]:
        quad = frame[y0:y0 + half_h, x0:x0 + half_w]
        quad_up = cv2.resize(quad, (half_w * 2, half_h * 2))
        rgb = cv2.cvtColor(quad_up, cv2.COLOR_BGR2RGB)
        for f in _detect_faces(quad_mesh, rgb, half_w * 2, half_h * 2):
            qcx, qcy = f["centroid"]
            faces.append({
                "centroid": (x0 + qcx / 2, y0 + qcy / 2),
                "face_w": f["face_w"] / 2,
                "face_h": f["face_h"] / 2,
                "mar": f["mar"],
            })
    return faces


def _analyze(video_path: str) -> _Analysis:
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    # Separate, stateless instance for the quadrant-split fallback — a
    # temporally-tracking FaceMesh fed alternating full-frame/quadrant crops
    # would get confused, since it assumes a continuous single stream.
    quad_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        min_detection_confidence=0.3,
    )

    per_frame_faces: List[List[Dict]] = []
    gray_small: List[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = _detect_faces(face_mesh, rgb, img_w, img_h)
            if not faces:
                faces = _detect_faces_via_quadrants(quad_mesh, frame, img_w, img_h)
            per_frame_faces.append(faces)
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
            gray_small.append(small)
    finally:
        cap.release()
        face_mesh.close()
        quad_mesh.close()

    n = len(per_frame_faces)
    cuts = _detect_cuts(gray_small)
    shots = _shots_from_cuts(cuts)
    frame_to_shot = [0] * n
    for idx, (s, e) in enumerate(shots):
        for f in range(s, e + 1):
            frame_to_shot[f] = idx

    rms = _read_audio_rms(video_path, n, fps)
    voice_threshold = float(np.percentile(rms, 40)) if rms.size else 0.0

    # --- track faces across frames (nearest-centroid matching) ---
    # A hard cut invalidates spatial continuity entirely (the whole frame
    # layout may have changed), so every track is dropped and re-seeded
    # fresh on the frame right after a cut. Track ids keep incrementing
    # across cuts (never reused), so a track id is a stable handle for a
    # single shot's face even though tracks don't persist across shots.
    tracks: Dict[int, _Track] = {}
    next_id = 0
    match_dist = max(img_w, img_h) * 0.12
    frame_track_ids: List[List[int]] = []
    for frame_i, faces in enumerate(per_frame_faces):
        if cuts[frame_i]:
            tracks = {}
        assigned = []
        used = set()
        for face in faces:
            best_id, best_dist = None, match_dist
            for tid, tr in tracks.items():
                if tid in used:
                    continue
                d = ((tr.centroid[0] - face["centroid"][0]) ** 2 + (tr.centroid[1] - face["centroid"][1]) ** 2) ** 0.5
                if d < best_dist:
                    best_id, best_dist = tid, d
            if best_id is None:
                best_id = next_id
                tracks[best_id] = _Track(face["centroid"])
                next_id += 1
            tr = tracks[best_id]
            tr.centroid = face["centroid"]
            tr.mar_history.append(face["mar"])
            if len(tr.mar_history) > 15:
                tr.mar_history.pop(0)
            used.add(best_id)
            assigned.append(best_id)
        frame_track_ids.append(assigned)

    all_track_ids = sorted({tid for ids in frame_track_ids for tid in ids})
    mar_by_track: Dict[int, List[Optional[float]]] = {tid: [None] * n for tid in all_track_ids}
    centroid_by_track: Dict[int, List[Optional[Tuple[float, float]]]] = {tid: [None] * n for tid in all_track_ids}
    width_by_track: Dict[int, List[Optional[float]]] = {tid: [None] * n for tid in all_track_ids}
    height_by_track: Dict[int, List[Optional[float]]] = {tid: [None] * n for tid in all_track_ids}
    for i, (faces, ids) in enumerate(zip(per_frame_faces, frame_track_ids)):
        for face, tid in zip(faces, ids):
            mar_by_track[tid][i] = face["mar"]
            centroid_by_track[tid][i] = face["centroid"]
            width_by_track[tid][i] = face["face_w"]
            height_by_track[tid][i] = face["face_h"]

    return _Analysis(
        n=n, fps=fps, img_w=img_w, img_h=img_h,
        cuts=cuts, shots=shots, frame_to_shot=frame_to_shot,
        all_track_ids=all_track_ids,
        centroid_by_track=centroid_by_track,
        width_by_track=width_by_track,
        height_by_track=height_by_track,
        mar_by_track=mar_by_track,
        rms=rms, voice_threshold=voice_threshold,
    )


def compute_active_speaker_centers(
    video_path: str,
    shot_overrides: Optional[Dict[int, int]] = None,
) -> List[Optional[SpeakerTarget]]:
    """One SpeakerTarget per frame of video_path — the face most likely
    talking at that moment, with a short-memory bias against flicker within
    a shot, but an instant reaction to real camera cuts. Returns None for
    frames where no face has ever been seen yet.

    `shot_overrides`: optional {shot_index: track_id} map (from
    `list_shots_for_review`) forcing which face is "the speaker" for
    specific shots, bypassing the mouth-motion/audio heuristic there.
    """
    a = _analyze(video_path)
    n = a.n

    win = max(3, int(round(a.fps * 0.4)))  # ~0.4s mouth-motion window

    def motion_score(tid: int, i: int) -> float:
        lo = max(0, i - win)
        vals = [v for v in a.mar_by_track[tid][lo:i + 1] if v is not None]
        return float(np.std(vals)) if len(vals) >= 2 else 0.0

    switch_hold_frames = int(round(a.fps * 0.6))
    centers: List[Optional[SpeakerTarget]] = [None] * n
    current_speaker: Optional[int] = None
    hold = 0
    for i in range(n):
        shot_idx = a.frame_to_shot[i]
        override_track = shot_overrides.get(shot_idx) if shot_overrides else None
        if override_track is not None:
            c = a.centroid_by_track.get(override_track, [None] * n)[i]
            if c is not None:
                current_speaker = override_track
                cx, cy = c
                fh = a.height_by_track[override_track][i] or 0.0
                centers[i] = SpeakerTarget(cx, cy, fh, a.cuts[i])
            else:
                centers[i] = centers[i - 1] if i > 0 else None
            continue

        present = [tid for tid in a.all_track_ids if a.centroid_by_track[tid][i] is not None]
        if not present:
            centers[i] = centers[i - 1] if i > 0 else None
            continue
        voiced = a.rms[i] > a.voice_threshold if i < len(a.rms) else True
        if voiced:
            candidate = max(present, key=lambda t: motion_score(t, i))
        else:
            candidate = current_speaker if current_speaker in present else present[0]

        # A cut resets track identity above, so current_speaker can't
        # possibly be in `present` — switch immediately, no hold delay.
        if a.cuts[i] or current_speaker not in present:
            current_speaker = candidate
            hold = 0
        elif candidate != current_speaker:
            hold += 1
            if hold >= switch_hold_frames:
                current_speaker = candidate
                hold = 0
        else:
            hold = 0

        cx, cy = a.centroid_by_track[current_speaker][i]
        fh = a.height_by_track[current_speaker][i] or 0.0
        centers[i] = SpeakerTarget(cx, cy, fh, a.cuts[i])

    return centers


def list_shots_for_review(video_path: str) -> List[Dict]:
    """Split video_path into camera shots and, for each, return the faces
    detected on a representative frame (track id + position/size), so a
    human can pick which track is the speaker per shot."""
    a = _analyze(video_path)
    shots_out = []
    for idx, (s, e) in enumerate(a.shots):
        mid = (s + e) // 2
        candidates = list(range(mid, e + 1)) + list(range(mid - 1, s - 1, -1))
        chosen_frame = None
        faces = []
        for f in candidates:
            present = [tid for tid in a.all_track_ids if a.centroid_by_track[tid][f] is not None]
            if present:
                chosen_frame = f
                for tid in present:
                    cx, cy = a.centroid_by_track[tid][f]
                    faces.append({
                        "track_id": tid,
                        "cx": cx, "cy": cy,
                        "face_w": a.width_by_track[tid][f] or 0.0,
                        "face_h": a.height_by_track[tid][f] or 0.0,
                    })
                break
        shots_out.append({
            "shot_index": idx,
            "start_time": s / a.fps,
            "end_time": e / a.fps,
            "frame_index": chosen_frame,
            "faces": faces,
        })
    return shots_out


def render_shot_image(video_path: str, frame_index: int, faces: List[Dict], out_path: str) -> str:
    """Save the given frame with each face boxed and labeled by track id."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"could not read frame {frame_index} from {video_path}")

    for f in faces:
        cx, cy, w, h = f["cx"], f["cy"], f["face_w"], f["face_h"]
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        x1, y1 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 3)
        label = str(f["track_id"])
        ty = max(30, y0 - 10)
        cv2.putText(frame, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6)
        cv2.putText(frame, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

    cv2.imwrite(out_path, frame)
    return out_path
