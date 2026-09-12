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


def _cut_subclip_multi(
    source_path: str,
    start: float,
    end: float,
    cut_segments: List[Tuple[float, float]],
    out_path: str,
) -> str:
    """Like `_cut_subclip`, but also removes each (start, end) clip-relative
    span in `cut_segments` (silence gaps / filler words -- see editing.py) in
    one ffmpeg filter_complex trim+concat pass, instead of a plain -ss/-to
    cut. Falls back to `_cut_subclip` when there's nothing left to cut after
    merging, so unaffected callers see identical behaviour to before."""
    duration = end - start
    keep = keep_segments_from_cuts(cut_segments, duration)
    if not keep or (len(keep) == 1 and keep[0][0] <= 1e-6 and keep[0][1] >= duration - 1e-6):
        return _cut_subclip(source_path, start, end, out_path)

    filter_parts = []
    concat_inputs = []
    for i, (s, e) in enumerate(keep):
        abs_s, abs_e = start + s, start + e
        filter_parts.append(
            f"[0:v]trim=start={abs_s:.3f}:end={abs_e:.3f},setpts=PTS-STARTPTS[v{i}];"
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
        "-i", source_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
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


def mix_background_music(in_path: str, music_path: str, volume: float, out_path: str,
                          fade_seconds: float = 1.0) -> str:
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
    filter_complex = (
        f"[1:a]atrim=0:{duration},afade=t=in:st=0:d={fade_seconds},"
        f"volume={volume}[music];"
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
    """(width, height, fps) of a media file's first video stream, via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "csv=p=0",
            path,
        ],
        capture_output=True, text=True, check=True,
    )
    w_str, h_str, fps_str = result.stdout.strip().split(",")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return int(w_str), int(h_str), fps


def _render_cta_text_lines(width: int, height: int, scale: int, font_path: str):
    """Renders the two CTA text lines ("Comenta 'EDUARDO'" / "y te mando la
    entrevista completa") onto tightly-cropped RGBA canvases. Font size 1
    starts proportional to height, then shrinks if needed so line 1 fits
    within the frame width -- on a narrow vertical crop (e.g. 606x1080 from
    a 16:9 source) a height-only size otherwise overflows and clips the
    text at both edges. Returns (line1_canvas, line2_canvas, size1)."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    canvas_w = width * scale
    max_text_w = int(canvas_w * 0.94)

    size1 = int(height * scale * 0.075)
    size2 = int(height * scale * 0.032)

    WHITE, GOLD, BLACK = (255, 255, 255, 255), (245, 185, 33, 255), (0, 0, 0, 255)
    part_a, part_b = "Comenta ", "\"EDUARDO\""
    line2 = "y te mando la entrevista completa"

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
    w2 = _line2_width(size2)
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
    d1.text((x0 + w_a, origin_y), part_b, font=font1, fill=GOLD, stroke_width=stroke1,
             stroke_fill=BLACK, anchor="lm")
    line1_canvas = line1_canvas.crop(line1_canvas.getbbox())

    line2_canvas = PILImage.new("RGBA", (canvas_w, scratch_h), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(line2_canvas)
    d2.text((line2_canvas.width / 2, scratch_h // 2), line2, font=font2, fill=WHITE,
             stroke_width=stroke2, stroke_fill=BLACK, anchor="mm")
    line2_canvas = line2_canvas.crop(line2_canvas.getbbox())

    return line1_canvas, line2_canvas, size1


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
    line1_canvas, line2_canvas, size1 = _render_cta_text_lines(width, height, scale, font_path)

    gap = int(size1 * 0.28)
    total_h = line1_canvas.height + gap + line2_canvas.height
    canvas = PILImage.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
    top = canvas.height // 2 - total_h // 2
    canvas.alpha_composite(line1_canvas, ((canvas.width - line1_canvas.width) // 2, top))
    canvas.alpha_composite(
        line2_canvas,
        ((canvas.width - line2_canvas.width) // 2, top + line1_canvas.height + gap),
    )

    canvas = canvas.resize((width, height), PILImage.LANCZOS)
    canvas.save(out_path)
    return out_path


def _render_cta_overlay_png(out_path: str, width: int, height: int) -> str:
    """Like `_render_cta_text_png`, but also stacks the show's "portada"
    graphic (photo collage + "LA OTRA CARA DEL BILLONARIO" headline, cropped
    from the original end-card image) above the CTA text, so the outro shows
    both -- not just the bare text -- on top of the blurred running video.
    The portada is scaled to fit width/height budgets so this still works on
    both landscape and portrait frames."""
    from PIL import Image as PILImage

    from .subtitles import FONTS_DIR

    scale = 3
    font_path = str(FONTS_DIR / "Montserrat-Black.ttf")
    line1_canvas, line2_canvas, size1 = _render_cta_text_lines(width, height, scale, font_path)

    text_gap = int(size1 * 0.28)
    text_block_h = line1_canvas.height + text_gap + line2_canvas.height
    text_block_w = max(line1_canvas.width, line2_canvas.width)

    portada_path = FONTS_DIR.parent / "cta_portada.png"
    portada = PILImage.open(portada_path).convert("RGBA")
    canvas_w, canvas_h = width * scale, height * scale

    # Budget the whole stack (portada + gap + text) to a fraction of the
    # frame so it fits comfortably on both landscape and portrait clips.
    portada_gap = int(size1 * 0.4)
    max_block_h = int(canvas_h * 0.82)
    max_portada_w = int(canvas_w * 0.62)
    portada_w = max_portada_w
    portada_h = int(portada_w * portada.height / portada.width)
    available_for_portada = max_block_h - portada_gap - text_block_h
    if portada_h > available_for_portada:
        portada_h = max(int(available_for_portada), 1)
        portada_w = int(portada_h * portada.width / portada.height)
    portada_resized = portada.resize((portada_w, portada_h), PILImage.LANCZOS)

    total_w = max(text_block_w, portada_w)
    total_h = portada_h + portada_gap + text_block_h
    canvas = PILImage.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    top = canvas_h // 2 - total_h // 2
    canvas.alpha_composite(portada_resized, ((canvas_w - portada_w) // 2, top))
    text_top = top + portada_h + portada_gap
    canvas.alpha_composite(line1_canvas, ((canvas_w - line1_canvas.width) // 2, text_top))
    canvas.alpha_composite(
        line2_canvas,
        ((canvas_w - line2_canvas.width) // 2, text_top + line1_canvas.height + text_gap),
    )

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
    the cut is already visually continuous."""
    w, h, fps = _ffprobe_video_info(in_path)

    raw_outro = out_path + ".rawoutro.mp4"
    cta_png = out_path + ".ctatext.png"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{cta_start_time:.3f}", "-i", source_path,
                "-t", f"{duration}",
                "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                raw_outro,
            ],
            check=True,
        )
        _render_cta_overlay_png(cta_png, w, h)

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
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            out_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        for tmp in (raw_outro, cta_png):
            if os.path.exists(tmp):
                os.remove(tmp)
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
            _cut_subclip_multi(source_path, start_time, end_time, cut_segments, out_path)
        else:
            _cut_subclip(source_path, start_time, end_time, out_path)
        return out_path

    cut_path = out_path + ".cut.mp4"
    try:
        if cut_segments:
            _cut_subclip_multi(source_path, start_time, end_time, cut_segments, cut_path)
        else:
            _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(cut_path, out_path, aspect_ratio, shot_overrides=shot_overrides)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def _ffprobe_resolution(path: str) -> Tuple[int, int]:
    """(width, height) of a video's first video stream, via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            path,
        ],
        capture_output=True, text=True, check=True,
    )
    w, h = result.stdout.strip().split("x")[:2]
    return int(w), int(h)


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
            _cut_subclip_multi(source_path, start_time, end_time, cut_segments, cut_path)
        else:
            _cut_subclip(source_path, start_time, end_time, cut_path)

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
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
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
