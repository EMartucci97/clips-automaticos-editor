"""Word-by-word burned-in subtitles (TikTok/Reels style).

Takes the transcript segments (with word-level timestamps from faster-whisper,
and optionally a per-segment `speaker` tag from diarizer.py) for one
highlight's [start_time, end_time] window, builds an .ass subtitle track
showing one word at a time, and burns it onto the rendered clip with
ffmpeg's `ass` filter.

Words are colored by speaker when diarization labeled the segment they came
from (see shorts_generator/local/diarizer.py): ROBERTO_LABEL -> white,
OTHER_LABEL -> amber. Unlabeled words default to white.
"""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from .diarizer import OTHER_LABEL, ROBERTO_LABEL

# The default Homebrew `ffmpeg` bottle ships without libass, so the `ass`
# filter used to burn subtitles doesn't exist there. `ffmpeg-full` (also
# Homebrew, keg-only) includes libass — prefer it here if present.
_FFMPEG_FULL = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"
FFMPEG_BIN = _FFMPEG_FULL if os.path.exists(_FFMPEG_FULL) else (shutil.which("ffmpeg") or "ffmpeg")

# Bundled instead of relying on a system install, so the font is always
# available to the `ass` filter (passed via fontsdir=) regardless of what's
# installed on the machine running the pipeline.
FONTS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"
FONT_NAME = "Montserrat ExtraBold"

# ASS colours are &HAABBGGRR (alpha, then BGR — reverse of usual RGB order).
COLOR_WHITE = "&H00FFFFFF"   # RGB FFFFFF
COLOR_AMBER = "&H0020B0FF"   # RGB FFB020, matches the reference screenshots

STYLE_ROBERTO = "Word_Roberto"
STYLE_OTHER = "Word_Other"

ASS_HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: {STYLE_ROBERTO},{FONT_NAME},110,{COLOR_WHITE},&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,8,0,2,60,60,220,1
Style: {STYLE_OTHER},{FONT_NAME},110,{COLOR_AMBER},&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,8,0,2,60,60,220,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def words_for_window(segments: List[Dict], start_time: float, end_time: float) -> List[Dict]:
    """Collect words falling inside [start_time, end_time], shifted to be
    relative to start_time (0 = first frame of the rendered clip). Carries
    over the parent segment's `speaker` tag (ROBERTO_LABEL/OTHER_LABEL/None)
    so build_ass can color each word by who's talking."""
    words: List[Dict] = []
    for seg in segments:
        speaker = seg.get("speaker")
        for w in seg.get("words", []) or []:
            w_start, w_end = float(w["start"]), float(w["end"])
            if w_end <= start_time or w_start >= end_time:
                continue
            words.append({
                "start": max(0.0, w_start - start_time),
                "end": max(0.0, min(w_end, end_time) - start_time),
                "word": w["word"],
                "speaker": speaker,
            })
    return words


def _style_for_speaker(speaker: str) -> str:
    return STYLE_OTHER if speaker == OTHER_LABEL else STYLE_ROBERTO


def build_ass(words: List[Dict], out_path: str) -> str:
    lines = [ASS_HEADER]
    for w in words:
        text = w["word"].replace("\n", " ").strip()
        if not text:
            continue
        start = _ass_timestamp(w["start"])
        end = _ass_timestamp(max(w["end"], w["start"] + 0.05))
        style = _style_for_speaker(w.get("speaker"))
        lines.append(f"Dialogue: 0,{start},{end},{style},,0,0,0,,{text}\n")
    Path(out_path).write_text("".join(lines), encoding="utf-8")
    return out_path


def burn_subtitles(video_path: str, words: List[Dict], out_path: str) -> str:
    """Burn word-by-word subtitles onto video_path, writing out_path.
    If there are no words (e.g. silent clip), just copies the source through."""
    if not words:
        subprocess.run(
            [FFMPEG_BIN, "-y", "-loglevel", "error", "-i", video_path, "-c", "copy", out_path],
            check=True,
        )
        return out_path

    ass_path = out_path + ".ass"
    build_ass(words, ass_path)
    try:
        # ffmpeg filter paths need escaping on all platforms; simplest is to
        # run with cwd set to the .ass file's directory and use a bare filename.
        ass_dir = os.path.dirname(os.path.abspath(ass_path)) or "."
        ass_name = os.path.basename(ass_path)
        cmd = [
            FFMPEG_BIN, "-y", "-loglevel", "error",
            "-i", os.path.abspath(video_path),
            "-vf", f"ass={ass_name}:fontsdir={FONTS_DIR}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "copy",
            os.path.abspath(out_path),
        ]
        subprocess.run(cmd, check=True, cwd=ass_dir)
    finally:
        if os.path.exists(ass_path):
            os.remove(ass_path)
    return out_path
