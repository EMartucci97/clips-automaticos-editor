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

# Per-clip font choice (set via the 'Fuente' control in the standalone editor,
# static/editor.html, and read back by dashboard.py's regenerate panel). Keyed
# by a short id instead of storing the raw ASS font name, so a highlight's
# JSON only ever carries a value from this fixed set and dashboard.py never
# has to trust free-form text going into the ASS header.
FONT_CHOICES = {
    "extrabold": "Montserrat ExtraBold",
    "black": "Montserrat Black",
}
DEFAULT_FONT_KEY = "extrabold"

# ASS colours are &HAABBGGRR (alpha, then BGR — reverse of usual RGB order).
COLOR_WHITE = "&H00FFFFFF"   # RGB FFFFFF
COLOR_AMBER = "&H0020B0FF"   # RGB FFB020, matches the reference screenshots

STYLE_ROBERTO = "Word_Roberto"
STYLE_OTHER = "Word_Other"
STYLE_FIXED = "Word_Fixed"

# Per-clip caption controls, set via the "Subtítulos" panel in the standalone
# editor (static/editor.html) and read back by dashboard.py's regenerate panel.
# Keyed by short ids for the same reason as FONT_CHOICES: a highlight's JSON
# only ever carries a value from these fixed sets.
CAPTION_COLORS = {
    "auto": None,  # keep the existing per-speaker white/amber split
    "white": "&H00FFFFFF",
    "yellow": "&H0000FFFF",
    "cyan": "&H00FFFF00",
    "pink": "&H00B469FF",
}
DEFAULT_CAPTION_COLOR = "auto"

CAPTION_SIZES = {"s": 80, "m": 110, "l": 140, "xl": 170}
DEFAULT_CAPTION_SIZE = "m"

# (Alignment, MarginV) — ASS numpad-style alignment (2=bottom-center,
# 5=middle-center, 8=top-center) plus the vertical margin from that edge.
CAPTION_POSITIONS = {"bottom": (2, 220), "middle": (5, 0), "top": (8, 160)}
DEFAULT_CAPTION_POSITION = "bottom"

CAPTION_STYLES = ("karaoke", "bold", "minimal", "highlight")
DEFAULT_CAPTION_STYLE = "karaoke"


def _ass_header(
    font_name: str,
    size: int = 110,
    position: str = DEFAULT_CAPTION_POSITION,
    style: str = DEFAULT_CAPTION_STYLE,
    color: str = DEFAULT_CAPTION_COLOR,
) -> str:
    alignment, margin_v = CAPTION_POSITIONS.get(position, CAPTION_POSITIONS[DEFAULT_CAPTION_POSITION])
    # "bold" swaps the outline for a solid box behind the text; "minimal" keeps
    # the outline but thins it out; karaoke/highlight use today's outline look.
    if style == "bold":
        border_style, outline, back_colour = 3, 4, "&H80000000"
    elif style == "minimal":
        border_style, outline, back_colour = 1, 3, "&H00000000"
    else:
        border_style, outline, back_colour = 1, 8, "&H00000000"

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.601",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    ]
    fixed_color = CAPTION_COLORS.get(color) if color != DEFAULT_CAPTION_COLOR else None
    if fixed_color:
        lines.append(
            f"Style: {STYLE_FIXED},{font_name},{size},{fixed_color},&H000000FF,&H00000000,{back_colour},0,0,0,0,100,100,0,0,{border_style},{outline},0,{alignment},60,60,{margin_v},1"
        )
    else:
        lines.append(
            f"Style: {STYLE_ROBERTO},{font_name},{size},{COLOR_WHITE},&H000000FF,&H00000000,{back_colour},0,0,0,0,100,100,0,0,{border_style},{outline},0,{alignment},60,60,{margin_v},1"
        )
        lines.append(
            f"Style: {STYLE_OTHER},{font_name},{size},{COLOR_AMBER},&H000000FF,&H00000000,{back_colour},0,0,0,0,100,100,0,0,{border_style},{outline},0,{alignment},60,60,{margin_v},1"
        )
    lines += ["", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text", ""]
    return "\n".join(lines)


def _ass_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _time_key(a: float, b: float) -> str:
    return f"{round(float(a), 3)}:{round(float(b), 3)}"


def words_for_window(
    segments: List[Dict],
    start_time: float,
    end_time: float,
    time_overrides: List[Dict] = None,
) -> List[Dict]:
    """Collect words falling inside [start_time, end_time], shifted to be
    relative to start_time (0 = first frame of the rendered clip). Carries
    over the parent segment's `speaker` tag (ROBERTO_LABEL/OTHER_LABEL/None),
    unless a manual per-word override (`speaker_override`, see
    editor_server._apply_transcript_overrides) says otherwise, so build_ass
    can color each word by who's talking. Also carries each word's own
    original absolute timestamps (`word_start`/`word_end`, unshifted) so
    callers can address it back for a manual correction regardless of which
    window it was fetched through.

    `time_overrides` (optional) is a highlight's `caption_time_overrides` list
    — manual adjustments to when a word appears/disappears, made via the
    editor's caption timing track. Each entry keys off the word's original
    absolute `word_start`/`word_end` (never rewritten, so the key survives
    repeated edits) and carries the new absolute `start`/`end` to burn
    instead. Window membership (`if w_end <= start_time or w_start >=
    end_time`) is still decided from the original timestamps, not the
    override, so a manual nudge can't pull a word into/out of a window it
    wasn't fetched for."""
    overrides_by_key = {
        _time_key(o["word_start"], o["word_end"]): o for o in (time_overrides or [])
    }
    words: List[Dict] = []
    for seg in segments:
        speaker = seg.get("speaker")
        for w in seg.get("words", []) or []:
            w_start, w_end = float(w["start"]), float(w["end"])
            if w_end <= start_time or w_start >= end_time:
                continue
            override = overrides_by_key.get(_time_key(w_start, w_end))
            eff_start = override["start"] if override else w_start
            eff_end = override["end"] if override else w_end
            words.append({
                "start": max(0.0, eff_start - start_time),
                "end": max(0.0, min(eff_end, end_time) - start_time),
                "word": w["word"],
                "speaker": w.get("speaker_override", speaker),
                "word_start": w_start,
                "word_end": w_end,
            })
    return words


def _style_for_speaker(speaker: str) -> str:
    return STYLE_OTHER if speaker == OTHER_LABEL else STYLE_ROBERTO


def build_ass(
    words: List[Dict],
    out_path: str,
    *,
    font_name: str = FONT_NAME,
    size: int = 110,
    position: str = DEFAULT_CAPTION_POSITION,
    style: str = DEFAULT_CAPTION_STYLE,
    color: str = DEFAULT_CAPTION_COLOR,
    words_per_group: int = 1,
) -> str:
    lines = [_ass_header(font_name, size=size, position=position, style=style, color=color)]
    fixed_style = color != DEFAULT_CAPTION_COLOR
    for i, w in enumerate(words):
        text = w["word"].replace("\n", " ").strip()
        if not text:
            continue
        start = _ass_timestamp(w["start"])
        end = _ass_timestamp(max(w["end"], w["start"] + 0.05))
        ass_style = STYLE_FIXED if fixed_style else _style_for_speaker(w.get("speaker"))

        if words_per_group > 1:
            # Show the whole group of words together (not built up one at a time):
            # words before/after the active one in its group are dimmed, and the
            # active one gets painted as time passes over it.
            group_start = (i // words_per_group) * words_per_group
            before_text = " ".join(
                c["word"].replace("\n", " ").strip()
                for c in words[group_start:i] if c["word"].strip()
            )
            after_text = " ".join(
                c["word"].replace("\n", " ").strip()
                for c in words[i + 1:group_start + words_per_group] if c["word"].strip()
            )
            # Each segment resets its own alpha/scale explicitly (rather than
            # relying on tag order) so the active word's highlight scale never
            # leaks into the dimmed segment that follows it.
            dim_tag = "{\\alpha&H80&\\fscx100\\fscy100}"
            active_tag = "{\\alpha&H00&\\fscx115\\fscy115}" if style == "highlight" else "{\\alpha&H00&}"
            parts = []
            if before_text:
                parts.append(dim_tag + before_text)
            parts.append(active_tag + text)
            if after_text:
                parts.append(dim_tag + after_text)
            display = " ".join(parts)
        else:
            display = "{\\fscx115\\fscy115}" + text if style == "highlight" else text

        lines.append(f"Dialogue: 0,{start},{end},{ass_style},,0,0,0,,{display}\n")
    Path(out_path).write_text("".join(lines), encoding="utf-8")
    return out_path


def burn_subtitles(
    video_path: str,
    words: List[Dict],
    out_path: str,
    *,
    font_name: str = FONT_NAME,
    size: int = 110,
    position: str = DEFAULT_CAPTION_POSITION,
    style: str = DEFAULT_CAPTION_STYLE,
    color: str = DEFAULT_CAPTION_COLOR,
    words_per_group: int = 1,
) -> str:
    """Burn word-by-word subtitles onto video_path, writing out_path.
    If there are no words (e.g. silent clip), just copies the source through."""
    if not words:
        subprocess.run(
            [FFMPEG_BIN, "-y", "-loglevel", "error", "-i", video_path, "-c", "copy", out_path],
            check=True,
        )
        return out_path

    ass_path = out_path + ".ass"
    build_ass(
        words, ass_path,
        font_name=font_name, size=size, position=position,
        style=style, color=color, words_per_group=words_per_group,
    )
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
