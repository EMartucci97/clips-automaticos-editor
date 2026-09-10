"""Word-by-word / phrase-grouped burned-in subtitles (TikTok/Reels style).

Takes the transcript segments (with word-level timestamps from faster-whisper,
and optionally a per-segment `speaker` tag from diarizer.py) for one
highlight's [start_time, end_time] window, builds an .ass subtitle track,
and burns it onto the rendered clip with ffmpeg's `ass` filter.

Two quality passes run by default (see `use_llm`) before the .ass is built:
  1. `_llm_cleanup_words` re-reads the window's words with an LLM to fix
     likely Whisper mis-hears and convert spoken numbers to digits, without
     paraphrasing words that are already correct.
  2. `_llm_group_words` (only when `words_per_group > 1`) groups the cleaned
     words into caption boxes that read as coherent phrases, instead of a
     fixed word-count stride.
Both fall back to the original/mechanical behaviour on any failure (bad LLM
response, network error, etc.) so a transient hiccup never breaks a render.

Words are colored by speaker when diarization labeled the segment they came
from (see shorts_generator/local/diarizer.py): ROBERTO_LABEL -> white,
OTHER_LABEL -> amber. Unlabeled words default to white. When words are shown
grouped (`words_per_group > 1`), the whole group takes a single color picked
by majority vote across its words, so diarizer noise on one word can't flip
the color mid-phrase.
"""
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from .diarizer import OTHER_LABEL, ROBERTO_LABEL
from .llm import call_local_llm

# The default Homebrew `ffmpeg` bottle ships without libass, so the `ass`
# filter used to burn subtitles doesn't exist there. `ffmpeg-full` (also
# Homebrew, keg-only) includes libass — prefer it here if present.
_FFMPEG_FULL = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"
_FFPROBE_FULL = "/opt/homebrew/opt/ffmpeg-full/bin/ffprobe"
FFMPEG_BIN = _FFMPEG_FULL if os.path.exists(_FFMPEG_FULL) else (shutil.which("ffmpeg") or "ffmpeg")
FFPROBE_BIN = _FFPROBE_FULL if os.path.exists(_FFPROBE_FULL) else (shutil.which("ffprobe") or "ffprobe")

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
    "auto": None,  # keep the existing per-speaker/per-group white/amber split
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

# Grouped/Karaoke-phrase captions (via the LLM grouping pass) are now the
# default everywhere, not just the interactive editor's "Regenerar clip" flow
# -- the batch "Generar clips" and fully-automated pipeline call sites call
# burn_subtitles() without an explicit words_per_group, so this is what
# decides their caption style. Any value > 1 here just enables grouping; the
# actual box sizes come from the LLM (2-4, up to 6 words) with this as the
# mechanical-fallback target, so the exact number is not sensitive.
DEFAULT_WORDS_PER_GROUP = 3

# Default output canvas — matches the vertical 9:16 crops production has
# always used. `burn_subtitles` auto-detects the real resolution of the clip
# it's burning onto (see `_probe_resolution`) and overrides this, so a
# differently-shaped output (e.g. horizontal 16:9) never gets stretched.
DEFAULT_RESOLUTION = (1080, 1920)


def _ass_header(
    font_name: str,
    size: int = 110,
    position: str = DEFAULT_CAPTION_POSITION,
    style: str = DEFAULT_CAPTION_STYLE,
    color: str = DEFAULT_CAPTION_COLOR,
    play_res_x: int = DEFAULT_RESOLUTION[0],
    play_res_y: int = DEFAULT_RESOLUTION[1],
) -> str:
    alignment, margin_v = CAPTION_POSITIONS.get(position, CAPTION_POSITIONS[DEFAULT_CAPTION_POSITION])

    # size/outline/margins below are tuned against a 1920px-tall vertical
    # canvas; scale them proportionally to whatever PlayResY we're actually
    # authoring for, so a clip rendered at a different aspect ratio (e.g.
    # horizontal 1920x1080) keeps the same *relative* on-screen caption size
    # instead of ballooning/shrinking to fill a differently-shaped frame.
    scale = play_res_y / 1920.0
    size = round(size * scale)
    margin_v = round(margin_v * scale)
    margin_lr = round(60 * scale)

    # "bold" swaps the outline for a solid box behind the text; "minimal" keeps
    # the outline but thins it out; karaoke/highlight use today's outline look.
    if style == "bold":
        border_style, outline, back_colour = 3, 4, "&H80000000"
    elif style == "minimal":
        border_style, outline, back_colour = 1, 3, "&H00000000"
    else:
        border_style, outline, back_colour = 1, 8, "&H00000000"
    outline = round(outline * scale) or 1

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
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
            f"Style: {STYLE_FIXED},{font_name},{size},{fixed_color},&H000000FF,&H00000000,{back_colour},0,0,0,0,100,100,0,0,{border_style},{outline},0,{alignment},{margin_lr},{margin_lr},{margin_v},1"
        )
    else:
        lines.append(
            f"Style: {STYLE_ROBERTO},{font_name},{size},{COLOR_WHITE},&H000000FF,&H00000000,{back_colour},0,0,0,0,100,100,0,0,{border_style},{outline},0,{alignment},{margin_lr},{margin_lr},{margin_v},1"
        )
        lines.append(
            f"Style: {STYLE_OTHER},{font_name},{size},{COLOR_AMBER},&H000000FF,&H00000000,{back_colour},0,0,0,0,100,100,0,0,{border_style},{outline},0,{alignment},{margin_lr},{margin_lr},{margin_v},1"
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


def _probe_resolution(video_path: str) -> Tuple[int, int]:
    """Actual pixel size of the (already-cropped) clip we're about to burn
    subtitles onto. Used so the ASS canvas (PlayResX/Y) always matches the
    real frame shape — a mismatch (e.g. authoring for a vertical 1080x1920
    canvas onto a horizontal 1920x1080 frame) is what caused captions to
    render distorted before this was auto-detected."""
    out = subprocess.run(
        [FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video_path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


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
    can color each word/group by who's talking. Also carries each word's own
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


def _strip_punct(text: str) -> str:
    return re.sub(r"[.,]", "", text)


def _extract_json_loose(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rsplit("```", 1)[0]
    return json.loads(text)


CLEANUP_PROMPT = """Sos un corrector de transcripciones en español rioplatense. Te paso una lista numerada de palabras (con su índice original, empezando en 0) tal como las detectó un transcriptor automático (Whisper) en un fragmento de un podcast.

Tu tarea:
1. Corregí SOLO palabras que un reconocimiento de voz automático transcribió mal -- es decir, casos donde la palabra que aparece NO es lo que la persona dijo (típicamente porque suena parecido a otra cosa, o quedó una palabra suelta sin sentido en esa posición). Para reconocer esto necesitás que la palabra actual rompa la coherencia gramatical o semántica de la frase.
2. IMPORTANTE - NO reemplaces una palabra solo porque a vos se te ocurre una alternativa que "suena mejor" o es más natural. Si la palabra que está ya tiene sentido gramatical y semántico en el contexto (aunque no sea la que vos hubieras elegido), DEJALA EXACTAMENTE IGUAL. Ejemplo: si dice "y ahí encontré también algún tiempo" y "encontré" tiene sentido ahí (aunque "estuve" también tendría sentido), NO la cambies -- encontré ya es válida, cambiarla sería inventar, no corregir.
3. NO reescribas ni reformules frases que ya tienen sentido, aunque te parezcan mejorables.
4. Cuando se mencionen números hablados (fechas, horas, cantidades, edades, etc.), convertilos a dígitos cuando tenga sentido natural (ej: "ocho y media" -> "8:30", "quince minutos" -> "15 minutos", "una hora" se mantiene si es una expresión genérica pero "de una a ocho y media" -> "de 1 a 8:30"). Usá tu criterio: si convertir a dígito suena raro o pierde naturalidad, dejalo en letras.
5. Tu respuesta tiene que cubrir TODAS las palabras originales, en orden, sin saltear ninguna y sin superponer rangos. Cada palabra original tiene que caer en exactamente un token de salida.

Respondé ÚNICAMENTE con JSON válido, sin texto adicional, con este formato exacto:
{{"tokens": [{{"orig_start": 0, "orig_end": 2, "text": "¿Qué tal?"}}, ...]}}

Donde "orig_start" es el índice de la primera palabra original cubierta por este token (inclusive) y "orig_end" es el índice siguiente a la última palabra cubierta (exclusive) -- igual que un slice de Python words[orig_start:orig_end]. La mayoría de los tokens van a cubrir 1 sola palabra original (orig_end = orig_start + 1); usá rangos más largos solo cuando fusionás varias palabras en un número (ej: "ocho" + "y" + "media" -> "8:30" sería orig_start=X, orig_end=X+3).

Palabras originales:
{numbered_words}
"""

GROUPING_PROMPT = """Sos un editor de subtítulos para redes sociales (estilo TikTok/Reels) en español. Te paso una lista numerada de tokens (palabras o números ya limpios) de un fragmento de podcast, cada uno con su índice.

Tu tarea es decidir cómo agrupar estos tokens en "cajas" de subtítulo que se muestran juntas en pantalla, de forma que:
- Cada caja, leída sola, tenga sentido como frase (puede estar incompleta, pero tiene que sonar como una unidad coherente, nunca cortar en un lugar raro).
- Preferí cajas de 2 a 4 palabras. Podés hacer cajas más cortas o más largas si el sentido lo pide, pero evitá cajas de más de 6 palabras.
- Respetá pausas naturales del habla (comas, puntos, conectores como "y", "que", "pero" deberían empezar una caja nueva, no terminarla colgada).
- Cubrí TODOS los tokens, en orden, sin saltear ninguno y sin superponer rangos.

Respondé ÚNICAMENTE con JSON válido, sin texto adicional:
{{"groups": [[0, 3], [3, 6], ...]}}

Donde cada par [start, end) es un rango de índices (igual que un slice de Python), start inclusive, end exclusive.

Tokens:
{numbered_tokens}
"""


def _llm_cleanup_words(words: List[Dict]) -> List[Dict]:
    """Re-reads the window's words with an LLM (full-context pass) to fix
    likely Whisper mis-hears and convert spoken numbers to digits, without
    paraphrasing words that are already correct. Falls back to the original
    words unchanged on any failure (bad response, network error, coverage
    mismatch) -- a transient LLM hiccup must never break clip generation."""
    if not words:
        return words
    try:
        numbered = "\n".join(f"{i}: {w['word']}" for i, w in enumerate(words))
        raw = call_local_llm(CLEANUP_PROMPT.format(numbered_words=numbered))
        tokens = _extract_json_loose(raw)["tokens"]

        expect = 0
        for t in tokens:
            if t["orig_start"] != expect:
                raise ValueError(f"coverage gap/overlap at {t} (expected orig_start={expect})")
            expect = t["orig_end"]
        if expect != len(words):
            raise ValueError(f"coverage incomplete: covered up to {expect}, need {len(words)}")

        merged = []
        for t in tokens:
            span = words[t["orig_start"]:t["orig_end"]]
            speakers = [w.get("speaker") for w in span]
            speaker = Counter(speakers).most_common(1)[0][0] if speakers else None
            merged.append({
                "start": span[0]["start"],
                "end": span[-1]["end"],
                "word": t["text"],
                "speaker": speaker,
                "word_start": span[0].get("word_start", span[0]["start"]),
                "word_end": span[-1].get("word_end", span[-1]["end"]),
            })
        return merged
    except Exception as e:
        print(f"[subtitles] LLM cleanup pass failed ({e!r}); using raw transcript words.")
        return words


def _llm_group_words(words: List[Dict], words_per_group: int) -> List[tuple]:
    """Groups (already-cleaned) words into caption boxes that read as
    coherent phrases (2-4, up to 6 words) instead of a fixed word-count
    stride. Falls back to the mechanical `_group_boundaries` heuristic on
    any failure."""
    if not words:
        return []
    try:
        numbered = "\n".join(f"{i}: {w['word']}" for i, w in enumerate(words))
        raw = call_local_llm(GROUPING_PROMPT.format(numbered_tokens=numbered))
        groups = [tuple(g) for g in _extract_json_loose(raw)["groups"]]

        expect = 0
        for g in groups:
            if g[0] != expect:
                raise ValueError(f"coverage gap/overlap at {g} (expected start={expect})")
            expect = g[1]
        if expect != len(words):
            raise ValueError(f"coverage incomplete: covered up to {expect}, need {len(words)}")
        return groups
    except Exception as e:
        print(f"[subtitles] LLM grouping pass failed ({e!r}); using mechanical grouping.")
        return _group_boundaries(words, words_per_group)


_DANGLING_CONNECTORS = {
    "y", "o", "u", "e", "ni", "que", "pero", "porque", "aunque",
    "pues", "sino", "cuando", "mientras", "si",
    "la", "las", "el", "los", "un", "una", "unos", "unas", "lo",
}


def _group_boundaries(words: List[Dict], words_per_group: int) -> List[tuple]:
    """Mechanical fallback: decide caption-group boundaries from natural
    speech pauses instead of a fixed word-count stride: always break at
    sentence-ending punctuation, and at a comma/semicolon/colon once the
    group has reached `words_per_group` words. A hard cap (`words_per_group
    + 2`) forces a break even without punctuation, so a long clause with no
    pause doesn't grow into one giant caption — but if the word landing on
    that cap is a connector ("y", "que", ...), it reads better opening the
    next caption than dangling at the end of this one, so the break moves
    one word earlier instead. Only used when `_llm_group_words` fails."""
    if words_per_group <= 1:
        return [(i, i + 1) for i in range(len(words))]

    hard_max = words_per_group + 2
    groups = []
    start = 0
    for i, w in enumerate(words):
        text = (w.get("word") or "").rstrip()
        size = i - start + 1
        if text.endswith((".", "!", "?", "…")):
            groups.append((start, i + 1))
            start = i + 1
        elif size >= words_per_group and text.endswith((",", ";", ":")):
            groups.append((start, i + 1))
            start = i + 1
        elif size >= hard_max:
            bare = text.strip(".,;:!?…").lower()
            if bare in _DANGLING_CONNECTORS and size > 1:
                groups.append((start, i))
                start = i
            else:
                groups.append((start, i + 1))
                start = i + 1
    if start < len(words):
        groups.append((start, len(words)))
    return groups


def build_ass(
    words: List[Dict],
    out_path: str,
    *,
    font_name: str = FONT_NAME,
    size: int = 110,
    position: str = DEFAULT_CAPTION_POSITION,
    style: str = DEFAULT_CAPTION_STYLE,
    color: str = DEFAULT_CAPTION_COLOR,
    words_per_group: int = DEFAULT_WORDS_PER_GROUP,
    resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
    strip_punctuation: bool = True,
    use_llm: bool = True,
) -> str:
    if use_llm:
        words = _llm_cleanup_words(words)

    play_res_x, play_res_y = resolution
    lines = [_ass_header(
        font_name, size=size, position=position, style=style, color=color,
        play_res_x=play_res_x, play_res_y=play_res_y,
    )]
    fixed_style = color != DEFAULT_CAPTION_COLOR

    def _display_end(i: int):
        """End = start of the very next non-empty word, or None if i is the
        last non-empty word overall (nothing to collide with). A word's
        Dialogue End must NEVER exceed the next word's Start regardless of
        group boundaries or any minimum-duration floor -- otherwise two
        Dialogue events are simultaneously active and libass renders them
        stacked (duplicated-looking captions) or repositioned."""
        for j in range(i + 1, len(words)):
            if words[j]["word"].strip():
                return words[j]["start"]
        return None

    groups: List[tuple] = []
    group_for_index: Dict[int, tuple] = {}
    group_style: Dict[tuple, str] = {}
    if words_per_group > 1:
        groups = _llm_group_words(words, words_per_group) if use_llm else _group_boundaries(words, words_per_group)
        for g_start, g_end in groups:
            for idx in range(g_start, g_end):
                group_for_index[idx] = (g_start, g_end)
        if not fixed_style:
            # Majority-vote speaker per GROUP (not per word) -> one style for
            # the whole group, so diarizer noise on a single word can't flip
            # the color mid-phrase.
            for g_start, g_end in groups:
                speakers = [words[i].get("speaker") for i in range(g_start, g_end) if words[i]["word"].strip()]
                majority = Counter(speakers).most_common(1)[0][0] if speakers else None
                group_style[(g_start, g_end)] = _style_for_speaker(majority)

    for i, w in enumerate(words):
        text = w["word"].replace("\n", " ").strip()
        if strip_punctuation:
            text = _strip_punct(text)
        if not text:
            continue
        start = _ass_timestamp(w["start"])
        next_start = _display_end(i)
        raw_end = next_start if next_start is not None else max(w["end"], w["start"] + 0.05)
        end = _ass_timestamp(max(raw_end, w["start"] + 0.001))

        if words_per_group > 1:
            group_key = group_for_index[i]
            ass_style = STYLE_FIXED if fixed_style else group_style[group_key]

            group_start, group_end = group_key
            before_text = " ".join(
                _strip_punct(c["word"].replace("\n", " ").strip()) if strip_punctuation
                else c["word"].replace("\n", " ").strip()
                for c in words[group_start:i] if c["word"].strip()
            )
            after_text = " ".join(
                _strip_punct(c["word"].replace("\n", " ").strip()) if strip_punctuation
                else c["word"].replace("\n", " ").strip()
                for c in words[i + 1:group_end] if c["word"].strip()
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
            ass_style = STYLE_FIXED if fixed_style else _style_for_speaker(w.get("speaker"))
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
    words_per_group: int = DEFAULT_WORDS_PER_GROUP,
    strip_punctuation: bool = True,
    use_llm: bool = True,
) -> str:
    """Burn subtitles onto video_path, writing out_path.
    If there are no words (e.g. silent clip), just copies the source through."""
    if not words:
        subprocess.run(
            [FFMPEG_BIN, "-y", "-loglevel", "error", "-i", video_path, "-c", "copy", out_path],
            check=True,
        )
        return out_path

    try:
        resolution = _probe_resolution(video_path)
    except Exception as e:
        print(f"[subtitles] resolution probe failed ({e!r}); assuming {DEFAULT_RESOLUTION}.")
        resolution = DEFAULT_RESOLUTION

    ass_path = out_path + ".ass"
    build_ass(
        words, ass_path,
        font_name=font_name, size=size, position=position,
        style=style, color=color, words_per_group=words_per_group,
        resolution=resolution, strip_punctuation=strip_punctuation, use_llm=use_llm,
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
