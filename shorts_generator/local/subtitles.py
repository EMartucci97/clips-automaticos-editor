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
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import ImageFont

from .diarizer import OTHER_LABEL, ROBERTO_LABEL
from .llm import call_local_llm
from .encoding import video_encode_args

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

# Real font files backing FONT_CHOICES below, used to measure actual glyph
# widths with PIL so the "bold" style's hand-drawn box (see _wrap_group_words
# / _rounded_rect_path) matches what libass will actually render -- instead
# of guessing at an average character width.
FONT_FILES = {
    "Montserrat ExtraBold": FONTS_DIR / "Montserrat-ExtraBold.ttf",
    "Montserrat Black": FONTS_DIR / "Montserrat-Black.ttf",
}

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
# Dummy style backing the hand-drawn "bold" box shapes (see _rounded_rect_path
# / _group_box_dialogue) -- its Fontname/size are irrelevant since a \p1
# drawing event never renders text, but every Dialogue line needs a Style
# name that exists in [V4+ Styles].
STYLE_BOX = "Word_Box"

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

# Subidos ~24% (65/90/115/140 -> 80/112/142/172) junto con el recorte del
# padding de la caja del estilo "bold": el texto crece pero la caja negra
# queda del mismo alto y casi del mismo ancho que antes. Un numero suelto
# guardado en un highlight viejo no se toca, solo estas cuatro etiquetas.
CAPTION_SIZES = {"s": 80, "m": 112, "l": 142, "xl": 172}
DEFAULT_CAPTION_SIZE = "m"

# The editor's size control is a plain number of script units at the canonical
# PlayResY=1920 canvas (it used to be the four s/m/l/xl chips above, which are
# still accepted so old highlights/presets keep rendering identically). The
# bounds are what still looks like a caption: below ~30 it's unreadable on a
# phone, above ~220 a 3-word group no longer fits one line at 1080 wide.
CAPTION_SIZE_RANGE = (30, 220)

# (Alignment, MarginV) — ASS numpad-style alignment (2=bottom-center,
# 5=middle-center, 8=top-center) plus the vertical margin from that edge.
CAPTION_POSITIONS = {"bottom": (2, 220), "middle": (5, 0), "top": (8, 160)}
DEFAULT_CAPTION_POSITION = "bottom"

_HEX_COLOR_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def resolve_caption_size(value) -> int:
    """Caption font size in script units, from either a legacy s/m/l/xl key or
    a raw number (what the editor's numeric stepper now sends). Anything
    unparseable falls back to the default rather than raising -- this runs deep
    inside an export, where a bad stored value must not abort the render;
    editor_server validates the value up front, at PATCH time."""
    if value is None:
        return CAPTION_SIZES[DEFAULT_CAPTION_SIZE]
    if isinstance(value, bool):
        return CAPTION_SIZES[DEFAULT_CAPTION_SIZE]
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        key = str(value).strip()
        if key in CAPTION_SIZES:
            return CAPTION_SIZES[key]
        try:
            number = float(key)
        except ValueError:
            return CAPTION_SIZES[DEFAULT_CAPTION_SIZE]
    lo, hi = CAPTION_SIZE_RANGE
    return int(round(max(lo, min(hi, number))))


def resolve_caption_color(value) -> Optional[str]:
    """ASS &HAABBGGRR colour for a stored caption_color, or None for "auto"
    (which means "keep the per-speaker white/amber split"). Accepts the named
    keys in CAPTION_COLORS and any "#RRGGBB" hex the editor's colour picker
    produces. ASS is BGR, not RGB -- swapping the byte order here is the whole
    reason this lives in one function."""
    if not value:
        return None
    key = str(value).strip()
    if key == DEFAULT_CAPTION_COLOR:
        return None
    if key in CAPTION_COLORS:
        return CAPTION_COLORS[key]
    match = _HEX_COLOR_RE.match(key)
    if not match:
        return None
    rr, gg, bb = match.group(1)[0:2], match.group(1)[2:4], match.group(1)[4:6]
    return f"&H00{bb}{gg}{rr}".upper()


def resolve_caption_pos(pos_x, pos_y) -> Optional[Tuple[float, float]]:
    """Free caption position as (x, y) fractions of the frame, or None to fall
    back to the bottom/middle/top preset. Both coordinates have to be present
    and numeric for the free position to apply; a half-set pair (one axis
    dragged, the other never written) would otherwise silently pin captions to
    the top-left corner."""
    if pos_x is None or pos_y is None:
        return None
    try:
        x, y = float(pos_x), float(pos_y)
    except (TypeError, ValueError):
        return None
    return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))

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


# Padding y radio de la caja que dibuja a mano el estilo "bold", como fraccion
# del tamaño de fuente (no en unidades fijas de script). Antes eran constantes
# -- 70/20/40 unidades, derivadas del CSS del preview ("4px 14px", radio 8px)
# suponiendo una fuente de 110 -- y por eso con una palabra corta quedaba mucho
# fondo negro para poco texto: el padding no bajaba nunca. Atadas a la fuente,
# la caja acompaña al texto a cualquier tamaño. Los ratios son justo los que
# reproducen la caja vieja a fuente 110 (70/110, 20/110*0.33, 40/110), asi que
# con los CAPTION_SIZES nuevos la caja queda practicamente igual de grande pero
# con el texto ~24% mas grande adentro. El espejo esta en static/editor.html
# (padding "0.06em 0.28em", border-radius 0.36em).
_BOLD_PAD_X_RATIO = 0.28
_BOLD_PAD_Y_RATIO = 0.06
_BOLD_RADIUS_RATIO = 0.36
_BEZIER_KAPPA = 0.5523  # cubic-bezier approximation factor for a quarter circle


def _layout_metrics(
    size: int, position: str, style: str, play_res_y: int,
    pos: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Shared scale/margin/box math used by both _ass_header (styles table)
    and build_ass (hand-drawn "bold" box + manual line wrap), so the two
    never drift apart. size/outline/margins/padding are tuned against a
    1920px-tall vertical canvas; scale them proportionally to whatever
    PlayResY we're actually authoring for, so a clip rendered at a different
    aspect ratio (e.g. horizontal 1920x1080) keeps the same *relative*
    on-screen caption size instead of ballooning/shrinking to fill a
    differently-shaped frame."""
    alignment, margin_v = CAPTION_POSITIONS.get(position, CAPTION_POSITIONS[DEFAULT_CAPTION_POSITION])
    if pos is not None:
        # A caption dragged to an arbitrary point in the editor's visualizer is
        # placed per-Dialogue with \an5\pos(...) (see build_ass), so the style
        # table's alignment/MarginV stop deciding anything vertical; pin them to
        # centre/0 so _group_box_geometry and the header agree on that.
        alignment, margin_v = 5, 0
    scale = play_res_y / 1920.0
    # libass interpreta el campo Fontsize del .ass con una convencion propia
    # (heredada de SSA/GDI) que NO da la misma altura real de glifo en pixeles
    # que el mismo numero nominal en CSS (preview del editor) o en PIL. Medido
    # directamente con Montserrat ExtraBold vs. un frame exportado real: a
    # Fontsize=107 (size=190, play_res_y=1080) libass dibuja una "h" de 50px,
    # pero el preview del editor (misma letra, mismo clip) implica ~69px. Un
    # Fontsize=148 en libass da 70px -- practicamente lineal, factor ~1.38.
    # Sin esta correccion, el texto quemado en el video final se ve mas chico
    # que en el visualizer para el mismo "Tamaño" configurado.
    _ASS_FONTSIZE_CORRECTION = 1.38
    scaled_size = round(size * scale)
    # La correccion de arriba SOLO debe afectar el campo Fontsize que libass usa
    # para dibujar el glifo (via _ass_header). El box a mano de "bold" (pad_x/
    # pad_y/radius aca abajo) y toda cuenta de PIL (_pil_font/_text_width/
    # _line_height en build_ass) miden el texto con OTRA convencion de tamaño
    # que no tiene el mismo corrimiento que libass -- de hecho, medido en vivo,
    # esas cuentas con el scaled_size SIN corregir ya reproducen casi exacto el
    # box que muestra el preview (ancho: 496.9px calculado vs 495.99px medido
    # en DOM para "hace ya"). Aplicarles el 1.38 tambien (como hacia la primera
    # version de este fix) infla el box muy por encima del texto real: alto
    # 199.8px vs 151.5px medido, ancho 687.9px vs 496.0px medido -- el bug de
    # "el fondo del texto es enorme" que reporto el usuario. Por eso
    # ass_fontsize (corregido) se usa nada mas para el campo Fontsize del
    # Style table; todo lo demas sigue usando scaled_size sin corregir.
    ass_fontsize = round(scaled_size * _ASS_FONTSIZE_CORRECTION)
    margin_v = round(margin_v * scale)
    margin_lr = round(60 * scale)

    # "bold" draws its own box (see _rounded_rect_path) instead of relying on
    # libass's native BorderStyle=3 box, which draws one rectangle PER WRAPPED
    # LINE rather than a single box around the whole (possibly multi-line)
    # caption -- so it gets no native outline/box at all here. "minimal" keeps
    # the outline but thins it out; karaoke/highlight use today's outline look.
    if style == "bold":
        border_style, outline, back_colour = 1, 0, "&H00000000"
    elif style == "minimal":
        border_style, outline, back_colour = 1, 3, "&H00000000"
    else:
        border_style, outline, back_colour = 1, 8, "&H00000000"
    outline = round(outline * scale) or (0 if style == "bold" else 1)

    return {
        "alignment": alignment,
        "margin_v": margin_v,
        "margin_lr": margin_lr,
        "size": scaled_size,
        "ass_fontsize": ass_fontsize,
        "scale": scale,
        "border_style": border_style,
        "outline": outline,
        "back_colour": back_colour,
        # Sobre scaled_size (= size * scale), asi que ya viene escalado al canvas.
        "pad_x": _BOLD_PAD_X_RATIO * scaled_size,
        "pad_y": _BOLD_PAD_Y_RATIO * scaled_size,
        "radius": _BOLD_RADIUS_RATIO * scaled_size,
    }


def _ass_header(
    font_name: str,
    size: int = 110,
    position: str = DEFAULT_CAPTION_POSITION,
    style: str = DEFAULT_CAPTION_STYLE,
    color: str = DEFAULT_CAPTION_COLOR,
    play_res_x: int = DEFAULT_RESOLUTION[0],
    play_res_y: int = DEFAULT_RESOLUTION[1],
    pos: Optional[Tuple[float, float]] = None,
) -> str:
    m = _layout_metrics(size, position, style, play_res_y, pos=pos)
    alignment, margin_v, margin_lr = m["alignment"], m["margin_v"], m["margin_lr"]
    # Fontsize del Style table: el valor CORREGIDO (ass_fontsize), es lo unico
    # que libass usa para dibujar el glifo -- ver nota en _layout_metrics.
    size, border_style, outline, back_colour = m["ass_fontsize"], m["border_style"], m["outline"], m["back_colour"]

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
    fixed_color = resolve_caption_color(color)
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
    lines.append(
        f"Style: {STYLE_BOX},{font_name},{size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1"
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
                "overridden": bool(w.get("overridden")),
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
- {size_guidance}
- Un punto, cierre de pregunta/exclamación o puntos suspensivos (. ! ? …) SIEMPRE termina la caja ahí mismo, sin excepción -- ahí termina la oración (o el turno de habla), así que nunca sigas agrupando después, aunque la caja quede con pocas palabras.
- Respetá también otras pausas naturales del habla (comas, conectores como "y", "que", "pero" deberían empezar una caja nueva, no terminarla colgada).
- Nunca termines una caja en un artículo, preposición, conjunción, pronombre sujeto (yo, vos, él...), verbo auxiliar (ha, había, estaba, va...) o adverbio corto ("más", "muy", "ya", "no"): esa palabra va con la que sigue. Mantené el verbo junto a su sujeto o su complemento inmediato.
- Cubrí TODOS los tokens, en orden, sin saltear ninguno y sin superponer rangos.

Respondé ÚNICAMENTE con JSON válido, sin texto adicional:
{{"groups": [[0, 3], [3, 6], ...]}}

Donde cada par [start, end) es un rango de índices (igual que un slice de Python), start inclusive, end exclusive.

Tokens:
{numbered_tokens}
"""


# Max words sent to the LLM per cleanup/grouping call. The prompts require a
# verbose per-word JSON object in the response, so a single call over a full
# multi-minute clip (300+ words) can blow past the provider's output-token
# cap and come back truncated/malformed -- silently degrading the whole clip
# to raw, uncleaned text via the except-fallback below. Chunking keeps each
# call small enough to always finish, and confines any one chunk's failure
# to just that chunk instead of the entire clip.
_LLM_CHUNK_SIZE = 80


def _chunk_bounds(words: List[Dict], target_size: int) -> List[Tuple[int, int]]:
    """Split `words` into chunks of ~target_size, each extended (up to +15
    words) to the next sentence-ending punctuation so a chunk boundary lands
    between sentences rather than mid-sentence -- in particular so it doesn't
    split a multi-word number expression like "ocho y media" across two
    separate LLM calls, which would leave each half uncombinable."""
    n = len(words)
    if n <= target_size:
        return [(0, n)]
    bounds = []
    start = 0
    while start < n:
        end = min(start + target_size, n)
        if end < n:
            hard_cap = min(start + target_size + 15, n)
            i = end
            while i < hard_cap:
                text = (words[i - 1].get("word") or "").rstrip()
                if text.endswith((".", "!", "?", "…")):
                    break
                i += 1
            end = i
        bounds.append((start, end))
        start = end
    return bounds


# Disk cache for the two LLM chunk passes above, keyed by chunk content (not
# by clip/highlight id) -- so the live editor preview (/api/captions_range,
# /api/caption_groups) and the real export (build_ass) hit the exact same
# cache entries whenever they process the same words, instead of each paying
# for its own LLM round-trip. Content-addressed rather than time/position-
# addressed so an edited transcript word naturally busts just its own chunk.
# Bump the "_v1" tags below if CLEANUP_PROMPT/GROUPING_PROMPT ever change in
# a way that would make old cached answers wrong.
_LLM_CACHE_DIR = Path(os.getenv("LOCAL_OUTPUT_DIR", "output")) / ".llm_cache"


def _llm_cache_key(tag: str, payload) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"{tag}_{digest}"


def _llm_cache_get(key: str) -> Optional[dict]:
    path = _LLM_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _llm_cache_set(key: str, value: dict) -> None:
    try:
        _LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _LLM_CACHE_DIR / f"{key}.json"
        tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        with open(tmp_path, "w") as f:
            json.dump(value, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"[subtitles] LLM cache write failed ({e!r}); continuing uncached.")


def _normalize_for_compare(text: str) -> str:
    """Lowercase, strip accents and punctuation, for a fidelity comparison
    that ignores cosmetic differences (case, tildes, signos)."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", "", text).strip()


def _token_is_faithful(span: List[Dict], candidate_text: str) -> bool:
    """Guard against the cleanup LLM hallucinating/reordering words, or
    silently dropping words when it merges several original words into one
    output token, instead of just fixing a mis-hear or converting a number.

    Two distinct failure modes were found in production, both passing the
    old index-coverage-only check:
    1. A straight word swap ("bien" -> "bufandas") on a 1-word span.
    2. A multi-word span (e.g. "bien ¿Vendías bufandas en Calle Florida
       vendías", 7 words) collapsed into fewer words ("bufandas en calle
       Florida ¿vendías?", 5 words) that still share enough characters with
       the original for a char-level similarity ratio to stay high (~0.83)
       even though two whole words -- "bien" and "¿Vendías" -- were quietly
       swallowed. A char-ratio check alone cannot catch this.

    CLEANUP_PROMPT only ever asks the LLM to merge multiple original words
    into one token for spoken-number conversion (e.g. "ocho" + "y" +
    "media" -> "8:30"). So any multi-word span that does NOT involve a
    digit is already a contract violation on its face, regardless of text
    similarity -- reject it outright rather than trying to score it."""
    original_text = " ".join(w["word"] for w in span)
    has_digit = any(c.isdigit() for c in original_text) or any(c.isdigit() for c in candidate_text)
    if len(span) > 1 and not has_digit:
        return False
    if has_digit:
        return True
    original_norm = _normalize_for_compare(original_text)
    candidate_norm = _normalize_for_compare(candidate_text)
    if not original_norm or not candidate_norm:
        return True
    ratio = difflib.SequenceMatcher(None, original_norm, candidate_norm).ratio()
    return ratio >= 0.5


def _llm_cleanup_words_chunk(words: List[Dict]) -> List[Dict]:
    """Single LLM cleanup call over one chunk. Falls back to the chunk's
    original words unchanged on any failure (bad response, network error,
    coverage mismatch, or a token that fails the fidelity check) -- a
    transient LLM hiccup must never break clip generation."""
    if not words:
        return words
    cache_key = _llm_cache_key("cleanup_v1", words)
    cached = _llm_cache_get(cache_key)
    if cached is not None:
        return cached["merged"]
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
            # Words the user manually corrected in the editor are ground
            # truth -- never let the cleanup LLM re-"fix" them (it has no
            # way to tell a deliberate correction from a mis-hear, and a
            # foreign/uncommon word like a proper noun is exactly the kind
            # of token it will otherwise "helpfully" hallucinate over).
            span_overridden = any(w.get("overridden") for w in span)
            # El escudo de "overridden" de arriba solo debe evitar que el LLM
            # re-"corrija" el TEXTO de una palabra que el usuario ya arregló a
            # mano -- no es licencia para fusionar un tramo arbitrariamente
            # largo alrededor de esa palabra. Bug real en producción
            # (highlight "Nordelta", 2026-09-23): con una sola palabra
            # overridden en el medio, el LLM fusionó 16 palabras / 6.7s en un
            # solo token; aguas abajo eso rompe el agrupado de subtítulos (el
            # preview del editor le asigna el párrafo entero a un solo bloque
            # chico) y también el burn real (_split_groups_to_fit no puede
            # partir un token ya fusionado). Tope de tamaño aplica siempre,
            # haya o no corrección manual en el tramo -- 4 alcanza de sobra
            # para el único merge legítimo del prompt (número hablado ->
            # dígitos, ej. "ocho y media" son 3 palabras).
            if len(span) > 4 and not any(c.isdigit() for c in t["text"]):
                raise ValueError(
                    f"token span demasiado largo ({len(span)} palabras, sin dígitos): "
                    f"original={[w['word'] for w in span]!r}"
                )
            if not span_overridden and not _token_is_faithful(span, t["text"]):
                raise ValueError(
                    f"unfaithful cleanup token: original={[w['word'] for w in span]!r} "
                    f"-> {t['text']!r}"
                )
            text = " ".join(w["word"] for w in span) if span_overridden else t["text"]
            speakers = [w.get("speaker") for w in span]
            speaker = Counter(speakers).most_common(1)[0][0] if speakers else None
            merged.append({
                "start": span[0]["start"],
                "end": span[-1]["end"],
                "word": text,
                "speaker": speaker,
                "word_start": span[0].get("word_start", span[0]["start"]),
                "word_end": span[-1].get("word_end", span[-1]["end"]),
                "overridden": span_overridden,
            })
        _llm_cache_set(cache_key, {"merged": merged})
        return merged
    except Exception as e:
        print(f"[subtitles] LLM cleanup pass failed ({e!r}); using raw transcript words.")
        return words


def _llm_cleanup_words(words: List[Dict]) -> List[Dict]:
    """Re-reads the window's words with an LLM (full-context pass, chunked --
    see `_LLM_CHUNK_SIZE`) to fix likely Whisper mis-hears and convert spoken
    numbers to digits, without paraphrasing words that are already correct."""
    if not words:
        return words
    merged: List[Dict] = []
    for start, end in _chunk_bounds(words, _LLM_CHUNK_SIZE):
        merged.extend(_llm_cleanup_words_chunk(words[start:end]))
    return merged


# El prompt no traía el tamaño de caja pedido por el usuario (caption_words):
# quedaba fijo en "2 a 4, evitá más de 6" sin importar si el usuario había
# elegido 1, 2 o 3 palabras por caja. Resultado: con caption_words=2 el LLM
# igual devolvía cajas de hasta 6 palabras -- lo que el usuario reportaba
# como subtítulos "que no se separan en formato coherente" (el tamaño real
# en pantalla no tenía nada que ver con lo configurado). Este mapeo traduce
# words_per_group a una instrucción de tamaño concreta para el prompt.
_GROUPING_SIZE_GUIDANCE = {
    1: "Preferí cajas de 1 palabra. Como mucho 2 si separarlas rompe una expresión fija, un número o una cifra compuesta.",
    2: "Preferí cajas de 2 palabras. Podés hacer cajas de 1 o 3 si el sentido lo pide, pero evitá cajas de más de 4 palabras.",
    3: "Preferí cajas de 3 a 4 palabras. Podés hacer cajas más cortas o más largas si el sentido lo pide, pero evitá cajas de más de 6 palabras.",
}


def _llm_group_words_chunk(words: List[Dict], words_per_group: int, strict: bool = False) -> List[tuple]:
    """Single LLM grouping call over one chunk. Falls back to the mechanical
    `_group_boundaries` heuristic (for just this chunk) on any failure --
    unless `strict`, where the failure is raised so the caller can tell the
    user the LLM regrouping did not actually happen."""
    if not words:
        return []
    cache_key = _llm_cache_key("group_v4", {"words": words, "words_per_group": words_per_group})
    cached = _llm_cache_get(cache_key)
    if cached is not None:
        return [tuple(g) for g in cached["groups"]]
    try:
        numbered = "\n".join(f"{i}: {w['word']}" for i, w in enumerate(words))
        size_guidance = _GROUPING_SIZE_GUIDANCE.get(words_per_group, _GROUPING_SIZE_GUIDANCE[3])
        raw = call_local_llm(GROUPING_PROMPT.format(numbered_tokens=numbered, size_guidance=size_guidance))
        groups = [tuple(g) for g in _extract_json_loose(raw)["groups"]]

        expect = 0
        for g in groups:
            if g[0] != expect:
                raise ValueError(f"coverage gap/overlap at {g} (expected start={expect})")
            expect = g[1]
        if expect != len(words):
            raise ValueError(f"coverage incomplete: covered up to {expect}, need {len(words)}")
        _llm_cache_set(cache_key, {"groups": groups})
        return groups
    except Exception as e:
        if strict:
            raise
        print(f"[subtitles] LLM grouping pass failed ({e!r}); using mechanical grouping.")
        return _group_boundaries(words, words_per_group)


_SENTENCE_END_RE = re.compile(r"[.!?…]$")


def _merge_lone_word_groups(groups: List[tuple], words: List[Dict]) -> List[tuple]:
    """Both the LLM grouping pass and its mechanical fallback size groups by
    word count / punctuation with no notion of a multi-word proper noun or
    fixed expression ("Eliott Wave", "8:30 de la mañana"): whichever word
    lands on the count/pause boundary can end up alone in the next box. On
    screen that reads as if the phrase's last word popped back up on its
    own right after the box that already said it. A box holding a single
    word that doesn't itself close a sentence is exactly that failure, so
    fold it into the previous box instead (or the next one, if it's the
    very first box in the clip)."""
    if len(groups) < 2:
        return groups
    merged = [list(g) for g in groups]
    i = 1
    while i < len(merged):
        s, e = merged[i]
        if e - s == 1 and not _SENTENCE_END_RE.search((words[s].get("word") or "").strip()):
            merged[i - 1][1] = e
            del merged[i]
        else:
            i += 1
    if len(merged) >= 2:
        s, e = merged[0]
        if e - s == 1 and not _SENTENCE_END_RE.search((words[s].get("word") or "").strip()):
            merged[1][0] = s
            del merged[0]
    return [tuple(g) for g in merged]


def _llm_group_words(words: List[Dict], words_per_group: int) -> List[tuple]:
    """Groups (already-cleaned) words into caption boxes that read as
    coherent phrases (2-4, up to 6 words) instead of a fixed word-count
    stride (chunked -- see `_LLM_CHUNK_SIZE`)."""
    if not words:
        return []
    groups: List[tuple] = []
    for start, end in _chunk_bounds(words, _LLM_CHUNK_SIZE):
        for g_start, g_end in _llm_group_words_chunk(words[start:end], words_per_group):
            groups.append((g_start + start, g_end + start))
    return _merge_lone_word_groups(groups, words)


def apply_group_keys(words: List[Dict], group_keys) -> List[tuple]:
    """Groups defined by the user's "Acomodar subtitulos": `group_keys` are the
    absolute `word_start` of the first word of every caption group (same
    stable key `caption_splits` uses). Blank words (merged-token tails) never
    open a group, they stay with the previous one."""
    keys = {round(float(k), 3) for k in (group_keys or [])}
    starts = [0]
    for i in range(1, len(words)):
        if (words[i].get("word") or "").strip() and round(float(words[i].get("word_start", -1)), 3) in keys:
            starts.append(i)
    return list(zip(starts, starts[1:] + [len(words)]))


CORRECTION_PROMPT = """Sos un corrector de subtítulos en español rioplatense. Te paso una lista numerada de palabras tal como las transcribió un reconocimiento de voz automático (Whisper) en un fragmento de un podcast de finanzas/economía.
{context}
Tu tarea es encontrar SOLO las palabras que están mal y devolver su reemplazo:
1. Palabras mal escritas (ortografía, tildes faltantes o sobrantes, palabras pegadas o partidas como "dec20").
2. Palabras que la persona claramente NO dijo: no tienen sentido en esa posición y el contexto deja claro qué dijo en realidad (normalmente algo que suena parecido). Reemplazalas por la palabra correcta según el contexto.

MUY IMPORTANTE -- nombres propios: todo nombre de persona, apellido, empresa, banco, marca, país, ciudad o institución va con mayúscula inicial (ej: "eduardo costantini" -> "Eduardo Costantini", "banco francés" -> "Banco Francés", "argentina" -> "Argentina", "alfonsín" -> "Alfonsín"). Los subtítulos pueden estar escritos todo en minúscula a propósito: igual devolvé cada nombre propio con mayúscula inicial. Pero NO pongas mayúscula a palabras comunes ni al comienzo de oración si el resto está en minúscula.

Reglas:
- Si una palabra ya tiene sentido gramatical y semántico, DEJALA IGUAL aunque se te ocurra una "mejor". No reformules, no resumas, no cambies el estilo oral (muletillas, repeticiones y "o sea" se quedan).
- Si no estás seguro de qué se dijo, no la cambies.
- Whisper se equivoca por SONIDO: el reemplazo de una palabra mal oída tiene que sonar parecido a lo transcripto (ej: "IPER" -> "hiper", "llevó" -> "llegó"). Si la palabra correcta según el contexto no suena parecido (ej: "Robert" -> "Francés"), NO la cambies: sería adivinar, no corregir.
- Nunca cambies una palabra que forma una frase válida con sus vecinas (ej: "en un momento", "creo que en 200 mil" se quedan como están).
- Cada reemplazo es para UNA palabra de la lista, y el reemplazo tiene como mucho 3 palabras (ej: "dec20" -> "de 20").
- Mantené la puntuación que tenga la palabra original (coma, punto, signos).
- No conviertas números entre letras y dígitos: eso no es un error.

Respondé ÚNICAMENTE con JSON válido, sin texto adicional:
{{"changes": [{{"i": 12, "text": "palabra corregida"}}, ...]}}
Si no hay nada que corregir: {{"changes": []}}

Palabras:
{numbered_words}
"""

# Si el LLM "corrige" más que esta proporción de un tramo, está reescribiendo, no
# corrigiendo: se descarta el tramo entero en vez de aplicar una reescritura.
_CORRECTION_MAX_RATIO = 0.25


def _match_word_shape(original: str, new: str) -> str:
    """Keep the original token's surrounding whitespace (the transcript's
    spacing convention) and its trailing punctuation if the LLM dropped it."""
    lead = original[:len(original) - len(original.lstrip())]
    trail = original[len(original.rstrip()):]
    core_orig, core_new = original.strip(), new.strip()
    punct = re.search(r"[.,;:!?…]+$", core_orig)
    if punct and not re.search(r"[.,;:!?…]$", core_new):
        core_new += punct.group(0)
    return lead + core_new + trail


def _is_capitalization_only(original: str, new: str) -> bool:
    """True if `new` is the same all-lowercase word with only its
    capitalization and/or accents fixed ("argentina" -> "Argentina",
    "frances" -> "Francés"). All-caps words are left alone (user style)."""
    import unicodedata

    def base(t: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")

    o, n = original.strip(), new.strip()
    return o != n and o == o.lower() and base(o) == base(n)


def correct_clip_words(words: List[Dict], context: str = "") -> List[Dict]:
    """LLM pass of the "Acomodar subtitulos" button that fixes misspelled
    words and words Whisper clearly mis-heard. Strictly 1:1 (one original word
    -> its replacement), so every word keeps its own timing and its override
    key (`word_start`/`word_end`). Words the user already corrected by hand
    (`overridden`) only get proper-name capitalization / missing accents (an
    all-lowercase word, same letters, e.g. after the "minúsculas" bulk edit); their text
    is never rewritten. Returns the changes to persist:
    [{"word_start", "word_end", "from", "to"}]. Raises if the LLM call fails."""
    candidates = [
        (i, w) for i, w in enumerate(words)
        if (w.get("word") or "").strip()
    ]
    if not candidates:
        return []
    plain = [{"word": w["word"].strip()} for _, w in candidates]
    context_line = f"\nContexto del clip (para nombres propios y términos): {context.strip()}\n" if context.strip() else ""

    changes: List[Dict] = []
    for start, end in _chunk_bounds(plain, _LLM_CHUNK_SIZE):
        chunk = plain[start:end]
        cache_key = _llm_cache_key("correct_v3", {"words": chunk, "context": context_line})
        cached = _llm_cache_get(cache_key)
        if cached is not None:
            chunk_changes = cached["changes"]
        else:
            numbered = "\n".join(f"{i}: {w['word']}" for i, w in enumerate(chunk))
            raw = call_local_llm(CORRECTION_PROMPT.format(context=context_line, numbered_words=numbered))
            chunk_changes = []
            for c in _extract_json_loose(raw).get("changes") or []:
                i, text = int(c["i"]), str(c.get("text") or "").strip()
                if not (0 <= i < len(chunk)) or not text or len(text.split()) > 3:
                    continue
                if text == chunk[i]["word"]:
                    continue
                chunk_changes.append({"i": i, "text": text})
            if len(chunk_changes) > max(3, _CORRECTION_MAX_RATIO * len(chunk)):
                print(f"[subtitles] correction discarded: {len(chunk_changes)} changes in "
                      f"{len(chunk)} words looks like a rewrite, not a fix", flush=True)
                chunk_changes = []
            _llm_cache_set(cache_key, {"changes": chunk_changes})
        for c in chunk_changes:
            _, w = candidates[start + c["i"]]
            new_text = _match_word_shape(w["word"], c["text"])
            if w.get("overridden") and not _is_capitalization_only(w["word"], new_text):
                continue
            if new_text != w["word"]:
                changes.append({
                    "word_start": w["word_start"], "word_end": w["word_end"],
                    "from": w["word"], "to": new_text,
                })
    return changes


def arrange_caption_group_keys(words: List[Dict], words_per_group: int) -> List[float]:
    """LLM regrouping of the RAW clip words into coherent caption boxes (the
    "Acomodar subtitulos" button). Returns the `word_start` key of every
    group's first word except the first group's (see `apply_group_keys`).
    Raises if the LLM call fails, so the caller never presents the mechanical
    fallback as an "arranged" result."""
    if words_per_group <= 1 or not words:
        return []
    # Number tails (".000" after "38") are glued to the previous token so the
    # LLM sees one unit and can never split a figure.
    tokens: List[Dict] = []
    for i, w in enumerate(words):
        text = (w.get("word") or "").strip()
        if not text:
            continue
        if tokens and _NUMBER_CONTINUATION_RE.match(text):
            tokens[-1]["word"] += " " + text
        else:
            tokens.append({"word": text, "idx": i})
    if len(tokens) < 2:
        return []
    plain = [{"word": t["word"]} for t in tokens]

    groups: List[List[int]] = []
    for start, end in _chunk_bounds(plain, _LLM_CHUNK_SIZE):
        for g_start, g_end in _llm_group_words_chunk(plain[start:end], words_per_group, strict=True):
            groups.append([g_start + start, g_end + start])

    # A sentence end always closes a box, whatever the LLM did.
    def _ends_sentence(t: int) -> bool:
        return plain[t]["word"].rstrip().endswith(_SENTENCE_END_CHARS)

    split: List[List[int]] = []
    for g_start, g_end in groups:
        cur = g_start
        for t in range(g_start, g_end - 1):
            if _ends_sentence(t):
                split.append([cur, t + 1])
                cur = t + 1
        split.append([cur, g_end])

    # Fold orphan single words (not closing a sentence) into the neighbour
    # from the SAME sentence: previous box unless that one closed a sentence.
    merged: List[List[int]] = []
    for g in split:
        if (merged and g[1] - g[0] == 1 and not _ends_sentence(g[0])
                and not _ends_sentence(merged[-1][1] - 1)):
            merged[-1][1] = g[1]
        else:
            merged.append(g)

    keys = []
    for g_start, _ in merged[1:]:
        keys.append(round(float(words[tokens[g_start]["idx"]]["word_start"]), 3))
    return keys


# Words that read as unfinished when they close a caption box: articles,
# prepositions, conjunctions, possessives and clitics/negation. The mechanical
# grouper avoids ending a box on any of them (the next box should open with it).
_DANGLING_CONNECTORS = {
    "y", "o", "u", "e", "ni", "que", "pero", "porque", "aunque",
    "pues", "sino", "cuando", "mientras", "si", "como",
    "la", "las", "el", "los", "un", "una", "unos", "unas", "lo",
    "a", "al", "ante", "bajo", "con", "contra", "de", "del", "desde", "en",
    "entre", "hacia", "hasta", "para", "por", "segun", "según", "sin",
    "sobre", "tras",
    "mi", "mis", "tu", "tus", "su", "sus", "nuestro", "nuestra",
    "nuestros", "nuestras",
    "no", "me", "te", "se", "le", "les", "nos",
    "yo", "vos", "tú", "él", "ella", "ellos", "ellas", "nosotros", "usted",
    "ya", "más", "muy", "tan", "también",
    "ha", "he", "has", "han", "hemos", "había", "habían",
    "estaba", "estaban", "va", "vas", "voy", "vamos", "van",
}

_SENTENCE_END_CHARS = (".", "!", "?", "…")
_NUMBER_CONTINUATION_RE = re.compile(r"^[.,]\d")
_BARE_STRIP_CHARS = ".,;:!?…¿¡\"'()«»"

# Cost model for `_segment_sentence` (lower = better). Tuned so that, for
# words_per_group=2, "en La Plata / y mi inglés" beats "La Plata y mi /
# inglés siempre era".
_COST_LONE_WORD = 2.5        # 1-word box inside a longer sentence (orphan)
_COST_DANGLING_END = 4.0     # box closes on an article/preposition/connector
_COST_INTERIOR_COMMA = 1.0   # a comma/semicolon buried inside a box
_BONUS_COMMA_BREAK = 1.5     # break right after a comma/semicolon/colon
_BONUS_LONG_PAUSE = 1.5      # audio gap >= 0.4s between the two words
_BONUS_SHORT_PAUSE = 0.5     # audio gap >= 0.2s
_COST_FORBIDDEN = 100.0      # never split "38" / ".000"


def _bare_token(word: Dict) -> str:
    return (word.get("word") or "").strip().strip(_BARE_STRIP_CHARS).lower()


def _break_cost(words: List[Dict], j: int) -> float:
    """Cost of ending a caption box between words[j-1] and words[j]."""
    prev_text = (words[j - 1].get("word") or "").strip()
    next_text = (words[j].get("word") or "").strip()
    if _NUMBER_CONTINUATION_RE.match(next_text):
        return _COST_FORBIDDEN
    cost = 0.0
    if prev_text.endswith((",", ";", ":")):
        cost -= _BONUS_COMMA_BREAK
    elif _bare_token(words[j - 1]) in _DANGLING_CONNECTORS:
        cost += _COST_DANGLING_END
    try:
        gap = float(words[j]["start"]) - float(words[j - 1]["end"])
    except (KeyError, TypeError, ValueError):
        gap = 0.0
    if gap >= 0.4:
        cost -= _BONUS_LONG_PAUSE
    elif gap >= 0.2:
        cost -= _BONUS_SHORT_PAUSE
    return cost


def _segment_sentence(words: List[Dict], s: int, e: int, words_per_group: int) -> List[tuple]:
    """Optimal split of the sentence words[s:e] into caption boxes via DP:
    boxes close to `words_per_group` words, breaking preferably at commas and
    audio pauses and never on a dangling connector, orphan word or number."""
    n = e - s
    hard_max = words_per_group + 2
    interior_commas = [
        1 if (words[s + k].get("word") or "").rstrip().endswith((",", ";", ":")) else 0
        for k in range(n)
    ]
    break_cost = {j: _break_cost(words, s + j) for j in range(1, n)}
    # "38" + ".000" is one number on screen: count it as a single word for sizing.
    continuation = [
        1 if _NUMBER_CONTINUATION_RE.match((words[s + k].get("word") or "").strip()) else 0
        for k in range(n)
    ]

    INF = float("inf")
    best = [INF] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(max(0, j - hard_max), j):
            if best[i] == INF:
                continue
            size = j - i - sum(continuation[i:j])
            cost = best[i] + (size - words_per_group) ** 2
            if size == 1 and n > 1:
                cost += _COST_LONE_WORD
            cost += _COST_INTERIOR_COMMA * sum(interior_commas[i:j - 1])
            if j < n:
                cost += break_cost[j]
            if cost < best[j]:
                best[j] = cost
                back[j] = i
    groups = []
    j = n
    while j > 0:
        i = back[j]
        groups.append((s + i, s + j))
        j = i
    groups.reverse()
    return groups


def _group_boundaries(words: List[Dict], words_per_group: int) -> List[tuple]:
    """Mechanical grouping (also the LLM fallback): a sentence-ending
    punctuation mark ALWAYS closes the box; inside each sentence,
    `_segment_sentence` picks balanced boxes of ~`words_per_group` words
    (hard cap `words_per_group + 2`) that break at commas / pauses and avoid
    dangling connectors, orphan words and split numbers."""
    if words_per_group <= 1:
        return [(i, i + 1) for i in range(len(words))]

    groups: List[tuple] = []
    start = 0
    for i, w in enumerate(words):
        if (w.get("word") or "").rstrip().endswith(_SENTENCE_END_CHARS):
            groups.extend(_segment_sentence(words, start, i + 1, words_per_group))
            start = i + 1
    if start < len(words):
        groups.extend(_segment_sentence(words, start, len(words), words_per_group))
    return groups


@lru_cache(maxsize=None)
def _pil_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    """Loads the real .ttf backing `font_name` at `size` px so caption-box
    math can measure actual glyph widths instead of guessing -- libass
    renders with the same underlying FreeType, so this stays in sync with
    what actually shows up on screen. Falls back to PIL's built-in bitmap
    font (a rough approximation) if the .ttf isn't found, rather than
    raising, since a slightly-off box beats a crashed render."""
    path = FONT_FILES.get(font_name)
    if path and path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _text_width(font: ImageFont.FreeTypeFont, text: str) -> float:
    if not text:
        return 0.0
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def _line_height(font: ImageFont.FreeTypeFont) -> float:
    ascent, descent = font.getmetrics()
    return float(ascent + descent)


def _wrap_group_words(word_texts: List[str], font: ImageFont.FreeTypeFont, max_width: float) -> List[List[int]]:
    """Greedily wraps one caption group's words into lines using real font
    metrics, returning each line as a list of indices into `word_texts`.
    This replaces libass's own WrapStyle=0 auto-wrap for the "bold" style
    specifically, so build_ass knows exactly where each line breaks *before*
    building the hand-drawn box -- letting libass choose the break on its own
    would make the box's line count (and therefore its height) unpredictable.
    A single word that alone exceeds max_width is still kept on its own line
    rather than split mid-word."""
    lines: List[List[int]] = []
    current: List[int] = []
    current_text = ""
    for i, word in enumerate(word_texts):
        candidate = f"{current_text} {word}" if current_text else word
        if current and _text_width(font, candidate) > max_width:
            lines.append(current)
            current = [i]
            current_text = word
        else:
            current.append(i)
            current_text = candidate
    if current:
        lines.append(current)

    # A trailing line holding a single word (typography's "widow") reads on
    # screen as if the previous line's word "repeated" alone right after it
    # -- e.g. a name split "fanático de la Eliott" / "Wave" purely because
    # the fifth word didn't fit, with no idea "Eliott Wave" is one unit. Pull
    # that last word's line-mate back from the previous line instead, as
    # long as the rebalanced two-word line still fits. Skip when the previous
    # line has exactly 2 words: taking one away would leave it with 1 word of
    # its own, just relocating the widow instead of removing it.
    if len(lines) >= 2 and len(lines[-1]) == 1:
        prev, last = lines[-2], lines[-1]
        if len(prev) != 2:
            moved_text = " ".join(word_texts[i] for i in (prev[-1:] + last))
            if _text_width(font, moved_text) <= max_width:
                lines[-1] = prev[-1:] + last
                lines[-2] = prev[:-1]
                if not lines[-2]:
                    del lines[-2]
    return lines


def _rounded_rect_path(width: float, height: float, radius: float) -> str:
    """ASS \\p1 drawing commands for a rounded rectangle spanning (0,0) to
    (width,height) -- meant to be placed with \\an7\\pos(left,top) so those
    absolute coordinates become its top-left corner. Corners are cubic-bezier
    approximations of a quarter circle (kappa ~0.5523), same technique as
    SVG/CSS rounded-rect rendering."""
    r = max(0.0, min(radius, width / 2, height / 2))
    if r <= 0:
        return f"m 0 0 l {width:.1f} 0 l {width:.1f} {height:.1f} l 0 {height:.1f}"
    k = _BEZIER_KAPPA * r
    w, h = width, height
    return (
        f"m {r:.1f} 0 "
        f"l {w - r:.1f} 0 "
        f"b {w - r + k:.1f} 0 {w:.1f} {r - k:.1f} {w:.1f} {r:.1f} "
        f"l {w:.1f} {h - r:.1f} "
        f"b {w:.1f} {h - r + k:.1f} {w - r + k:.1f} {h:.1f} {w - r:.1f} {h:.1f} "
        f"l {r:.1f} {h:.1f} "
        f"b {r - k:.1f} {h:.1f} 0 {h - r + k:.1f} 0 {h - r:.1f} "
        f"l 0 {r:.1f} "
        f"b 0 {r - k:.1f} {r - k:.1f} 0 {r:.1f} 0"
    )


def _split_groups_to_fit(
    groups: List[Tuple[int, int]],
    words: List[Dict],
    font: ImageFont.FreeTypeFont,
    max_width: float,
    strip_punctuation: bool,
) -> List[Tuple[int, int]]:
    """Re-splits any caption group whose text wouldn't fit on one line (per
    real font metrics) into consecutive single-line sub-groups, at the same
    word boundaries _wrap_group_words would break lines at. Captions must
    render as a single line -- letting a group wrap to 2+ lines (whether via
    libass's own auto-wrap or the "bold" style's hand-drawn box) produced
    incoherent, near-one-word-per-line captions on narrower/vertical canvases
    -- so oversized groups become multiple time-sequenced one-line captions
    instead of a single multi-line one. Word timing is untouched: each
    sub-group's Dialogue end is still the next word's start (see
    _display_end), so this only affects word grouping, not playback speed."""
    result: List[Tuple[int, int]] = []
    for g_start, g_end in groups:
        idxs = [i for i in range(g_start, g_end) if words[i]["word"].strip()]
        if not idxs:
            result.append((g_start, g_end))
            continue
        word_texts = [
            _strip_punct(words[i]["word"].replace("\n", " ").strip()) if strip_punctuation
            else words[i]["word"].replace("\n", " ").strip()
            for i in idxs
        ]
        line_groups = _wrap_group_words(word_texts, font, max_width)
        if len(line_groups) <= 1:
            result.append((g_start, g_end))
            continue
        for local_idxs in line_groups:
            sub_idxs = [idxs[li] for li in local_idxs]
            result.append((sub_idxs[0], sub_idxs[-1] + 1))
    return result


def _group_box_geometry(
    block_width: float, block_height: float,
    play_res_x: int, play_res_y: int,
    alignment: int, margin_v: float, pad_x: float, pad_y: float,
    pos_px: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float, float]:
    """Top-left (x, y) and (width, height) of the hand-drawn box for a group
    whose measured text block is block_width x block_height, positioned the
    same way libass would position that text under the given alignment/
    margin -- box_x/box_y are always computed from alignment+margin (never
    text_align's MarginL/MarginR asymmetry) because _ass_header always emits
    equal MarginL/MarginR, so horizontal centering is always PlayResX/2.

    With pos_px (a free position dragged in the editor) the box is centred on
    that pixel instead, matching the \\an5\\pos(...) the text itself carries.
    It is deliberately NOT clamped to the frame: the clamp would have to depend
    on each group's own width, so wide and narrow phrases would land at
    different places and the caption would jitter horizontally between groups.
    The editor clamps the draggable point instead."""
    box_w = block_width + 2 * pad_x
    box_h = block_height + 2 * pad_y
    if pos_px is not None:
        return pos_px[0] - box_w / 2, pos_px[1] - box_h / 2, box_w, box_h
    box_x = (play_res_x - box_w) / 2
    if alignment == 8:  # top
        box_y = margin_v
    elif alignment == 5:  # middle
        box_y = (play_res_y - box_h) / 2
    else:  # bottom (2)
        box_y = play_res_y - margin_v - box_h
    return box_x, box_y, box_w, box_h


def apply_group_splits(groups: List[tuple], words: List[Dict], split_keys) -> List[tuple]:
    """Manual "cortar bloque" (CapCut/Premiere razor): `split_keys` are the
    absolute `word_start` of the first word of each block the user cut open.
    Every group holding such a word strictly after its first shown word is cut
    there in two. A cut never lands on a blank (merged-token tail) word: it
    slides forward to the next shown word, or is ignored if none is left, so
    neither half can end up as an all-blank block. Same rule as splitCaptionBlockAt
    in the editor."""
    keys = {round(float(k), 3) for k in (split_keys or [])}
    if not keys:
        return list(groups)
    out: List[tuple] = []
    for g_start, g_end in groups:
        cuts = []
        for i in range(g_start + 1, g_end):
            w = words[i]
            if round(float(w.get("word_start", -1)), 3) in keys:
                j = i
                while j < g_end and not words[j]["word"].strip():
                    j += 1
                if j < g_end and j > g_start and any(x["word"].strip() for x in words[g_start:j]):
                    cuts.append(j)
        prev = g_start
        for c in sorted(set(cuts)):
            out.append((prev, c))
            prev = c
        out.append((prev, g_end))
    return out


# Una pausa entre dos bloques de subtitulo de hasta este largo (s) no deja la
# pantalla vacia: el bloque sigue visible hasta que arranca el siguiente. Pausas
# mas largas si lo ocultan, para no dejar texto colgado en silencios largos.
CAPTION_BRIDGE_MAX_GAP = 2.0


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
    use_llm: bool = False,
    pos: Optional[Tuple[float, float]] = None,
    caption_splits=None,
    caption_group_keys=None,
) -> str:
    if use_llm:
        words = _llm_cleanup_words(words)

    size = resolve_caption_size(size)
    play_res_x, play_res_y = resolution
    lines = [_ass_header(
        font_name, size=size, position=position, style=style, color=color,
        play_res_x=play_res_x, play_res_y=play_res_y, pos=pos,
    )]
    fixed_style = resolve_caption_color(color) is not None

    # Free position (dragged in the editor) -> every Dialogue carries its own
    # \an5\pos, and the hand-drawn "bold" box is centred on the same point.
    pos_px = (pos[0] * play_res_x, pos[1] * play_res_y) if pos is not None else None
    pos_tag = f"{{\\an5\\pos({pos_px[0]:.1f},{pos_px[1]:.1f})}}" if pos_px is not None else ""

    layout = _layout_metrics(size, position, style, play_res_y, pos=pos)
    pil_font = _pil_font(font_name, layout["size"])
    max_text_width = play_res_x - 2 * layout["margin_lr"] - 2 * layout["pad_x"]

    def _group_span(g_start: int, g_end: int):
        """[start, end) of a caption block = min start / max end of its words.
        Blank words at the very start of a group are the tail of the previous
        block's merged word (or deleted words), so they leave this block and
        extend the previous one -- the same rule the editor's timeline block
        uses, so the block on the timeline, the preview and this burn
        appear/disappear at exactly the same instants."""
        ws = words[g_start:g_end]
        first = next((k for k, x in enumerate(ws) if x["word"].strip()), None)
        if first is None:
            return min(x["start"] for x in ws), max(x["end"] for x in ws)
        e = g_end
        while e < len(words) and not words[e]["word"].strip():
            e += 1
        ws = words[g_start + first:e]
        return min(x["start"] for x in ws), max(x["end"] for x in ws)

    def _block_span(g_start: int, g_end: int):
        """_group_span with the end stretched to the next block's start when
        the pause between them is <= CAPTION_BRIDGE_MAX_GAP, so the caption
        changes instead of blinking out between blocks. Same rule as
        showEnd in the editor's getCaptionBlocks()."""
        s, e = _group_span(g_start, g_end)
        k = g_end
        while k < len(words) and not words[k]["word"].strip():
            k += 1
        if k < len(words):
            nxt = group_for_index.get(k, (k, k + 1)) if words_per_group > 1 else (k, k + 1)
            ns = _group_span(*nxt)[0]
            if e < ns <= e + CAPTION_BRIDGE_MAX_GAP:
                e = ns
        return s, e

    def _token_span(i: int):
        """(start, end) of the Dialogue for non-empty word i. Blocks are the
        source of truth: the block's first shown word starts at the block start,
        its last one ends at the block end (no bridging across pauses, and
        blocks may overlap -- both are drawn). Inside a block each word lasts
        until the next shown word of the same block starts (karaoke)."""
        if words_per_group > 1:
            g_start, g_end = group_for_index[i]
        else:
            g_start, g_end = i, i + 1
        shown = [k for k in range(g_start, g_end) if words[k]["word"].strip()]
        pos = shown.index(i)
        span_start, span_end = _block_span(g_start, g_end)
        start_v = span_start if pos == 0 else words[i]["start"]
        end_v = span_end if pos == len(shown) - 1 else words[shown[pos + 1]]["start"]
        return start_v, max(end_v, start_v + 0.001)

    groups: List[tuple] = []
    group_for_index: Dict[int, tuple] = {}
    group_style: Dict[tuple, str] = {}
    if words_per_group > 1:
        if caption_group_keys:
            groups = apply_group_keys(words, caption_group_keys)
        else:
            groups = _llm_group_words(words, words_per_group) if use_llm else _group_boundaries(words, words_per_group)
        # Captions must render on ONE line -- a group whose text doesn't fit
        # the available width at this font size gets re-split at word
        # boundaries into consecutive one-line groups instead of wrapping
        # (wrapping either looked like N stacked lines with no box, or forced
        # the "bold" box to stack lines with awkward per-line word counts on
        # narrower/vertical canvases). Word timing is untouched -- each new
        # sub-group is just a smaller slice of the same word list.
        groups = _split_groups_to_fit(groups, words, pil_font, max_text_width, strip_punctuation)
        groups = apply_group_splits(groups, words, caption_splits)
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

    # "bold" draws its own box behind each caption group (see
    # _rounded_rect_path) instead of relying on libass's native BorderStyle=3
    # box, which draws a separate rectangle PER WRAPPED LINE -- so a group
    # that wraps to 2 lines used to render as two disjoint boxes instead of
    # one continuous one. That means line breaks must be decided *here*, with
    # real font metrics, before any Dialogue text is written, instead of
    # leaving WrapStyle=0 auto-wrap to libass (which could break the text
    # somewhere this box didn't account for). Applies to single-word
    # captions too (words_per_group<=1) by treating each word as its own
    # one-word "group" for box purposes.
    group_line_of: Dict[int, int] = {}
    group_scale: Dict[tuple, float] = {}
    if style == "bold":
        line_height = _line_height(pil_font)
        bold_groups = groups if words_per_group > 1 else [
            (idx, idx + 1) for idx, bw in enumerate(words) if bw["word"].strip()
        ]
        for g_start, g_end in bold_groups:
            idxs = [idx for idx in range(g_start, g_end) if words[idx]["word"].strip()]
            if not idxs:
                continue
            word_texts = [
                _strip_punct(words[idx]["word"].replace("\n", " ").strip()) if strip_punctuation
                else words[idx]["word"].replace("\n", " ").strip()
                for idx in idxs
            ]
            line_groups = _wrap_group_words(word_texts, pil_font, max_text_width)
            for line_no, local_idxs in enumerate(line_groups):
                for local_i in local_idxs:
                    group_line_of[idxs[local_i]] = line_no
            raw_block_width = max(
                _text_width(pil_font, " ".join(word_texts[li] for li in local_idxs))
                for local_idxs in line_groups
            )
            # Groups are already pre-split to one line each (see
            # _split_groups_to_fit); the only way raw_block_width can still
            # exceed max_text_width here is a single word too wide to fit on
            # its own (long compound Spanish words at bigger sizes on a
            # narrow vertical canvas) -- splitting mid-word isn't an option,
            # so shrink that one caption's font instead of letting its box
            # and text bleed past the frame edge.
            scale = min(1.0, max_text_width / raw_block_width) if raw_block_width > 0 else 1.0
            group_scale[(g_start, g_end)] = scale
            block_width = raw_block_width * scale
            block_height = len(line_groups) * line_height * scale
            box_x, box_y, box_w, box_h = _group_box_geometry(
                block_width, block_height, play_res_x, play_res_y,
                layout["alignment"], layout["margin_v"], layout["pad_x"], layout["pad_y"],
                pos_px=pos_px,
            )
            first_idx, last_idx = idxs[0], idxs[-1]
            box_s, box_e = _block_span(g_start, g_end)
            box_start = _ass_timestamp(box_s)
            box_end = _ass_timestamp(max(box_e, box_s + 0.001))
            path = _rounded_rect_path(box_w, box_h, layout["radius"])
            box_tags = f"{{\\an7\\pos({box_x:.1f},{box_y:.1f})\\1c&H000000&\\1a&H80&\\bord0\\shad0\\p1}}"
            lines.append(f"Dialogue: 0,{box_start},{box_end},{STYLE_BOX},,0,0,0,,{box_tags}{path}{{\\p0}}\n")

    for i, w in enumerate(words):
        text = w["word"].replace("\n", " ").strip()
        if strip_punctuation:
            text = _strip_punct(text)
        if not text:
            continue
        tok_start, tok_end = _token_span(i)
        start = _ass_timestamp(tok_start)
        end = _ass_timestamp(tok_end)

        if words_per_group > 1:
            group_key = group_for_index[i]
            ass_style = STYLE_FIXED if fixed_style else group_style[group_key]
            group_start, group_end = group_key

            def _word_text(idx: int) -> str:
                t = words[idx]["word"].replace("\n", " ").strip()
                return _strip_punct(t) if strip_punctuation else t

            # Each segment resets its own alpha/scale explicitly (rather than
            # relying on tag order) so the active word's highlight scale never
            # leaks into the dimmed segment that follows it.
            scale = group_scale.get(group_key, 1.0)
            dim_scale = 100 * scale
            dim_tag = f"{{\\alpha&H80&\\fscx{dim_scale:.0f}\\fscy{dim_scale:.0f}}}"
            if style == "highlight":
                active_tag = "{\\alpha&H00&\\fscx115\\fscy115}"
            elif scale < 1.0:
                active_scale = 100 * scale
                active_tag = f"{{\\alpha&H00&\\fscx{active_scale:.0f}\\fscy{active_scale:.0f}}}"
            else:
                active_tag = "{\\alpha&H00&}"
            parts = []
            prev_line = None
            for idx in range(group_start, group_end):
                if not words[idx]["word"].strip():
                    continue
                if parts:
                    # Reuse the exact line breaks the box geometry was built
                    # from, so the text never wraps somewhere the hand-drawn
                    # box didn't account for.
                    cur_line = group_line_of.get(idx)
                    parts.append("\\N" if style == "bold" and cur_line != prev_line else " ")
                tag = active_tag if idx == i else dim_tag
                parts.append(tag + _word_text(idx))
                prev_line = group_line_of.get(idx)
            display = "".join(parts)
        else:
            ass_style = STYLE_FIXED if fixed_style else _style_for_speaker(w.get("speaker"))
            if style == "highlight":
                display = "{\\fscx115\\fscy115}" + text
            else:
                one_word_scale = group_scale.get((i, i + 1), 1.0)
                if style == "bold" and one_word_scale < 1.0:
                    s = 100 * one_word_scale
                    display = f"{{\\fscx{s:.0f}\\fscy{s:.0f}}}" + text
                else:
                    display = text

        layer = 1 if style == "bold" else 0
        lines.append(f"Dialogue: {layer},{start},{end},{ass_style},,0,0,0,,{pos_tag}{display}\n")
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
    use_llm: bool = False,
    pos: Optional[Tuple[float, float]] = None,
    caption_splits=None,
    caption_group_keys=None,
    speed: float = 1.0,
) -> str:
    """Burn subtitles onto video_path, writing out_path.
    If there are no words (e.g. silent clip), just copies the source through.

    `speed` != 1 aplica además el cambio de velocidad (setpts/atempo, igual
    que clipper.apply_speed_local) en la MISMA pasada, después del ass: los
    tiempos de los subtítulos están en segundos del clip sin acelerar, así
    que quemarlos antes del setpts los deja sincronizados. Ahorra un encode
    entero del clip en el export."""
    if not words:
        if speed != 1.0:
            from .clipper import apply_speed_local
            return apply_speed_local(video_path, speed, out_path) if os.path.abspath(video_path) != os.path.abspath(out_path) else out_path
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
        pos=pos, caption_splits=caption_splits, caption_group_keys=caption_group_keys,
    )
    try:
        # ffmpeg filter paths need escaping on all platforms; simplest is to
        # run with cwd set to the .ass file's directory and use a bare filename.
        ass_dir = os.path.dirname(os.path.abspath(ass_path)) or "."
        ass_name = os.path.basename(ass_path)
        ass_filter = f"ass={ass_name}:fontsdir={FONTS_DIR}"
        if speed != 1.0:
            filter_args = [
                "-filter_complex", f"[0:v]{ass_filter},setpts=PTS/{speed}[v];[0:a]atempo={speed}[a]",
                "-map", "[v]", "-map", "[a]",
            ]
            audio_args = ["-c:a", "aac", "-b:a", "128k"]
        else:
            filter_args = ["-vf", ass_filter]
            audio_args = ["-c:a", "copy"]
        cmd = [
            FFMPEG_BIN, "-y", "-loglevel", "error",
            "-i", os.path.abspath(video_path),
            *filter_args,
            *video_encode_args(),
            *audio_args,
            os.path.abspath(out_path),
        ]
        subprocess.run(cmd, check=True, cwd=ass_dir)
    finally:
        if os.path.exists(ass_path):
            os.remove(ass_path)
    return out_path
