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

CAPTION_SIZES = {"s": 65, "m": 90, "l": 115, "xl": 140}
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


# Reference values (at the canonical PlayResY=1920 canvas) for the "bold"
# style's hand-drawn box: horizontal/vertical padding around the measured
# text block and corner radius, in script units. Picked to match the
# preview's CSS box (static/editor.html: background rgba(0,0,0,.5), padding
# "4px 14px", border-radius 8px) at its known 5x preview-to-canvas downscale
# (CLIP_CAPTION_SIZE_PREVIEW maps CAPTION_SIZES 1:5, e.g. m=110 vs 22px) --
# so 14px*5=70, 4px*5=20, 8px*5=40. Scaled by the same `scale` factor as
# everything else below for non-1920-tall canvases.
_BOLD_PAD_X_REF = 70.0
_BOLD_PAD_Y_REF = 20.0
_BOLD_RADIUS_REF = 40.0
_BEZIER_KAPPA = 0.5523  # cubic-bezier approximation factor for a quarter circle


def _layout_metrics(
    size: int, position: str, style: str, play_res_y: int,
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
    scale = play_res_y / 1920.0
    scaled_size = round(size * scale)
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
        "scale": scale,
        "border_style": border_style,
        "outline": outline,
        "back_colour": back_colour,
        "pad_x": _BOLD_PAD_X_REF * scale,
        "pad_y": _BOLD_PAD_Y_REF * scale,
        "radius": _BOLD_RADIUS_REF * scale,
    }


def _ass_header(
    font_name: str,
    size: int = 110,
    position: str = DEFAULT_CAPTION_POSITION,
    style: str = DEFAULT_CAPTION_STYLE,
    color: str = DEFAULT_CAPTION_COLOR,
    play_res_x: int = DEFAULT_RESOLUTION[0],
    play_res_y: int = DEFAULT_RESOLUTION[1],
) -> str:
    m = _layout_metrics(size, position, style, play_res_y)
    alignment, margin_v, margin_lr = m["alignment"], m["margin_v"], m["margin_lr"]
    size, border_style, outline, back_colour = m["size"], m["border_style"], m["outline"], m["back_colour"]

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
            if not _token_is_faithful(span, t["text"]):
                raise ValueError(
                    f"unfaithful cleanup token: original={[w['word'] for w in span]!r} "
                    f"-> {t['text']!r}"
                )
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


def _llm_group_words_chunk(words: List[Dict], words_per_group: int) -> List[tuple]:
    """Single LLM grouping call over one chunk. Falls back to the mechanical
    `_group_boundaries` heuristic (for just this chunk) on any failure."""
    if not words:
        return []
    cache_key = _llm_cache_key("group_v1", {"words": words, "words_per_group": words_per_group})
    cached = _llm_cache_get(cache_key)
    if cached is not None:
        return [tuple(g) for g in cached["groups"]]
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
        _llm_cache_set(cache_key, {"groups": groups})
        return groups
    except Exception as e:
        print(f"[subtitles] LLM grouping pass failed ({e!r}); using mechanical grouping.")
        return _group_boundaries(words, words_per_group)


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
    return groups


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
) -> Tuple[float, float, float, float]:
    """Top-left (x, y) and (width, height) of the hand-drawn box for a group
    whose measured text block is block_width x block_height, positioned the
    same way libass would position that text under the given alignment/
    margin -- box_x/box_y are always computed from alignment+margin (never
    text_align's MarginL/MarginR asymmetry) because _ass_header always emits
    equal MarginL/MarginR, so horizontal centering is always PlayResX/2."""
    box_w = block_width + 2 * pad_x
    box_h = block_height + 2 * pad_y
    box_x = (play_res_x - box_w) / 2
    if alignment == 8:  # top
        box_y = margin_v
    elif alignment == 5:  # middle
        box_y = (play_res_y - box_h) / 2
    else:  # bottom (2)
        box_y = play_res_y - margin_v - box_h
    return box_x, box_y, box_w, box_h


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

    layout = _layout_metrics(size, position, style, play_res_y)
    pil_font = _pil_font(font_name, layout["size"])
    max_text_width = play_res_x - 2 * layout["margin_lr"] - 2 * layout["pad_x"]

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
        # Captions must render on ONE line -- a group whose text doesn't fit
        # the available width at this font size gets re-split at word
        # boundaries into consecutive one-line groups instead of wrapping
        # (wrapping either looked like N stacked lines with no box, or forced
        # the "bold" box to stack lines with awkward per-line word counts on
        # narrower/vertical canvases). Word timing is untouched -- each new
        # sub-group is just a smaller slice of the same word list.
        groups = _split_groups_to_fit(groups, words, pil_font, max_text_width, strip_punctuation)
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
            )
            first_idx, last_idx = idxs[0], idxs[-1]
            box_start = _ass_timestamp(words[first_idx]["start"])
            next_start = _display_end(last_idx)
            box_raw_end = next_start if next_start is not None else max(words[last_idx]["end"], words[last_idx]["start"] + 0.05)
            box_end = _ass_timestamp(max(box_raw_end, words[first_idx]["start"] + 0.001))
            path = _rounded_rect_path(box_w, box_h, layout["radius"])
            box_tags = f"{{\\an7\\pos({box_x:.1f},{box_y:.1f})\\1c&H000000&\\1a&H80&\\bord0\\shad0\\p1}}"
            lines.append(f"Dialogue: 0,{box_start},{box_end},{STYLE_BOX},,0,0,0,,{box_tags}{path}{{\\p0}}\n")

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
        lines.append(f"Dialogue: {layer},{start},{end},{ass_style},,0,0,0,,{display}\n")
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
