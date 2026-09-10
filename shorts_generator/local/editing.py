"""Silence and filler-word ("muletilla") removal for local clips.

Everything here operates in clip-relative time -- the same coordinate space
`subtitles.words_for_window()` already returns (0 = start of the highlight
window). Callers are expected to:

  1. Get the window's words via `words_for_window()` (BEFORE cropping).
  2. Call `find_cut_segments()` to get a list of (start, end) spans to
     remove -- deterministic silence gaps plus (optionally) LLM-flagged
     filler words.
  3. Pass those segments to `clipper.crop_clip_local(..., cut_segments=...)`
     (or the split-screen equivalent) so the physical cut happens in ONE
     ffmpeg pass alongside the outer [start_time, end_time] crop.
  4. Call `remap_words_across_cuts()` on the same word list to shift
     `start`/`end` onto the post-cut timeline before burning captions --
     words that fell inside a removed segment are dropped entirely, which
     is also how cut filler words disappear from the captions with no
     separate text-side filtering needed.

`word_start`/`word_end` (the original absolute source-video timestamps used
by the editor's manual caption_time_overrides lookup) are never touched --
only `start`/`end` are remapped.
"""
from typing import Dict, List, Optional, Tuple

from .llm import call_local_llm
from .subtitles import _extract_json_loose

# Gaps between words shorter than this are left alone -- normal speech
# pacing, breathing room between sentences, etc.
SILENCE_MIN_GAP = 0.5
# A flagged gap is trimmed down to this much (not to zero), split across
# both sides of the cut, so the join doesn't feel like an unnaturally hard
# splice -- a quick jump-cut, not a mid-word slam.
SILENCE_KEEP_GAP = 0.15
# Small padding added around LLM-flagged filler-word spans before cutting,
# so the cut doesn't clip the tail/head of the word right before/after it.
FILLER_PAD = 0.04


def find_silence_cuts(
    words: List[Dict],
    duration: float,
    min_gap: float = SILENCE_MIN_GAP,
    keep_gap: float = SILENCE_KEEP_GAP,
) -> List[Tuple[float, float]]:
    """Flag dead air longer than `min_gap` -- leading silence before the
    first word, trailing silence after the last word, and internal gaps
    between consecutive words -- and return the (start, end) clip-relative
    spans to cut, trimmed to leave `keep_gap` of padding rather than
    removing the pause entirely."""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: w["start"])
    cuts = []

    first_start = sorted_words[0]["start"]
    if first_start > min_gap:
        cuts.append((0.0, max(0.0, first_start - keep_gap)))

    for a, b in zip(sorted_words, sorted_words[1:]):
        gap_start, gap_end = a["end"], b["start"]
        if gap_end - gap_start > min_gap:
            cut_start = gap_start + keep_gap / 2
            cut_end = gap_end - keep_gap / 2
            if cut_end > cut_start:
                cuts.append((cut_start, cut_end))

    last_end = sorted_words[-1]["end"]
    if duration - last_end > min_gap:
        cuts.append((last_end + keep_gap, duration))

    return cuts


FILLER_PROMPT = """Sos un editor de video que corta muletillas y relleno vacío de transcripciones para redes sociales (estilo TikTok/Reels), en español rioplatense. Te paso una lista numerada de palabras de un fragmento de podcast, con su índice.

Tu tarea es identificar SOLO palabras o frases cortas que son puro relleno vocal sin ningún aporte de sentido: muletillas de disfluencia como "eh", "este", "o sea" (cuando es un tic verbal sin función, no cuando conecta una idea), "viste", "digo" (cuando es autocorrección sin agregar información), "tipo" (como muletilla, no cuando significa "clase de"), repeticiones inmediatas de la misma palabra ("yo yo creo"), y arranques de frase vacíos ("bueno... bueno, este...").

MUY IMPORTANTE - SÉ CONSERVADOR. Ante la duda, NO la marques:
- Si la palabra o frase cumple una función gramatical o semántica real en la oración (conecta ideas, aporta información, es parte de una expresión con sentido), NO la marques, aunque sea una palabra que a veces funciona como muletilla.
- "o sea que" casi siempre conecta una conclusión con lo anterior y tiene sentido -- NO la marques salvo que sea un tic aislado sin ninguna conexión lógica real.
- NO marques nada que, si se saca, cambie el significado o rompa la gramática de lo que queda.
- NO marques pausas, respiraciones ni silencios -- eso se maneja aparte, esto es solo sobre palabras.
- Si tenés dudas, dejala sin marcar.

Respondé ÚNICAMENTE con JSON válido, sin texto adicional:
{{"cuts": [[3, 4], [12, 14]]}}

Donde cada par [start, end) es un rango de índices de palabras a cortar (igual que un slice de Python), start inclusive, end exclusive. Si no hay nada para cortar, respondé {{"cuts": []}}.

Palabras:
{numbered_words}
"""


def find_filler_cuts(words: List[Dict], pad: float = FILLER_PAD) -> List[Tuple[float, float]]:
    """LLM pass over the window's words flagging filler/disfluency spans to
    cut. Conservative by design (see FILLER_PROMPT) -- falls back to
    cutting nothing on any failure (bad response, network error, etc.) so a
    transient hiccup never breaks a render."""
    if not words:
        return []
    try:
        numbered = "\n".join(f"{i}: {w['word']}" for i, w in enumerate(words))
        raw = call_local_llm(FILLER_PROMPT.format(numbered_words=numbered))
        raw_cuts = _extract_json_loose(raw)["cuts"]

        cuts = []
        for pair in raw_cuts:
            i0, i1 = int(pair[0]), int(pair[1])
            if not (0 <= i0 < i1 <= len(words)):
                raise ValueError(f"cut index range {pair} out of bounds for {len(words)} words")
            seg_start = words[i0]["start"] - pad
            if i0 > 0:
                # never bite into the previous word's tail when there's no
                # natural gap (back-to-back words) -- otherwise the padding
                # can shave off just enough to drop that whole word on remap
                seg_start = max(seg_start, words[i0 - 1]["end"])
            seg_end = words[i1 - 1]["end"] + pad
            if i1 < len(words):
                seg_end = min(seg_end, words[i1]["start"])
            cuts.append((max(0.0, seg_start), seg_end))
        return cuts
    except Exception as e:
        print(f"[editing] filler LLM pass failed ({e!r}); cutting nothing.")
        return []


def merge_cut_segments(segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sort and merge overlapping/touching (start, end) spans into a
    minimal, non-overlapping list."""
    if not segments:
        return []
    ordered = sorted(segments)
    merged = [list(ordered[0])]
    for s, e in ordered[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def keep_segments_from_cuts(
    cut_segments: List[Tuple[float, float]], duration: float
) -> List[Tuple[float, float]]:
    """Complement of (merged) cut_segments within [0, duration] -- the spans
    that survive the cut, in original clip-relative time. Empty when the
    whole clip is cut away (degenerate, but handled)."""
    cuts = merge_cut_segments(cut_segments)
    keep = []
    cursor = 0.0
    for s, e in cuts:
        s = max(0.0, min(s, duration))
        e = max(0.0, min(e, duration))
        if s > cursor:
            keep.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        keep.append((cursor, duration))
    return keep


def remap_time(t: float, keep_segments: List[Tuple[float, float]]) -> Optional[float]:
    """Map an original clip-relative timestamp onto the post-cut timeline.
    Returns None if `t` falls inside a removed segment (i.e. isn't part of
    any kept segment)."""
    cum = 0.0
    for s, e in keep_segments:
        if t < s:
            return None
        if t <= e:
            return cum + (t - s)
        cum += e - s
    return None


def remap_words_across_cuts(
    words: List[Dict], keep_segments: List[Tuple[float, float]]
) -> List[Dict]:
    """Shift each word's `start`/`end` onto the post-cut timeline, dropping
    words that fall (even partially) inside a removed segment -- this is
    also how cut filler words vanish from the burned captions, with no
    separate text-side filtering step. `word_start`/`word_end` (used by the
    editor's manual caption_time_overrides lookup) are left untouched."""
    if not keep_segments:
        return list(words)
    out = []
    for w in words:
        new_start = remap_time(w["start"], keep_segments)
        new_end = remap_time(w["end"], keep_segments)
        if new_start is None or new_end is None or new_end <= new_start:
            continue
        nw = dict(w)
        nw["start"] = new_start
        nw["end"] = new_end
        out.append(nw)
    return out


def find_cut_segments(
    words: List[Dict],
    duration: float,
    remove_silence: bool = True,
    remove_fillers: bool = True,
) -> List[Tuple[float, float]]:
    """Convenience entry point: run the deterministic silence pass and/or
    the LLM filler pass and merge their results into one cut-segment list,
    ready for `clipper.crop_clip_local(..., cut_segments=...)` and
    `remap_words_across_cuts()`."""
    segments = []
    if remove_silence:
        segments += find_silence_cuts(words, duration)
    if remove_fillers:
        segments += find_filler_cuts(words)
    return merge_cut_segments(segments)
