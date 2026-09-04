"""LLM pass that catches likely Whisper ASR errors (dropped acronym letters,
garbled financial jargon) before the transcript feeds highlight selection and
clipping. Runs once per video, right after transcribe_local(); results are
cached alongside the .srt/.words.json sidecars so re-runs are free.

Deliberately narrow: it only fixes words a human would recognize as
mis-heard (e.g. "fore mexicana" -> "AFORE mexicana"). It must never touch
grammar, filler words, repetitions, or anything that changes what was said.
"""
import difflib
import json
from pathlib import Path
from typing import Callable, Dict, List

from ..config import LOCAL_OUTPUT_DIR, LOCAL_WHISPER_INITIAL_PROMPT, TRANSCRIPT_REVIEW_ENABLED
from ..highlights import _parse_json_loose
from .transcriber import _words_cache_path, _write_srt_cache

LLMFn = Callable[[str], str]

SEGMENTS_PER_CALL = 60

REVIEW_SYSTEM_PROMPT = """Sos un corrector de transcripciones automáticas (Whisper) de contenido \
financiero en español, para Ruarte Reports (mercados, inversión, Argentina/LatAm).

Vocabulario de referencia del canal: {glossary}

Tarea: te paso una lista numerada de segmentos de una transcripción automática. Tu ÚNICO trabajo es \
detectar errores de reconocimiento de voz evidentes — típicamente siglas financieras que perdieron una \
letra inicial o quedaron mal formadas (ej: "fore mexicana" -> "AFORE mexicana", "FP peruana" -> "AFP \
peruana"), o términos de jerga financiera/mercados irreconocibles que claramente son otra palabra mal \
transcripta.

Reglas estrictas:
- NO corrijas gramática, puntuación, muletillas, repeticiones de palabras, ni nada de estilo.
- NO cambies el significado de la frase. Si la palabra rara podría ser lo que la persona realmente dijo \
(dudó, repitió, usó una expresión coloquial), DEJALA IGUAL.
- Solo tocá una palabra si estás razonablemente seguro de que es un error de reconocimiento de voz, no \
una elección real del hablante.
- Devolvé el segmento corregido COMPLETO (no solo la palabra cambiada), pero SOLO con las palabras de ESE \
segmento — nunca copies ni completes con palabras de un segmento anterior o siguiente, aunque la oración \
siga cortada entre segmentos. La cantidad de palabras del segmento corregido tiene que ser prácticamente \
igual a la del original (a lo sumo cambia 1 o 2 palabras).
- Si ningún segmento necesita corrección, devolvé una lista vacía.

Segmentos:
{segments}

Respondé SOLO JSON con este formato exacto, sin texto adicional ni markdown:
{{"corrections": [{{"i": <índice del segmento, entero>, "text": "<texto completo corregido>", "why": "<motivo breve>"}}]}}
Incluí en "corrections" únicamente los segmentos que cambiaste."""


def _review_cache_path(media_path: str) -> Path:
    cache_dir = Path(LOCAL_OUTPUT_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (Path(media_path).stem + ".review.json")


def _align_words(old_words: List[Dict], old_text: str, new_text: str) -> List[Dict]:
    """Re-derive word-level timestamps for a corrected segment text by
    diffing old vs new tokens. Equal/replace blocks of matching length keep
    the original timestamps (just swap the word string); anything trickier
    (a fix that changes token count) borrows the nearest original word's
    timestamps rather than inventing new ones."""
    old_tokens = old_text.split()
    new_tokens = new_text.split()
    if not old_words or len(old_tokens) != len(old_words):
        # Word list doesn't line up with the text it supposedly came from —
        # bail out and just keep the original words (timestamps still valid,
        # subtitles may show the old spelling for the touched word).
        return old_words

    matcher = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    new_words: List[Dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            new_words.extend(old_words[i1:i2])
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            for k, idx in enumerate(range(i1, i2)):
                w = dict(old_words[idx])
                w["word"] = new_tokens[j1 + k]
                new_words.append(w)
        else:
            # token count changed (split/merge) — approximate by borrowing
            # the timestamp span of the original block for every new token.
            span_start = old_words[i1]["start"] if i1 < len(old_words) else old_words[-1]["end"]
            span_end = old_words[i2 - 1]["end"] if i2 - 1 < len(old_words) else span_start
            n = max(1, j2 - j1)
            step = (span_end - span_start) / n
            for k, tok in enumerate(new_tokens[j1:j2]):
                new_words.append({
                    "start": span_start + step * k,
                    "end": span_start + step * (k + 1),
                    "word": tok,
                })
    return new_words


def review_transcript(media_path: str, transcript: Dict, llm_fn: LLMFn) -> Dict:
    """Best-effort ASR-error review. Mutates and returns `transcript` with
    corrected segment text (and word timestamps where possible). Caches its
    output next to the .srt/.words.json sidecars so repeat runs on the same
    source skip the LLM call entirely."""
    if not TRANSCRIPT_REVIEW_ENABLED:
        return transcript

    segments = transcript.get("segments", [])
    if not segments:
        return transcript

    cache_path = _review_cache_path(media_path)
    if cache_path.exists():
        try:
            cached_corrections = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"[review] reusing cached review: {cache_path} ({len(cached_corrections)} correction(s))", flush=True)
        except (json.JSONDecodeError, OSError):
            cached_corrections = None
    else:
        cached_corrections = None

    if cached_corrections is None:
        cached_corrections = []
        for start in range(0, len(segments), SEGMENTS_PER_CALL):
            chunk = segments[start:start + SEGMENTS_PER_CALL]
            numbered = "\n".join(f"{start + i}: {s.get('text', '').strip()}" for i, s in enumerate(chunk))
            prompt = REVIEW_SYSTEM_PROMPT.format(
                glossary=LOCAL_WHISPER_INITIAL_PROMPT or "(sin glosario configurado)",
                segments=numbered,
            )
            try:
                raw = llm_fn(prompt)
                parsed = _parse_json_loose(raw)
                for c in parsed.get("corrections", []):
                    i = int(c["i"])
                    if 0 <= i < len(segments) and c.get("text", "").strip():
                        cached_corrections.append({"i": i, "text": c["text"].strip(), "why": c.get("why", "")})
            except Exception as e:
                print(f"[review] LLM call failed for segments {start}-{start + len(chunk)}: {e}", flush=True)
        cache_path.write_text(json.dumps(cached_corrections, ensure_ascii=False, indent=2), encoding="utf-8")

    if not cached_corrections:
        print("[review] no ASR errors detected, transcript unchanged", flush=True)
        return transcript

    changed = False
    for c in cached_corrections:
        seg = segments[c["i"]]
        old_text = seg.get("text", "").strip()
        new_text = c["text"]
        if old_text == new_text:
            continue
        old_tokens, new_tokens = old_text.split(), new_text.split()
        max_delta = max(2, round(0.2 * len(old_tokens)))
        if abs(len(new_tokens) - len(old_tokens)) > max_delta:
            # A real ASR fix (dropped letter, one garbled word) barely moves
            # the word count. A big jump usually means the model bled in
            # text from a neighboring segment — reject rather than corrupt
            # the transcript with duplicated content.
            print(
                f"[review] segment {c['i']}: rejected suspicious correction "
                f"({len(old_tokens)} -> {len(new_tokens)} words) — \"{old_text}\" -> \"{new_text}\"",
                flush=True,
            )
            continue
        print(f"[review] segment {c['i']}: \"{old_text}\" -> \"{new_text}\" ({c.get('why', '')})", flush=True)
        seg["words"] = _align_words(seg.get("words", []), old_text, new_text)
        seg["text"] = new_text
        changed = True

    if changed:
        # Keep the .srt/.words.json sidecars in sync with what the rest of
        # the pipeline (and anyone reading the .srt by hand) actually sees.
        _write_srt_cache(media_path, transcript)
        _words_cache_path(media_path).write_text(
            json.dumps([s.get("words", []) for s in segments], ensure_ascii=False), encoding="utf-8"
        )
        print(f"[review] applied {len(cached_corrections)} correction(s), sidecars updated", flush=True)
    else:
        print(f"[review] {len(cached_corrections)} correction(s) already applied from a previous run", flush=True)

    return transcript
