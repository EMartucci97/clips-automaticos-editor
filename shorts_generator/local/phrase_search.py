"""Find where a specific phrase is said in a transcript, with context.

Deterministic substring search (no LLM call) — used by the dashboard's
"clip específico" mode, as opposed to highlights.get_highlights which asks
an LLM to pick topics/moments.
"""
import unicodedata
from typing import Dict, List


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip()


def find_phrase_clips(
    transcript: Dict,
    phrase: str,
    context_before: float = 8.0,
    context_after: float = 12.0,
) -> List[Dict]:
    """Search transcript segments for `phrase` (accent/case-insensitive).

    Also checks each pair of adjacent segments concatenated, so a phrase
    split across a segment boundary by Whisper still matches. Returns one
    highlight-shaped dict per match, windowed by `context_before`/`after`
    seconds and merged if overlapping.
    """
    needle = _normalize(phrase)
    if not needle:
        return []

    segments = transcript.get("segments", [])
    duration = transcript.get("duration", segments[-1]["end"] if segments else 0)

    windows: List[Dict] = []
    for i, seg in enumerate(segments):
        haystack = _normalize(seg["text"])
        matched_text = seg["text"].strip()
        found = needle in haystack

        end_seg = seg
        if not found and i + 1 < len(segments):
            nxt = segments[i + 1]
            combined = _normalize(seg["text"] + " " + nxt["text"])
            if needle in combined:
                found = True
                end_seg = nxt
                matched_text = f"{seg['text'].strip()} {nxt['text'].strip()}"

        if not found:
            continue

        start = max(0.0, seg["start"] - context_before)
        end = end_seg["end"] + context_after
        if duration:
            end = min(end, duration)
        if end <= start:
            continue

        windows.append(
            {
                "title": f"Frase: \"{phrase.strip()}\"",
                "start_time": start,
                "end_time": end,
                "score": 100,
                "sales_angle_score": 0,
                "sales_angle": "",
                "combined_score": 100,
                "hook_sentence": matched_text,
                "virality_reason": "Coincide con la frase buscada",
            }
        )

    return _merge_overlapping(windows)


def _merge_overlapping(windows: List[Dict]) -> List[Dict]:
    """Merge windows whose time ranges overlap into a single wider clip."""
    if not windows:
        return []
    windows = sorted(windows, key=lambda w: w["start_time"])
    merged = [windows[0]]
    for w in windows[1:]:
        last = merged[-1]
        if w["start_time"] <= last["end_time"]:
            last["end_time"] = max(last["end_time"], w["end_time"])
            last["hook_sentence"] = last["hook_sentence"] + " (...) " + w["hook_sentence"]
        else:
            merged.append(w)
    return merged
