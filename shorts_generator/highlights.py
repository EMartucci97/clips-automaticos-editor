"""Find the most viral-worthy highlights in a transcript.

Logic ported from ViralVadoo's transcript_analysis/highlight_generator.py:
  - content-type / density detection
  - chunking for long videos with overlap
  - virality-criteria prompt
  - score-based dedupe with overlap suppression

The LLM call is pluggable via the `llm_fn` argument so the same prompts can
drive either MuAPI (default, --mode api) or a direct local LLM client
(--mode local).
"""
import json
import re
from typing import Callable, Dict, List, Optional

from . import muapi
from .config import SALES_ANGLE_MIN_SCORE, SALES_ANGLE_WEIGHT, SALES_ANGLES_ENABLED


LLMFn = Callable[[str], str]


CONTENT_TYPE_PROMPT = """Analyze this video transcript sample and classify the content type.
Choose one: podcast, interview, tutorial, lecture, commentary, debate, vlog, other.
Also estimate content density: low (mostly filler/chit-chat), medium, or high (dense info/stories).
Respond with JSON only: {"content_type": "...", "density": "..."}"""


VIRALITY_CRITERIA = """
Virality signals to prioritize (ranked by impact):
1. HOOK MOMENTS — statements that create immediate curiosity ("The secret is...", "Nobody talks about...", "I was completely wrong about...")
2. EMOTIONAL PEAKS — genuine surprise, laughter, anger, vulnerability, excitement; raw unscripted reactions
3. OPINION BOMBS — strong, polarizing or counter-intuitive statements that trigger agree/disagree
4. REVELATION MOMENTS — surprising facts, stats, or confessions that reframe how the viewer thinks
5. CONFLICT/TENSION — disagreement, pushback, or a problem being confronted head-on
6. QUOTABLE ONE-LINERS — a sentence that works as a standalone quote card
7. STORY PEAKS — the climax or twist of an anecdote; the payoff moment
8. PRACTICAL VALUE — a concrete tip, hack, or insight the viewer can immediately apply
"""


SALES_ANGLES = """
Proven sales angles (Ruarte Reports — technical analysis / wealth management education, target avatar: investors with $100k-$3M already invested, 40-65yo, who do NOT identify as traders and want criteria/structure/timing/capital protection, not lifestyle promises):
1. ECONOMIC CYCLE AWARENESS — why recognizing the cycle matters (getting stuck in a bad investment for years/decades, e.g. buying a top and never recovering)
2. SP500 / MAINSTREAM INDEX AS A BAD BET RIGHT NOW — an index or asset that already ran hard is a mature-cycle trap, not an opportunity
3. WEAK DOLLAR → ROTATION INTO EMERGING MARKETS / LATAM / COMMODITIES — macro thesis connecting currency weakness to where value migrates next
4. UNDERPERFORMING REGION/ASSET AS A HISTORIC 5-YEAR OPPORTUNITY — a multi-year underperformance setting up a mean-reversion entry
5. ELLIOTT WAVE / CYCLE-BASED TIMING — anticipating tops/bottoms ahead of the crowd, ignoring news noise
6. AUTHORITY / TRACK RECORD — decades of experience, calling a crash or bottom correctly, managing large portfolios, credentials — stated as evidence, not bragging
7. OBJECTION HANDLING IN THE MOMENT — "do you guarantee results", "I got scammed before", "I don't have time", "I want to live off trading" answered directly and firmly
8. FEARS OF THE WEALTHY-INVESTOR AVATAR — no time to manage it themselves, fear of missing the next opportunity, fear of trusting again after being burned, bad timing/entering late
9. NAMED ENEMIES — gurus/"smart money" influencers who sell hype with no real track record, lifestyle-flexing educators with no demonstrable results
10. SOCIAL PROOF / COMMUNITY PROOF — documented calls, real results shown, a real community/team behind the claims (not just promises)
"""


HIGHLIGHT_SYSTEM_PROMPT = """You are an elite short-form video editor who has studied thousands of viral clips on TikTok, Instagram Reels, and YouTube Shorts. You know exactly what makes viewers stop scrolling, watch to the end, and share.

{virality_criteria}
{sales_angle_criteria}

Content type: {content_type} | Density: {density}

Your task: identify the most viral-worthy highlights from the transcript.

Rules:
- Every highlight must open with a strong HOOK — a line that grabs attention within the first 3 seconds
- start_time and end_time must bound the ENTIRE clip, from the hook line all the way through the full thought/story — NOT just the timestamp of the hook sentence itself. A highlight that is only the hook line's own 2-4 second span is WRONG.
- Duration sweet spot: 45-90 seconds, measured as end_time minus start_time. Go shorter (20-44s) only for a perfect standalone one-liner. Go longer (91-180s) only when a story arc needs full context to land. Never return end_time - start_time under 20 seconds.
- Never cut mid-sentence or mid-thought — each clip must feel complete and self-contained
- Clips must not overlap significantly with each other
- Score 0-100 on viral potential (not general quality) — call this "score"
- {sales_angle_instruction}
- {num_clips_instruction}
- For each highlight, identify the single best "hook_sentence" — the opening line that would make someone stop scrolling
- Explain in one sentence why this clip is viral ("virality_reason")

Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","virality_reason":"string"{sales_angle_json_fields}}}]}}"""


CHUNK_SIZE_SECONDS = 1200       # 20-min chunks for long videos
LONG_VIDEO_THRESHOLD = 1800     # chunk videos longer than 30 min
CHUNK_OVERLAP_SECONDS = 60
MIN_HIGHLIGHT_DURATION_SECONDS = 15  # below the prompt's stated 20-44s floor; almost always a model error
GPT_CALL_TIMEOUT_SECONDS = 300  # cap LLM polls at 5 min — a wedged call should fail fast
MAX_HIGHLIGHT_API_ATTEMPTS = 3


def call_muapi_llm(prompt: str) -> str:
    """Default LLM backend: MuAPI gpt-5-mini."""
    result = muapi.run(
        "gpt-5-mini",
        {"prompt": prompt},
        label="gpt-5-mini",
        timeout=GPT_CALL_TIMEOUT_SECONDS,
    )

    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], str) and outputs[0].strip():
        return outputs[0]

    for key in ("output", "text", "response", "result", "content"):
        v = result.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            inner = v.get("text") or v.get("content")
            if isinstance(inner, str) and inner.strip():
                return inner
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0]

    raise RuntimeError(f"Could not extract gpt-5-mini text from response: {result}")


def _parse_json_loose(raw: str) -> Dict:
    """gpt-5-4 sometimes wraps JSON in markdown fences — strip and parse."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _sanitize_highlights(raw_highlights: object, duration: float) -> List[Dict]:
    """Normalize model output into the expected shape; skip invalid entries."""
    if not isinstance(raw_highlights, list):
        return []

    max_end = duration if duration > 0 else float("inf")
    cleaned: List[Dict] = []
    for item in raw_highlights:
        if not isinstance(item, dict):
            continue

        start = _coerce_float(item.get("start_time"), default=-1.0)
        end = _coerce_float(item.get("end_time"), default=-1.0)
        if start < 0 or end <= start:
            continue

        if max_end != float("inf"):
            start = min(start, max_end)
            end = min(end, max_end)
            if end <= start:
                continue

        if end - start < MIN_HIGHLIGHT_DURATION_SECONDS:
            continue

        score = max(0, min(100, _coerce_int(item.get("score"), default=0)))
        sales_angle_score = max(0, min(100, _coerce_int(item.get("sales_angle_score"), default=0)))
        weight = SALES_ANGLE_WEIGHT if SALES_ANGLES_ENABLED else 0.0
        combined_score = score * (1 - weight) + sales_angle_score * weight

        cleaned.append(
            {
                "title": str(item.get("title") or "Untitled Highlight").strip(),
                "start_time": start,
                "end_time": end,
                "score": score,
                "sales_angle_score": sales_angle_score,
                "sales_angle": str(item.get("sales_angle") or "").strip(),
                "combined_score": combined_score,
                "hook_sentence": str(item.get("hook_sentence") or "").strip(),
                "virality_reason": str(item.get("virality_reason") or "").strip(),
            }
        )

    return cleaned


def detect_content_type(transcript: Dict, llm_fn: LLMFn = call_muapi_llm) -> Dict[str, str]:
    segments = transcript.get("segments", [])
    sample = " ".join(s["text"] for s in segments[:25])[:3000]
    prompt = f"{CONTENT_TYPE_PROMPT}\n\nTranscript sample:\n{sample}"
    try:
        raw = llm_fn(prompt)
        return _parse_json_loose(raw)
    except Exception:
        return {"content_type": "other", "density": "medium"}


def build_transcript_text(transcript: Dict) -> str:
    segments = transcript.get("segments", [])
    return "\n".join(f"[{s['start']:.1f}s] {s['text'].strip()}" for s in segments)


def chunk_transcript(transcript: Dict) -> List[Dict]:
    segments = transcript.get("segments", [])
    duration = transcript.get("duration", segments[-1]["end"] if segments else 0)
    chunks = []
    start = 0
    while start < duration:
        end = min(start + CHUNK_SIZE_SECONDS, duration)
        chunk_segs = [
            s for s in segments
            if s["start"] >= start and s["end"] <= end + CHUNK_OVERLAP_SECONDS
        ]
        if chunk_segs:
            # Rewrite timestamps relative to the chunk start so the LLM sees a
            # 0-based transcript (matching chunk["duration"]) instead of the
            # video's absolute timeline — the offset is re-added once the LLM
            # response comes back, in get_highlights().
            relative_segs = [
                {**s, "start": s["start"] - start, "end": s["end"] - start}
                for s in chunk_segs
            ]
            chunk = dict(transcript)
            chunk["segments"] = relative_segs
            chunk["duration"] = end - start
            chunk["_offset"] = start
            chunks.append(chunk)
        start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
    return chunks


def call_highlight_api(
    transcript_text: str,
    content_info: Dict,
    duration: float,
    num_clips: int,
    is_chunk: bool = False,
    llm_fn: LLMFn = call_muapi_llm,
) -> Dict:
    # Ask for ~2× the user's target so dedupe has headroom, but cap so the model
    # doesn't have to generate a huge JSON payload (which times out gpt-5-mini).
    target = max(num_clips * 2, 5)
    natural_max = max(2 if is_chunk else 3, int(duration / 90))
    min_clips = min(target, natural_max, 8)
    if SALES_ANGLES_ENABLED:
        sales_angle_criteria = SALES_ANGLES
        sales_angle_instruction = (
            'ALSO score 0-100 how strongly this moment lands one of the sales angles above — call this '
            '"sales_angle_score". If none apply, use 0. Name which angle it is in "sales_angle" '
            '(one short label, or "" if none)'
        )
        sales_angle_json_fields = ',"sales_angle_score":int,"sales_angle":"string"'
    else:
        sales_angle_criteria = ""
        sales_angle_instruction = ""
        sales_angle_json_fields = ""
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        sales_angle_criteria=sales_angle_criteria,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        sales_angle_instruction=sales_angle_instruction,
        num_clips_instruction=f"Generate at least {min_clips} highlights",
        sales_angle_json_fields=sales_angle_json_fields,
    )
    system = "\n".join(line for line in system.splitlines() if line.strip() != "-")
    base_prompt = f"{system}\n\nTranscript:\n{transcript_text}"
    prompt = base_prompt
    last_error = "unknown"

    for attempt in range(1, MAX_HIGHLIGHT_API_ATTEMPTS + 1):
        raw = llm_fn(prompt)
        try:
            parsed = _parse_json_loose(raw)
            highlights = _sanitize_highlights(parsed.get("highlights"), duration=duration)
            if highlights:
                return {"highlights": highlights}
            last_error = "no valid highlights in response"
        except Exception as e:
            last_error = str(e)

        if attempt < MAX_HIGHLIGHT_API_ATTEMPTS:
            print(
                f"[highlights] invalid model output on attempt {attempt}/{MAX_HIGHLIGHT_API_ATTEMPTS}; retrying",
                flush=True,
            )
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Return ONLY valid JSON with a top-level 'highlights' array."
                + " Each item must include: title, start_time, end_time, score, hook_sentence, virality_reason."
                + " No markdown fences, no commentary."
                + " Your previous response used end_time - start_time spans that were too short (just the hook"
                + " sentence's own timestamp). start_time/end_time must bound the WHOLE highlight clip —"
                + " end_time - start_time must be at least 20 seconds, ideally 45-90."
            )

    raise RuntimeError(
        f"Highlight generator produced invalid output after {MAX_HIGHLIGHT_API_ATTEMPTS} attempts: {last_error}"
    )


def _apply_sales_angle_fallback(highlights: List[Dict]) -> List[Dict]:
    """If no candidate clears SALES_ANGLE_MIN_SCORE, this video just doesn't
    have a clear sales angle — fall back to ranking by virality alone instead
    of blending in a near-zero sales angle score for every clip."""
    if not SALES_ANGLES_ENABLED or not highlights:
        return highlights

    best = max((h.get("sales_angle_score", 0) for h in highlights), default=0)
    if best >= SALES_ANGLE_MIN_SCORE:
        return highlights

    print(
        f"[highlights] no candidate clears the sales-angle threshold "
        f"(best sales_angle_score={best:.0f} < {SALES_ANGLE_MIN_SCORE:.0f}) — "
        "falling back to virality-only ranking for this video",
        flush=True,
    )
    for h in highlights:
        h["combined_score"] = h.get("score", 0)
        h["sales_angle_fallback"] = True
    return highlights


def dedupe_highlights(highlights: List[Dict]) -> List[Dict]:
    """Drop a highlight if it overlaps >50% with a higher-(combined-)scoring one already kept."""
    highlights = sorted(highlights, key=lambda x: float(x.get("combined_score", x.get("score", 0))), reverse=True)
    kept: List[Dict] = []
    for h in highlights:
        h_start = float(h["start_time"])
        h_end = float(h["end_time"])
        h_dur = h_end - h_start
        overlapping = False
        for k in kept:
            latest_start = max(h_start, float(k["start_time"]))
            earliest_end = min(h_end, float(k["end_time"]))
            overlap = earliest_end - latest_start
            if overlap > 0 and overlap > 0.5 * h_dur:
                overlapping = True
                break
        if not overlapping:
            kept.append(h)
    return kept


def get_highlights(
    transcript: Dict,
    num_clips: int = 3,
    llm_fn: Optional[LLMFn] = None,
) -> Dict:
    """Main entry point — returns {highlights: [...]} sorted by score.

    `llm_fn` swaps the underlying LLM. Defaults to MuAPI gpt-5-mini; local
    mode passes in a local LLM-backed callable.
    """
    llm_fn = llm_fn or call_muapi_llm
    duration = transcript.get("duration", 0)
    content_info = detect_content_type(transcript, llm_fn=llm_fn)
    print(f"[highlights] content={content_info.get('content_type')} density={content_info.get('density')} duration={duration:.0f}s", flush=True)

    if duration >= LONG_VIDEO_THRESHOLD:
        chunks = chunk_transcript(transcript)
        print(f"[highlights] long video — splitting into {len(chunks)} chunks", flush=True)
        all_highlights: List[Dict] = []
        for i, chunk in enumerate(chunks):
            offset = chunk.get("_offset", 0)
            text = build_transcript_text(chunk)
            print(f"[highlights] chunk {i + 1}/{len(chunks)} (offset {offset:.0f}s)", flush=True)
            result = call_highlight_api(text, content_info, chunk["duration"], num_clips=num_clips, is_chunk=True, llm_fn=llm_fn)
            for h in result.get("highlights", []):
                h["start_time"] = float(h["start_time"]) + offset
                h["end_time"] = float(h["end_time"]) + offset
                all_highlights.append(h)
        all_highlights = _apply_sales_angle_fallback(all_highlights)
        highlights = dedupe_highlights(all_highlights)
    else:
        text = build_transcript_text(transcript)
        result = call_highlight_api(text, content_info, duration, num_clips=num_clips, llm_fn=llm_fn)
        candidates = _apply_sales_angle_fallback(result.get("highlights", []))
        highlights = dedupe_highlights(candidates)

    return {"highlights": highlights}
