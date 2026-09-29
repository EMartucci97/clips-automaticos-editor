"""Step 3 (quality): WER against hand-corrected reference transcripts, plus a
difflib-based "edit fraction" proxy for how much a manual editor would have
to touch the output.

Two separate measurements:
  - WER + edit_fraction_raw: per TRANSCRIPTION config, comparing the raw
    transcript text to the reference. Measures the transcription step alone.
  - edit_fraction_cleaned: per (transcription, llm) combo, comparing the
    post-cleanup word text (output of run_grouping.py) to the reference.
    Measures what actually reaches a manual editor after both passes.

Both normalize with subtitles._normalize_for_compare -- the same function
production's _token_is_faithful guard uses -- so the number is comparable to
the fidelity bar the pipeline already enforces.

Requires hand-corrected reference files at
benchmark/reference/{clip_id}.reference.txt (plain text, no timestamps).
Clips without a reference file are silently skipped.
"""
import difflib
import json

import jiwer

from shorts_generator.local.subtitles import _normalize_for_compare

from .config import CAPTIONS_DIR, REFERENCE_DIR, TEST_CLIPS, TRANSCRIPTION_CONFIGS, TRANSCRIPTS_DIR, WORK_DIR


def _load_reference(clip_id: str):
    path = REFERENCE_DIR / f"{clip_id}.reference.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _transcript_text(transcript: dict) -> str:
    return " ".join(seg.get("text", "") for seg in transcript.get("segments", []))


def _words_text(words: list) -> str:
    return " ".join(w["word"] for w in words)


def _edit_fraction(ref_norm: str, hyp_norm: str):
    if not ref_norm or not hyp_norm:
        return None
    return 1 - difflib.SequenceMatcher(None, ref_norm, hyp_norm).ratio()


def score_transcription_wer() -> dict:
    results = {}
    for clip_id, _ in TEST_CLIPS:
        reference = _load_reference(clip_id)
        if reference is None:
            continue
        ref_norm = _normalize_for_compare(reference)
        for config_id in TRANSCRIPTION_CONFIGS:
            transcript_path = TRANSCRIPTS_DIR / config_id / f"{clip_id}.json"
            if not transcript_path.exists():
                continue
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            hyp_norm = _normalize_for_compare(_transcript_text(transcript))
            wer = jiwer.wer(ref_norm, hyp_norm) if ref_norm and hyp_norm else None
            results.setdefault(config_id, {})[clip_id] = {
                "wer": wer,
                "edit_fraction_raw": _edit_fraction(ref_norm, hyp_norm),
            }
    return results


def score_cleaned_edit_fraction() -> dict:
    results = {}
    if not CAPTIONS_DIR.exists():
        return results
    for clip_id, _ in TEST_CLIPS:
        reference = _load_reference(clip_id)
        if reference is None:
            continue
        ref_norm = _normalize_for_compare(reference)
        for combo_dir in sorted(CAPTIONS_DIR.iterdir()):
            if not combo_dir.is_dir():
                continue
            caption_path = combo_dir / f"{clip_id}.json"
            if not caption_path.exists():
                continue
            data = json.loads(caption_path.read_text(encoding="utf-8"))
            hyp_norm = _normalize_for_compare(_words_text(data.get("words", [])))
            results.setdefault(combo_dir.name, {})[clip_id] = {
                "edit_fraction_cleaned": _edit_fraction(ref_norm, hyp_norm),
            }
    return results


def main():
    if not any((REFERENCE_DIR).glob("*.reference.txt")):
        print(f"No reference files found in {REFERENCE_DIR} -- nothing to score.")
        print("Add {clip_id}.reference.txt files (hand-corrected transcripts) and re-run.")
        return

    transcription_scores = score_transcription_wer()
    cleaned_scores = score_cleaned_edit_fraction()
    out = {"transcription": transcription_scores, "cleaned": cleaned_scores}
    out_path = WORK_DIR / "wer_scores.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    for config_id, per_clip in transcription_scores.items():
        for clip_id, scores in per_clip.items():
            wer = scores["wer"]
            ef = scores["edit_fraction_raw"]
            print(f"  {config_id} / {clip_id}: WER={wer:.3f} edit_fraction_raw={ef:.3f}" if wer is not None else f"  {config_id} / {clip_id}: (no data)")

    for combo_id, per_clip in cleaned_scores.items():
        for clip_id, scores in per_clip.items():
            ef = scores["edit_fraction_cleaned"]
            print(f"  {combo_id} / {clip_id}: edit_fraction_cleaned={ef:.3f}" if ef is not None else f"  {combo_id} / {clip_id}: (no data)")


if __name__ == "__main__":
    main()
