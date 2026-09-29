"""Step 4: final cost/quality report.

Aggregates everything the earlier steps wrote to benchmark/work/ into one
table, one row per (transcription_config, llm_config) combo actually run
(i.e. one row per subdirectory under work/captions/). Metrics are averaged
across the test clips in TEST_CLIPS.

Cost/quality ratio: costo_estimado_por_video_usd / (1 - fraccion_edicion).
Lower is better -- cheaper AND less manual editing wins. fraccion_edicion
uses edit_fraction_cleaned (post grouping/cleanup, what actually reaches a
manual editor) when a reference transcript exists for scoring, falling back
to edit_fraction_raw (transcription-only) when the LLM step wasn't scored
separately, and is left blank (ratio not computed) when neither is
available -- a WER/edit-fraction number requires a hand-corrected reference
(see score_wer.py), so combos run before the user finishes correcting
reference transcripts will show BLOCKED for these columns rather than a
misleading number.

Cost is projected to REFERENCE_CLIP_SECONDS (60s, the user's stated typical
final-clip length) by scaling each combo's real per-clip cost linearly with
audio duration -- a reasonable approximation since both transcription cost
(per-minute pricing) and LLM cost (roughly proportional to word count/audio
length) scale with duration.
"""
import csv
import json

from shorts_generator.local.subtitles import _DANGLING_CONNECTORS

from .config import (
    CAPTIONS_DIR,
    REFERENCE_CLIP_SECONDS,
    REVENUE_PER_VIDEO_USD,
    TEST_CLIPS,
    TRANSCRIPTS_DIR,
    WORK_DIR,
)

GROUP_MIN_WORDS = 2
GROUP_MAX_WORDS = 6

COLUMNS = [
    "config_id", "modelo_transcripcion", "config_llm", "WER", "fraccion_edicion",
    "conectores_colgantes", "grupos_fuera_de_rango", "tokens_input", "tokens_output",
    "costo_est_por_video_usd", "costo_pct_de_ingreso", "ratio_costo_calidad",
    "tiempo_real_seg", "notas",
]


def _load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _coherence_for_combo(combo_id: str) -> dict:
    combo_dir = CAPTIONS_DIR / combo_id
    dangling = 0
    out_of_range = 0
    total_groups = 0
    for clip_id, _ in TEST_CLIPS:
        data = _load_json(combo_dir / f"{clip_id}.json")
        if not data:
            continue
        words = data.get("words", [])
        for g_start, g_end in data.get("groups", []):
            total_groups += 1
            size = g_end - g_start
            if size < GROUP_MIN_WORDS or size > GROUP_MAX_WORDS:
                out_of_range += 1
            if 0 <= g_end - 1 < len(words):
                bare = "".join(ch for ch in words[g_end - 1]["word"].lower() if ch.isalpha())
                if bare in _DANGLING_CONNECTORS:
                    dangling += 1
    return {"dangling_connectors": dangling, "out_of_range_groups": out_of_range, "total_groups": total_groups}


def _transcription_meta_avg(transcription_id: str) -> dict:
    elapsed_total, duration_total, cost_total, n = 0.0, 0.0, 0.0, 0
    for clip_id, _ in TEST_CLIPS:
        meta = _load_json(TRANSCRIPTS_DIR / transcription_id / f"{clip_id}.meta.json")
        if not meta:
            continue
        elapsed_total += meta.get("elapsed_seconds", 0.0)
        duration_total += meta.get("duration_seconds", 0.0)
        cost_total += meta.get("estimated_cost_usd", 0.0)
        n += 1
    if n == 0:
        return {}
    return {
        "elapsed_seconds": elapsed_total / n,
        "duration_seconds": duration_total / n,
        "cost_usd": cost_total,
        "n_clips": n,
    }


def _grouping_meta_sum(combo_id: str) -> dict:
    elapsed_total, in_tok, out_tok, cost_total, n = 0.0, 0, 0, 0.0, 0
    for clip_id, _ in TEST_CLIPS:
        meta = _load_json(CAPTIONS_DIR / combo_id / f"{clip_id}.meta.json")
        if not meta:
            continue
        elapsed_total += meta.get("elapsed_seconds", 0.0)
        in_tok += meta.get("input_tokens", 0)
        out_tok += meta.get("output_tokens", 0)
        cost_total += meta.get("estimated_cost_usd", 0.0)
        n += 1
    return {"elapsed_seconds": elapsed_total, "input_tokens": in_tok, "output_tokens": out_tok, "cost_usd": cost_total, "n_clips": n}


def _wer_scores():
    return _load_json(WORK_DIR / "wer_scores.json") or {"transcription": {}, "cleaned": {}}


def build_rows() -> list:
    if not CAPTIONS_DIR.exists():
        return []
    wer = _wer_scores()
    rows = []
    for combo_dir in sorted(CAPTIONS_DIR.iterdir()):
        if not combo_dir.is_dir():
            continue
        combo_id = combo_dir.name
        if "__" not in combo_id:
            continue
        transcription_id, llm_id = combo_id.split("__", 1)

        t_meta = _transcription_meta_avg(transcription_id)
        g_meta = _grouping_meta_sum(combo_id)
        coherence = _coherence_for_combo(combo_id)

        avg_duration = t_meta.get("duration_seconds") or 0.0
        scale = (REFERENCE_CLIP_SECONDS / avg_duration) if avg_duration > 0 else None
        transcription_cost_per_clip = (t_meta.get("cost_usd", 0.0) / t_meta["n_clips"]) if t_meta.get("n_clips") else 0.0
        llm_cost_per_clip = (g_meta.get("cost_usd", 0.0) / g_meta["n_clips"]) if g_meta.get("n_clips") else 0.0
        cost_per_video = (transcription_cost_per_clip + llm_cost_per_clip) * scale if scale is not None else None

        wer_vals = [v["wer"] for v in wer["transcription"].get(transcription_id, {}).values() if v.get("wer") is not None]
        wer_avg = sum(wer_vals) / len(wer_vals) if wer_vals else None

        ef_cleaned_vals = [v["edit_fraction_cleaned"] for v in wer["cleaned"].get(combo_id, {}).values() if v.get("edit_fraction_cleaned") is not None]
        ef_raw_vals = [v["edit_fraction_raw"] for v in wer["transcription"].get(transcription_id, {}).values() if v.get("edit_fraction_raw") is not None]
        edit_fraction = (sum(ef_cleaned_vals) / len(ef_cleaned_vals)) if ef_cleaned_vals else ((sum(ef_raw_vals) / len(ef_raw_vals)) if ef_raw_vals else None)

        ratio = None
        if cost_per_video is not None and edit_fraction is not None and edit_fraction < 1.0:
            ratio = cost_per_video / (1 - edit_fraction)

        notas = []
        if not t_meta:
            notas.append("sin transcript")
        if edit_fraction is None:
            notas.append("BLOCKED: falta referencia corregida a mano (score_wer.py)")
        if scale is None:
            notas.append("sin duracion de audio")

        rows.append({
            "config_id": combo_id,
            "modelo_transcripcion": transcription_id,
            "config_llm": llm_id,
            "WER": round(wer_avg, 4) if wer_avg is not None else None,
            "fraccion_edicion": round(edit_fraction, 4) if edit_fraction is not None else None,
            "conectores_colgantes": coherence["dangling_connectors"],
            "grupos_fuera_de_rango": coherence["out_of_range_groups"],
            "tokens_input": g_meta.get("input_tokens", 0),
            "tokens_output": g_meta.get("output_tokens", 0),
            "costo_est_por_video_usd": round(cost_per_video, 4) if cost_per_video is not None else None,
            "costo_pct_de_ingreso": round(100 * cost_per_video / REVENUE_PER_VIDEO_USD, 2) if cost_per_video is not None else None,
            "ratio_costo_calidad": round(ratio, 5) if ratio is not None else None,
            "tiempo_real_seg": round(t_meta.get("elapsed_seconds", 0.0) + g_meta.get("elapsed_seconds", 0.0), 1),
            "notas": "; ".join(notas),
        })
    return rows


def _summary(rows: list) -> str:
    scored = [r for r in rows if r["fraccion_edicion"] is not None and r["costo_est_por_video_usd"] is not None]
    if not scored:
        return "Sin filas con fraccion_edicion calculada todavia -- falta correr score_wer.py con referencias corregidas a mano."

    lines = []
    below_threshold = [r for r in scored if r["fraccion_edicion"] <= 0.05]
    if below_threshold:
        cheapest = min(below_threshold, key=lambda r: r["costo_est_por_video_usd"])
        lines.append(
            f"- Mas barata con fraccion_edicion <= 5%: **{cheapest['config_id']}** "
            f"(${cheapest['costo_est_por_video_usd']:.4f}/video, {cheapest['fraccion_edicion']*100:.1f}% edicion)"
        )
    else:
        lines.append("- Ninguna config baja de 5% de fraccion_edicion todavia.")

    best_quality = min(scored, key=lambda r: r["fraccion_edicion"])
    lines.append(
        f"- Mejor calidad (sin importar costo): **{best_quality['config_id']}** "
        f"({best_quality['fraccion_edicion']*100:.1f}% edicion, ${best_quality['costo_est_por_video_usd']:.4f}/video)"
    )

    ratio_scored = [r for r in scored if r["ratio_costo_calidad"] is not None]
    if ratio_scored:
        best_ratio = min(ratio_scored, key=lambda r: r["ratio_costo_calidad"])
        lines.append(f"- Mejor ratio costo/calidad: **{best_ratio['config_id']}** (ratio={best_ratio['ratio_costo_calidad']:.5f})")

    return "\n".join(lines)


def main():
    rows = build_rows()
    if not rows:
        print(f"No hay combos en {CAPTIONS_DIR} todavia -- corre run_transcription.py y run_grouping.py primero.")
        return

    csv_path = WORK_DIR / "report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        f"# Benchmark: transcripcion + edicion de subtitulos ({REFERENCE_CLIP_SECONDS:.0f}s referencia, ${REVENUE_PER_VIDEO_USD:.2f}/video ingreso)",
        "",
        "Formula ratio_costo_calidad = costo_est_por_video_usd / (1 - fraccion_edicion). Mas bajo = mejor.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "|".join(["---"] * len(COLUMNS)) + "|",
    ]
    for r in rows:
        md_lines.append("| " + " | ".join("" if r[c] is None else str(r[c]) for c in COLUMNS) + " |")
    md_lines += ["", "## Resumen", "", _summary(rows)]

    md_path = WORK_DIR / "report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print()
    print(_summary(rows))


if __name__ == "__main__":
    main()
