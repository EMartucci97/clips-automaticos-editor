"""Dashboard: elegí una fuente, decime qué clipear, y generá los clips.

Wizard de 4 pasos (todo corre sobre el pipeline local existente, no
reimplementa nada de transcripción/highlights/corte):
  1. Fuente:        Google Drive o YouTube
  2. Link:          el link/carpeta de esa fuente
  3. Qué clipear:    una frase específica, o varios clips por tema
  4. Formato:        corte crudo (sin edición) o con edición (vertical + subs)

Después de generar, cada clip resultante tiene un botón "Editar" que abre
el flujo de revisión de shots / regeneración ya existente (detectar quién
habla, corregir el track, volver a recortar).

Run con:
    streamlit run dashboard.py
"""
import datetime
import glob
import json
import os

import streamlit as st

from shorts_generator.config import LOCAL_OUTPUT_DIR
from shorts_generator.local import active_speaker, clipper, subtitles, transcriber, two_camera
from shorts_generator.local.drive import (
    download_drive_file,
    download_drive_folder,
    is_drive_folder_url,
)
from shorts_generator.local.downloader import download_youtube_local
from shorts_generator.local.llm import call_local_llm
from shorts_generator.highlights import get_highlights
from shorts_generator.local.phrase_search import find_phrase_clips

st.set_page_config(page_title="Clips dashboard", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #0e0f13;
        --bg-alt: #15171d;
        --card: #1a1c23;
        --card-border: #2a2d37;
        --text: #e8e9ed;
        --text-dim: #9a9ca6;
        --accent: #ff4b4b;
        --accent-hover: #ff6b6b;
    }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    }
    .stApp {
        background: var(--bg);
        color: var(--text);
    }
    section[data-testid="stSidebar"] {
        background: var(--bg-alt);
        border-right: 1px solid var(--card-border);
    }
    h1, h2, h3 {
        color: var(--text);
        letter-spacing: -0.01em;
    }
    p, span, label, .stCaption, div[data-testid="stCaptionContainer"] {
        color: var(--text-dim);
    }

    /* card-like wrapper around each numbered step + sidebar entries */
    div[data-testid="stVerticalBlockBorderWrapper"],
    section[data-testid="stSidebar"] div[data-testid="stExpander"] {
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 0.25rem 0.5rem;
    }
    div[data-testid="stExpander"] {
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        overflow: hidden;
    }
    div[data-testid="stExpander"] summary {
        border-radius: 14px;
    }

    /* buttons */
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid var(--card-border);
        background: var(--card);
        color: var(--text);
        font-weight: 600;
        transition: all 0.15s ease;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--accent);
        color: var(--accent-hover);
        background: #20222b;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: #ffffff;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--accent-hover);
        border-color: var(--accent-hover);
        color: #ffffff;
    }

    /* inputs */
    .stTextInput input, .stNumberInput input, textarea,
    div[data-baseweb="select"] > div, div[data-baseweb="input"] {
        background: var(--card) !important;
        border-radius: 10px !important;
        border: 1px solid var(--card-border) !important;
        color: var(--text) !important;
    }

    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 0.6rem 0.9rem;
    }

    hr, div[data-testid="stDivider"] {
        border-color: var(--card-border) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TMP_DIR = os.path.join(LOCAL_OUTPUT_DIR, "_dashboard_tmp")
os.makedirs(TMP_DIR, exist_ok=True)
TMP_CUT_PATH = os.path.join(TMP_DIR, "review_cut.mp4")

st.title("Clips — generar y revisar")

defaults = {
    "source_type": None,
    "source_link": None,
    "video_paths": None,
    "clip_mode": None,
    "results": None,
    "edit_job": None,
}
for key, val in defaults.items():
    st.session_state.setdefault(key, val)


# ------------------------------------------------------- galería (disco) --
def _list_generated_clips():
    """Todos los .mp4 en output/ que parecen clips (no fuentes descargadas
    ni archivos temporales), leído directo de disco así sobrevive a un
    refresh o a que se reinicie el session_state del wizard."""
    clips = []
    for path in sorted(glob.glob(os.path.join(LOCAL_OUTPUT_DIR, "*.mp4")), key=os.path.getmtime, reverse=True):
        name = os.path.basename(path)
        if name.startswith("_") or name.startswith("source_") or ".silent." in name:
            continue
        if os.path.getsize(path) > 200 * 1024 * 1024:  # descarta videos fuente completos
            continue
        clips.append(path)
    return clips


def _metadata_path(clip_path: str) -> str:
    return clip_path + ".meta.json"


def _save_clip_metadata(clip_path: str, source: str, start: float, end: float, **extra) -> None:
    """Guarda de qué video fuente y qué rango [start, end] salió un clip, así
    se lo puede volver a editar (reencuadrar, split, subs) después de un
    refresh, sin depender de session_state. `extra` guarda título/hook/scores
    cuando están disponibles (generación por tema), para mostrarlos en la
    galería."""
    data = {"source": source, "start": start, "end": end, **extra}
    with open(_metadata_path(clip_path), "w") as f:
        json.dump(data, f)


def _find_result_json_metadata(clip_path: str):
    """Fallback para clips generados por CLI (main.py --output-json), que no
    pasan por _save_clip_metadata: busca en los *_result.json de output/ un
    'short' cuyo clip_url coincida con este archivo."""
    name = os.path.basename(clip_path)
    for json_path in glob.glob(os.path.join(LOCAL_OUTPUT_DIR, "*.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        shorts = data.get("shorts") if isinstance(data, dict) else None
        if not shorts:
            continue
        for s in shorts:
            if os.path.basename(str(s.get("clip_url", ""))) == name:
                return {
                    "source": data.get("source_video_url"),
                    "start": s.get("start_time"),
                    "end": s.get("end_time"),
                    "title": s.get("title"),
                    "hook_sentence": s.get("hook_sentence"),
                    "score": s.get("score"),
                    "sales_angle_score": s.get("sales_angle_score"),
                    "sales_angle": s.get("sales_angle"),
                    "central_theme": s.get("central_theme"),
                    "transcript": s.get("transcript"),
                }
    return None


def _load_clip_metadata(clip_path: str):
    try:
        with open(_metadata_path(clip_path)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return _find_result_json_metadata(clip_path)


# --------------------------------------------------------------- feedback --
FEEDBACK_PATH = os.path.join(LOCAL_OUTPUT_DIR, "clip_feedback.jsonl")


def _save_feedback(clip_path: str, text: str, meta: dict | None) -> None:
    """Feedback en texto libre por clip, para releer más adelante y ajustar
    el prompt de highlights.py / los criterios de selección. Append-only en
    un jsonl, no se pisa nada."""
    entry = {
        "clip": os.path.basename(clip_path),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "feedback": text.strip(),
        "title": (meta or {}).get("title"),
        "score": (meta or {}).get("score"),
        "sales_angle": (meta or {}).get("sales_angle"),
    }
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_feedback(clip_path: str) -> list:
    name = os.path.basename(clip_path)
    entries = []
    if not os.path.exists(FEEDBACK_PATH):
        return entries
    with open(FEEDBACK_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("clip") == name:
                entries.append(entry)
    return entries


generated_clips = _list_generated_clips()


def _render_gallery_card(path: str) -> None:
    name = os.path.basename(path)
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d/%m %H:%M")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    meta = _load_clip_metadata(path)
    title = (meta or {}).get("title")
    with st.container(border=True):
        st.markdown(f"**{title or name}**")
        st.caption(f"{mtime} · {size_mb:.1f} MB")
        if meta and meta.get("hook_sentence"):
            st.write(f"“{meta['hook_sentence']}”")
        if meta and meta.get("central_theme"):
            st.caption(f"🎯 {meta['central_theme']}")
        if meta and (meta.get("score") is not None or meta.get("sales_angle_score") is not None):
            score_cols = st.columns(2)
            if meta.get("score") is not None:
                score_cols[0].metric("score", meta["score"])
            if meta.get("sales_angle_score") is not None:
                score_cols[1].metric("sales_angle_score", meta["sales_angle_score"])
            if meta.get("sales_angle"):
                st.caption(f"Ángulo: {meta['sales_angle']}")
        st.video(path)
        if meta and meta.get("transcript"):
            if st.checkbox("📝 Ver transcripción del clip", key=f"gallery_tx_{name}"):
                st.text_area(
                    "Transcripción", meta["transcript"], height=120,
                    disabled=True, label_visibility="collapsed", key=f"gallery_tx_area_{name}",
                )
        dl_col, edit_col = st.columns(2)
        with dl_col:
            with open(path, "rb") as f:
                st.download_button("Descargar", f, file_name=name, key=f"gallery_dl_{name}")
        with edit_col:
            if meta and meta.get("source") and meta.get("start") is not None and meta.get("end") is not None:
                if st.button("✏️ Editar", key=f"gallery_edit_{name}"):
                    st.session_state.edit_job = {
                        "source": meta["source"],
                        "start": float(meta["start"]),
                        "end": float(meta["end"]),
                        "filename": name,
                    }
            else:
                st.caption("Sin metadata\n(clip viejo)")

        with st.expander("💬 Feedback"):
            past_feedback = _load_feedback(path)
            if past_feedback:
                for entry in past_feedback:
                    st.caption(f"{entry['timestamp']}: {entry['feedback']}")
            fb_text = st.text_area(
                "Feedback", key=f"fb_input_{name}", label_visibility="collapsed",
                placeholder="¿Qué está bien o mal de este clip? (hook flojo, corte mal hecho, ángulo equivocado, etc.)",
            )
            if st.button("Guardar feedback", key=f"fb_save_{name}"):
                if fb_text.strip():
                    _save_feedback(path, fb_text, meta)
                    st.success("Guardado")
                    st.rerun()
                else:
                    st.warning("Escribí algo antes de guardar.")


def _render_gallery(clips: list, columns: int = 3) -> None:
    if not clips:
        st.caption("Todavía no generaste ningún clip.")
        return
    cols = st.columns(columns)
    for i, path in enumerate(clips):
        with cols[i % columns]:
            _render_gallery_card(path)


# ---------------------------------------------------------------- editor --
# Se resuelve antes que los pasos 1-4 (y sus st.stop()) para que el botón
# "Editar" de la galería funcione aunque todavía no se haya elegido fuente.
job = st.session_state.edit_job
if job:
    st.divider()
    st.subheader(f"Editar: {job['filename']}")

    st.caption("Ajustá el inicio/fin si el corte quedó desfasado (empieza antes/después de lo ideal)")
    trim_cols = st.columns(2)
    with trim_cols[0]:
        adj_start = st.number_input(
            "Inicio (s)", min_value=0.0, value=float(job["start"]), step=0.5,
            key=f"start_{job['filename']}",
        )
    with trim_cols[1]:
        adj_end = st.number_input(
            "Fin (s)", min_value=0.0, value=float(job["end"]), step=0.5,
            key=f"end_{job['filename']}",
        )
    if adj_end <= adj_start:
        st.warning("El fin debe ser mayor que el inicio; se ignoró el ajuste.")
    else:
        job["start"], job["end"] = adj_start, adj_end
    st.caption(f"Duración actual: {job['end'] - job['start']:.1f}s")

    out_path = os.path.join(LOCAL_OUTPUT_DIR, job["filename"])
    job_key = f"{job['source']}::{job['start']}::{job['end']}"

    if st.button("1) Analizar shots (detectar quién habla)"):
        with st.spinner("Cortando subclip y analizando shots..."):
            clipper._cut_subclip(job["source"], job["start"], job["end"], TMP_CUT_PATH)
            shots = active_speaker.list_shots_for_review(TMP_CUT_PATH)
            shot_images = []
            for shot in shots:
                if shot["faces"] and shot["frame_index"] is not None:
                    img_path = os.path.join(TMP_DIR, f"shot_{shot['shot_index']:02d}.png")
                    active_speaker.render_shot_image(TMP_CUT_PATH, shot["frame_index"], shot["faces"], img_path)
                else:
                    img_path = None
                shot_images.append(img_path)
            st.session_state.shots = shots
            st.session_state.shot_images = shot_images
            st.session_state.shots_job_key = job_key

    if st.session_state.get("shots_job_key") == job_key and st.session_state.get("shots"):
        st.write("Shots detectados")
        overrides = {}
        for shot, img_path in zip(st.session_state.shots, st.session_state.shot_images):
            idx = shot["shot_index"]
            cols = st.columns([2, 1])
            with cols[0]:
                caption = f"Shot {idx} · {shot['start_time']:.1f}s–{shot['end_time']:.1f}s"
                if img_path:
                    st.image(img_path, caption=caption)
                else:
                    st.write(f"{caption} (sin caras detectadas)")
            with cols[1]:
                track_ids = [f["track_id"] for f in shot["faces"]]
                choice = st.selectbox(
                    f"Hablante shot {idx}", ["auto"] + track_ids, key=f"track_{job_key}_{idx}",
                )
                if choice != "auto":
                    overrides[idx] = choice
        st.session_state.overrides = overrides

        aspect_ratio = st.selectbox("Aspect ratio", ["9:16", "1:1", "4:5"], index=0)
        burn_subs = st.checkbox("Quemar subtítulos", value=True)

        split_screen = st.checkbox(
            "Split screen (arriba/abajo)", value=False, key=f"split_{job['filename']}",
        )

        reframe_mode = "Seguimiento activo"
        if not split_screen:
            reframe_mode = st.radio(
                "Cámara",
                ["Seguimiento activo", "2 cámaras fijas (diarización + zoom sutil)"],
                key=f"reframe_mode_{job['filename']}",
                help=(
                    "Seguimiento activo: sigue frame a frame a quien habla. "
                    "2 cámaras: dos planos fijos (izquierda/derecha) que cambian entre sí "
                    "según quién habla (diarización), con un leve zoom de respiración."
                ),
            )

        top_pane = bottom_pane = None
        if split_screen:
            gallery_options = _list_generated_clips()
            gallery_labels = {p: os.path.basename(p) for p in gallery_options}
            main_duration = job["end"] - job["start"]

            split_cols = st.columns(2)
            for label, slot_key in (("Arriba", "top"), ("Abajo", "bottom")):
                with split_cols[0 if slot_key == "top" else 1]:
                    st.markdown(f"**{label}**")
                    choice = st.radio(
                        f"Contenido {label.lower()}",
                        ["Hablante del clip", "Otro clip de la galería"],
                        key=f"{slot_key}_mode_{job['filename']}",
                        label_visibility="collapsed",
                    )
                    pane = None
                    if choice == "Otro clip de la galería":
                        if gallery_options:
                            picked = st.selectbox(
                                f"Clip para {label.lower()}",
                                gallery_options,
                                format_func=lambda p: gallery_labels.get(p, os.path.basename(p)),
                                key=f"{slot_key}_pick_{job['filename']}",
                            )
                            st.caption(
                                "Usa el clip elegido desde su inicio, en loop/recortado para "
                                "durar lo mismo que el clip principal."
                            )
                            pane = {"source": picked, "start": 0.0, "end": main_duration}
                        else:
                            st.caption("Todavía no hay otros clips generados en la galería.")
                    if slot_key == "top":
                        top_pane = pane
                    else:
                        bottom_pane = pane

        if st.button("2) Regenerar clip"):
            with st.spinner("Regenerando clip..."):
                if split_screen:
                    clipper.crop_clip_split(
                        job["source"], job["start"], job["end"], aspect_ratio, out_path,
                        top=top_pane, bottom=bottom_pane,
                        shot_overrides=st.session_state.overrides or None,
                    )
                elif reframe_mode == "2 cámaras fijas (diarización + zoom sutil)":
                    two_camera.crop_clip_two_camera(
                        job["source"], job["start"], job["end"], aspect_ratio, out_path,
                    )
                else:
                    clipper.crop_clip_local(
                        job["source"], job["start"], job["end"], aspect_ratio, out_path,
                        shot_overrides=st.session_state.overrides or None,
                    )
                if burn_subs:
                    transcript = transcriber.transcribe_local(job["source"])
                    words = subtitles.words_for_window(transcript["segments"], job["start"], job["end"])
                    tmp_subbed = out_path + ".subbed.mp4"
                    subtitles.burn_subtitles(out_path, words, tmp_subbed)
                    os.replace(tmp_subbed, out_path)
                _save_clip_metadata(out_path, job["source"], job["start"], job["end"])
                st.session_state.regenerated_path = out_path
            st.success(f"Listo: {out_path}")

    if st.session_state.get("regenerated_path") and os.path.exists(st.session_state["regenerated_path"]):
        st.video(st.session_state["regenerated_path"])
        with open(st.session_state["regenerated_path"], "rb") as f:
            st.download_button("Descargar clip editado", f, file_name=os.path.basename(st.session_state["regenerated_path"]))
    st.divider()


tab_generar, tab_galeria = st.tabs(["🎬 Generar clips", f"📂 Galería ({len(generated_clips)})"])

with tab_galeria:
    _render_gallery(generated_clips)

with tab_generar:
    # ------------------------------------------------------------ paso 1 --
    st.subheader("1) ¿De dónde son los clips?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Google Drive", icon="📁", use_container_width=True,
                      type="primary" if st.session_state.source_type == "drive" else "secondary"):
            st.session_state.source_type = "drive"
            st.session_state.video_paths = None
            st.session_state.results = None
    with col2:
        if st.button("YouTube", icon="▶️", use_container_width=True,
                      type="primary" if st.session_state.source_type == "youtube" else "secondary"):
            st.session_state.source_type = "youtube"
            st.session_state.video_paths = None
            st.session_state.results = None

    if not st.session_state.source_type:
        st.info("Elegí una fuente para continuar.")
        st.stop()

    # ------------------------------------------------------------ paso 2 --
    st.divider()
    st.subheader("2) Link")
    if st.session_state.source_type == "drive":
        link = st.text_input(
            "Link de Google Drive (archivo o carpeta, compartido como 'Cualquiera con el enlace')",
            value=st.session_state.source_link or "",
        )
    else:
        link = st.text_input("Link de YouTube", value=st.session_state.source_link or "")

    if st.button("Descargar / preparar video"):
        st.session_state.source_link = link
        st.session_state.results = None
        with st.spinner("Descargando..."):
            try:
                if st.session_state.source_type == "drive":
                    if is_drive_folder_url(link):
                        st.session_state.video_paths = download_drive_folder(link)
                    else:
                        st.session_state.video_paths = [download_drive_file(link)]
                else:
                    st.session_state.video_paths = [download_youtube_local(link)]
            except Exception as e:
                st.session_state.video_paths = None
                st.error(f"No se pudo descargar: {e}")

    if not st.session_state.video_paths:
        st.info("Pegá el link y descargá el/los video(s) para continuar.")
        st.stop()

    st.success(f"Listo: {len(st.session_state.video_paths)} video(s) — " + ", ".join(
        os.path.basename(p) for p in st.session_state.video_paths
    ))

    # ------------------------------------------------------------ paso 3 --
    st.divider()
    st.subheader("3) ¿Qué querés clipear?")
    clip_mode = st.radio(
        "Modo",
        ["Una frase específica", "Varios clips por tema"],
        label_visibility="collapsed",
    )

    phrase = None
    context_before = context_after = None
    num_clips = None
    if clip_mode == "Una frase específica":
        phrase = st.text_input("Frase o palabras clave a buscar en el video")
        c1, c2 = st.columns(2)
        context_before = c1.slider("Segundos de contexto antes", 0, 30, 8)
        context_after = c2.slider("Segundos de contexto después", 0, 30, 12)
    else:
        num_clips = st.number_input("Cantidad de clips a generar", min_value=1, max_value=10, value=3)

    # ------------------------------------------------------------ paso 4 --
    st.divider()
    st.subheader("4) Formato")
    format_mode = st.radio(
        "Formato",
        ["Sin edición (corte crudo, tal cual el original)", "Con edición (recorte vertical 9:16 + subtítulos)"],
        label_visibility="collapsed",
    )
    edited = format_mode.startswith("Con edición")

    generate = st.button("🎬 Generar clips", type="primary")

    def _process_video(source_path: str, mode: str) -> list:
        transcript = transcriber.transcribe_local(source_path)
        if not transcript["segments"]:
            raise RuntimeError("No se detectó audio/voz en el video.")

        if mode == "Una frase específica":
            if not phrase or not phrase.strip():
                raise RuntimeError("Ingresá una frase para buscar.")
            highlights = find_phrase_clips(transcript, phrase, context_before, context_after)
            if not highlights:
                raise RuntimeError(f"No se encontró la frase \"{phrase}\" en la transcripción.")
        else:
            result = get_highlights(transcript, num_clips=int(num_clips), llm_fn=call_local_llm)
            highlights = sorted(
                result.get("highlights", []),
                key=lambda h: float(h.get("combined_score", h.get("score", 0))),
                reverse=True,
            )[: int(num_clips)]
            if not highlights:
                raise RuntimeError("No se generaron highlights para este video.")

        out = []
        used_names = set()
        for i, h in enumerate(highlights, 1):
            slug = clipper._clip_slug(h)
            filename = f"{i:02d}_{slug}.mp4"
            n = 2
            while filename in used_names:
                filename = f"{i:02d}_{slug}-{n}.mp4"
                n += 1
            used_names.add(filename)
            out_path = os.path.join(LOCAL_OUTPUT_DIR, filename)
            start, end = float(h["start_time"]), float(h["end_time"])
            if edited:
                clipper.crop_clip_local(source_path, start, end, "9:16", out_path)
                words = subtitles.words_for_window(transcript["segments"], start, end)
                tmp_subbed = out_path + ".subbed.mp4"
                subtitles.burn_subtitles(out_path, words, tmp_subbed)
                os.replace(tmp_subbed, out_path)
            else:
                clipper._cut_subclip(source_path, start, end, out_path)
            _save_clip_metadata(
                out_path, source_path, start, end,
                title=h.get("title"),
                hook_sentence=h.get("hook_sentence"),
                score=h.get("score"),
                sales_angle_score=h.get("sales_angle_score"),
                sales_angle=h.get("sales_angle"),
                central_theme=h.get("central_theme"),
                transcript=h.get("transcript"),
            )
            out.append({**h, "clip_url": out_path, "source": source_path, "edited": edited})
        return out

    if generate:
        all_results = []
        errors = []
        for source_path in st.session_state.video_paths:
            with st.spinner(f"Procesando {os.path.basename(source_path)}..."):
                try:
                    all_results.extend(_process_video(source_path, clip_mode))
                except Exception as e:
                    errors.append(f"{os.path.basename(source_path)}: {e}")
        st.session_state.results = all_results
        st.session_state.edit_job = None
        for err in errors:
            st.error(err)

    # -------------------------------------------------------- resultados --
    if st.session_state.results:
        st.divider()
        st.subheader("Resultado")
        for i, r in enumerate(st.session_state.results):
            label = f"{i + 1:02d}. {r.get('sales_angle') or r.get('title')} — [{r['start_time']:.1f}s–{r['end_time']:.1f}s]"
            with st.expander(label, expanded=(i == 0)):
                cols = st.columns([2, 1])
                with cols[0]:
                    if r.get("hook_sentence"):
                        st.write(r["hook_sentence"])
                    if r.get("central_theme"):
                        st.caption(f"🎯 {r['central_theme']}")
                    clip_path = r.get("clip_url")
                    if clip_path and os.path.exists(clip_path):
                        st.video(clip_path)
                        with open(clip_path, "rb") as f:
                            st.download_button(
                                "Descargar clip", f, file_name=os.path.basename(clip_path),
                                key=f"dl_{i}",
                            )
                    if r.get("transcript"):
                        if st.checkbox("📝 Ver transcripción del clip", key=f"result_tx_{i}"):
                            st.text_area(
                                "Transcripción", r["transcript"], height=120,
                                disabled=True, label_visibility="collapsed", key=f"result_tx_area_{i}",
                            )
                with cols[1]:
                    st.metric("score", r.get("score"))
                    if r.get("sales_angle_score"):
                        st.metric("sales_angle_score", r.get("sales_angle_score"))
                    st.caption("editado" if r.get("edited") else "sin editar")
                    if st.button("✏️ Editar este clip", key=f"edit_{i}"):
                        st.session_state.edit_job = {
                            "source": r["source"],
                            "start": float(r["start_time"]),
                            "end": float(r["end_time"]),
                            "filename": os.path.basename(r["clip_url"]),
                        }
