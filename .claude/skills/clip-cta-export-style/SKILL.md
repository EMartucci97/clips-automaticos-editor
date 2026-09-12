---
name: clip-cta-export-style
description: Documents the standard export "estilo" for clips-automaticos highlights — crop → captions → speed → blurred CTA outro (real background footage, incremental blur, portada + "Comenta EDUARDO" text) → background music mix. Use this whenever exporting a highlight from the local editor (editor_server.py), tuning the CTA outro/blur/overlay timing, or explaining/reproducing this pipeline for a new source video. Not a script to run — it's the reference for how `POST /api/highlights/<id>/export` already behaves and how to adjust it.
---

# Estilo de export de clips (CTA blur outro + música)

Este estilo ya está **wireado como comportamiento por defecto** de `POST /api/highlights/<id>/export` en `editor_server.py` — se aplica automáticamente a **cualquier** highlight de **cualquier** video fuente que se cargue en el editor local, sin pasos extra por video. Esta skill documenta cómo funciona y qué tocar si se quiere afinar el estilo más adelante, para no tener que re-derivar la lógica desde cero en cada sesión.

## Pipeline (orden fijo, en `editor_server.py::api_export_highlight`)

1. **Crop** (`clipper.crop_clip_local`) — recorte del highlight + cortes internos (`cuts`), aspect ratio 9:16 si `crop_enabled`.
2. **Captions** (`burn_subtitles`) — si `captions_enabled` (default true).
3. **Speed** (`clipper.apply_speed_local`) — si `speed != 1.0`.
4. **CTA outro con blur** (`clipper.append_blurred_cta_outro`) — **siempre**, mientras el video fuente tenga ≥0.5s de metraje después del `end_time` del highlight (clamp automático si no).
5. **Música de fondo** (`clipper.mix_background_music`) — solo si el highlight tiene `music_track` asignado desde la UI.

Todo corre sobre `SOURCE_PATH` (el video con el que se lanzó `editor_server.py`), así que el fondo real del outro siempre sale del mismo video fuente que el clip — no hay imagen/video hardcodeado por proyecto salvo la portada/CTA (ver abajo).

## CTA outro: qué hace `append_blurred_cta_outro`

Extrae video real de `source_path` arrancando en `cta_start_time` (el `end_time` del highlight) — mismo escenario, sigue corriendo, sin el audio del diálogo (se reemplaza por silencio). Encima se compone un overlay con blur incremental + portada/texto:

- **Blur:** filtro `gblur` con sigma controlado en tiempo real vía `sendcmd` (NO `xfade`/crossfade de opacidad — eso fue un bug real, ver [[project_clips_automaticos_musica]] sección "Bug real en el blur incremental"). Rampea `sigma` de 0 a `blur_sigma` (default 22.0) en `blur_ramp_seconds` (default **0.25s**), y se mantiene ahí el resto del outro.
- **Overlay (portada + texto CTA):** PNG generado en runtime por `_render_cta_overlay_png` (portada `assets/cta_portada.png` + texto "Comenta EDUARDO" / "y te mando la entrevista completa" renderizado con `assets/fonts/Montserrat-Black.ttf`, sin el contorno dentado del PNG original). Invisible hasta `overlay_delay_seconds` (default **0.25s**), después hace fade-in de alpha durante `overlay_fade_seconds` (default **0.25s**) hasta quedar opaco.
- **Secuencia resultante:** blur ya fuerte a los 0.25s, overlay recién visible del todo a los 0.5s — blur primero, imagen después, medio segundo en total. Este timing es el pedido explícito del usuario (bajado de ~2s en iteraciones previas).
- **Duración del outro:** controlada por `editor_server.py::CTA_OUTRO_DURATION_SECONDS = 6.5` (no por el default `3.5` de la función, que solo aplica si se llama sin ese argumento).
- **Audio:** silencioso (`anullsrc`), la música de fondo (paso 5) es lo único que se escucha ahí.

## Música de fondo

`mix_background_music`: hace loop del track elegido, `afade` **solo de entrada** (sin fade-out al final — el usuario prefirió que la música sostenga el volumen hasta el último frame), `amix` con `normalize=0` (crítico: si no, se pisa la voz original solo por sumar un segundo input de audio). Selección de track + volumen es **por highlight**, vía la UI del modal (`editor.html`), no un default global.

## Assets usados (fijos por ahora, no por-video)

- `assets/cta_portada.png` — imagen "LA OTRA CARA DEL BILLONARIO" (collage + titular).
- `assets/fonts/Montserrat-Black.ttf` — fuente del texto CTA.
- `assets/music/*` — 22 tracks disponibles, elegidos por highlight desde la UI.

Si en el futuro se quiere una portada/texto distinto por proyecto/video (no solo por highlight), hoy está hardcodeado en `_render_cta_overlay_png` — habría que parametrizarlo (imagen + copy) en vez de asumir siempre "Comenta EDUARDO".

## Cómo aplicar esto a un video nuevo

No hace falta nada especial — el pipeline ya es automático:

```bash
./venv/bin/python3 editor_server.py "<nuevo_source.mov>" --port 8502
```

Abrir el editor, elegir/ajustar el highlight, opcionalmente asignar música por clip, y exportar (`POST /api/highlights/<id>/export` desde la UI). El CTA outro con blur sale solo, con el fondo real de ESE video.

**Importante:** el server corre con `debug=False` (no autoreload). Cualquier cambio a `clipper.py`/`editor_server.py` requiere matar y relanzar el proceso antes de que un nuevo export lo refleje:

```bash
lsof -ti:8502 | xargs -r kill
nohup ./venv/bin/python3 editor_server.py "<source.mov>" --port 8502 >> /tmp/editor_server.log 2>&1 & disown
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8502/   # esperar 200
```

## Knobs si se quiere afinar más el estilo

| Parámetro | Dónde | Valor actual |
|---|---|---|
| Duración total del outro | `editor_server.py::CTA_OUTRO_DURATION_SECONDS` | 6.5s |
| Intensidad máxima del blur | `editor_server.py::CTA_OUTRO_BLUR_SIGMA` | 22.0 |
| Velocidad del ramp de blur | `clipper.append_blurred_cta_outro(blur_ramp_seconds=...)` | 0.25s |
| Delay antes de que aparezca la portada/CTA | `clipper.append_blurred_cta_outro(overlay_delay_seconds=...)` | 0.25s |
| Duración del fade-in de la portada/CTA | `clipper.append_blurred_cta_outro(overlay_fade_seconds=...)` | 0.25s |
| Volumen default de música | `editor_server.py::DEFAULT_MUSIC_VOLUME` | 0.18 |
| Fade-in de la música al arrancar | `clipper.mix_background_music(fade_seconds=...)` | 1.0s |

## Verificación al tocar cualquiera de estos valores (no saltear)

1. Reiniciar el server (no autoreload).
2. Export real vía `POST /api/highlights/<id>/export` (nunca solo un test aislado de la función — tiene que pasar por el pipeline real).
3. Extraer frames del tramo del outro con `ffmpeg -ss <inicio_outro> -i <archivo> -t <ventana> -vf fps=<N>` — usar fps alto (8-10) si la ventana a verificar es de menos de 1-2s.
4. **Leer visualmente varios frames intermedios**, no solo el primero y el último — un mecanismo técnicamente correcto puede seguir siendo perceptualmente plano a mitad de rango (esto fue exactamente el bug real que motivó el fix con `sendcmd`, ver [[project_clips_automaticos_musica]]).
5. Confirmar con `ffprobe -show_entries format=duration -of csv=p=0` que la duración total no cambió si el ajuste no debía tocarla.

## Estado

Implementado, wireado como default incondicional del pipeline de export, y verificado end-to-end (incluyendo el timing final de 0.5s blur→imagen). Ver [[project_clips_automaticos_musica]] para el historial completo de iteraciones y bugs encontrados en el camino.
