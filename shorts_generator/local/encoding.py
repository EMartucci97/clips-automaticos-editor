"""Args de encode de video compartidos por todas las pasadas del export.

El export encadena varias pasadas que re-codifican (corte, crop, subtítulos,
velocidad, overlays, outro). Solo la ÚLTIMA necesita libx264 (archivo final,
chico y compatible); las intermedias van a ser decodificadas de nuevo al toque,
así que usan el encoder por hardware (h264_videotoolbox, ~2x más rápido y con
SSIM >= libx264 crf20 a q 85 -- medido sobre el source real, 2026-09-26).

`_render_export` marca qué pasada es intermedia con `intermediate_encodes()`;
fuera de ese contexto todo encode es final (comportamiento de siempre)."""
import threading
from contextlib import contextmanager
import json
import subprocess
from typing import List, Optional, Tuple

FINAL_VIDEO_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]  # veryfast: ~1.8x faster than fast, SSIM 0.986 vs 0.988, smaller files
INTERMEDIATE_VIDEO_ARGS = ["-c:v", "h264_videotoolbox", "-q:v", "85", "-pix_fmt", "yuv420p"]

_state = threading.local()


def video_encode_args(intermediate: bool = False) -> List[str]:
    if intermediate or getattr(_state, "intermediate", False):
        return list(INTERMEDIATE_VIDEO_ARGS)
    return list(FINAL_VIDEO_ARGS)


@contextmanager
def intermediate_encodes(on: bool = True):
    prev = getattr(_state, "intermediate", False)
    _state.intermediate = on
    try:
        yield
    finally:
        _state.intermediate = prev


def probe_display_video(path: str, ffprobe_bin: str = "ffprobe") -> Tuple[int, int, Optional[str]]:
    """(width, height, r_frame_rate) del primer stream de video TAL COMO LO VE
    ffmpeg al decodificar: si el archivo trae rotación de 90/270 (MOV de
    iPhone grabado vertical: 1920x1080 + rotation=90) se intercambian ancho y
    alto, porque ffmpeg auto-rota los cuadros. JSON en vez de csv porque la
    side_data de rotación agrega campos extra a la salida csv."""
    out = subprocess.run(
        [ffprobe_bin, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:stream_side_data=rotation",
         "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    st = json.loads(out)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    rot = 0
    for sd in st.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if abs(rot) % 180 == 90:
        w, h = h, w
    return w, h, st.get("r_frame_rate")
