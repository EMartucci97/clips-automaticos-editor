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
from typing import List

FINAL_VIDEO_ARGS = ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]
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
