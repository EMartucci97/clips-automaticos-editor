"""Proceso persistente de DeepFilterNet3, corrido con el intérprete de
~/.deepfilter-venv (no el venv del proyecto). Carga el modelo UNA sola vez y
después procesa un wav por línea de stdin -- evita pagar ~2s de carga de
modelo en cada clip, que era el costo real de invocar el binario CLI
`deep-filter-py` (un proceso nuevo, un modelo nuevo, por cada request).

Protocolo (JSON, una línea por mensaje):
  stdin  -> {"in": "<wav de entrada>", "out": "<wav de salida>"}
  stdout <- {"ok": true} | {"ok": false, "error": "..."}
Al arrancar, antes de aceptar requests, imprime {"ready": true} (o
{"ready": false, "error": "..."} si el modelo no cargó).
{"cmd": "shutdown"} termina el proceso prolijamente.
"""
import json
import sys

from df.enhance import enhance, init_df, load_audio, save_audio
from df.io import resample
from df.model import ModelParams


def main() -> None:
    try:
        model, df_state, _ = init_df(
            None,
            post_filter=False,
            # "NONE" salta todo el setup de loguru (df/logger.py sólo agrega el
            # sink a sys.stdout/stderr si level != "none") -- con "INFO" el
            # logger de DeepFilterNet escribe líneas sueltas a stdout que
            # rompen este protocolo JSON-por-línea (JSONDecodeError: Extra data).
            log_level="NONE",
            config_allow_defaults=True,
            epoch="best",
            mask_only=False,
        )
        df_sr = ModelParams().sr
    except Exception as exc:  # noqa: BLE001 - reportar cualquier falla de carga al padre
        print(json.dumps({"ready": False, "error": str(exc)}), flush=True)
        return
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": f"request inválido: {exc}"}), flush=True)
            continue
        if req.get("cmd") == "shutdown":
            break
        try:
            audio, info = load_audio(req["in"], sr=df_sr, verbose=False)
            enhanced = enhance(model, df_state, audio, pad=True, atten_lim_db=None)
            enhanced = resample(enhanced.to("cpu"), df_sr, info.sample_rate)
            save_audio(req["out"], enhanced, sr=info.sample_rate, log=False)
            print(json.dumps({"ok": True}), flush=True)
        except Exception as exc:  # noqa: BLE001 - una falla puntual no debe matar el worker
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
