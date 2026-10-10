"""Local LLM backend — OpenAI, Gemini, or Anthropic, selected by LLM_PROVIDER."""
import time

from ..config import (
    ANTHROPIC_MODEL,
    ANTHROPIC_WORKSPACE_ID,
    GEMINI_MODEL,
    LLM_FALLBACK_PROVIDER,
    LLM_PROVIDER,
    OPENAI_MODEL,
    require_anthropic_key,
    require_gemini_key,
    require_openai_key,
)


def call_openai_llm(prompt: str) -> str:
    """OpenAI Chat Completions backend used by --mode local."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(api_key=require_openai_key())
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def call_gemini_llm(prompt: str) -> str:
    """Gemini backend used by --mode local when LLM_PROVIDER=gemini."""
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-genai is required for LLM_PROVIDER=gemini. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = genai.Client(api_key=require_gemini_key())
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 8192,
        },
    )
    text = response.text or ""
    if not text.strip():
        # Respuesta vacía (MAX_TOKENS, bloqueo de seguridad, etc.): tirar error
        # para que call_local_llm use el backup en vez de devolver "" al parser.
        cand = (response.candidates or [None])[0]
        reason = getattr(cand, "finish_reason", None)
        raise RuntimeError(f"gemini devolvió respuesta vacía (finish_reason={reason})")
    return text


def call_anthropic_llm(prompt: str) -> str:
    """Anthropic Messages API backend used by --mode local when LLM_PROVIDER=anthropic."""
    try:
        import anthropic  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "anthropic is required for LLM_PROVIDER=anthropic. Install it with:\n"
            "    pip install anthropic"
        ) from e

    client = anthropic.Anthropic(api_key=require_anthropic_key())
    extra_headers = {"anthropic-workspace-id": ANTHROPIC_WORKSPACE_ID} if ANTHROPIC_WORKSPACE_ID else None
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
        extra_headers=extra_headers,
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise RuntimeError(f"anthropic devolvió respuesta vacía (stop_reason={response.stop_reason})")
    return text


def _call_provider(provider: str, prompt: str) -> str:
    if provider == "openai":
        return call_openai_llm(prompt)
    if provider == "gemini":
        return call_gemini_llm(prompt)
    if provider == "anthropic":
        return call_anthropic_llm(prompt)
    raise RuntimeError(
        f"Unknown LLM provider {provider!r}. Use 'openai', 'gemini', or 'anthropic'."
    )


_TRANSIENT_MARKERS = ("503", "500", "unavailable", "overloaded", "high demand", "timeout", "timed out")
_QUOTA_MARKERS = ("429", "resource_exhausted", "quota")
_RETRY_DELAYS = (3, 10, 30)
# Cuota agotada (tier gratis): reintentar no sirve, se salta directo al backup y
# se evita volver a pegarle al proveedor durante este rato.
_QUOTA_COOLDOWN_S = 120
_quota_blocked_until: dict[str, float] = {}


def call_local_llm(prompt: str) -> str:
    """Dispatch to LLM_PROVIDER; retry transient errors, then use LLM_FALLBACK_PROVIDER.
    Quota errors (429) skip retries and go straight to the fallback."""
    provider = (LLM_PROVIDER or "openai").strip().lower()
    last_err: Exception | None = None
    has_fallback = bool(LLM_FALLBACK_PROVIDER and LLM_FALLBACK_PROVIDER != provider)
    if has_fallback and time.time() < _quota_blocked_until.get(provider, 0):
        return _call_provider(LLM_FALLBACK_PROVIDER, prompt)
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            return _call_provider(provider, prompt)
        except Exception as e:  # noqa: BLE001 — cualquier fallo del proveedor cae al backup
            last_err = e
            msg = str(e).lower()
            if has_fallback and any(m in msg for m in _QUOTA_MARKERS):
                _quota_blocked_until[provider] = time.time() + _QUOTA_COOLDOWN_S
                break
            transient = any(m in msg for m in _TRANSIENT_MARKERS) or any(m in msg for m in _QUOTA_MARKERS)
            if not transient or attempt == len(_RETRY_DELAYS):
                break
            print(f"[llm] {provider} falló ({str(e)[:80]}), reintento en {_RETRY_DELAYS[attempt]}s", flush=True)
            time.sleep(_RETRY_DELAYS[attempt])

    fallback = LLM_FALLBACK_PROVIDER
    if fallback and fallback != provider:
        print(f"[llm] {provider} no disponible ({str(last_err)[:120]}), uso backup: {fallback}", flush=True)
        return _call_provider(fallback, prompt)
    raise last_err  # type: ignore[misc]
