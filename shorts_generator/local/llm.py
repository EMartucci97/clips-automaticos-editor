"""Local LLM backend — OpenAI, Gemini, or Anthropic, selected by LLM_PROVIDER."""
from ..config import (
    ANTHROPIC_MODEL,
    ANTHROPIC_WORKSPACE_ID,
    GEMINI_MODEL,
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
    return response.text or ""


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
    return "".join(block.text for block in response.content if block.type == "text")


def call_local_llm(prompt: str) -> str:
    """Dispatch to the configured local LLM provider."""
    provider = (LLM_PROVIDER or "openai").strip().lower()
    if provider == "openai":
        return call_openai_llm(prompt)
    if provider == "gemini":
        return call_gemini_llm(prompt)
    if provider == "anthropic":
        return call_anthropic_llm(prompt)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'openai', 'gemini', or 'anthropic'."
    )
