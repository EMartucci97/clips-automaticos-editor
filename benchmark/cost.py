"""Token usage + cost tracking for the benchmark's LLM grouping/cleanup passes.

llm.py's call_openai_llm/call_anthropic_llm (shorts_generator/local/llm.py)
return only the response text -- they never surface response.usage. Rather
than touch production code, this module duplicates their exact API-call
shape (same client construction, same params, confirmed by reading llm.py
before writing this) so the benchmark can capture real token usage from the
real response objects. The duplicated call is a handful of lines, so keeping
it in sync with llm.py if that file ever changes is a non-issue.

subtitles.py (shorts_generator/local/subtitles.py) does
`from .llm import call_local_llm` -- imported by value into subtitles.py's
own namespace. So to intercept every call subtitles.py makes, we must patch
`shorts_generator.local.subtitles.call_local_llm`, not
`shorts_generator.local.llm.call_local_llm` (patching the latter would have
zero effect on subtitles.py, which already holds its own reference).
"""
import subprocess
import threading
from contextlib import contextmanager
from typing import Dict

from shorts_generator.config import (
    ANTHROPIC_WORKSPACE_ID,
    require_anthropic_key,
    require_openai_key,
)
from shorts_generator.local.subtitles import FFPROBE_BIN

_local = threading.local()


def _accumulator() -> Dict[str, int]:
    if not hasattr(_local, "usage"):
        _local.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    return _local.usage


def reset_usage() -> None:
    _local.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def get_usage() -> Dict[str, int]:
    return dict(_accumulator())


def audio_duration_seconds(media_path: str) -> float:
    out = subprocess.run(
        [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", media_path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def _call_openai_with_usage(prompt: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=require_openai_key())
    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = _accumulator()
    if response.usage:
        usage["input_tokens"] += response.usage.prompt_tokens
        usage["output_tokens"] += response.usage.completion_tokens
    usage["calls"] += 1
    return response.choices[0].message.content or ""


def _call_anthropic_with_usage(prompt: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=require_anthropic_key())
    extra_headers = {"anthropic-workspace-id": ANTHROPIC_WORKSPACE_ID} if ANTHROPIC_WORKSPACE_ID else None
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
        extra_headers=extra_headers,
    )
    usage = _accumulator()
    if response.usage:
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
    usage["calls"] += 1
    return "".join(block.text for block in response.content if block.type == "text")


@contextmanager
def track_llm_usage(provider: str, model: str):
    """Patches subtitles.call_local_llm (see module docstring for why that
    specific name) to dispatch to our usage-capturing reimplementation for
    the given provider/model, for the duration of the `with` block. Resets
    the accumulator on entry so nested/sequential calls don't leak into each
    other's totals."""
    import shorts_generator.local.subtitles as subtitles_mod

    reset_usage()
    original = subtitles_mod.call_local_llm

    def patched(prompt: str) -> str:
        if provider == "openai":
            return _call_openai_with_usage(prompt, model)
        if provider == "anthropic":
            return _call_anthropic_with_usage(prompt, model)
        raise ValueError(f"Unknown provider {provider!r}")

    subtitles_mod.call_local_llm = patched
    try:
        yield
    finally:
        subtitles_mod.call_local_llm = original
