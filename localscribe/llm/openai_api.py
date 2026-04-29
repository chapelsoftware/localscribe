"""OpenAI-compatible HTTP backend.

Speaks the OpenAI Chat Completions schema, which is the de facto standard
for LLM HTTP APIs. Works with:

  - OpenAI                (https://api.openai.com/v1)
  - Anthropic API         (https://api.anthropic.com/v1/openai-compat)
  - Ollama                (http://localhost:11434/v1)
  - LM Studio             (http://localhost:1234/v1)
  - vLLM, Together, Groq, OpenRouter, ...

Configured via env vars (or .env):
  OPENAI_API_KEY    Bearer token. Required for hosted services; can be
                    blank for local services like Ollama / LM Studio.
  OPENAI_BASE_URL   Default: https://api.openai.com/v1
  OPENAI_MODEL      Default: gpt-4o
"""
from __future__ import annotations

import httpx

from ..config import get_env
from . import LLMError


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o"


def ask(prompt: str, *, system: str | None = None, timeout: int = 900) -> str:
    api_key = get_env("OPENAI_API_KEY") or ""
    base_url = (get_env("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = get_env("OPENAI_MODEL") or DEFAULT_MODEL

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    url = f"{base_url}/chat/completions"
    try:
        resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
    except httpx.TimeoutException as e:
        raise LLMError(f"openai_api request to {url} timed out after {timeout}s") from e
    except httpx.HTTPError as e:
        raise LLMError(f"openai_api request to {url} failed: {e}") from e

    if resp.status_code != 200:
        raise LLMError(
            f"openai_api {resp.status_code} from {url} "
            f"(model={model}): {resp.text[:500]}"
        )

    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, ValueError, TypeError) as e:
        raise LLMError(
            f"openai_api: unexpected response shape from {url}: "
            f"{resp.text[:500]}"
        ) from e
