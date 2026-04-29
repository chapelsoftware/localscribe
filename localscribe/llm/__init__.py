"""LLM backend dispatch.

Selects between four shipped backends:
  - claude_cli (default): shells out to the `claude` CLI; uses your
    Claude Pro/Max subscription. No API key.
  - codex_cli: shells out to the `codex` CLI; uses your ChatGPT
    Plus/Pro/Team subscription. No API key.
  - gemini_cli: shells out to the `gemini` CLI; uses your Google
    Gemini Advanced / Google AI Pro subscription. No API key.
  - openai_api: HTTP POST to any OpenAI-compatible Chat Completions
    endpoint -- OpenAI, Anthropic API, Ollama, vLLM, LM Studio, Groq,
    OpenRouter, etc.

Picked via LOCALSCRIBE_LLM_BACKEND (env var or .env). Default: claude_cli.
"""
from __future__ import annotations

from ..config import get_env


KNOWN_BACKENDS = ("claude_cli", "codex_cli", "gemini_cli", "openai_api")


class LLMError(RuntimeError):
    """Raised by any backend when the LLM call fails."""


def ask(prompt: str, *, system: str | None = None, timeout: int = 900) -> str:
    """Send a prompt to the configured LLM backend and return its text reply.

    The system prompt, if provided, is passed to the backend however
    that backend supports it. Backends without a dedicated system-prompt
    flag (codex_cli, gemini_cli) prepend it to the user prompt.
    """
    backend = (get_env("LOCALSCRIBE_LLM_BACKEND") or "claude_cli").strip()
    if backend == "claude_cli":
        from . import claude_cli
        return claude_cli.ask(prompt, system=system, timeout=timeout)
    if backend == "codex_cli":
        from . import codex_cli
        return codex_cli.ask(prompt, system=system, timeout=timeout)
    if backend == "gemini_cli":
        from . import gemini_cli
        return gemini_cli.ask(prompt, system=system, timeout=timeout)
    if backend == "openai_api":
        from . import openai_api
        return openai_api.ask(prompt, system=system, timeout=timeout)
    raise LLMError(
        f"Unknown LOCALSCRIBE_LLM_BACKEND={backend!r}. "
        f"Use one of: {', '.join(KNOWN_BACKENDS)}."
    )
