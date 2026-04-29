"""Google Gemini CLI backend.

Uses the `gemini` CLI in headless mode. Auth via `gemini auth login`
against your personal Google account (Gemini Advanced / Google AI Pro
subscription). API-key auth via `GEMINI_API_KEY` env var also works.

Gemini has no dedicated system-prompt flag, so we prepend the system
instructions to the user prompt with explicit markers.
"""
from __future__ import annotations

import shutil
import subprocess

from . import LLMError


def _bin() -> str:
    path = shutil.which("gemini")
    if not path:
        raise LLMError(
            "`gemini` CLI not found on PATH. Install with "
            "`npm install -g @google/gemini-cli` and run `gemini auth`, "
            "or switch backends with LOCALSCRIBE_LLM_BACKEND="
            "openai_api / claude_cli."
        )
    return path


def ask(prompt: str, *, system: str | None = None, timeout: int = 900) -> str:
    if system:
        prompt = f"SYSTEM INSTRUCTIONS:\n{system}\n\nUSER:\n{prompt}"

    # `gemini` reads stdin in non-TTY environments, treating it as the
    # full prompt. Avoids argv size limits on long transcripts.
    cmd = [_bin()]

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise LLMError(f"gemini CLI timed out after {timeout}s") from e

    if result.returncode != 0:
        raise LLMError(
            f"gemini CLI exited {result.returncode}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()
