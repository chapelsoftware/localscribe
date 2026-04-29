"""OpenAI Codex CLI backend.

Uses the `codex` CLI in headless mode. Auth via `codex login` against
your ChatGPT Plus/Pro/Team subscription (browser OAuth, no API key
required) -- or `codex login --with-api-key` to use API credits instead.

Codex has no dedicated system-prompt flag, so we prepend the system
instructions to the user prompt with explicit markers.

Known issue: codex 0.124.0 introduced a regression where `codex exec`
can silently crash with no output when stdio is detached from a TTY
(https://github.com/openai/codex/issues/19945). If you hit empty
returns, downgrade codex or switch backends.
"""
from __future__ import annotations

import shutil
import subprocess

from . import LLMError


def _bin() -> str:
    path = shutil.which("codex")
    if not path:
        raise LLMError(
            "`codex` CLI not found on PATH. Install with "
            "`npm install -g @openai/codex` and run `codex login`, "
            "or switch backends with LOCALSCRIBE_LLM_BACKEND="
            "openai_api / claude_cli."
        )
    return path


def ask(prompt: str, *, system: str | None = None, timeout: int = 900) -> str:
    if system:
        prompt = f"SYSTEM INSTRUCTIONS:\n{system}\n\nUSER:\n{prompt}"

    # `codex exec -` reads the prompt from stdin -- avoids argv size limits
    # on long transcripts, same as we do for claude.
    cmd = [_bin(), "exec", "-"]

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
        raise LLMError(f"codex CLI timed out after {timeout}s") from e

    if result.returncode != 0:
        raise LLMError(
            f"codex CLI exited {result.returncode}\nstderr:\n{result.stderr}"
        )
    out = result.stdout.strip()
    if not out:
        raise LLMError(
            "codex CLI returned empty output. If you're on codex 0.124.0+, "
            "this may be the known TTY-detach regression "
            "(https://github.com/openai/codex/issues/19945). Try downgrading "
            "or switch backends."
        )
    return out
