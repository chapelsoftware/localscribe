"""Claude CLI backend.

Uses the user's Claude Pro/Max subscription (not API credits) by invoking
the `claude` binary in headless mode with `-p` (print) and stdin input.
"""
from __future__ import annotations

import shutil
import subprocess

from . import LLMError


def _claude_bin() -> str:
    path = shutil.which("claude")
    if not path:
        raise LLMError(
            "`claude` CLI not found on PATH. Install Claude Code "
            "(https://claude.com/claude-code), or switch backends with "
            "LOCALSCRIBE_LLM_BACKEND=openai_api."
        )
    return path


def ask(prompt: str, *, system: str | None = None, timeout: int = 900) -> str:
    """Run `claude -p` with the prompt on stdin, return stdout.

    The prompt is passed via stdin to avoid argv size limits on long
    transcripts.
    """
    cmd = [_claude_bin(), "-p"]
    if system:
        cmd.extend(["--append-system-prompt", system])

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
        raise LLMError(f"claude CLI timed out after {timeout}s") from e

    if result.returncode != 0:
        raise LLMError(
            f"claude CLI exited {result.returncode}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()
