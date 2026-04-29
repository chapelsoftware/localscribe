"""Logging setup for the CLI.

Each pipeline stage owns a short-named logger ('download', 'transcribe',
'diarize', etc.) so the prefix in the rendered output stays terse:

    [transcribe] running on /path/to/audio.wav...
"""
from __future__ import annotations

import logging


def setup_logging(verbosity: int = 0) -> None:
    """Configure the root logger.

    verbosity = -1 -> WARNING (--quiet)
              =  0 -> INFO     (default)
              >= 1 -> DEBUG    (--verbose)
    """
    if verbosity <= -1:
        level = logging.WARNING
    elif verbosity == 0:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        format="[%(name)s] %(message)s",
        level=level,
        force=True,
    )
