"""Pull the original video stream for clip extraction.

The main pipeline's `download` stage only pulls audio (16 kHz mono WAV) --
that's all transcribe + diarize need, and it's much smaller. For clip
generation we additionally need the original video at full resolution.

We download once per video and cache to `<output>/<id>/video.mp4`. yt-dlp
is reused for format selection + post-processing (the main pipeline
already declares it as a dependency).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from ..config import Paths, cached

log = logging.getLogger("clips.download")


def video_path(paths: Paths) -> Path:
    """Where the original video file lives. Always mp4."""
    return paths.root / "video.mp4"


def fetch_video(paths: Paths, force: bool = False) -> Path:
    """Download the original video for a video id that's already been
    through the main pipeline.

    Reads the source URL from the cached metadata.json. Local-file inputs
    can't be re-downloaded -- they raise.
    """
    out = video_path(paths)
    if cached(out, force):
        log.info("cached: %s", out.name)
        return out

    metadata = paths.metadata
    if not metadata.exists():
        raise FileNotFoundError(
            f"No metadata.json for {paths.video_id}; run the main "
            f"pipeline first (`localscribe <url>`)."
        )

    import json
    meta = json.loads(metadata.read_text())

    if meta.get("source_kind") == "local":
        # Local files: copy/symlink the original into video.mp4 so the
        # rest of the clip pipeline doesn't have to special-case it.
        src = Path(meta["source_path"])
        if not src.exists():
            raise FileNotFoundError(f"Original local file is gone: {src}")
        log.info("symlinking local source -> %s", out.name)
        if out.exists() or out.is_symlink():
            out.unlink()
        out.symlink_to(src.resolve())
        return out

    url = meta.get("webpage_url") or f"https://www.youtube.com/watch?v={paths.video_id}"
    log.info("fetching video for %s...", paths.video_id)

    import yt_dlp  # lazy

    ydl_opts = {
        # Pick best mp4-compatible video + audio, then merge to mp4.
        # Fallback "best" if no separate streams available.
        "format": (
            "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/"
            "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"
        ),
        "outtmpl": str(out.with_suffix("") ) + ".%(ext)s",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    cookies = os.environ.get("YOUTUBE_COOKIES")
    if cookies and os.path.exists(cookies):
        ydl_opts["cookiefile"] = cookies

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not out.exists():
        # yt-dlp might have written video.mkv if the merge fell back.
        for cand in paths.root.glob("video.*"):
            if cand.suffix.lower() in (".mp4", ".mkv", ".webm") and cand != out:
                log.info("renaming %s -> %s", cand.name, out.name)
                cand.rename(out)
                break

    if not out.exists():
        raise RuntimeError(f"yt-dlp finished but {out} is missing")

    log.info("ok: %s (%.1f MB)", out.name, out.stat().st_size / 1024 / 1024)
    return out
