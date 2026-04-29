"""Stage 1: Get audio + metadata from either a YouTube URL or a local file.

- YouTube URL: downloaded via yt-dlp, converted to 16 kHz mono WAV.
- Local file (any ffmpeg-readable audio or video): converted to 16 kHz
  mono WAV via ffmpeg directly; metadata is synthesized from the filename
  and ffprobe.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..config import Paths, cached, save_json

log = logging.getLogger("download")


_YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _looks_like_local_path(s: str) -> bool:
    """Heuristic: does this input look like a local filesystem path?"""
    if s.startswith(("http://", "https://", "www.")):
        return False
    if s.startswith(("/", "~", ".")):
        return True
    # Bare filename with an extension we recognize
    p = Path(s)
    if p.suffix.lower() in {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus",
                             ".webm", ".mp4", ".mkv", ".mov", ".aac", ".wma"}:
        return True
    return False


def _local_file_id(path: Path) -> str:
    """Derive a stable, readable video_id from an absolute path.

    Same file -> same id across runs, so caching works.
    """
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", path.stem)[:24].strip("_") or "file"
    h = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
    return f"{stem}-{h}"


def _ffprobe_duration(path: Path) -> float:
    """Get duration (seconds) of an audio/video file via ffprobe."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            text=True,
        ).strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def _run_local_file(src_path: Path, force: bool) -> Paths:
    """Handle a local audio/video file: convert to wav + synth metadata."""
    if not src_path.exists():
        raise FileNotFoundError(f"Local file not found: {src_path}")

    video_id = _local_file_id(src_path)
    paths = Paths.for_video(video_id)

    if cached(paths.audio, force) and cached(paths.metadata, force):
        log.info("cached: %s (local: %s)", video_id, src_path.name)
        return paths

    log.info("converting local file %s -> %s...", src_path, paths.audio)
    # ffmpeg convert to 16 kHz mono WAV (same format whisper/pyannote want)
    subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-loglevel", "error",
         "-i", str(src_path),
         "-ac", "1", "-ar", "16000",
         str(paths.audio)],
        check=True,
    )

    duration = _ffprobe_duration(src_path)
    meta = {
        "video_id": video_id,
        "title": src_path.stem,
        "uploader": None,
        "channel": None,
        "upload_date": None,
        "duration": int(duration) if duration else None,
        "description": f"Local file: {src_path}",
        "chapters": None,
        "tags": None,
        "webpage_url": None,
        "source_kind": "local",
        "source_path": str(src_path.resolve()),
    }
    save_json(paths.metadata, meta)

    log.info("ok: %s (%ds)", paths.audio, int(duration))
    return paths


def extract_video_id(url: str) -> str:
    """Extract the 11-char YouTube video ID from any common URL form."""
    if _YT_ID_RE.match(url):
        return url
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        vid = parsed.path.lstrip("/")
        if _YT_ID_RE.match(vid):
            return vid
    if parsed.hostname and "youtube.com" in parsed.hostname:
        qs = parse_qs(parsed.query)
        if "v" in qs and _YT_ID_RE.match(qs["v"][0]):
            return qs["v"][0]
        # /shorts/<id> or /embed/<id>
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and _YT_ID_RE.match(parts[1]):
            return parts[1]
    raise ValueError(f"Could not extract YouTube video ID from: {url}")


def resolve_video_id(source: str) -> str:
    """Same id-resolution as run(), but pure: no I/O, no downloads.

    Useful for things like `localscribe --clean URL` that need to know
    which output dir to delete without re-running the download.
    """
    if _looks_like_local_path(source):
        return _local_file_id(Path(os.path.expanduser(source)))
    return extract_video_id(source)


def run(source: str, force: bool = False) -> Paths:
    """Get audio + metadata from a URL or local file path.

    Returns Paths with video_id set.
    """
    # Local file?
    if _looks_like_local_path(source):
        return _run_local_file(Path(os.path.expanduser(source)), force)

    # URL path (original behavior)
    url = source
    video_id = extract_video_id(url)
    paths = Paths.for_video(video_id)

    if cached(paths.audio, force) and cached(paths.metadata, force):
        log.info("cached: %s", video_id)
        return paths

    log.info("fetching %s...", video_id)

    import yt_dlp  # lazy -- heavy import

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(paths.root / "audio.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        # Downsample to 16kHz mono -- what whisper and pyannote both want
        "postprocessor_args": [
            "-ar", "16000",
            "-ac", "1",
        ],
    }

    # Optional: pass a Netscape-format cookies file to bypass age/region/bot
    # checks. Set YOUTUBE_COOKIES=/path/to/cookies.txt to enable.
    cookies = os.environ.get("YOUTUBE_COOKIES")
    if cookies and os.path.exists(cookies):
        ydl_opts["cookiefile"] = cookies

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # Slim the metadata down to what later stages need
    meta = {
        "video_id": video_id,
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "channel": info.get("channel"),
        "upload_date": info.get("upload_date"),
        "duration": info.get("duration"),
        "description": info.get("description"),
        "chapters": info.get("chapters"),
        "tags": info.get("tags"),
        "webpage_url": info.get("webpage_url"),
    }
    save_json(paths.metadata, meta)

    if not paths.audio.exists():
        raise RuntimeError(f"yt-dlp finished but {paths.audio} is missing")

    log.info("ok: %s", paths.audio)
    return paths
