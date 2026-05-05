"""Audio peak extraction for the clip-review timeline waveform.

Decodes a clip's audio to mono 8 kHz s16 LE PCM via ffmpeg, then
buckets samples into N peak bins (max-absolute per bin) returned as
0..1 floats. Cached as JSON next to the mp4 so the review server only
recomputes when the clip is re-rendered.
"""
from __future__ import annotations

import json
import logging
import struct
import subprocess
from pathlib import Path

log = logging.getLogger("clips.waveform")

DEFAULT_BINS = 600  # ~10 bins/second for a 60s clip


def compute_peaks(video_path: Path, n_bins: int = DEFAULT_BINS) -> list[float]:
    """Return `n_bins` audio peak values in [0, 1] for `video_path`."""
    if not video_path.exists():
        return []
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin",
                "-i", str(video_path),
                "-vn",
                "-ac", "1",
                "-ar", "8000",
                "-f", "s16le", "-",
            ],
            capture_output=True, check=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.warning("waveform extract failed for %s: %s", video_path.name, e)
        return []
    raw = proc.stdout
    n_samples = len(raw) // 2
    if n_samples == 0:
        return []

    samples_per_bin = max(1, n_samples // n_bins)
    bins: list[float] = []
    fmt = f"<{samples_per_bin}h"
    chunk_bytes = samples_per_bin * 2
    for i in range(n_bins):
        offset = i * chunk_bytes
        if offset + chunk_bytes > len(raw):
            break
        chunk = raw[offset:offset + chunk_bytes]
        vals = struct.unpack(fmt, chunk)
        peak = max(abs(v) for v in vals) / 32768.0
        bins.append(round(peak, 4))
    return bins


def get_or_compute_peaks(
    video_path: Path,
    cache_path: Path,
    n_bins: int = DEFAULT_BINS,
) -> list[float]:
    """Return cached peaks or compute + cache them.

    Cache invalidated when `cache_path` is older than `video_path` so a
    re-cut clip always gets a fresh waveform on first request.
    """
    if (
        cache_path.exists()
        and cache_path.stat().st_mtime >= video_path.stat().st_mtime
    ):
        try:
            cached = json.loads(cache_path.read_text())
            if isinstance(cached, list) and cached:
                return cached
        except (json.JSONDecodeError, OSError):
            pass
    peaks = compute_peaks(video_path, n_bins)
    if peaks:
        try:
            cache_path.write_text(json.dumps(peaks))
        except OSError:
            pass
    return peaks


def probe_clip(video_path: Path) -> dict:
    """Return `{"fps": float, "duration": float}` for a clip mp4."""
    if not video_path.exists():
        return {"fps": 30.0, "duration": 0.0}
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,duration",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(r.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        rfr = stream.get("r_frame_rate") or "30/1"
        num, _, den = rfr.partition("/")
        fps = float(num) / float(den) if (num and den and float(den)) else 30.0
        duration = float(stream.get("duration") or 0.0)
        return {"fps": round(fps, 3), "duration": round(duration, 3)}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            json.JSONDecodeError, ValueError):
        return {"fps": 30.0, "duration": 0.0}
